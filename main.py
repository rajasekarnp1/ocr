import logging
import logging.config
import os
import shutil # For directory cleanup
import numpy as np
import cv2 # For creating dummy image
import yaml # For loading simple config if config_loader is not used directly

# Assuming these modules are in PYTHONPATH or same directory level
from config_loader import load_config # Use the project's config loader
from ocr_workflow_orchestrator import OCRWorkflowOrchestrator

# Global logger for this test script
logger = logging.getLogger(__name__)

# --- Test Configuration & Setup ---
TEST_CONFIG_FILE = "test_engines_config.yaml"
DUMMY_MODEL_DIR = "models_test" # Relative to script execution
DUMMY_DET_MODEL_FILE = os.path.join(DUMMY_MODEL_DIR, "dummy_det.onnx")
DUMMY_REC_MODEL_FILE = os.path.join(DUMMY_MODEL_DIR, "dummy_rec.onnx")
DUMMY_CHARS_FILE = os.path.join(DUMMY_MODEL_DIR, "dummy_chars.txt")
DUMMY_IMAGE_FILE = "dummy_integration_test_image.png"

# Google Cloud API Key Path - User should set this environment variable for live test
# OR update test_engines_config.yaml directly (but env var is safer for shared code)
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_TEST_API_KEY_PATH"

def setup_dummy_files_for_local_engine():
    """Creates dummy directories and files required by LocalOCREngine config."""
    logger.info(f"Setting up dummy files in {DUMMY_MODEL_DIR}...")
    os.makedirs(DUMMY_MODEL_DIR, exist_ok=True)

    # Create empty dummy files (or with minimal valid content if needed by file readers)
    with open(DUMMY_DET_MODEL_FILE, 'w') as f: f.write("dummy_onnx_det_content")
    with open(DUMMY_REC_MODEL_FILE, 'w') as f: f.write("dummy_onnx_rec_content")
    with open(DUMMY_CHARS_FILE, 'w') as f:
        f.write("a\n"); f.write("b\n"); f.write("c\n"); # Minimal char set
    logger.info("Dummy files created.")

def cleanup_dummy_files():
    """Removes dummy files and directories created during setup."""
    logger.info("Cleaning up dummy files and directories...")
    if os.path.exists(DUMMY_IMAGE_FILE):
        os.remove(DUMMY_IMAGE_FILE)
        logger.debug(f"Removed {DUMMY_IMAGE_FILE}")
    if os.path.exists(DUMMY_MODEL_DIR):
        shutil.rmtree(DUMMY_MODEL_DIR) # Recursively remove directory
        logger.debug(f"Removed directory {DUMMY_MODEL_DIR}")
    logger.info("Cleanup complete.")

def create_dummy_image():
    """Creates a dummy image file for testing."""
    logger.info(f"Creating dummy image: {DUMMY_IMAGE_FILE}...")
    image = np.full((200, 400, 3), (220, 220, 220), dtype=np.uint8) # Light gray background
    cv2.putText(image, "OCR Test Text", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
    cv2.putText(image, "Engine Integration", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.imwrite(DUMMY_IMAGE_FILE, image)
    logger.info("Dummy image created.")
    return DUMMY_IMAGE_FILE

def main():
    """
    Main integration test script for OCR Workflow Orchestrator and OCR Engines.
    """
    # 1. Load Configuration (includes logging setup from config file)
    try:
        config = load_config(TEST_CONFIG_FILE) # This also applies logging config
        logger.info(f"Successfully loaded configuration from {TEST_CONFIG_FILE}")
    except FileNotFoundError:
        logger.error(f"FATAL: Test configuration file '{TEST_CONFIG_FILE}' not found. Aborting.")
        return
    except Exception as e:
        logger.error(f"FATAL: Error loading configuration: {e}. Aborting.", exc_info=True)
        return

    # 2. Setup Dummy Files for Local Engine
    # These paths are hardcoded here but match test_engines_config.yaml
    # LocalOCREngine's initialize() will try to load these.
    # If ONNX runtime fails to load them as valid models, LocalOCREngine should report itself as unavailable.
    setup_dummy_files_for_local_engine()

    # 3. Handle Google Cloud API Key for Google Engine
    # The config file has a placeholder path. Override if env var is set.
    google_engine_config = None
    try:
        # Find the Google engine config to potentially update its api_key_path
        for name, cfg_dict in config.get('ocr_engines', {}).items():
            if cfg_dict.get('class') == 'GoogleCloudOCREngine':
                google_engine_config = cfg_dict['config'] # Get the nested 'config' dictionary
                break
    except AttributeError as e: # Handle if config is not a dict
        logger.error(f"Configuration 'ocr_engines' is not a dictionary or is malformed: {e}")
        google_engine_config = {} # Ensure it's a dict to avoid further errors

    # Ensure google_engine_config is a dictionary before trying to access/modify it
    if not isinstance(google_engine_config, dict):
        logger.warning("Google Cloud Vision engine configuration not found or not a dictionary in test_engines_config.yaml.")
        google_engine_config = {} # Initialize to empty dict to prevent errors below

    # Check environment variable for actual key path
    actual_google_api_key_path = os.getenv(GOOGLE_API_KEY_ENV_VAR)
    skip_google_tests = False

    if actual_google_api_key_path and os.path.exists(actual_google_api_key_path):
        logger.info(f"Using Google API key from environment variable {GOOGLE_API_KEY_ENV_VAR}: {actual_google_api_key_path}")
        if google_engine_config is not None: # Should exist if config structure is correct
             google_engine_config['api_key_path'] = actual_google_api_key_path
        else: # Should not happen if config is correct
             logger.warning("Google engine 'config' section not found to update api_key_path.")
    else:
        # Check if the path in the config file (placeholder or user-set) exists
        config_google_key_path = google_engine_config.get('api_key_path', "path/to/your/service_account_key.json")
        if os.path.exists(config_google_key_path) and config_google_key_path != "path/to/your/service_account_key.json":
            logger.info(f"Using Google API key from config file: {config_google_key_path}")
            # No change needed to config dict if path in file is valid and not placeholder
        else:
            logger.warning(f"Google API key path not found or is placeholder ('{config_google_key_path}'). "
                           f"Set the {GOOGLE_API_KEY_ENV_VAR} environment variable or update config to run Google Cloud tests.")
            skip_google_tests = True


    # 4. Create Dummy Image for Processing
    dummy_image_path = create_dummy_image()

    # 5. Instantiate OCRWorkflowOrchestrator
    logger.info("Initializing OCRWorkflowOrchestrator...")
    try:
        orchestrator = OCRWorkflowOrchestrator(config)
    except Exception as e:
        logger.error(f"Failed to initialize OCRWorkflowOrchestrator: {e}", exc_info=True)
        cleanup_dummy_files()
        return
    logger.info("OCRWorkflowOrchestrator initialized.")

    # 6. Test Engine Availability
    available_engines = orchestrator.engine_manager.get_available_engines()
    logger.info(f"Available OCR engines: {list(available_engines.keys())}")

    # Assertions for availability (depends on dummy files and keys)
    # Local engine *might* be available if it considers empty files sufficient for init structure,
    # but its recognize() would likely fail if ONNX models are not valid.
    # Or, it might correctly report unavailable if ONNX session creation fails with dummy files.
    if "PaddleOCR ONNX (Test)" in available_engines:
        logger.info("LocalOCREngine (PaddleOCR ONNX (Test)) reported as available.")
    else:
        logger.warning("LocalOCREngine (PaddleOCR ONNX (Test)) reported as NOT available. This might be expected if dummy ONNX files cause its initialization to fail.")

    if not skip_google_tests:
        if "Google Cloud Vision API (Test)" in available_engines:
            logger.info("GoogleCloudOCREngine (Google Cloud Vision API (Test)) reported as available.")
        else:
            logger.error("GoogleCloudOCREngine (Google Cloud Vision API (Test)) reported as NOT available, but key was provided/found.")
    else:
        if "Google Cloud Vision API (Test)" in available_engines:
             logger.error("GoogleCloudOCREngine reported as available, but tests were marked to be skipped (key likely missing initially). This is unexpected.")
        else:
             logger.info("GoogleCloudOCREngine tests are skipped as API key is not available.")


    # 7. Test Processing with Local Engine
    logger.info("\n--- Testing with Local Engine (PaddleOCR ONNX (Test)) ---")
    try:
        # Even if "available", recognition might fail if dummy ONNX files are not loadable by runtime
        result_local = orchestrator.process_document(dummy_image_path, requested_engine_name="local_paddle_ocr")
        logger.info(f"Result from Local Engine: {result_local}")
        assert isinstance(result_local, dict), "Local engine result should be a dict."
        assert "text" in result_local, "Local engine result should contain 'text' key."
        # Further assertions could check if segments are empty if dummy models can't be processed.
    except RuntimeError as e: # Orchestrator raises RuntimeError if engine not available or processing fails
        logger.error(f"Error processing with Local Engine: {e}")
        if "not available" in str(e).lower():
             logger.info("This error is expected if the local engine's dummy models made it unavailable.")
    except Exception as e:
        logger.error(f"Unexpected error during local engine processing test: {e}", exc_info=True)


    # 8. Test Processing with Google Cloud Engine (Conditional)
    if not skip_google_tests:
        logger.info("\n--- Testing with Google Cloud Engine (Google Cloud Vision API (Test)) ---")
        try:
            result_google = orchestrator.process_document(dummy_image_path, requested_engine_name="google_cloud_vision", language_hint="en")
            logger.info(f"Result from Google Cloud Engine: {result_google.get('text', '')[:200]}...") # Print start of text
            if result_google.get("segments"): logger.info(f"First segment from Google: {result_google['segments'][0]}")
            assert isinstance(result_google, dict), "Google engine result should be a dict."
            assert "text" in result_google, "Google engine result should contain 'text' key."
            if result_google.get("error"):  # Check if API returned an error message
                logger.warning(f"Google Cloud API returned an error: {result_google['error']}")
        except RuntimeError as e: # Orchestrator raises RuntimeError if engine not available or processing fails
            logger.error(f"Error processing with Google Cloud Engine: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Google Cloud engine processing test: {e}", exc_info=True)
    else:
        logger.info("\n--- Skipping Google Cloud Engine Test (API key not configured) ---")

    # 9. Test Default Engine Processing
    logger.info("\n--- Testing with Default Engine ---")
    default_engine_name = config.get("app_settings", {}).get("default_ocr_engine")
    if default_engine_name == "google_cloud_vision" and skip_google_tests:
        logger.warning(f"Default engine is Google Cloud, but tests are skipped. Cannot test default engine processing.")
    else:
        try:
            logger.info(f"Attempting processing with default engine: {default_engine_name}")
            result_default = orchestrator.process_document(dummy_image_path)
            logger.info(f"Result from Default Engine ({result_default.get('engine_name')}): {result_default.get('text', '')[:100]}...")
            assert isinstance(result_default, dict), "Default engine result should be a dict."
            assert "text" in result_default, "Default engine result should contain 'text' key."
            assert result_default.get("engine_name", "").lower().startswith(default_engine_name.split('_')[0].lower()) # Check if correct engine was used
        except RuntimeError as e:
            logger.error(f"Error processing with Default Engine: {e}")
            if default_engine_name == "local_paddle_ocr" and "not available" in str(e).lower():
                logger.info("This error is expected for local engine if dummy models made it unavailable.")
        except Exception as e:
            logger.error(f"Unexpected error during default engine processing test: {e}", exc_info=True)


    # 10. Cleanup
    cleanup_dummy_files()
    logger.info("\nIntegration Test Run Finished.")

if __name__ == "__main__":
    # Setup basic logging if this script is run directly and config doesn't load/setup logging
    if not logging.getLogger().hasHandlers(): # Check if logging is already configured
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler()])
    main()
```
