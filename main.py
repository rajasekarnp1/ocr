"""
Main entry point for the OCR-X application (conceptual).

This script demonstrates the initialization and use of the OCRWorkflowOrchestrator
to process a dummy image using a dummy OCR engine.
"""

import logging
import os
import sys
import yaml # PyYAML must be installed
from pathlib import Path
from typing import Dict, Any

# --- Path Setup ---
# Ensure project root is in sys.path for direct module imports
# This assumes main.py is in the project root.
# If it's in a 'src' or 'app' subdirectory, this might need adjustment
# or preferably, the project is structured as an installable package.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Setup ---

try:
    from ocr_workflow_orchestrator import OCRWorkflowOrchestrator
    from config_loader import load_config, DEFAULT_LOGGING_CONFIG # load_config also handles logging setup
    from PIL import Image # Pillow must be installed
except ImportError as e:
    print(f"Error: Missing critical dependencies. Please ensure all required packages are installed. Details: {e}")
    print("You might need to run: pip install PyYAML Pillow Pillow") # Pillow is for PIL
    sys.exit(1)

# --- Global Variables ---
CONFIG_FILE_NAME = "temp_main_config.yaml"
DUMMY_IMAGE_NAME = "temp_dummy_image.png"

# Logger will be configured by load_config() called within main()
logger: logging.Logger


def create_dummy_config_file(config_path: Path) -> None:
    """Creates a dummy YAML configuration file for demonstration."""
    global logger
    logger.info(f"Creating dummy configuration file at: {config_path}")
    
    # Ensure the stubs directory is correctly referenced relative to project_root
    # For dynamic module loading, the path needs to be understood by importlib
    # Assuming 'stubs' is a top-level directory alongside 'ocr_workflow_orchestrator.py' etc.
    # If main.py is at project_root, then "stubs.dummy_engine" is correct.
    
    dummy_config_content: Dict[str, Any] = {
        "app_settings": {
            "default_ocr_engine": "dummy_local_engine", # Setting a default for testing
            "some_other_setting": "value123"
        },
        "ocr_engines": {
            "dummy_local_engine": {
                "enabled": True,
                "module": "stubs.dummy_engine",  # Path for importlib
                "class": "DummyLocalEngine",
                "name": "My Main Test Dummy Engine", # Custom name for the engine instance
                "config": { # Engine-specific configuration map
                    "model_path": "models/main_dummy_model.onnx",
                    "custom_param": "engine_specific_value"
                }
            },
            "another_engine_example": {
                "enabled": False, # This engine won't be loaded
                "module": "stubs.dummy_engine",
                "class": "DummyLocalEngine",
                "name": "My Disabled Dummy",
                "config": {"model_path": "models/another.onnx"}
            }
        },
        "logging": DEFAULT_LOGGING_CONFIG.copy() # Start with default logging
    }
    # Optionally customize logging for the main example
    if "loggers" in dummy_config_content["logging"]:
        dummy_config_content["logging"]["loggers"]["MainApp"] = {
            "handlers": ["console"], "level": "INFO", "propagate": False
        }
        dummy_config_content["logging"]["loggers"]["OCRWorkflowOrchestrator"] = {
             "handlers": ["console"], "level": "INFO", "propagate": False
        }
        dummy_config_content["logging"]["loggers"]["OCREngineManager"] = {
             "handlers": ["console"], "level": "INFO", "propagate": False
        }
        dummy_config_content["logging"]["loggers"]["stubs.dummy_engine"] = { # Configure stub logger
             "handlers": ["console"], "level": "INFO", "propagate": False
        }


    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(dummy_config_content, f)
    logger.debug(f"Dummy configuration content: \n{yaml.dump(dummy_config_content)}")


def create_dummy_image_file(image_path: Path) -> None:
    """Creates a simple dummy PNG image file using Pillow."""
    global logger
    logger.info(f"Creating dummy image file at: {image_path}")
    try:
        img = Image.new('RGB', (200, 100), color = 'skyblue')
        # You could add text to the image if you had a font file and wanted more complex dummy data
        # from PIL import ImageDraw, ImageFont
        # draw = ImageDraw.Draw(img)
        # try:
        #     font = ImageFont.truetype("arial.ttf", 15) # Or some other common font
        # except IOError:
        #     font = ImageFont.load_default()
        # draw.text((10,10), "Hello OCR-X", fill=(0,0,0), font=font)
        img.save(image_path, "PNG")
        logger.debug(f"Dummy image '{image_path}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create dummy image using Pillow: {e}. Please ensure Pillow is installed.", exc_info=True)
        # Create an empty file to avoid FileNotFoundError later if Pillow fails,
        # though the orchestrator's image loading will likely fail then.
        if not image_path.exists():
            image_path.touch()


def main():
    """
    Main function to demonstrate OCR-X workflow.
    """
    # --- Setup Logging ---
    # Load config which also initializes logging.
    # We pass a path, but it might not exist yet; load_config handles this.
    # For the main app, we usually expect config_loader to find or create the config.
    # Here, we explicitly create it for the demo.
    
    # Get a preliminary logger based on defaults before load_config fully configures it.
    # This ensures that logging for create_dummy_config_file itself is captured.
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
    global logger # Make it accessible to helper functions
    logger = logging.getLogger("MainApp") # Use a specific name for this main script's logger
    
    config_file_path = project_root / CONFIG_FILE_NAME
    dummy_image_file_path = project_root / DUMMY_IMAGE_NAME

    try:
        # 1. Create Dummy Configuration
        create_dummy_config_file(config_file_path)

        # 2. Create Dummy Image
        create_dummy_image_file(dummy_image_file_path)

        # 3. Instantiate Orchestrator
        # The orchestrator will load the config (and re-configure logging if specified in the file)
        logger.info("Initializing OCRWorkflowOrchestrator...")
        orchestrator = OCRWorkflowOrchestrator(config_path=str(config_file_path))
        
        # Re-assign logger in case load_config within orchestrator changed its config
        logger = logging.getLogger("MainApp") 

        # 4. Log Available Engines
        available_engines = orchestrator.engine_manager.get_available_engines()
        logger.info(f"Available OCR engines after orchestrator init: {available_engines}")

        if not available_engines:
            logger.error("No OCR engines were loaded by the orchestrator. Cannot proceed with OCR.")
            return

        # 5. Process Document
        # Test with the default engine specified in the dummy config
        logger.info(f"\n--- Processing document '{DUMMY_IMAGE_NAME}' with default engine logic ---")
        result_default = orchestrator.process_document(str(dummy_image_file_path), language_hint="en")

        if result_default and "error" not in result_default:
            logger.info(f"OCR Result (default engine): Text = '{result_default.get('text')}'")
            logger.info(f"Full result (default engine): {result_default}")
        else:
            logger.error(f"OCR processing failed or returned an error (default engine): {result_default}")

        # Test with a specifically requested (and available) engine
        if "dummy_local_engine" in available_engines:
            logger.info(f"\n--- Processing document '{DUMMY_IMAGE_NAME}' with requested engine 'dummy_local_engine' ---")
            result_specific = orchestrator.process_document(str(dummy_image_file_path), requested_engine_name="dummy_local_engine", language_hint="de")
            if result_specific and "error" not in result_specific:
                logger.info(f"OCR Result (specific engine): Text = '{result_specific.get('text')}'")
                logger.info(f"Full result (specific engine): {result_specific}")
            else:
                logger.error(f"OCR processing failed or returned an error (specific engine): {result_specific}")
        else:
            logger.warning("Skipping specific engine test as 'dummy_local_engine' is not available.")

    except FileNotFoundError as e:
        logger.error(f"File not found during main execution: {e}", exc_info=True)
    except ImportError as e: # Catching potential import errors from dynamic loading
        logger.error(f"Import error during main execution, possibly an engine module: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        # 6. Cleanup
        logger.info("Cleaning up temporary files...")
        if config_file_path.exists():
            try:
                os.remove(config_file_path)
                logger.info(f"Removed dummy config file: {config_file_path}")
            except Exception as e:
                logger.error(f"Error removing dummy config file '{config_file_path}': {e}")
        
        if dummy_image_file_path.exists():
            try:
                os.remove(dummy_image_file_path)
                logger.info(f"Removed dummy image file: {dummy_image_file_path}")
            except Exception as e:
                logger.error(f"Error removing dummy image file '{dummy_image_file_path}': {e}")
        
        log_file_path = project_root / "ocrx_app_debug.log" # From default logging config
        if log_file_path.exists():
            logger.info(f"Note: A log file may have been created at '{log_file_path}' if file logging was enabled.")


if __name__ == '__main__':
    # Basic logging setup for the very start, will be overridden by load_config in main()
    # This ensures that if main() itself has an early issue, it's logged.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger_bootstrap = logging.getLogger("BootstrapMain") # Different name to avoid conflict
    main_logger_bootstrap.info("Starting OCR-X main application...")
    main()
    main_logger_bootstrap.info("OCR-X main application finished.")

```
The `main.py` file has been created with the specified structure and functionality.

**Key features implemented:**

1.  **Path Setup:**
    *   The project root is determined using `Path(__file__).resolve().parent`.
    *   This project root is added to `sys.path` to ensure that modules like `ocr_workflow_orchestrator`, `config_loader`, `ocr_engine_interface`, and the `stubs` directory are correctly importable when `main.py` is run directly from the project root.

2.  **Imports:**
    *   Imports `OCRWorkflowOrchestrator`, `load_config`, `DEFAULT_LOGGING_CONFIG`, `logging`, `os`, `sys`, `yaml`, `pathlib.Path`, `typing.Dict`, `typing.Any`, and `PIL.Image`.
    *   Includes a try-except block for initial imports to provide a user-friendly message if critical dependencies like PyYAML or Pillow are missing.

3.  **Global Variables:**
    *   `CONFIG_FILE_NAME` ("temp\_main\_config.yaml") and `DUMMY_IMAGE_NAME` ("temp\_dummy\_image.png") are defined for clarity.
    *   A global `logger` variable is declared, to be initialized within `main()`.

4.  **`create_dummy_config_file(config_path: Path)` Function:**
    *   Creates a YAML configuration file (`temp_main_config.yaml`) programmatically.
    *   The configuration includes:
        *   `app_settings` with a `default_ocr_engine` set to "dummy\_local\_engine".
        *   `ocr_engines` section defining `dummy_local_engine` (enabled, pointing to `stubs.dummy_engine.DummyLocalEngine`, with a custom name and specific config parameters) and a disabled engine example.
        *   A `logging` section initialized from `DEFAULT_LOGGING_CONFIG` and then customized to set INFO level for key application loggers (`MainApp`, `OCRWorkflowOrchestrator`, `OCREngineManager`, `stubs.dummy_engine`) for the demonstration.

5.  **`create_dummy_image_file(image_path: Path)` Function:**
    *   Uses `PIL.Image` to create a simple, small sky-blue PNG image (`temp_dummy_image.png`).
    *   Includes basic error handling in case Pillow fails to create the image.

6.  **`main()` Function:**
    *   **Logging Setup:** Initializes a global `logger` named "MainApp" using `DEFAULT_LOGGING_CONFIG` *before* creating dummy files, ensuring that the creation process itself is logged. The `OCRWorkflowOrchestrator` will later re-apply logging based on the content of the dummy config file.
    *   **Dummy File Creation:** Calls `create_dummy_config_file()` and `create_dummy_image_file()`.
    *   **Orchestrator Instantiation:** Creates an instance of `OCRWorkflowOrchestrator`, passing the path to the dummy config file. The logger is re-obtained after this step.
    *   **Log Available Engines:** Retrieves and logs the list of available engines from the `engine_manager`.
    *   **Process Document:**
        *   Calls `orchestrator.process_document()` first using the default engine logic (since `default_ocr_engine` is set in the dummy config).
        *   Then, calls `orchestrator.process_document()` again, explicitly requesting the "dummy\_local\_engine".
        *   Language hints (`"en"`, `"de"`) are passed for demonstration.
    *   **Print Result:** Logs the text result or an error message based on the outcome.
    *   **Cleanup:** Includes a `finally` block to ensure the dummy config and image files are deleted. It also logs a message about the potential creation of `ocrx_app_debug.log` if file logging was active.
    *   **Error Handling:** A main `try...except...finally` block wraps the core logic to catch and log major exceptions (`FileNotFoundError`, `ImportError`, generic `Exception`).

7.  **`if __name__ == '__main__':` Block:**
    *   Sets up a very basic bootstrap logger.
    *   Calls the `main()` function.
    *   Logs start and finish messages.

This script should now provide a runnable example demonstrating the core orchestration logic, dynamic engine loading (via the dummy engine), and configuration/logging setup. It creates its own temporary dependencies (config and image) and cleans them up.
