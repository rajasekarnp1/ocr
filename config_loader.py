import yaml
import json
import logging
import logging.config
import os
import sys # For basic stdout logging if config fails catastrophically

# Assuming custom_exceptions.py is in the same directory or PYTHONPATH
try:
    from custom_exceptions import OCRConfigurationError, OCRFileNotFoundError
except ImportError:
    # Fallback if custom_exceptions.py is not available during early setup phase
    # This allows the module to be imported, but custom exceptions won't be used here.
    class OCRConfigurationError(Exception): pass
    class OCRFileNotFoundError(FileNotFoundError): pass


# Define a default logging configuration in case the file is missing or incomplete
# This is applied if the config file is missing or its logging section is invalid.
DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO', # Default level for console
        },
        # Example: File handler (can be added to config.yaml)
        # 'file': {
        #     'class': 'logging.FileHandler',
        #     'formatter': 'standard',
        #     'filename': 'ocrx_app.log',
        #     'level': 'DEBUG',
        # }
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG', # Default root level if not specified by user config
    },
    'loggers': { # Example: ensure orchestrator and main CLI have a decent default
        'OCRWorkflowOrchestrator': {
            'handlers': ['console'],
            'level': 'INFO', # Default for orchestrator
            'propagate': False
        },
        'OCRMain': { # Logger for main.py
            'handlers': ['console'],
            'level': 'INFO', # Default for CLI messages
            'propagate': False
        }
    }
}

# Get a logger for this module itself
logger = logging.getLogger(__name__)

def create_default_config_if_not_exists(config_path="config_dev.yaml"):
    """
    Creates a default configuration file if it doesn't already exist.
    This function now uses the module-level logger.
    """
    if not os.path.exists(config_path):
        logger.info(f"Configuration file '{config_path}' not found. Creating a default one.")
        default_content = {
            "app_settings": {
                "version": "0.0.1-default", # Added app_settings to default
                "default_output_format": "txt",
                "model_paths": {
                    "geometric_corrector": "models/dummy_geometric_model.onnx", # Updated path
                    "svtr_recognizer": "models/dummy_recognition_model.onnx",
                    # Keeping other potential model paths for completeness
                    "paddle_ocr_det": "models/paddle_det_v4.onnx",
                    "paddle_ocr_rec": "models/paddle_rec_v4_en.onnx",
                    "nlp_model_path": "models/dummy_nlp_model.onnx" # Used by postprocessing_module's old __main__
                },
                "performance": {
                    "use_directml": True,
                    "onnx_intra_op_threads": 0
                },
                # Added postprocessing_settings for TextCleaner whitelist and SpellCorrector dictionary
                "postprocessing_settings": {
                    "whitelist_chars": None,
                    "dictionary_path": "default_dict.txt"
                },
                # Added deskewer_settings (can be empty to use defaults in ImageDeskewer)
                "deskewer_settings": {
                    "angle_threshold_degrees": 0.5,
                    "min_contour_area_ratio": 0.01,
                    "max_contour_area_ratio": 0.95
                }
            },
            "logging": DEFAULT_LOGGING_CONFIG
        }
        # Ensure the /app/models directory exists for the default config
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created directory: {models_dir} for default ONNX models.")

        # Ensure dummy model files are created if they don't exist when creating default config
        geom_model_default_path = default_content["app_settings"]["model_paths"]["geometric_corrector"]
        if not os.path.exists(geom_model_default_path):
            try:
                with open(geom_model_default_path, "w") as f: f.write("dummy geometric model placeholder content")
                logger.warning(f"Created placeholder file for missing {geom_model_default_path}. Run model generator for actual ONNX.")
            except IOError as e:
                logger.error(f"Failed to create placeholder for {geom_model_default_path}: {e}", exc_info=True)


        rec_model_default_path = default_content["app_settings"]["model_paths"]["svtr_recognizer"]
        if not os.path.exists(rec_model_default_path):
            try:
                with open(rec_model_default_path, "w") as f: f.write("dummy recognition model placeholder content")
                logger.warning(f"Created placeholder file for missing {rec_model_default_path}. Run model generator for actual ONNX.")
            except IOError as e:
                 logger.error(f"Failed to create placeholder for {rec_model_default_path}: {e}", exc_info=True)

        postproc_nlp_model_path = default_content["app_settings"]["model_paths"].get("nlp_model_path", "models/dummy_nlp_model.onnx")
        if not os.path.exists(postproc_nlp_model_path):
            nlp_dir = os.path.dirname(postproc_nlp_model_path)
            if nlp_dir and not os.path.exists(nlp_dir):
                try:
                    os.makedirs(nlp_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"Failed to create directory {nlp_dir} for NLP model: {e}", exc_info=True)
            try:
                with open(postproc_nlp_model_path, "w") as f: f.write("dummy nlp model placeholder content")
                logger.warning(f"Created placeholder file for missing {postproc_nlp_model_path}.")
            except IOError as e:
                logger.error(f"Failed to create placeholder for {postproc_nlp_model_path}: {e}", exc_info=True)

        default_dict_file_path = default_content["app_settings"]["postprocessing_settings"]["dictionary_path"]
        if default_dict_file_path and not os.path.exists(default_dict_file_path):
            logger.warning(f"Default dictionary '{default_dict_file_path}' not found. Creating a minimal one.")
            try:
                with open(default_dict_file_path, "w", encoding='utf-8') as f:
                    f.write("the\na\nis\ntext\nimage\nocr\npython\ncode\nexample\n")
            except IOError as e:
                 logger.error(f"Failed to create minimal default dictionary at {default_dict_file_path}: {e}", exc_info=True)
        try:
            with open(config_path, "w") as f:
                yaml.dump(default_content, f, sort_keys=False)
            logger.info(f"Default configuration file created at '{config_path}'.")
        except IOError as e:
            logger.error(f"Failed to write default configuration file to '{config_path}': {e}", exc_info=True)
            # If writing default config fails, subsequent operations depending on it will likely fail.

def load_config(config_path: str = "config_dev.yaml") -> dict:
    """
    Loads configuration from a YAML or JSON file.
    Sets up logging based on the configuration.
    If config_path is None, uses "config_dev.yaml".
    """
    actual_config_path = config_path if config_path else "config_dev.yaml"

    # Ensure default config_dev.yaml exists if it's the target and is missing
    if actual_config_path == "config_dev.yaml" and not os.path.exists(actual_config_path):
        try:
            create_default_config_if_not_exists(actual_config_path)
        except Exception as e_create: # Catch errors during default config creation
            # Log to basic stdout/stderr if full logging isn't up yet
            print(f"CRITICAL: Failed to create default config '{actual_config_path}': {e_create}", file=sys.stderr)
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # Basic fallback
            logger.critical(f"Failed to create default config '{actual_config_path}': {e_create}", exc_info=True)
            # Depending on policy, either raise or return minimal config for app to potentially limp along / fail later
            raise OCRConfigurationError(f"Failed to create default config '{actual_config_path}'") from e_create


    config_data = None
    try:
        if not os.path.exists(actual_config_path):
            logger.warning(f"Configuration file '{actual_config_path}' not found. Attempting to use default logging and minimal config.")
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG) # Apply default logging
            logger.info("Applied default logging configuration as config file was not found.")
            # Raise OCRFileNotFoundError to be caught by main.py for user-friendly message
            raise OCRFileNotFoundError(f"Configuration file '{actual_config_path}' not found.")

        with open(actual_config_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            if not file_content.strip(): # Check if file is empty or only whitespace
                 logger.warning(f"Configuration file '{actual_config_path}' is empty. Using default logging and minimal config.")
                 logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
                 return {"app_settings": {"default_setting": True, "config_status": "empty_config_loaded"}, "logging": DEFAULT_LOGGING_CONFIG}

            # Reset file pointer to beginning after reading for emptiness check
            f.seek(0)
            if actual_config_path.endswith((".yaml", ".yml")):
                config_data = yaml.safe_load(f)
            elif actual_config_path.endswith(".json"):
                config_data = json.load(f)
            else: # Try YAML for unknown extensions
                logger.info(f"Attempting to load '{actual_config_path}' as YAML due to unknown/missing extension.")
                config_data = yaml.safe_load(f) # This might raise YAMLError

        if not config_data: # Handles cases like empty file after comments or successfully parsed empty content
            logger.warning(f"Configuration file '{actual_config_path}' parsed to empty content. Using default logging.")
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
            return {"app_settings": {"default_setting": True, "config_status": "empty_content_parsed"}, "logging": DEFAULT_LOGGING_CONFIG}

        # Setup logging using the configuration from the file
        logging_config_from_file = config_data.get('logging', DEFAULT_LOGGING_CONFIG)
        logging.config.dictConfig(logging_config_from_file)

        logger.info(f"Configuration loaded and logging configured from '{actual_config_path}'.")
        return config_data

    except OCRFileNotFoundError: # Re-raise specific error
        raise
    except (yaml.YAMLError, json.JSONDecodeError) as parse_err:
        logger.error(f"Error parsing configuration file '{actual_config_path}': {parse_err}", exc_info=True)
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG) # Fallback logging
        raise OCRConfigurationError(f"Failed to parse configuration file '{actual_config_path}'. Error: {parse_err}") from parse_err
    except IOError as io_err: # Catch file I/O errors during open/read
        logger.error(f"I/O error reading configuration file '{actual_config_path}': {io_err}", exc_info=True)
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        raise OCRConfigurationError(f"I/O error with configuration file '{actual_config_path}'. Error: {io_err}") from io_err
    except Exception as e: # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred while loading configuration '{actual_config_path}': {e}", exc_info=True)
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG) # Fallback logging
        raise OCRConfigurationError(f"Unexpected error loading configuration '{actual_config_path}'. Error: {e}") from e

if __name__ == '__main__':
    # Configure basic logging for this __main__ block to see its specific messages
    # This will be overridden if load_config is successful and sets up its own config.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("Running config_loader.py as main script for testing.")

    # Test 1: Default config creation and loading
    logger.info("\n--- Test 1: Default config_dev.yaml ---")
    default_test_path = "config_dev_test_main.yaml"
    if os.path.exists(default_test_path):
        os.remove(default_test_path)
    try:
        config = load_config(default_test_path) # Should create and load
        if config and config.get("app_settings", {}).get("version") == "0.0.1-default":
            logger.info(f"Test 1 PASSED: Successfully created and loaded '{default_test_path}'. App version: {config['app_settings']['version']}")
        else:
            logger.error(f"Test 1 FAILED: Problem with default config. Config data: {config}")
    except Exception as e:
        logger.error(f"Test 1 FAILED with exception: {e}", exc_info=True)
    finally:
        if os.path.exists(default_test_path):
            os.remove(default_test_path)

    # Test 2: Loading a non-existent config file (not named config_dev.yaml)
    logger.info("\n--- Test 2: Non-existent config file ---")
    non_existent_path = "non_existent_test_config.yaml"
    try:
        load_config(non_existent_path)
        logger.error(f"Test 2 FAILED: Expected OCRFileNotFoundError for '{non_existent_path}' but no exception was raised.")
    except OCRFileNotFoundError:
        logger.info(f"Test 2 PASSED: Correctly raised OCRFileNotFoundError for '{non_existent_path}'.")
    except Exception as e:
        logger.error(f"Test 2 FAILED: Expected OCRFileNotFoundError but got {type(e).__name__}: {e}", exc_info=True)

    # Test 3: Loading a malformed YAML file
    logger.info("\n--- Test 3: Malformed YAML file ---")
    malformed_path = "malformed_test_config.yaml"
    with open(malformed_path, "w") as f:
        f.write("app_settings: version: 1.0\nlogging: - level: DEBUG") # Invalid YAML
    try:
        load_config(malformed_path)
        logger.error(f"Test 3 FAILED: Expected OCRConfigurationError for malformed YAML but no exception was raised.")
    except OCRConfigurationError as e:
        if "Failed to parse" in str(e):
            logger.info(f"Test 3 PASSED: Correctly raised OCRConfigurationError for malformed YAML.")
        else:
            logger.error(f"Test 3 FAILED: Expected parse error message but got: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Test 3 FAILED: Expected OCRConfigurationError but got {type(e).__name__}: {e}", exc_info=True)
    finally:
        if os.path.exists(malformed_path):
            os.remove(malformed_path)

    logger.info("\n--- Config loader __main__ tests finished. ---")
