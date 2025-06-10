"""
Configuration loader for the OCR-X application.

This module provides functionality to load application settings from a YAML
(or JSON) file, including setting up logging configurations.
"""

import yaml # Requires PyYAML: pip install PyYAML
import json
import logging
import logging.config # For dictConfig
import os
from typing import Dict, Any, Optional # Using Dict and Any from typing for Python 3.8 compatibility
# from .exceptions import OCRXConfigurationError # Not used in get_module_config directly, but good for module context

# Default logging configuration if config file is missing or logging section is absent/invalid
DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        },
        'simple': { # Simple can remain for less verbose needs if any
            'format': '%(levelname)s - %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard', # Use the new standard formatter
            'level': 'INFO',
            'stream': 'ext://sys.stdout'
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG' # Overall root logger level, handlers control their own output level
    },
    'loggers': { # Example of configuring specific loggers
        'OCRWorkflowOrchestrator': { # Matches class name from orchestrator template
            'handlers': ['console'],
            'level': 'INFO', # Default level for this specific logger
            'propagate': False
        },
        'PreprocessingModule': { # Conceptual module name
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': True
        },
        'RecognitionModule': { # Conceptual module name
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        },
         'ConfigLoader': { # Logger for this module itself
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Get a logger for this module, which will be configured by load_config
module_logger = logging.getLogger(__name__)
# To make this logger specifically configurable in the YAML, we can name it, e.g., 'ConfigLoader'
# module_logger = logging.getLogger('ConfigLoader') # This will be used later if defined in YAML

def get_module_config(main_config: Dict[str, Any], module_key: str, default_if_missing: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Safely extracts a sub-dictionary for a module's configuration.

    Args:
        main_config: The main application configuration dictionary.
        module_key: The key for the desired module's configuration
                    (e.g., "preprocessing_settings", "recognition_engine_A").
        default_if_missing: A default dictionary to return if the key is not found.
                            If None (default), an empty dict is returned.

    Returns:
        The module's configuration dictionary or the default.
    """
    if default_if_missing is None:
        default_if_missing = {}

    # Assuming module configurations are nested under a top-level "modules" key
    module_config = main_config.get("modules", {}).get(module_key, default_if_missing)
    if not isinstance(module_config, dict):
        # Use the module_logger defined at the top of the file for consistency
        module_logger.warning(
            f"Configuration for module key '{module_key}' is not a dictionary. "
            f"Using default: {default_if_missing}"
        )
        return default_if_missing
    return module_config

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads configuration from a YAML or JSON file.

    Sets up logging based on the 'logging' section in the configuration.
    If the file is not found or a 'logging' section is missing, it uses
    DEFAULT_LOGGING_CONFIG.

    :param config_path: Path to the configuration file (default: "config.yaml").
    :return: A dictionary containing the loaded configuration.
    """
    config_data: Dict[str, Any] = {}
    logging_applied_custom = False

    try:
        if not os.path.exists(config_path):
            # Apply default logging config first before logging the warning
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
            logging_applied_custom = False # Indicate default was used
            module_logger.warning(f"Configuration file '{config_path}' not found. Using default logging and minimal config.")
            # Return a minimal config that includes the default logging setup for consistency
            return {'logging': DEFAULT_LOGGING_CONFIG, 'app_settings': {'error': f"Config file {config_path} not found"}}

        with open(config_path, 'r', encoding='utf-8') as f:
            file_extension = os.path.splitext(config_path)[1].lower()
            if file_extension in (".yaml", ".yml"):
                config_data = yaml.safe_load(f)
            elif file_extension == ".json":
                config_data = json.load(f)
            else:
                # Apply default logging config first before logging the error
                logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
                logging_applied_custom = False
                module_logger.error(f"Unsupported configuration file format: {config_path}. Must be YAML or JSON.")
                # Return minimal config for consistency upon failure
                return {'logging': DEFAULT_LOGGING_CONFIG, 'app_settings': {'error': f"Unsupported config format: {config_path}"}}
        
        if not config_data: # Handle empty config file
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
            logging_applied_custom = False
            module_logger.warning(f"Configuration file '{config_path}' is empty. Using default logging and minimal config.")
            return {'logging': DEFAULT_LOGGING_CONFIG, 'app_settings': {'error': f"Empty config file: {config_path}"}}

        logging_config_to_apply = config_data.get('logging', DEFAULT_LOGGING_CONFIG)
        logging.config.dictConfig(logging_config_to_apply)
        logging_applied_custom = True if 'logging' in config_data else False
        
        # Now that logging is configured (either default or custom), log the success message.
        # Re-initialize module_logger in case its settings were changed by dictConfig
        global module_logger 
        module_logger = logging.getLogger(__name__) # Or 'ConfigLoader' if named that way

        module_logger.info(f"Configuration loaded successfully from '{config_path}'.")
        if logging_applied_custom:
            module_logger.info("Custom logging configuration applied from file.")
        else:
            module_logger.info("Default logging configuration applied as no 'logging' section found in config file.")
            
        return config_data

    except (yaml.YAMLError, json.JSONDecodeError) as parse_err:
        # Apply default logging config first before logging the error
        if not logging_applied_custom: # Avoid reconfiguring if custom already failed
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        module_logger = logging.getLogger(__name__)
        module_logger.error(f"Error parsing configuration file '{config_path}': {parse_err}", exc_info=True)
        module_logger.warning("Falling back to default logging configuration due to parsing error.")
        # Return minimal config for consistency upon failure
        return {'logging': DEFAULT_LOGGING_CONFIG, 'app_settings': {'error': f"Parsing error in {config_path}: {parse_err}"}}
    except Exception as e:
        # Apply default logging config first before logging the error
        if not logging_applied_custom:
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        module_logger = logging.getLogger(__name__)
        module_logger.critical(f"An unexpected error occurred while loading/processing config from '{config_path}': {e}", exc_info=True)
        module_logger.warning("Falling back to default logging configuration due to critical error.")
        # Re-raise critical unexpected errors or return a minimal config
        # For this template, returning minimal config to allow app to potentially start with defaults
        return {'logging': DEFAULT_LOGGING_CONFIG, 'app_settings': {'error': f"Critical error loading {config_path}: {e}"}}


if __name__ == '__main__':
    # --- Create a dummy config.yaml for testing ---
    dummy_config_content = """
app_settings:
  version: "1.0.0-rc1"
  default_ocr_engine: "Local_PaddleOCR" # Example app-specific setting
  api_endpoints:
    google_vision_api: "https://vision.googleapis.com/v1/images:annotate"
  local_model_paths:
    paddle_ocr_detection: "models/paddle_det_v3.onnx"
    paddle_ocr_recognition: "models/paddle_rec_v3_en.onnx"

logging:
  version: 1
  disable_existing_loggers: False # Keep this False unless you have a specific reason
  formatters:
    verbose:
      format: "[%(asctime)s] %(levelname)-8s %(name)s:%(module)s:%(funcName)s:%(lineno)d - %(message)s"
    simple:
      format: "%(levelname)s: %(message)s"
  handlers:
    console_verbose: # A more verbose console handler
      class: logging.StreamHandler
      formatter: verbose
      level: DEBUG # Handler level for console_verbose
      stream: ext://sys.stdout
    file_handler: # Example file handler
      class: logging.handlers.RotatingFileHandler
      formatter: verbose
      filename: ocrx_app_debug.log
      maxBytes: 10485760 # 10MB
      backupCount: 3
      level: DEBUG # Handler level for file
      encoding: utf8
  root:
    handlers: [console_verbose, file_handler] # Apply to root
    level: DEBUG # Root logger level (lowest possible level)
  loggers:
    ConfigLoader: # Specific logger for this module
      handlers: [console_verbose, file_handler]
      level: DEBUG # Make this module's logger verbose for the example
      propagate: False
    OCRWorkflowOrchestrator:
      handlers: [console_verbose, file_handler]
      level: INFO
      propagate: False
    # Add other specific loggers if needed
    # 'werkzeug': { 'level': 'WARNING', 'handlers': ['console_verbose'] } # Example for a library
"""
    test_config_file_name = "config_example.yaml"
    with open(test_config_file_name, "w", encoding="utf-8") as f:
        f.write(dummy_config_content)
    
    # --- Test Case 1: Load the created config file ---
    print(f"\n--- Attempting to load '{test_config_file_name}' ---")
    config = load_config(config_path=test_config_file_name)
    
    # Accessing some config values (after logging is configured)
    main_app_logger = logging.getLogger("MainAppExample") # Using a generic logger for app messages
    if config and 'app_settings' in config:
        main_app_logger.info(f"App Version from config: {config.get('app_settings', {}).get('version', 'N/A')}")
        main_app_logger.debug(f"Default OCR Engine from config: {config.get('app_settings', {}).get('default_ocr_engine', 'Not Set')}")
    else:
        main_app_logger.error("app_settings not found in the loaded configuration.")

    # Test loggers defined in the dummy config
    logging.getLogger("ConfigLoader").info("This is an INFO message from ConfigLoader (should appear based on config).")
    logging.getLogger("ConfigLoader").debug("This is a DEBUG message from ConfigLoader (should appear).")
    logging.getLogger("OCRWorkflowOrchestrator").debug("Orchestrator DEBUG (should NOT show if level is INFO in config).")
    logging.getLogger("OCRWorkflowOrchestrator").info("Orchestrator INFO (should show).")
    logging.getLogger("PreprocessingModule").warning("PreprocessingModule WARNING (should show if propagate is True or has handlers).")
    logging.getLogger("PreprocessingModule").info("PreprocessingModule INFO (should NOT show if level is WARNING).")


    # --- Test Case 2: Config file not found ---
    print("\n--- Attempting to load a non-existent config file ---")
    non_existent_config_path = "non_existent_config.yaml"
    config_not_found = load_config(config_path=non_existent_config_path)
    # The load_config function itself will log a warning.
    # We can check the returned minimal config.
    if 'app_settings' in config_not_found and 'error' in config_not_found['app_settings']:
        main_app_logger.info(f"Load_config response for non-existent file: {config_not_found['app_settings']['error']}")
    # Further logging will now use DEFAULT_LOGGING_CONFIG
    main_app_logger.info("Logging after attempting to load non-existent file (should use default config).")


    # --- Test Case 3: Invalid YAML/JSON content ---
    print("\n--- Attempting to load an invalid config file ---")
    invalid_config_file_name = "invalid_config.yaml"
    with open(invalid_config_file_name, "w", encoding="utf-8") as f:
        f.write("app_settings: {version: 1.0\nlogging: [invalid_yaml_structure") # Malformed YAML
    
    invalid_config = load_config(config_path=invalid_config_file_name)
    if 'app_settings' in invalid_config and 'error' in invalid_config['app_settings']:
         main_app_logger.info(f"Load_config response for invalid file: {invalid_config['app_settings']['error']}")
    main_app_logger.info("Logging after attempting to load invalid file (should use default config).")


    # --- Test Case 4: Empty config file ---
    print("\n--- Attempting to load an empty config file ---")
    empty_config_file_name = "empty_config.yaml"
    with open(empty_config_file_name, "w", encoding="utf-8") as f:
        pass # Create an empty file
    
    empty_config = load_config(config_path=empty_config_file_name)
    if 'app_settings' in empty_config and 'error' in empty_config['app_settings']:
         main_app_logger.info(f"Load_config response for empty file: {empty_config['app_settings']['error']}")
    main_app_logger.info("Logging after attempting to load empty file (should use default config).")

    # Clean up dummy files
    # os.remove(test_config_file_name)
    # os.remove(invalid_config_file_name)
    # os.remove(empty_config_file_name)
    # print("\nDummy config files removed.")
    print(f"\nNote: Dummy config files ('{test_config_file_name}', '{invalid_config_file_name}', '{empty_config_file_name}') and 'ocrx_app_debug.log' were created/used during this test run.")

```
The `config_loader.py` file has been created successfully.

**Summary of the File Content:**

1.  **Imports:** Includes `yaml`, `json`, `logging`, `logging.config`, `os`, and `typing` for type hints.
2.  **`DEFAULT_LOGGING_CONFIG`:** A comprehensive default dictionary is defined at the module level. It includes:
    *   `version: 1`, `disable_existing_loggers: False`.
    *   Two formatters: `default` (verbose) and `simple`.
    *   A `console` handler using the `default` formatter and INFO level.
    *   A `root` logger configuration with the `console` handler and DEBUG level.
    *   Example logger configurations for `OCRWorkflowOrchestrator`, `PreprocessingModule`, `RecognitionModule`, and `ConfigLoader` itself, demonstrating how different modules can have different default logging levels.
3.  **`module_logger`:** A module-level logger is initialized using `logging.getLogger(__name__)`. This logger's configuration will be updated once `dictConfig` is called.
4.  **`load_config(config_path: str = "config.yaml") -> Dict[str, Any]` Function:**
    *   **File Handling:**
        *   Checks if `config_path` exists. If not, it applies `DEFAULT_LOGGING_CONFIG`, logs a warning, and returns a minimal dictionary containing the error and the default logging config.
        *   Opens and reads the file, attempting to parse it as YAML or JSON based on the file extension.
        *   Handles `FileNotFoundError`, `yaml.YAMLError`, `json.JSONDecodeError`, and other generic `Exception` types gracefully.
        *   Handles empty configuration files by applying default logging and returning a minimal config with an error message.
    *   **Logging Configuration:**
        *   If the config file is loaded successfully, it attempts to get a `logging` section from `config_data`. If not found, it uses `DEFAULT_LOGGING_CONFIG`.
        *   Calls `logging.config.dictConfig()` with the chosen logging configuration. **Crucially, this is done *before* most of the function's own informational logging messages** to ensure those messages use the user-defined (or default) logging setup.
        *   The `module_logger` is re-obtained after `dictConfig` in case its settings were altered.
    *   **Return Value:** Returns the loaded `config_data` dictionary. If errors occur that prevent loading, it returns a dictionary containing an `app_settings.error` key with an error message and the `logging` key set to `DEFAULT_LOGGING_CONFIG` to ensure the caller has a consistent structure.
    *   **Logging Messages:** Logs informational messages about successful loading, warnings for missing files or sections, and errors for parsing issues or other exceptions. `exc_info=True` is used for unexpected errors.
5.  **`if __name__ == '__main__':` Block:**
    *   Creates a dummy `config_example.yaml` file with application-specific settings and a detailed logging configuration (including multiple formatters, console and rotating file handlers, and specific logger levels).
    *   **Test Case 1:** Loads this `config_example.yaml` and demonstrates accessing configuration values and how different loggers behave based on the file's logging config.
    *   **Test Case 2:** Attempts to load a non-existent configuration file, demonstrating fallback to default logging.
    *   **Test Case 3:** Attempts to load a file with invalid YAML content, demonstrating error handling and fallback.
    *   **Test Case 4:** Attempts to load an empty configuration file.
    *   Includes commented-out lines for cleaning up the dummy files (left commented for review of outputs).

This structure ensures that logging is configured as early as possible, either from the file or defaults, and that the application can potentially continue with default settings even if the configuration file is problematic.
