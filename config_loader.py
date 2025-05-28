import yaml # Requires PyYAML to be installed: pip install PyYAML
import json
import logging
import logging.config
import os

# Define a default logging configuration in case the file is missing or incomplete
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
    'root': { # Root logger
        'handlers': ['console'], # Default to console
        'level': 'DEBUG', # Capture all DEBUG level messages and above at root
    },
    # Example: Specific logger configuration (can be added to config.yaml)
    # 'loggers': {
    #     'OCRWorkflowOrchestrator': {
    #         'handlers': ['console', 'file'], # Use both console and file
    #         'level': 'DEBUG',
    #         'propagate': False # Don't pass to root logger if handled here
    #     }
    # }
}

def create_default_config_if_not_exists(config_path="config_dev.yaml"):
    if not os.path.exists(config_path):
        logging.info(f"'{config_path}' not found, creating a default one.")
        default_content = {
            "app_settings": {
                "version": "0.0.1-default",
                "default_output_format": "txt",
                "model_paths": {
                    "geometric_corrector": "models/geometric_v1.onnx",
                    "paddle_ocr_det": "models/paddle_det_v4.onnx",
                    "paddle_ocr_rec": "models/paddle_rec_v4_en.onnx",
                    "svtr_recognizer": "models/svtr_large_en.onnx",
                    "byt5_corrector": "models/byt5_ocr_corrector.onnx"
                },
                "performance": {
                    "use_directml": True,
                    "onnx_intra_op_threads": 0
                }
            },
            "logging": DEFAULT_LOGGING_CONFIG # Use the same default logging
        }
        # Ensure directory for config exists if it's nested, though for "config_dev.yaml" it's not
        # config_dir = os.path.dirname(config_path)
        # if config_dir and not os.path.exists(config_dir):
        #    os.makedirs(config_dir)

        with open(config_path, "w") as f:
            yaml.dump(default_content, f, sort_keys=False)
        logging.info(f"Default configuration file created at '{config_path}'.")


def load_config(config_path="config.yaml"):
    """
    Loads configuration from a YAML or JSON file.
    Sets up logging based on the configuration.
    """
    # Create a default config_dev.yaml if the specified config_path doesn't exist,
    # especially to support the __main__ block of other modules.
    # This is a bit of a workaround for the subtask's requirement.
    # Ideally, only the main application or its specific test would create this.
    if config_path == "config_dev.yaml" and not os.path.exists(config_path):
        create_default_config_if_not_exists(config_path)


    config_data = None
    try:
        if not os.path.exists(config_path):
            logging.warning(f"Configuration file '{config_path}' not found. Attempting to use defaults.")
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
            logging.info("Applied default logging configuration as config file was not found.")
            return {"app_settings": {"default_setting": True}, "logging": DEFAULT_LOGGING_CONFIG} 

        with open(config_path, 'r') as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config_data = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config_data = json.load(f)
            else:
                try:
                    config_data = yaml.safe_load(f)
                    logging.info(f"Attempting to load '{config_path}' as YAML due to unknown extension.")
                except yaml.YAMLError:
                    logging.error(f"Unsupported configuration file format: {config_path}. Must be YAML or JSON.")
                    raise ValueError(f"Unsupported configuration file format: {config_path}")

        if not config_data: 
            logging.warning(f"Configuration file '{config_path}' is empty. Using default logging.")
            logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
            return {"app_settings": {"default_setting": True}, "logging": DEFAULT_LOGGING_CONFIG}

        logging_config_from_file = config_data.get('logging', DEFAULT_LOGGING_CONFIG)
        logging.config.dictConfig(logging_config_from_file)
        
        logging.info(f"Configuration loaded and logging configured from '{config_path}'.")
        return config_data

    except FileNotFoundError: 
        logging.error(f"Critical error: Config file '{config_path}' not found despite check. Using default logging.")
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        return {"app_settings": {"default_setting": True}, "logging": DEFAULT_LOGGING_CONFIG}
    except (yaml.YAMLError, json.JSONDecodeError) as parse_err:
        logging.error(f"Error parsing configuration file '{config_path}': {parse_err}", exc_info=True)
        logging.warning("Falling back to default logging configuration due to parsing error.")
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        raise ValueError(f"Failed to parse config file: {config_path}") from parse_err
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading configuration '{config_path}': {e}", exc_info=True)
        logging.warning("Falling back to default logging configuration due to unexpected error.")
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        raise RuntimeError(f"Unexpected error loading config: {config_path}") from e

if __name__ == '__main__':
    # This will now create "config_dev.yaml" if it doesn't exist,
    # because load_config (when called with "config_dev.yaml") calls create_default_config_if_not_exists.
    dummy_config_file_for_main = "config_dev.yaml"
    
    # The content that would be in the dummy config_dev.yaml for this main block.
    # create_default_config_if_not_exists will handle creating it with suitable defaults.
    # We no longer need to write it manually here if load_config ensures it.

    try:
        # load_config will ensure "config_dev.yaml" is created if it's missing.
        config = load_config(config_path=dummy_config_file_for_main)
        main_logger = logging.getLogger(__name__) 
        if config:
            main_logger.info(f"App version from config: {config.get('app_settings', {}).get('version')}")
            main_logger.debug("This is a debug message from the main example script in config_loader.")
            main_logger.warning("This is a warning message from config_loader.")
            
            test_module_logger = logging.getLogger("MyTestModuleInConfigLoader")
            test_module_logger.info("Info message from MyTestModuleInConfigLoader.")
            test_module_logger.debug("Debug message from MyTestModuleInConfigLoader.")

    except Exception as e:
        # Use a basic logger if config loading failed catastrophically before logging was set up
        logging.basicConfig(level=logging.ERROR)
        logging.critical(f"Failed to run config loader example: {e}", exc_info=True)
