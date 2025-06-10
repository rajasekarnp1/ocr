import unittest
import os
import yaml
import json
import logging
import shutil
import unittest # Ensure unittest is imported

# Assuming custom_exceptions.py is in the same directory or PYTHONPATH
try:
    from custom_exceptions import OCRConfigurationError, OCRFileNotFoundError
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from custom_exceptions import OCRConfigurationError, OCRFileNotFoundError
    # Fallback if custom_exceptions is not found (e.g. during isolated testing)
    if "OCRConfigurationError" not in globals(): # Check if already defined by try block
        class OCRConfigurationError(RuntimeError): pass
        class OCRFileNotFoundError(FileNotFoundError): pass


from config_loader import load_config, DEFAULT_LOGGING_CONFIG, create_default_config_if_not_exists

# Configure logging for tests (basic if not already configured by a test runner)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO) # Set to DEBUG for more verbose test logging


class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_config_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.test_dir)

        # Reset logging to a known state before each test
        logging.shutdown()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Ensure that the default config_dev.yaml, if created by a previous test run's module, is removed
        if os.path.exists("config_dev.yaml"):
            os.remove("config_dev.yaml")


    def test_load_config_yaml_success(self):
        dummy_yaml_path = os.path.join(self.test_dir, "test_config.yaml")
        dummy_config_data = {
            "app_settings": {"version": "1.0.0-test-yaml"},
            "logging": {
                "version": 1,
                "root": {"level": "WARNING", "handlers": ["console"]},
                "handlers": {"console": {"class": "logging.StreamHandler", "level": "WARNING"}}
            }
        }
        with open(dummy_yaml_path, 'w') as f:
            yaml.dump(dummy_config_data, f)

        config = load_config(dummy_yaml_path)
        self.assertEqual(config["app_settings"]["version"], "1.0.0-test-yaml")
        self.assertEqual(logging.getLogger().getEffectiveLevel(), logging.WARNING)

    def test_load_config_json_success(self):
        dummy_json_path = os.path.join(self.test_dir, "test_config.json")
        dummy_config_data = {
            "app_settings": {"version": "1.0.0-test-json"},
            "logging": {
                "version": 1,
                "root": {"level": "ERROR", "handlers": ["console"]},
                "handlers": {"console": {"class": "logging.StreamHandler", "level": "ERROR"}}
            }
        }
        with open(dummy_json_path, 'w') as f:
            json.dump(dummy_config_data, f)

        config = load_config(dummy_json_path)
        self.assertEqual(config["app_settings"]["version"], "1.0.0-test-json")
        self.assertEqual(logging.getLogger().getEffectiveLevel(), logging.ERROR)

    def test_load_config_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "non_existent_config_for_test.yaml")
        # load_config (non-config_dev.yaml) should raise OCRFileNotFoundError if file doesn't exist.
        logger.info(f"TestConfigLoader: Testing load_config with non-existent path: {non_existent_path}")
        with self.assertRaises(OCRFileNotFoundError) as context:
            load_config(non_existent_path)
        self.assertTrue(f"Configuration file '{non_existent_path}' not found" in str(context.exception))
        # Default logging should be applied by load_config before raising the error
        # if the error occurs after the initial os.path.exists check.
        # The refined load_config applies default logging then raises.
        self.assertEqual(logging.getLogger().getEffectiveLevel(), DEFAULT_LOGGING_CONFIG['root']['level'])


    def test_load_config_empty_file(self):
        # Test behavior when a config file (not config_dev.yaml) is empty.
        # load_config should parse it as empty, log a warning, and return a default-like structure.
        logger.info("TestConfigLoader: test_load_config_empty_file")
        empty_yaml_path = os.path.join(self.test_dir, "actually_empty_config.yaml")
        with open(empty_yaml_path, 'w') as f:
            pass # Create an empty file

        config = load_config(empty_yaml_path)
        self.assertTrue(config["app_settings"]["default_setting"])
        self.assertEqual(logging.getLogger().getEffectiveLevel(), logging.DEBUG)

    def test_load_config_invalid_yaml_file(self):
        invalid_yaml_path = os.path.join(self.test_dir, "invalid_config.yaml")
        with open(invalid_yaml_path, 'w') as f:
            f.write("app_settings: version: 1.0\nlogging:\n  handlers:\n    console:\n      level: INVALID_LEVEL_VALUE_TYPE") # Invalid level value type

        logger.info(f"TestConfigLoader: Testing load_config with invalid (malformed) YAML: {invalid_yaml_path}")
        with self.assertRaises(OCRConfigurationError) as context:
            load_config(invalid_yaml_path)
        self.assertTrue("Failed to parse configuration file" in str(context.exception) or "Error parsing configuration file" in str(context.exception))

        # Check that default logging is applied after a parse error by load_config's exception handling
        self.assertEqual(logging.getLogger().getEffectiveLevel(), DEFAULT_LOGGING_CONFIG['root']['level'])

    def test_logging_configured_from_valid_file(self):
        dummy_yaml_path = os.path.join(self.test_dir, "logging_test_config.yaml")
        dummy_config_data = {
            "logging": {
                "version": 1,
                "formatters": {
                    "test_formatter": {"format": "%(levelname)s-%(name)s-%(message)s"}
                },
                "handlers": {
                    "test_handler": {
                        "class": "logging.StreamHandler",
                        "formatter": "test_formatter",
                        "level": "INFO"
                    }
                },
                "loggers": {
                    "TestLogger": {"handlers": ["test_handler"], "level": "INFO", "propagate": False}
                },
                "root": {"handlers": ["test_handler"], "level": "DEBUG"} # Root also uses test_handler
            }
        }
        with open(dummy_yaml_path, 'w') as f:
            yaml.dump(dummy_config_data, f)

        load_config(dummy_yaml_path)

        # Check root logger
        self.assertEqual(logging.getLogger().getEffectiveLevel(), logging.DEBUG)
        # Check specific logger
        specific_logger = logging.getLogger("TestLogger")
        self.assertEqual(specific_logger.getEffectiveLevel(), logging.INFO)
        self.assertFalse(specific_logger.propagate)
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in specific_logger.handlers))
        # Check if formatter is somewhat applied (hard to check exact format string directly)
        # For simplicity, we'll assume if levels and handlers are set, formatters are too.

    def test_create_default_config_dev_yaml(self):
        # Test the specific case where load_config is called with "config_dev.yaml"
        # and it's expected to create it if it doesn't exist.
        config_dev_path = "config_dev.yaml" # In the root directory
        if os.path.exists(config_dev_path):
            os.remove(config_dev_path) # Ensure it doesn't exist

        self.assertFalse(os.path.exists(config_dev_path))

        # load_config should call create_default_config_if_not_exists internally
        config = load_config(config_dev_path)

        self.assertTrue(os.path.exists(config_dev_path))
        self.assertIn("app_settings", config)
        self.assertEqual(config["app_settings"]["version"], "0.0.1-default")

        # Clean up the created config_dev.yaml
        if os.path.exists(config_dev_path):
            os.remove(config_dev_path)

    def test_create_default_config_explicit_call(self):
        # Test the create_default_config_if_not_exists function directly
        default_config_path = os.path.join(self.test_dir, "default_test_config.yaml")
        if os.path.exists(default_config_path):
            os.remove(default_config_path)

        self.assertFalse(os.path.exists(default_config_path))
        create_default_config_if_not_exists(default_config_path)
        self.assertTrue(os.path.exists(default_config_path))

        with open(default_config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        self.assertIn("app_settings", config_data)
        self.assertEqual(config_data["app_settings"]["version"], "0.0.1-default")
        self.assertIn("logging", config_data)


if __name__ == '__main__':
    unittest.main()
