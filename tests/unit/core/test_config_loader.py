"""
Unit tests for the config_loader.py module using pytest.
"""

import pytest
import yaml
import json
import os
import logging
from typing import Dict, Any

# Ensure the config_loader module can be found (adjust path if necessary, e.g., using sys.path.append)
# This assumes that 'config_loader.py' is in the parent directory or PYTHONPATH is set.
# For a typical project structure, you might need:
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocrx.core.config_loader import load_config, DEFAULT_LOGGING_CONFIG, get_module_config
from typing import Optional # Added for the helper function typing


# --- Helper function to create dummy config content ---
def create_dummy_config_content(include_logging: bool = True, custom_logging_level: Optional[str] = None) -> Dict[str, Any]:
    """Creates a dictionary representing dummy config content."""
    content: Dict[str, Any] = {
        "app_settings": {
            "default_ocr_engine": "dummy_local_test_engine",
            "feature_flags": {"new_feature_x": True},
        },
        "ocr_engines": {
            "dummy_local_test_engine": {
                "module": "tests.stubs.test_dummy_engine",
                "class": "TestDummyEngine",
                "enabled": True,
                "config": {"model_path": "models/dummy_test.onnx"}
            }
        }
    }
    if include_logging:
        log_level = custom_logging_level if custom_logging_level else "INFO"
        content["logging"] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "test_formatter": {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
            },
            "handlers": {
                "test_console": {
                    "class": "logging.StreamHandler",
                    "formatter": "test_formatter",
                    "level": "DEBUG", # Handler level
                }
            },
            "loggers": {
                "TestAppLogger": { # A specific logger for testing
                    "handlers": ["test_console"],
                    "level": log_level, # Configurable for tests
                    "propagate": False
                }
            },
            "root": {
                "handlers": ["test_console"],
                "level": "WARNING" # Root level
            }
        }
    return content

# --- Test Cases ---

def test_load_valid_yaml_config(tmp_path):
    """Test loading a valid YAML configuration file."""
    config_content = create_dummy_config_content()
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    loaded_config = load_config(str(config_file))

    assert loaded_config is not None
    assert loaded_config["app_settings"]["default_ocr_engine"] == "dummy_local_test_engine"
    assert loaded_config["ocr_engines"]["dummy_local_test_engine"]["module"] == "stubs.test_dummy_engine"
    assert "logging" in loaded_config # Check that logging config is part of the returned dict

    # Verify logging was configured (check a specific logger from the config)
    test_app_logger = logging.getLogger("TestAppLogger")
    assert test_app_logger.level == logging.INFO # Default level from create_dummy_config_content
    # Check if the handler from config is attached (can be a bit brittle if handler names change)
    assert any(isinstance(h, logging.StreamHandler) for h in test_app_logger.handlers)


def test_load_valid_json_config(tmp_path):
    """Test loading a valid JSON configuration file."""
    config_content = create_dummy_config_content()
    config_file = tmp_path / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_content, f, indent=2)

    loaded_config = load_config(str(config_file))

    assert loaded_config is not None
    assert loaded_config["app_settings"]["default_ocr_engine"] == "dummy_local_test_engine"
    assert "logging" in loaded_config

    # Verify basic logging setup (at least default or custom was applied without error)
    # More specific checks like in YAML test are also good
    assert logging.getLogger("TestAppLogger").level == logging.INFO


def test_load_non_existent_config(caplog):
    """Test loading a non-existent configuration file."""
    # Ensure the root logger captures warnings for this test if not already configured
    logging.getLogger().setLevel(logging.WARNING)
    
    config_path = "non_existent_config_file.yaml"
    loaded_config = load_config(config_path)

    assert "app_settings" in loaded_config
    assert "error" in loaded_config["app_settings"]
    assert config_path in loaded_config["app_settings"]["error"]
    assert loaded_config.get("logging") == DEFAULT_LOGGING_CONFIG # Should return default logging

    # Check caplog for the warning message
    assert any(
        f"Configuration file '{config_path}' not found. Using default logging and minimal config." in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )
    # Verify that default logging was applied by checking a known default logger's level
    # (e.g., root or a specific one from DEFAULT_LOGGING_CONFIG)
    # This assumes DEFAULT_LOGGING_CONFIG sets root to DEBUG and console handler to INFO
    assert logging.getLogger("ConfigLoader").level == logging.INFO # From DEFAULT_LOGGING_CONFIG


def test_load_invalid_yaml_config(tmp_path, caplog):
    """Test loading an invalid (malformed) YAML configuration file."""
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("app_settings: {version: 1.0\nlogging: [invalid_yaml_structure") # Malformed

    loaded_config = load_config(str(config_file))

    assert "app_settings" in loaded_config
    assert "error" in loaded_config["app_settings"]
    assert "Parsing error" in loaded_config["app_settings"]["error"]
    assert loaded_config.get("logging") == DEFAULT_LOGGING_CONFIG

    assert any(
        f"Error parsing configuration file '{str(config_file)}'" in record.message
        for record in caplog.records
        if record.levelname == "ERROR"
    )

# --- Tests for get_module_config ---

@pytest.fixture
def sample_main_config() -> Dict[str, Any]:
    """Provides a sample main configuration dictionary for testing get_module_config."""
    return {
        "app_settings": {"global_setting": "app_wide_value"},
        "modules": {
            "module_A": {
                "param1": "valueA1",
                "param2": 100
            },
            "module_B": {
                "param1": "valueB1",
                "enabled": False
            },
            "module_C_is_not_dict": "not_a_dictionary"
        },
        "logging": {
            "version": 1
            # ... other logging config ...
        }
    }

def test_get_module_config_success(sample_main_config):
    """Test successfully extracting a module's configuration."""
    module_a_config = get_module_config(sample_main_config, "module_A")
    assert module_a_config == {"param1": "valueA1", "param2": 100}

    module_b_config = get_module_config(sample_main_config, "module_B")
    assert module_b_config == {"param1": "valueB1", "enabled": False}

def test_get_module_config_key_not_found_no_default(sample_main_config):
    """Test getting config for a non-existent module key with no default provided."""
    # Logger is not passed to get_module_config, it uses logging.getLogger(__name__)
    # which is ocrx.core.config_loader. So we don't need caplog here unless we want to check its output.
    non_existent_config = get_module_config(sample_main_config, "module_X")
    assert non_existent_config == {} # Should return empty dict by default

def test_get_module_config_key_not_found_with_default(sample_main_config):
    """Test getting config for a non-existent module key with a custom default."""
    default_settings = {"default_param": True, "another": 123}
    non_existent_config = get_module_config(sample_main_config, "module_Y", default_if_missing=default_settings)
    assert non_existent_config == default_settings

def test_get_module_config_modules_key_missing(sample_main_config):
    """Test when the top-level 'modules' key is missing from main_config."""
    config_without_modules = {"app_settings": {"setting": "val"}}

    module_config = get_module_config(config_without_modules, "module_A")
    assert module_config == {} # Default empty dict

    default_settings = {"info": "using default"}
    module_config_with_default = get_module_config(config_without_modules, "module_A", default_if_missing=default_settings)
    assert module_config_with_default == default_settings

def test_get_module_config_is_not_dict_warning(sample_main_config, caplog):
    """Test that a warning is logged if the retrieved module config is not a dictionary."""
    default_settings = {"using_default": True}
    with caplog.at_level(logging.WARNING):
        module_c_config = get_module_config(sample_main_config, "module_C_is_not_dict", default_if_missing=default_settings)

    assert module_c_config == default_settings # Should return the default
    assert any(
        "Configuration for module key 'module_C_is_not_dict' is not a dictionary." in record.message and
        "Using default: {'using_default': True}" in record.message
        for record in caplog.records
        if record.levelname == "WARNING" and record.name == "ocrx.core.config_loader" # Check logger name
    )

def test_get_module_config_empty_main_config():
    """Test with an entirely empty main configuration."""
    empty_config = {}
    module_config = get_module_config(empty_config, "any_module")
    assert module_config == {}

    default_val = {"is_default": True}
    module_config_def = get_module_config(empty_config, "any_module", default_if_missing=default_val)
    assert module_config_def == default_val
    assert any(
        "Falling back to default logging configuration due to parsing error." in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )

def test_load_empty_config_file(tmp_path, caplog):
    """Test loading an empty configuration file."""
    config_file = tmp_path / "empty_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        pass # Create empty file

    loaded_config = load_config(str(config_file))

    assert "app_settings" in loaded_config
    assert "error" in loaded_config["app_settings"]
    assert "Empty config file" in loaded_config["app_settings"]["error"]
    assert loaded_config.get("logging") == DEFAULT_LOGGING_CONFIG

    assert any(
        f"Configuration file '{str(config_file)}' is empty. Using default logging and minimal config." in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )

def test_load_config_precedence_logging_config_missing(tmp_path, caplog):
    """Test loading a config file with app settings but no 'logging' section."""
    config_content_no_logging = create_dummy_config_content(include_logging=False)
    config_file = tmp_path / "config_no_logging.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content_no_logging, f)

    # Temporarily set a distinct level for a known logger to see if it's overwritten by defaults
    # Note: This relies on the logger already existing from a previous test or default setup.
    # A cleaner way might be to check the structure of logging.config.dictConfig arguments if possible,
    # or check the level of a logger known to be *only* in DEFAULT_LOGGING_CONFIG.
    logging.getLogger("ConfigLoader").setLevel(logging.DEBUG) # Set to something different than default

    loaded_config = load_config(str(config_file))

    assert loaded_config["app_settings"]["default_ocr_engine"] == "dummy_local_test_engine"
    assert "logging" not in loaded_config # The original loaded_config should not have 'logging' section
                                        # but load_config internally applies defaults.

    # Check that the ConfigLoader logger now has the level from DEFAULT_LOGGING_CONFIG
    # This assumes 'ConfigLoader' is defined in DEFAULT_LOGGING_CONFIG with INFO level.
    default_config_loader_level = DEFAULT_LOGGING_CONFIG.get('loggers', {}).get('ConfigLoader', {}).get('level', 'INFO').upper()
    assert logging.getLogger("ConfigLoader").level == getattr(logging, default_config_loader_level)
    
    assert any(
        "Default logging configuration applied as no 'logging' section found in config file." in record.message
        for record in caplog.records
        if record.levelname == "INFO"
    )

def test_load_config_applies_custom_logging(tmp_path):
    """Test that a custom logging configuration from the file is applied."""
    custom_log_level_str = "CRITICAL"
    custom_logger_name = "TestAppLogger" # Defined in create_dummy_config_content helper

    config_content = create_dummy_config_content(include_logging=True, custom_logging_level=custom_log_level_str)
    config_file = tmp_path / "custom_log_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    load_config(str(config_file)) # This call applies the logging config

    # Verify that the custom logging setting was applied
    test_app_logger = logging.getLogger(custom_logger_name)
    assert test_app_logger.level == getattr(logging, custom_log_level_str)
    
    # Check if the specific handler from the custom config is present
    # This is a bit more robust than checking all handlers if the default also adds some.
    handler_found = False
    for handler in test_app_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and \
           handler.formatter._fmt == config_content["logging"]["formatters"]["test_formatter"]["format"]:
            handler_found = True
            break
    assert handler_found, "Custom handler defined in test config was not found on the logger."

def test_load_unsupported_file_format(tmp_path, caplog):
    """Test loading a file with an unsupported extension."""
    config_file = tmp_path / "config.txt"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("app_settings: { version: '1.0' }")

    loaded_config = load_config(str(config_file))

    assert "app_settings" in loaded_config
    assert "error" in loaded_config["app_settings"]
    assert "Unsupported config format" in loaded_config["app_settings"]["error"]
    assert loaded_config.get("logging") == DEFAULT_LOGGING_CONFIG

    assert any(
        f"Unsupported configuration file format: {str(config_file)}" in record.message
        for record in caplog.records
        if record.levelname == "ERROR"
    )
