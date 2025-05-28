"""
Unit tests for the stubs.dummy_engine.py module using pytest.
"""

import pytest
import os
import sys
import logging
from unittest.mock import MagicMock
from typing import Dict, Any

# --- Path Setup (Simulating conftest.py behavior for standalone execution) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Go up two levels (tests/ -> project_root/)
stubs_dir = os.path.join(project_root, "stubs") # Path to the stubs directory itself

# Add project root and stubs directory to sys.path
# This ensures 'ocr_engine_interface' can be imported from project root
# and 'dummy_engine' can be imported from 'stubs'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if stubs_dir not in sys.path: # Though dummy_engine is in this test file's parent, explicit is better
    sys.path.insert(0, stubs_dir)
# --- End Path Setup ---

# Now try importing the modules
try:
    from ocr_engine_interface import OCREngine # Should be found from project_root
    from dummy_engine import DummyLocalEngine # Should be found from stubs_dir (or project_root.stubs)
except ImportError as e:
    # This fallback might be needed if the above path manipulation isn't perfect
    # or if the test runner has a different current working directory.
    print(f"Initial import failed: {e}. Trying alternative sys.path modification for tests.")
    # This assumes the tests are being run from the 'tests' directory or project root
    if os.path.basename(os.getcwd()) == "tests":
        sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".."))) # Add project root
    elif os.path.basename(os.getcwd()) == os.path.basename(project_root): # Already in project root
        pass # No change needed
    else: # Fallback if CWD is unexpected, try relative to this file's location
        sys.path.insert(0, project_root)

    # Re-try imports after potential path correction
    from ocr_engine_interface import OCREngine
    from dummy_engine import DummyLocalEngine


# --- Fixtures ---

@pytest.fixture
def mock_logger() -> MagicMock:
    """Returns a mock logger."""
    logger = MagicMock(spec=logging.Logger)
    # Configure the mock logger to also output to console for easier debugging if needed
    # This can be achieved by adding a real handler to the mock, or using caplog.
    # For simplicity in this stub, we'll rely on caplog if actual log capture is needed for assertions.
    return logger

@pytest.fixture
def dummy_engine_config() -> Dict[str, Any]:
    """Provides a sample engine configuration for DummyLocalEngine."""
    return {"model_path": "dummy/test_model.onnx", "name": "ConfiguredTestDummy"}

@pytest.fixture
def dummy_engine_instance(dummy_engine_config: Dict[str, Any], mock_logger: MagicMock) -> DummyLocalEngine:
    """Provides a non-initialized DummyLocalEngine instance."""
    engine = DummyLocalEngine(engine_config, mock_logger)
    return engine

@pytest.fixture
def dummy_engine_instance_initialized(dummy_engine_instance: DummyLocalEngine) -> DummyLocalEngine:
    """Provides an initialized DummyLocalEngine instance."""
    dummy_engine_instance.initialize()
    return dummy_engine_instance


# --- Test Cases ---

def test_dummy_engine_init(mock_logger: MagicMock):
    """
    Test DummyLocalEngine initialization.
    1. Create a sample `engine_config` dictionary.
    2. Instantiate `DummyLocalEngine(engine_config, mock_logger)`.
    3. Assert that `self.logger` is set and `self.model_path` (or any other config-driven attribute) is correctly initialized.
    4. Assert that `self._is_initialized` is `False` initially.
    """
    engine_config = {"model_path": "dummy/specific_model.onnx", "name": "SpecificTestDummy"}
    engine = DummyLocalEngine(engine_config, mock_logger)

    assert engine.logger is mock_logger
    assert engine.model_path == "dummy/specific_model.onnx"
    assert engine.engine_config["name"] == "SpecificTestDummy" # Check if other parts of config are stored
    assert not engine._is_initialized, "Engine should not be initialized immediately after __init__"

def test_dummy_engine_initialize_success(dummy_engine_instance: DummyLocalEngine, mock_logger: MagicMock, caplog: pytest.LogCaptureFixture):
    """
    Test successful initialization of DummyLocalEngine.
    1. Use a fixture `dummy_engine_instance` that provides an initialized `DummyLocalEngine`.
    2. Call `dummy_engine_instance.initialize()`.
    3. Assert that `dummy_engine_instance._is_initialized` becomes `True`.
    4. Assert that an appropriate INFO message was logged by the engine's `initialize` method.
    """
    with caplog.at_level(logging.INFO): # Ensure INFO logs are captured
        dummy_engine_instance.initialize()

    assert dummy_engine_instance._is_initialized, "Engine should be initialized after calling initialize()"
    
    # Check logger calls on the mock_logger (if not using caplog for this specific check)
    # Example: mock_logger.info.assert_any_call(f"DummyLocalEngine '{dummy_engine_instance.get_engine_name()}' initialized. Model path: {dummy_engine_instance.model_path}")
    
    # Check caplog for the specific message
    expected_log_message = f"DummyLocalEngine '{dummy_engine_instance.get_engine_name()}' initialized. Model path: {dummy_engine_instance.model_path}"
    assert any(expected_log_message in record.message for record in caplog.records if record.levelname == "INFO")


def test_dummy_engine_get_engine_name(dummy_engine_instance: DummyLocalEngine, dummy_engine_config: Dict[str, Any]):
    """
    Test that get_engine_name returns the expected name.
    It should use the 'name' from config if provided, otherwise a default.
    """
    # Case 1: Name provided in config
    assert dummy_engine_instance.get_engine_name() == dummy_engine_config["name"]

    # Case 2: Name not provided in config, should use default
    config_no_name = {"model_path": "some/path.onnx"}
    engine_no_name = DummyLocalEngine(config_no_name, MagicMock(spec=logging.Logger))
    assert engine_no_name.get_engine_name() == "Dummy_Local_Engine_v1" # Default from DummyLocalEngine

def test_dummy_engine_is_available(dummy_engine_instance: DummyLocalEngine):
    """
    Test the is_available method.
    1. Test an engine instance that has *not* had `initialize()` called. Assert `is_available()` returns `False`.
    2. Then, call `initialize()` on the instance. Assert `is_available()` now returns `True`.
    """
    assert not dummy_engine_instance.is_available(), "Engine should not be available before initialization"
    
    dummy_engine_instance.initialize()
    assert dummy_engine_instance.is_available(), "Engine should be available after successful initialization"

def test_dummy_engine_recognize_before_init(dummy_engine_instance: DummyLocalEngine, caplog: pytest.LogCaptureFixture):
    """Test that recognize raises an error if called before initialization."""
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError, match=f"Engine {dummy_engine_instance.get_engine_name()} not initialized."):
        dummy_engine_instance.recognize("dummy_image_data.png")
    
    assert any(f"{dummy_engine_instance.get_engine_name()} called before initialization." in record.message for record in caplog.records if record.levelname == "ERROR")


def test_dummy_engine_recognize_success(dummy_engine_instance_initialized: DummyLocalEngine, caplog: pytest.LogCaptureFixture):
    """
    Test the recognize method of an initialized DummyLocalEngine.
    1. Use a fixture `dummy_engine_instance_initialized`.
    2. Call `recognize`.
    3. Assert the returned dictionary matches the expected structure and placeholder content.
    4. Assert an INFO message was logged.
    """
    image_path_data = "dummy_image_data.png"
    language_hint_data = "eng-US"
    
    with caplog.at_level(logging.INFO):
        result = dummy_engine_instance_initialized.recognize(image_path_data, language_hint=language_hint_data)

    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result
    assert "segments" in result
    assert "engine_name" in result

    expected_image_info = f"simulated_image_from_path_{image_path_data}"
    assert result["text"] == f"Dummy OCR result from {dummy_engine_instance_initialized.get_engine_name()} for {expected_image_info}"
    assert result["confidence"] == 0.99
    assert isinstance(result["segments"], list)
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "Dummy segment"
    assert result["segments"][0]["bounding_box"] == [0,0,10,10]
    assert result["segments"][0]["confidence"] == 0.99
    assert result["engine_name"] == dummy_engine_instance_initialized.get_engine_name()

    expected_log_message = f"DummyLocalEngine recognizing image (type: <class 'str'>). Language hint: {language_hint_data}"
    assert any(expected_log_message in record.message for record in caplog.records if record.levelname == "INFO")

```
The `tests/stubs/test_dummy_engine.py` file has been created with the specified pytest test cases and fixtures.

**Summary of the Test File Content:**

1.  **Path Setup:** Includes a section at the beginning to dynamically add the project root and `stubs` directory to `sys.path`. This ensures that `ocr_engine_interface` (from project root) and `dummy_engine` (from `stubs` directory) can be imported correctly during test execution. Fallback logic is also included if initial imports fail.
2.  **Imports:** All necessary modules and classes are imported: `pytest`, `os`, `sys`, `logging`, `unittest.mock.MagicMock`, `typing.Dict`, `typing.Any`, `ocr_engine_interface.OCREngine`, and `stubs.dummy_engine.DummyLocalEngine`.
3.  **Pytest Fixtures:**
    *   `mock_logger()`: Creates and returns a `MagicMock` instance for `logging.Logger`.
    *   `dummy_engine_config()`: Returns a sample dictionary for engine configuration, including `model_path` and `name`.
    *   `dummy_engine_instance(dummy_engine_config, mock_logger)`: Provides a non-initialized `DummyLocalEngine` instance using the config and mock logger.
    *   `dummy_engine_instance_initialized(dummy_engine_instance)`: Takes a `dummy_engine_instance`, calls `initialize()` on it, and returns the initialized instance.
4.  **Test Cases:**
    *   **`test_dummy_engine_init(mock_logger)`:**
        *   Creates a specific `engine_config`.
        *   Instantiates `DummyLocalEngine`.
        *   Asserts that the logger, `model_path`, and other parts of `engine_config` are correctly set as attributes.
        *   Asserts `_is_initialized` is `False` after `__init__`.
    *   **`test_dummy_engine_initialize_success(dummy_engine_instance, mock_logger, caplog)`:**
        *   Takes an uninitialized `dummy_engine_instance`.
        *   Calls `initialize()`.
        *   Asserts `_is_initialized` becomes `True`.
        *   Uses `caplog` to assert that the expected INFO message about initialization (including engine name and model path) was logged.
    *   **`test_dummy_engine_get_engine_name(dummy_engine_instance, dummy_engine_config)`:**
        *   Asserts that `get_engine_name()` returns the name from the configuration if provided.
        *   Tests a separate instance with no 'name' in config to ensure it returns the hardcoded default name from `DummyLocalEngine`.
    *   **`test_dummy_engine_is_available(dummy_engine_instance)`:**
        *   Asserts `is_available()` returns `False` for a newly created (uninitialized) instance.
        *   Calls `initialize()` on the instance.
        *   Asserts `is_available()` now returns `True`.
    *   **`test_dummy_engine_recognize_before_init(dummy_engine_instance, caplog)`:**
        *   Uses an uninitialized `dummy_engine_instance`.
        *   Asserts that calling `recognize()` raises a `RuntimeError` with the expected message.
        *   Uses `caplog` to check that an ERROR message about the engine not being initialized was logged.
    *   **`test_dummy_engine_recognize_success(dummy_engine_instance_initialized, caplog)`:**
        *   Uses an initialized `dummy_engine_instance_initialized`.
        *   Calls `recognize()` with sample image data (a string path) and a language hint.
        *   Asserts that the returned dictionary has the correct structure (`text`, `confidence`, `segments`, `engine_name`).
        *   Asserts that the content of these fields matches the placeholder/simulated output defined in `DummyLocalEngine.recognize`.
        *   Uses `caplog` to verify that an INFO message about the recognition process (including image data type and language hint) was logged.

This test suite covers the basic functionality of the `DummyLocalEngine`, including its initialization, state changes, and core methods, along with logging behavior.
