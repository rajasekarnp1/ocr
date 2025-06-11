"""
Unit tests for ocr_workflow_orchestrator.py using pytest.
"""

import pytest
import os
import sys
import logging
import importlib
from unittest.mock import patch, MagicMock, PropertyMock

# --- Path Setup (Simulating conftest.py behavior for standalone execution) ---
# In a real project, this would typically be in conftest.py
# or the project would be installed as a package.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
stubs_path = os.path.join(project_root, "stubs")

# Add project root and stubs to sys.path to allow imports
# This is crucial for importlib.import_module to find 'stubs.dummy_engine'
# and for direct imports of orchestrator modules.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if stubs_path not in sys.path:
    sys.path.insert(0, stubs_path)
# --- End Path Setup ---

from ocr_workflow_orchestrator import OCRWorkflowOrchestrator, OCREngineManager
from ocr_engine_interface import OCREngine
from config_loader import load_config, DEFAULT_LOGGING_CONFIG # Used by orchestrator
from stubs.dummy_engine import DummyLocalEngine # For direct use in some tests
from PIL import Image, UnidentifiedImageError
import numpy as np
import yaml # For creating dummy config files

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Returns a mock logger."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def basic_app_config_dict() -> dict:
    """A basic application configuration dictionary for tests."""
    return {
        "app_settings": {
            "default_ocr_engine": "dummy_engine_1",
            "temp_file_dir": "temp_test_files" # Example app setting
        },
        "ocr_engines": {
            "dummy_engine_1": {
                "enabled": True,
                "module": "stubs.dummy_engine",
                "class": "DummyLocalEngine",
                "config": {"name": "TestDummy1", "model_path": "models/dummy1.onnx"}
            },
            "dummy_engine_2": {
                "enabled": True,
                "module": "stubs.dummy_engine",
                "class": "DummyLocalEngine", # Assuming same class, different config
                "config": {"name": "TestDummy2", "model_path": "models/dummy2.onnx"}
            },
            "disabled_engine": {
                "enabled": False,
                "module": "stubs.dummy_engine",
                "class": "DummyLocalEngine",
                "config": {"name": "DisabledTestDummy", "model_path": "models/disabled.onnx"}
            }
        },
        # Include a basic logging config to prevent errors if not overridden by specific tests
        "logging": DEFAULT_LOGGING_CONFIG
    }

@pytest.fixture
def temp_config_file(tmp_path, basic_app_config_dict) -> str:
    """Creates a temporary YAML config file and returns its path."""
    config_file_path = tmp_path / "test_config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        yaml.dump(basic_app_config_dict, f)
    return str(config_file_path)

@pytest.fixture
def orchestrator(temp_config_file, monkeypatch) -> OCRWorkflowOrchestrator:
    """Fixture to get an initialized OCRWorkflowOrchestrator instance."""
    # Ensure the stubs directory is in path for engine loading
    monkeypatch.syspath_prepend(stubs_path) # Ensure stubs/ is discoverable

    # Patch load_image to prevent actual file operations unless specifically tested
    with patch.object(OCRWorkflowOrchestrator, 'load_image', return_value=np.zeros((10,10,3), dtype=np.uint8)) as _:
        orch = OCRWorkflowOrchestrator(config_path=temp_config_file)
    return orch

@pytest.fixture
def orchestrator_with_dummy_engine(temp_config_file, monkeypatch) -> OCRWorkflowOrchestrator:
    """
    Fixture for an orchestrator confirmed to have dummy_engine_1 loaded.
    This is similar to 'orchestrator' but useful for tests that assume an engine is ready.
    """
    monkeypatch.syspath_prepend(stubs_path)
    # Patch load_image to prevent actual file operations unless specifically tested
    with patch.object(OCRWorkflowOrchestrator, 'load_image', return_value=np.zeros((10,10,3), dtype=np.uint8)) as _:
        orch = OCRWorkflowOrchestrator(config_path=temp_config_file)

    # Check if the engine was loaded, if not, something is wrong with the test setup
    assert "dummy_engine_1" in orch.engine_manager.get_available_engines(), \
        "Setup Error: dummy_engine_1 not loaded in orchestrator_with_dummy_engine fixture."
    return orch

@pytest.fixture
def orchestrator_no_engines(tmp_path, monkeypatch, basic_app_config_dict) -> OCRWorkflowOrchestrator:
    """Fixture for an orchestrator where no engines are configured/enabled."""
    empty_engine_config = basic_app_config_dict.copy()
    empty_engine_config["ocr_engines"] = {} # No engines defined

    config_file_path = tmp_path / "no_engines_config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        yaml.dump(empty_engine_config, f)

    monkeypatch.syspath_prepend(stubs_path)
    with patch.object(OCRWorkflowOrchestrator, 'load_image', return_value=np.zeros((10,10,3), dtype=np.uint8)) as _:
        orch = OCRWorkflowOrchestrator(config_path=str(config_file_path))
    return orch


# --- OCREngineManager Tests ---

def test_engine_manager_init(basic_app_config_dict, mock_logger):
    """Test OCREngineManager initialization."""
    manager = OCREngineManager(app_config=basic_app_config_dict, parent_logger=mock_logger)
    assert manager.engines == {}
    assert manager.engine_configs == basic_app_config_dict.get('ocr_engines')
    mock_logger.getChild.assert_called_with('OCREngineManager')


def test_discover_and_load_engines_success(basic_app_config_dict, mock_logger, monkeypatch):
    """Test successful discovery and loading of enabled engines."""
    monkeypatch.syspath_prepend(stubs_path) # Ensure dummy_engine is findable

    manager = OCREngineManager(app_config=basic_app_config_dict, parent_logger=mock_logger)
    manager.discover_and_load_engines()

    assert "dummy_engine_1" in manager.engines
    assert isinstance(manager.engines["dummy_engine_1"], DummyLocalEngine)
    assert manager.engines["dummy_engine_1"].get_engine_name() == "TestDummy1"
    assert "dummy_engine_2" in manager.engines
    assert "disabled_engine" not in manager.engines
    assert len(manager.get_available_engines()) == 2
    assert "dummy_engine_1" in manager.get_available_engines()


def test_discover_and_load_engines_module_not_found(basic_app_config_dict, caplog, mock_logger):
    """Test engine loading when module is not found."""
    bad_config = basic_app_config_dict.copy()
    bad_config["ocr_engines"]["bad_module_engine"] = {
        "enabled": True, "module": "non_existent_stubs.non_existent_engine", "class": "SomeClass"
    }
    manager = OCREngineManager(app_config=bad_config, parent_logger=mock_logger)
    manager.discover_and_load_engines()

    assert "bad_module_engine" not in manager.engines
    assert any(
        "Failed to import module 'non_existent_stubs.non_existent_engine' for engine 'bad_module_engine'" in record.message
        for record in caplog.records if record.levelname == "ERROR"
    )

def test_discover_and_load_engines_class_not_found(basic_app_config_dict, caplog, mock_logger, monkeypatch):
    """Test engine loading when class is not found in an existing module."""
    monkeypatch.syspath_prepend(stubs_path)
    bad_config = basic_app_config_dict.copy()
    bad_config["ocr_engines"]["bad_class_engine"] = {
        "enabled": True, "module": "stubs.dummy_engine", "class": "NonExistentClassInDummy"
    }
    manager = OCREngineManager(app_config=bad_config, parent_logger=mock_logger)
    manager.discover_and_load_engines()

    assert "bad_class_engine" not in manager.engines
    assert any(
        "Failed to get class 'NonExistentClassInDummy' from module 'stubs.dummy_engine' for engine 'bad_class_engine'" in record.message
        for record in caplog.records if record.levelname == "ERROR"
    )

@patch("stubs.dummy_engine.DummyLocalEngine.initialize", side_effect=RuntimeError("Initialization failed!"))
def test_discover_and_load_engines_initialization_fails(mock_init, basic_app_config_dict, caplog, mock_logger, monkeypatch):
    """Test engine loading when engine's initialize() method fails."""
    monkeypatch.syspath_prepend(stubs_path)

    manager = OCREngineManager(app_config=basic_app_config_dict, parent_logger=mock_logger)
    manager.discover_and_load_engines()

    # dummy_engine_1 should fail initialization
    assert "dummy_engine_1" not in manager.engines
    # dummy_engine_2 should also try to load, and if its init is not mocked, it might load
    # For this test, we only care about the one we mocked.
    assert any(
        "Failed to initialize engine 'dummy_engine_1': Initialization failed!" in record.message
        for record in caplog.records if record.levelname == "ERROR"
    )

@patch("stubs.dummy_engine.DummyLocalEngine.is_available", return_value=False)
def test_discover_and_load_engines_not_available(mock_is_available, basic_app_config_dict, caplog, mock_logger, monkeypatch):
    """Test engine loading when engine's is_available() returns False after initialization."""
    monkeypatch.syspath_prepend(stubs_path)

    manager = OCREngineManager(app_config=basic_app_config_dict, parent_logger=mock_logger)
    manager.discover_and_load_engines()

    assert "dummy_engine_1" not in manager.engines # Because is_available is False
    assert any(
        "Engine 'dummy_engine_1' loaded but reported not available after initialization." in record.message
        for record in caplog.records if record.levelname == "WARNING"
    )

@pytest.fixture
def engine_manager_with_loaded_engine(basic_app_config_dict, mock_logger, monkeypatch) -> OCREngineManager:
    """Fixture for an OCREngineManager with dummy_engine_1 successfully loaded."""
    monkeypatch.syspath_prepend(stubs_path)
    manager = OCREngineManager(app_config=basic_app_config_dict, parent_logger=mock_logger)
    # We only want dummy_engine_1 for this test to be predictable
    manager.engine_configs = {"dummy_engine_1": basic_app_config_dict["ocr_engines"]["dummy_engine_1"]}
    manager.discover_and_load_engines()
    assert "dummy_engine_1" in manager.engines, "Fixture setup failed: dummy_engine_1 not loaded."
    return manager

def test_get_engine_success_and_fail(engine_manager_with_loaded_engine, caplog):
    """Test get_engine for success and failure cases."""
    manager = engine_manager_with_loaded_engine

    # Success
    engine = manager.get_engine("dummy_engine_1")
    assert engine is not None
    assert isinstance(engine, DummyLocalEngine)
    assert engine.get_engine_name() == "TestDummy1"

    # Fail (non-existent engine)
    engine_none = manager.get_engine("non_existent_engine")
    assert engine_none is None
    assert any(
        "Engine 'non_existent_engine' was requested but not found" in record.message
        for record in caplog.records if record.levelname == "WARNING"
    )

    # Fail (engine loaded but made unavailable - conceptually)
    # To test this properly, we'd need to mock is_available on the instance
    if "dummy_engine_1" in manager.engines:
        with patch.object(manager.engines["dummy_engine_1"], 'is_available', return_value=False):
            unavailable_engine = manager.get_engine("dummy_engine_1")
            assert unavailable_engine is None
            assert any(
                "Engine 'dummy_engine_1' was requested but is not currently available." in record.message
                for record in caplog.records if record.levelname == "WARNING"
            )


# --- OCRWorkflowOrchestrator Tests ---

def test_orchestrator_init_success(temp_config_file, monkeypatch, caplog):
    """Test successful initialization of OCRWorkflowOrchestrator."""
    monkeypatch.syspath_prepend(stubs_path)
    caplog.set_level(logging.INFO) # Ensure INFO messages are captured

    orchestrator_instance = OCRWorkflowOrchestrator(config_path=temp_config_file)

    assert orchestrator_instance.config is not None
    assert "dummy_engine_1" in orchestrator_instance.config.get("ocr_engines", {})
    assert orchestrator_instance.engine_manager is not None
    assert "dummy_engine_1" in orchestrator_instance.engine_manager.get_available_engines()
    assert any("OCRWorkflowOrchestrator initialized." in message for message in caplog.messages)
    assert any("Successfully loaded and initialized engine: 'dummy_engine_1'" in message for message in caplog.messages)


def test_orchestrator_init_no_config_file(caplog):
    """Test orchestrator initialization when config file is not found."""
    caplog.set_level(logging.WARNING)
    # This will use default logging because load_config applies it on file not found
    orchestrator_instance = OCRWorkflowOrchestrator(config_path="this_config_does_not_exist.yaml")

    assert orchestrator_instance.config is not None # load_config returns a default structure
    assert "error" in orchestrator_instance.config.get("app_settings", {})
    assert "this_config_does_not_exist.yaml" in orchestrator_instance.config["app_settings"]["error"]
    assert len(orchestrator_instance.engine_manager.get_available_engines()) == 0
    assert any("Configuration file 'this_config_does_not_exist.yaml' not found." in message for message in caplog.messages)
    assert any("No OCR engines available after discovery and loading." in message for message in caplog.messages)


def test_load_image_success(orchestrator, tmp_path):
    """Test successful image loading."""
    image_file = tmp_path / "test_image.png"
    try:
        Image.new('RGB', (60, 30), color='red').save(image_file)
    except ImportError:
        pytest.skip("Pillow not available or issue creating image, skipping image load test.")

    # Unpatch the load_image for this specific test
    with patch.object(OCRWorkflowOrchestrator, 'load_image', new=OCRWorkflowOrchestrator.load_image.__get__(orchestrator)):
        loaded_image = orchestrator.load_image(str(image_file))
        assert isinstance(loaded_image, np.ndarray)
        assert loaded_image.shape == (30, 60, 3)


def test_load_image_file_not_found(orchestrator, caplog):
    """Test loading a non-existent image file."""
    caplog.set_level(logging.ERROR)
    with patch.object(OCRWorkflowOrchestrator, 'load_image', new=OCRWorkflowOrchestrator.load_image.__get__(orchestrator)):
        with pytest.raises(FileNotFoundError):
            orchestrator.load_image("non_existent_image.png")
    assert any("Image file not found at: non_existent_image.png" in message for message in caplog.messages)


def test_load_image_unsupported_format(orchestrator, tmp_path, caplog):
    """Test loading an unsupported image format."""
    caplog.set_level(logging.ERROR)
    invalid_image_file = tmp_path / "fake_image.txt"
    invalid_image_file.write_text("This is not an image.")

    with patch.object(OCRWorkflowOrchestrator, 'load_image', new=OCRWorkflowOrchestrator.load_image.__get__(orchestrator)):
        with pytest.raises(ValueError) as excinfo: # Expecting ValueError from our handler
            orchestrator.load_image(str(invalid_image_file))
        assert "Unsupported image format or corrupted file" in str(excinfo.value)

    assert any(f"Cannot identify image file (unsupported format or corrupted): {str(invalid_image_file)}" in message for message in caplog.messages)

@patch.object(OCRWorkflowOrchestrator, 'preprocess_image', side_effect=lambda x: x) # Mock to pass through
@patch.object(OCRWorkflowOrchestrator, 'postprocess_text', side_effect=lambda x: x) # Mock to pass through
def test_process_document_e2e_with_dummy_engine(mock_postprocess, mock_preprocess, orchestrator_with_dummy_engine, tmp_path, caplog):
    """End-to-end style test for process_document with a dummy engine."""
    caplog.set_level(logging.INFO)

    dummy_image_file = tmp_path / "e2e_test_image.png"
    try:
        Image.new('RGB', (60, 30), color='blue').save(dummy_image_file)
    except ImportError:
        pytest.skip("Pillow not available or issue creating image, skipping e2e test.")

    # Unpatch load_image for this test to use the real one
    with patch.object(orchestrator_with_dummy_engine, 'load_image', new=OCRWorkflowOrchestrator.load_image.__get__(orchestrator_with_dummy_engine)):
        result = orchestrator_with_dummy_engine.process_document(str(dummy_image_file), requested_engine_name="dummy_engine_1")

    assert result is not None
    assert "text" in result
    assert "Dummy OCR result from TestDummy1" in result["text"]
    assert result["engine_name"] == "TestDummy1"

    mock_preprocess.assert_called_once()
    mock_postprocess.assert_called_once()

    assert any(f"Starting document processing for: '{str(dummy_image_file)}'" in record.message for record in caplog.records)
    assert any("Selected engine for recognition: 'dummy_engine_1'" in record.message for record in caplog.records)
    assert any(f"Recognition by 'dummy_engine_1' completed." in record.message for record in caplog.records)
    assert any(f"Document processing completed successfully for: '{str(dummy_image_file)}'" in record.message for record in caplog.records)

def test_process_document_engine_selection_logic(orchestrator_with_dummy_engine, tmp_path, caplog, monkeypatch):
    """Test various engine selection scenarios in process_document."""
    caplog.set_level(logging.INFO)
    dummy_image_file = tmp_path / "selection_test_image.png"
    try:
        Image.new('RGB', (60, 30), color='green').save(dummy_image_file)
    except ImportError:
        pytest.skip("Pillow not available or issue creating image, skipping selection logic test.")

    # Unpatch load_image for this test to use the real one
    with patch.object(orchestrator_with_dummy_engine, 'load_image', new=OCRWorkflowOrchestrator.load_image.__get__(orchestrator_with_dummy_engine)):
        # 1. Explicitly request valid dummy_engine_2
        result = orchestrator_with_dummy_engine.process_document(str(dummy_image_file), requested_engine_name="dummy_engine_2")
        assert result is not None and "TestDummy2" in result["text"]
        assert any("Selected engine for recognition: 'dummy_engine_2'" in record.message for record in caplog.records)
        caplog.clear()

        # 2. Request non-existent engine (should use default: dummy_engine_1)
        result = orchestrator_with_dummy_engine.process_document(str(dummy_image_file), requested_engine_name="non_existent_engine")
        assert result is not None and "TestDummy1" in result["text"] # Default from basic_app_config_dict
        assert any("Requested engine 'non_existent_engine' is not available. Trying default or first available." in record.message for record in caplog.records)
        assert any("Using default engine: 'dummy_engine_1'" in record.message for record in caplog.records)
        caplog.clear()

        # 3. No engine requested, rely on default_ocr_engine from config (which is dummy_engine_1)
        result = orchestrator_with_dummy_engine.process_document(str(dummy_image_file))
        assert result is not None and "TestDummy1" in result["text"]
        assert any("Using default engine: 'dummy_engine_1'" in record.message for record in caplog.records)
        caplog.clear()

        # 4. No engine requested, no default, rely on first available (monkeypatch config)
        current_config = orchestrator_with_dummy_engine.config.copy()
        del current_config['app_settings']['default_ocr_engine']

        # Need to re-init orchestrator or manager with this modified config
        # For simplicity, let's mock the config directly on the orchestrator for this sub-test
        monkeypatch.setattr(orchestrator_with_dummy_engine, 'config', current_config)

        result = orchestrator_with_dummy_engine.process_document(str(dummy_image_file))
        # The first available engine could be dummy_engine_1 or dummy_engine_2 depending on dict ordering in test
        # So, we check if it's one of them
        assert result is not None and ("TestDummy1" in result["text"] or "TestDummy2" in result["text"])
        first_available = orchestrator_with_dummy_engine.engine_manager.get_available_engines()[0]
        assert any(f"Using first available engine: '{first_available}'" in record.message for record in caplog.records)


def test_process_document_no_engine_available(orchestrator_no_engines, tmp_path, caplog):
    """Test process_document when no engines are loaded/available."""
    caplog.set_level(logging.ERROR)
    dummy_image_file = tmp_path / "no_engine_test.png"
    try:
        Image.new('RGB', (60, 30), color='yellow').save(dummy_image_file)
    except ImportError:
        pytest.skip("Pillow not available or issue creating image, skipping no engine test.")

    # Unpatch load_image for this test to use the real one
    with patch.object(orchestrator_no_engines, 'load_image', new=OCRWorkflowOrchestrator.load_image.__get__(orchestrator_no_engines)):
        result = orchestrator_no_engines.process_document(str(dummy_image_file))

    assert result is None # Should return None as no engine can process
    assert any("No OCR engine available to process the document." in record.message for record in caplog.records)

```
The `tests/test_ocr_workflow_orchestrator.py` file has been created with a comprehensive suite of pytest tests.

**Summary of the Test File Content:**

1.  **Path Setup:** Includes a section to dynamically add the project root and `stubs` directory to `sys.path`. This is a workaround for running tests in an environment where the main project isn't formally installed as a package.
2.  **Imports:** All necessary modules and classes are imported.
3.  **Fixtures:**
    *   `mock_logger()`: Provides a `MagicMock` for `logging.Logger`.
    *   `basic_app_config_dict()`: Returns a dictionary representing a typical application configuration, including definitions for `dummy_engine_1`, `dummy_engine_2`, and a `disabled_engine`.
    *   `temp_config_file(tmp_path, basic_app_config_dict)`: Creates a temporary YAML configuration file using `tmp_path` and the `basic_app_config_dict`.
    *   `orchestrator(temp_config_file, monkeypatch)`: Initializes an `OCRWorkflowOrchestrator` instance using the `temp_config_file`. It also patches `load_image` by default to avoid file I/O unless specifically unpatched in a test.
    *   `orchestrator_with_dummy_engine(temp_config_file, monkeypatch)`: Similar to `orchestrator` but asserts that `dummy_engine_1` is loaded, ensuring it's ready for tests that depend on an active engine.
    *   `orchestrator_no_engines(tmp_path, monkeypatch, basic_app_config_dict)`: Creates an orchestrator instance with a configuration that has no engines defined or enabled.
4.  **OCREngineManager Tests:**
    *   `test_engine_manager_init`: Checks correct initialization of attributes.
    *   `test_discover_and_load_engines_success`: Verifies that enabled dummy engines are loaded correctly and disabled ones are skipped.
    *   `test_discover_and_load_engines_module_not_found`: Ensures appropriate error logging when a configured engine module cannot be found.
    *   `test_discover_and_load_engines_class_not_found`: Checks error logging when an engine class is not found within an existing module.
    *   `test_discover_and_load_engines_initialization_fails`: Uses `@patch` to make a dummy engine's `initialize()` method raise an exception and verifies the engine is not loaded and an error is logged.
    *   `test_discover_and_load_engines_not_available`: Uses `@patch` to make a dummy engine's `is_available()` return `False` and verifies it's not loaded and a warning is logged.
    *   `test_get_engine_success_and_fail`: Uses a pre-configured manager to test retrieving existing, non-existing, and unavailable engines.
5.  **OCRWorkflowOrchestrator Tests:**
    *   `test_orchestrator_init_success`: Checks successful orchestrator initialization, including engine loading, using a valid temporary config.
    *   `test_orchestrator_init_no_config_file`: Verifies fallback to default logging and no engines loaded when the config file is missing.
    *   `test_load_image_success`: Tests successful image loading using Pillow (unpatches the default `load_image` mock for this test).
    *   `test_load_image_file_not_found`: Checks for `FileNotFoundError` and appropriate logging.
    *   `test_load_image_unsupported_format`: Checks for `ValueError` (due to `UnidentifiedImageError`) and logging when an invalid image file is provided.
    *   `test_process_document_e2e_with_dummy_engine`: An end-to-end style test that uses an orchestrator with a loaded dummy engine, processes a dummy image, and checks if the expected dummy text is returned. It also mocks `preprocess_image` and `postprocess_text` to verify they are called.
    *   `test_process_document_engine_selection_logic`: Tests various scenarios for engine selection: explicitly requested, non-existent requested (fallback to default), no request (use default), and no request with no default (use first available).
    *   `test_process_document_no_engine_available`: Uses an orchestrator with no engines loaded and ensures `process_document` handles this gracefully by returning `None` and logging an error.

The tests make extensive use of `pytest` features like `tmp_path` for creating temporary files, `monkeypatch` for modifying `sys.path` and object attributes, and `caplog` for capturing and asserting log messages. `unittest.mock.patch` is also used for mocking methods. This provides a solid foundation for testing the orchestrator's core logic.
