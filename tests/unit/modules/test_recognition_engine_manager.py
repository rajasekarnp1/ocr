import pytest
from unittest.mock import patch, MagicMock, call
import logging

from ocrx.modules.recognition_engine_manager import RecognitionCoreManager
from ocrx.core.data_objects import PageContext, TextRegion, RecognitionResult
from ocrx.core.ocr_engine_interface import OCREngine # For mocking
from ocrx.core.exceptions import OCRXConfigurationError, OCRXProcessingError

# --- Mock Engine Wrapper ---
class MockEngine(OCREngine):
    def __init__(self, engine_config: dict, logger: logging.Logger):
        super().__init__(engine_config, logger)
        self.engine_name = engine_config.get("name", "MockEngineInstance")
        self._is_initialized = False # Will be set by initialize

    def initialize(self) -> None:
        self.logger.info(f"MockEngine {self.engine_name} initializing...")
        if self.engine_config.get("fail_init", False):
            self._is_initialized = False
            raise OCRXModelLoadError(f"MockEngine {self.engine_name} configured to fail initialization.")
        self._is_initialized = True
        self.logger.info(f"MockEngine {self.engine_name} initialized.")

    def recognize(self, image_region: np.ndarray, language_hint: str = None) -> List[RecognitionResult]: # type: ignore
        if not self._is_initialized:
            raise OCRXProcessingError(f"{self.engine_name} not initialized.")

        rec_text = self.engine_config.get("rec_text", f"Recognized by {self.engine_name}")
        rec_confidence = self.engine_config.get("rec_confidence", 0.98)

        # Simulate creating one RecognitionResult per call for simplicity
        return [RecognitionResult(text=rec_text, confidence=rec_confidence, engine_id=self.engine_name)]

    def get_engine_name(self) -> str:
        return self.engine_name

    def is_available(self) -> bool:
        return self._is_initialized


# --- Fixtures ---
@pytest.fixture
def mock_logger_parent():
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def basic_manager_config_one_engine() -> dict:
    return {
        "module_id": "test_rec_manager",
        "default_engine_id": "mock_ocr_engine_1",
        "engines_config": {
            "mock_ocr_engine_1": {
                "enabled": True,
                "class_path": "tests.unit.modules.test_recognition_engine_manager.MockEngine", # Path to MockEngine
                "params": {"name": "TestEngine1", "rec_text": "Text from Engine1"}
            }
        }
    }

@pytest.fixture
def manager_with_one_engine(basic_manager_config_one_engine, mock_logger_parent) -> RecognitionCoreManager:
    # Patch importlib for this fixture's scope if MockEngine isn't in sys.path during real tests
    # For now, assume it's discoverable or use direct patching of the class loading if needed.
    manager = RecognitionCoreManager(
        module_id=basic_manager_config_one_engine["module_id"],
        config=basic_manager_config_one_engine
    )
    # _initialize_resources is called by __init__ of OCRXModuleBase
    return manager


# --- Test Cases ---

def test_manager_init_success(basic_manager_config_one_engine, mock_logger_parent):
    manager = RecognitionCoreManager(
        module_id=basic_manager_config_one_engine["module_id"],
        config=basic_manager_config_one_engine
    )
    assert "mock_ocr_engine_1" in manager.engines
    assert isinstance(manager.engines["mock_ocr_engine_1"], MockEngine)
    assert manager.engines["mock_ocr_engine_1"].is_available()
    assert manager.config["default_engine_id"] == "mock_ocr_engine_1"

def test_manager_init_engine_disabled(mock_logger_parent):
    config = {
        "module_id": "test_disabled",
        "engines_config": {
            "disabled_engine": {
                "enabled": False, # This engine is disabled
                "class_path": "tests.unit.modules.test_recognition_engine_manager.MockEngine",
                "params": {"name": "DisabledEngine"}
            }
        }
    }
    manager = RecognitionCoreManager(module_id=config["module_id"], config=config)
    assert "disabled_engine" not in manager.engines

def test_manager_init_engine_init_fails(mock_logger_parent):
    config = {
        "module_id": "test_init_fail",
        "engines_config": {
            "fail_engine": {
                "enabled": True,
                "class_path": "tests.unit.modules.test_recognition_engine_manager.MockEngine",
                "params": {"name": "FailInitEngine", "fail_init": True} # MockEngine will fail init
            }
        }
    }
    manager = RecognitionCoreManager(module_id=config["module_id"], config=config)
    assert "fail_engine" not in manager.engines # Should not be added if not available

def test_manager_init_no_engines_config(mock_logger_parent):
    config = {"module_id": "no_engines_cfg"} # Missing "engines_config"
    manager = RecognitionCoreManager(module_id=config["module_id"], config=config)
    assert not manager.engines # No engines should be loaded
    # Check for warning log? (Covered by _validate_config tests for OCRXModuleBase generally)

def test_manager_init_bad_class_path(mock_logger_parent):
    config = {
        "module_id": "bad_path",
        "engines_config": {
            "bad_path_engine": {"enabled": True, "class_path": "non.existent.EngineClass"}
        }
    }
    manager = RecognitionCoreManager(module_id=config["module_id"], config=config)
    assert "bad_path_engine" not in manager.engines


@pytest.fixture
def sample_page_ctx_with_regions() -> PageContext:
    """PageContext with a few TextRegions having dummy image_crop data."""
    page = PageContext(page_number=0)
    # Dummy crop (e.g., 30x100 BGR image)
    dummy_crop1 = np.random.randint(0, 255, (30, 100, 3), dtype=np.uint8)
    dummy_crop2 = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)

    page.layout_regions = [
        TextRegion(region_id="r1", bounding_box=(0,0,100,30), image_crop=dummy_crop1),
        TextRegion(region_id="r2", bounding_box=(0,40,120,80), image_crop=dummy_crop2),
        TextRegion(region_id="r3_no_crop", bounding_box=(10,90,50,120), image_crop=None) # No crop
    ]
    return page

def test_manager_process_success(manager_with_one_engine: RecognitionCoreManager, sample_page_ctx_with_regions: PageContext):
    # Mock the actual engine's recognize method to check it's called correctly
    # The engine instance is already a MockEngine due to manager_with_one_engine fixture setup.
    # We can spy on its `recognize` method or further mock it if needed.
    # For this test, let's trust MockEngine's recognize and check its output.

    manager_with_one_engine.process(sample_page_ctx_with_regions)

    assert len(sample_page_ctx_with_regions.layout_regions[0].raw_ocr_results) == 1
    assert sample_page_ctx_with_regions.layout_regions[0].raw_ocr_results[0].text == "Text from Engine1"
    assert sample_page_ctx_with_regions.layout_regions[0].raw_ocr_results[0].engine_id == "TestEngine1"

    assert len(sample_page_ctx_with_regions.layout_regions[1].raw_ocr_results) == 1
    assert sample_page_ctx_with_regions.layout_regions[1].raw_ocr_results[0].text == "Text from Engine1"

    # Region r3 had no crop, so it should have no results and an error logged by manager
    assert len(sample_page_ctx_with_regions.layout_regions[2].raw_ocr_results) == 0
    # (Logging check would require caplog fixture)

def test_manager_process_no_regions(manager_with_one_engine: RecognitionCoreManager):
    page_no_regions = PageContext(page_number=0, layout_regions=[])
    # Spy on engine's recognize to ensure it's not called
    manager_with_one_engine.engines["mock_ocr_engine_1"].recognize = MagicMock() # type: ignore

    manager_with_one_engine.process(page_no_regions)
    manager_with_one_engine.engines["mock_ocr_engine_1"].recognize.assert_not_called() # type: ignore

def test_manager_process_engine_unavailable_fallback(mock_logger_parent):
    config = {
        "module_id": "test_fallback",
        "default_engine_id": "engine_A_fails",
        "engines_config": {
            "engine_A_fails": {
                "enabled": True,
                "class_path": "tests.unit.modules.test_recognition_engine_manager.MockEngine",
                "params": {"name": "EngineA", "fail_init": True} # Fails to init
            },
            "engine_B_works": {
                "enabled": True,
                "class_path": "tests.unit.modules.test_recognition_engine_manager.MockEngine",
                "params": {"name": "EngineB", "rec_text": "Text from EngineB"}
            }
        }
    }
    manager = RecognitionCoreManager(module_id=config["module_id"], config=config)
    assert "engine_A_fails" not in manager.engines # Failed init
    assert "engine_B_works" in manager.engines

    page_ctx = PageContext(page_number=0, layout_regions=[
        TextRegion(region_id="r1", bounding_box=(0,0,1,1), image_crop=np.zeros((10,10,3),dtype=np.uint8))
    ])
    manager.process(page_ctx) # Should use engine_B_works as fallback

    assert len(page_ctx.layout_regions[0].raw_ocr_results) == 1
    assert page_ctx.layout_regions[0].raw_ocr_results[0].text == "Text from EngineB"
    assert page_ctx.layout_regions[0].raw_ocr_results[0].engine_id == "EngineB"


def test_manager_process_no_engines_available_raises_error(mock_logger_parent):
    config_no_active_engines = {
        "module_id": "no_active_engines",
        "engines_config": {
            "engine_X": {"enabled": True, "class_path": "...", "params": {"fail_init": True}}
        }
    }
    # Path to MockEngine for dynamic loading
    config_no_active_engines["engines_config"]["engine_X"]["class_path"] = "tests.unit.modules.test_recognition_engine_manager.MockEngine"

    manager = RecognitionCoreManager(module_id=config_no_active_engines["module_id"], config=config_no_active_engines)
    assert not manager.engines # No engines successfully loaded

    page_ctx_dummy = PageContext(page_number=0, layout_regions=[
         TextRegion(region_id="r1", bounding_box=(0,0,1,1), image_crop=np.zeros((10,10,3),dtype=np.uint8))
    ])
    with pytest.raises(OCRXConfigurationError, match="No OCR engines available for recognition"):
        manager.process(page_ctx_dummy)


def test_manager_process_engine_recognize_fails(manager_with_one_engine: RecognitionCoreManager, sample_page_ctx_with_regions: PageContext, caplog):
    # Make the mock engine's recognize method raise an error
    mock_engine_instance = manager_with_one_engine.engines["mock_ocr_engine_1"]
    mock_engine_instance.recognize = MagicMock(side_effect=OCRXProcessingError("Mocked engine recognize error")) # type: ignore

    with caplog.at_level(logging.ERROR):
        manager_with_one_engine.process(sample_page_ctx_with_regions)

    # Check that an error was logged for each region that had a crop
    assert len(caplog.records) == 2 # For r1 and r2 (r3 has no crop)
    assert "Error recognizing text in region r1" in caplog.text
    assert "Error recognizing text in region r2" in caplog.text

    # Results list should be empty for those regions
    assert len(sample_page_ctx_with_regions.layout_regions[0].raw_ocr_results) == 0
    assert len(sample_page_ctx_with_regions.layout_regions[1].raw_ocr_results) == 0
    # Page errors should be populated
    assert len(sample_page_ctx_with_regions.errors) == 2
    assert "Error recognizing text in region r1" in sample_page_ctx_with_regions.errors[0]

def test_manager_disabled(basic_manager_config_one_engine, sample_page_ctx_with_regions: PageContext):
    config = basic_manager_config_one_engine.copy()
    config["enabled"] = False # Disable the manager itself
    manager = RecognitionCoreManager(module_id=config["module_id"], config=config)

    # Spy on the engine's recognize method
    # Need to get the engine instance after manager init if it was created
    # However, if manager is disabled, its _initialize_resources might not even load engines.
    # Let's assume engines are loaded but process() bails out early.

    # For this test, let's ensure an engine would be available if manager was enabled
    # We can patch the process method of the engine to see if it's called.
    with patch.object(MockEngine, 'recognize', new_callable=MagicMock) as mock_engine_recognize:
        # Re-create manager with the patched MockEngine if needed, or patch instance after creation
        # Easiest: if engines are loaded, patch on the instance.
        # The current setup loads engines in _initialize_resources called by base __init__.
        # So, the engine instance exists.
        if "mock_ocr_engine_1" in manager.engines: # It should be there if config was ok
             manager.engines["mock_ocr_engine_1"].recognize = mock_engine_recognize # type: ignore

        manager.process(sample_page_ctx_with_regions)
        mock_engine_recognize.assert_not_called() # Because manager itself is disabled.
