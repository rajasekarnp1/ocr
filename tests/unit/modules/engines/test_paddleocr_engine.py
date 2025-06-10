import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

# Import the wrapper and related classes
from ocrx.modules.engines.paddleocr_engine import PaddleOCREngineWrapper, PADDLEOCR_AVAILABLE
from ocrx.core.data_objects import RecognitionResult
from ocrx.core.exceptions import OCRXModelLoadError, OCRXProcessingError
import logging

# --- Mock for PaddleOCR library ---
# This mock will be used if PaddleOCR is not actually installed, or to control its behavior.

class MockPaddleOCR:
    def __init__(self, **kwargs):
        self.config_params = kwargs
        logging.info(f"MockPaddleOCR initialized with params: {kwargs}")
        # Simulate model loading failure based on a dummy config param
        if kwargs.get("lang", "") == "fail_load":
            raise OCRXModelLoadError("Mock PaddleOCR load failure due to 'fail_load' lang.")

    def ocr(self, img_data, cls=True):
        logging.info(f"MockPaddleOCR.ocr called with image (shape: {img_data.shape}), cls: {cls}")
        # Simulate some OCR results based on image properties or a known state
        # Standard PaddleOCR output:
        # result = [[[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]], (text, confidence)], ...]
        # Or result = [None] if no text detected

        # Simple mock: return one line of text if image is not tiny
        if img_data.shape[0] > 10 and img_data.shape[1] > 10:
            mock_line = [
                [[[10, 10], [100, 10], [100, 30], [10, 30]]], # Bounding box (quadrilateral)
                ("Mocked text from PaddleOCR", 0.95)          # Text and confidence
            ]
            # PaddleOCR often returns a list containing one sublist of results
            return [[mock_line]]
        else:
            return [[None]] # Simulate no text detected for very small images

# --- Fixtures ---
@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def basic_paddle_config() -> dict:
    return {
        "lang": "en",
        "use_angle_cls": True,
        "show_log": False, # Suppress PaddleOCR's own logging during tests
        # Add other relevant params if your wrapper uses them
    }

@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """A sample BGR image region."""
    return np.random.randint(0, 256, size=(50, 200, 3), dtype=np.uint8)


# --- Test Cases ---

@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', new_callable=lambda: MockPaddleOCR) # Always mock for tests
def test_paddle_engine_init_success(MockPaddleOCRLib, basic_paddle_config, mock_logger):
    """Test successful initialization of PaddleOCREngineWrapper."""
    # Ensure PADDLEOCR_AVAILABLE is True for this test path if it's checked in __init__ beyond just lib presence
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        assert engine.is_available()
        assert engine.ocr_instance is not None
        assert isinstance(engine.ocr_instance, MockPaddleOCR) # Check it used our mock
        mock_logger.info.assert_any_call("PaddleOCR engine 'PaddleOCR_Engine_UnknownPaddleVersion' initialized successfully.")


@patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', False) # Simulate lib not installed
@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', None) # Ensure PaddleOCR is None
def test_paddle_engine_init_lib_not_available(basic_paddle_config, mock_logger):
    """Test initialization when PaddleOCR library is not available."""
    engine = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
    assert not engine.is_available()
    assert engine.ocr_instance is None
    mock_logger.error.assert_called_with("PaddleOCR library not found. Please install paddleocr and paddlepaddle.")


@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', new_callable=lambda: MockPaddleOCR)
def test_paddle_engine_init_model_load_failure(MockPaddleOCRLib, mock_logger):
    """Test initialization when PaddleOCR's internal model loading fails (simulated)."""
    failing_config = {"lang": "fail_load"} # MockPaddleOCR will raise on this
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine = PaddleOCREngineWrapper(engine_config=failing_config, logger=mock_logger)
        assert not engine.is_available() # Should be False due to init failure
        mock_logger.error.assert_any_call("Failed to initialize PaddleOCR instance: Mock PaddleOCR load failure due to 'fail_load' lang.", exc_info=True)

@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', new_callable=lambda: MockPaddleOCR)
def test_paddle_engine_recognize_success(MockPaddleOCRLib, basic_paddle_config, mock_logger, sample_bgr_image):
    """Test successful text recognition."""
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        assert engine.is_available()

        results = engine.recognize(sample_bgr_image)

        assert isinstance(results, list)
        assert len(results) == 1 # MockPaddleOCR returns one line
        result = results[0]
        assert isinstance(result, RecognitionResult)
        assert result.text == "Mocked text from PaddleOCR"
        assert result.confidence == 0.95
        assert result.engine_id == engine.get_engine_name()
        assert result.char_boxes is not None # Wrapper creates a line-level box here
        assert result.char_boxes[0] == (10, 10, 100, 30) # Min/max of the quad

@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', new_callable=lambda: MockPaddleOCR)
def test_paddle_engine_recognize_no_text_detected(MockPaddleOCRLib, basic_paddle_config, mock_logger):
    """Test recognition when PaddleOCR detects no text."""
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        assert engine.is_available()

        # Mock ocr_instance.ocr to return no text
        engine.ocr_instance.ocr = MagicMock(return_value=[[None]]) # type: ignore

        small_image = np.zeros((5,5,3), dtype=np.uint8) # MockPaddleOCR returns no text for small images
        results = engine.recognize(small_image)

        assert isinstance(results, list)
        assert len(results) == 0 # No RecognitionResult objects should be created


@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', new_callable=lambda: MockPaddleOCR)
def test_paddle_engine_recognize_engine_not_available(MockPaddleOCRLib, basic_paddle_config, mock_logger, sample_bgr_image):
    """Test recognize call when engine is not available."""
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        # Manually make it unavailable for this test part
        engine._is_initialized = False
        engine.ocr_instance = None

        with pytest.raises(OCRXProcessingError, match="PaddleOCR engine .* is not available"):
            engine.recognize(sample_bgr_image)

@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', new_callable=lambda: MockPaddleOCR)
def test_paddle_engine_recognize_paddle_ocr_call_fails(MockPaddleOCRLib, basic_paddle_config, mock_logger, sample_bgr_image):
    """Test recognition if the ocr_instance.ocr() call itself raises an exception."""
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        assert engine.is_available()

        engine.ocr_instance.ocr = MagicMock(side_effect=Exception("Paddle Internal Error")) # type: ignore

        with pytest.raises(OCRXProcessingError, match="PaddleOCR recognition failed: Paddle Internal Error"):
            engine.recognize(sample_bgr_image)

def test_get_engine_name(basic_paddle_config, mock_logger):
    """Test the get_engine_name method."""
    # Test without PADDLEOCR_AVAILABLE to ensure it works even if lib is missing (name comes from config)
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', False):
        engine = PaddleOCREngineWrapper(engine_config={"ocr_version": "TestV2"}, logger=mock_logger)
        assert engine.get_engine_name() == "PaddleOCR_Engine_TestV2"

        engine_no_version = PaddleOCREngineWrapper(engine_config={}, logger=mock_logger)
        assert engine_no_version.get_engine_name() == "PaddleOCR_Engine_UnknownPaddleVersion"


@patch('ocrx.modules.engines.paddleocr_engine.PaddleOCR', new_callable=lambda: MockPaddleOCR)
def test_paddle_engine_initialize_method(MockPaddleOCRLib, basic_paddle_config, mock_logger):
    """Test the separate initialize() method for re-init attempt or error."""
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        engine.initialize() # Should log "already initialized"
        mock_logger.info.assert_any_call("PaddleOCR engine 'PaddleOCR_Engine_UnknownPaddleVersion' is already initialized.")

    # Simulate scenario where __init__ failed to set self.ocr_instance but PADDLEOCR_AVAILABLE is True
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', True):
        engine_reinit = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        engine_reinit.ocr_instance = None # Manually unset
        engine_reinit._is_initialized = False
        engine_reinit.initialize()
        assert engine_reinit.is_available()
        mock_logger.info.assert_any_call("PaddleOCR re-initialized successfully during initialize().")

    # Test initialize() when library is not available
    with patch('ocrx.modules.engines.paddleocr_engine.PADDLEOCR_AVAILABLE', False):
        engine_no_lib = PaddleOCREngineWrapper(engine_config=basic_paddle_config, logger=mock_logger)
        with pytest.raises(OCRXModelLoadError, match="PaddleOCR library is not installed"):
            engine_no_lib.initialize()
