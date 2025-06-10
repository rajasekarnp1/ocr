import pytest
from ocrx.core.exceptions import (
    OCRXError,
    OCRXConfigurationError,
    OCRXProcessingError,
    OCRXInputError,
    OCRXModelLoadError
)

def test_ocrx_error_can_be_raised():
    with pytest.raises(OCRXError, match="Generic OCRX error"):
        raise OCRXError("Generic OCRX error")

def test_ocrx_configuration_error_can_be_raised():
    with pytest.raises(OCRXConfigurationError, match="Configuration issue"):
        raise OCRXConfigurationError("Configuration issue")

def test_ocrx_processing_error_can_be_raised():
    with pytest.raises(OCRXProcessingError, match="Processing problem"):
        raise OCRXProcessingError("Processing problem")

def test_ocrx_input_error_can_be_raised():
    with pytest.raises(OCRXInputError, match="Invalid input"):
        raise OCRXInputError("Invalid input")

def test_ocrx_model_load_error_can_be_raised():
    with pytest.raises(OCRXModelLoadError, match="Model loading failed"):
        raise OCRXModelLoadError("Model loading failed")

def test_exception_hierarchy():
    assert issubclass(OCRXConfigurationError, OCRXError)
    assert issubclass(OCRXProcessingError, OCRXError)
    assert issubclass(OCRXInputError, OCRXError)
    assert issubclass(OCRXModelLoadError, OCRXError)

def test_exception_message_persists():
    message = "A very specific error message"
    try:
        raise OCRXProcessingError(message)
    except OCRXProcessingError as e:
        assert str(e) == message
