"""
Custom exceptions for the OCR Application.
"""

class BaseOCRError(Exception):
    """Base class for custom exceptions in the OCR application."""
    pass

class OCRConfigurationError(BaseOCRError):
    """Exception raised for errors in configuration loading or validation."""
    pass

class OCRPipelineError(BaseOCRError):
    """Exception raised for errors during the OCR processing pipeline."""
    pass

class OCRFileNotFoundError(BaseOCRError, FileNotFoundError):
    """Exception raised when a required file (e.g., model, image, dictionary) is not found."""
    # Inherits from FileNotFoundError for compatibility where FileNotFoundError is specifically caught.
    pass

class OCRModelError(BaseOCRError):
    """Exception raised for errors related to ONNX model loading or execution."""
    pass

class OCRImageProcessingError(BaseOCRError):
    """Exception raised for errors during image processing steps like binarization or deskewing."""
    pass
# Add more specific exceptions as needed, e.g.:
# class OCRTextProcessingError(BaseOCRError):
#     """Exception raised for errors during text cleaning or spell checking."""
#     pass
