class OCRXError(Exception):
    """Base class for exceptions in the OCR-X application."""
    pass

class OCRXConfigurationError(OCRXError):
    """Raised for errors in configuration."""
    pass

class OCRXProcessingError(OCRXError):
    """Raised for errors during a processing step in a module."""
    pass

class OCRXInputError(OCRXError):
    """Raised for invalid input data provided to a module or function."""
    pass

class OCRXModelLoadError(OCRXError):
    """Raised when a model fails to load."""
    pass
