import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from ocrx.core.exceptions import OCRXConfigurationError, OCRXProcessingError, OCRXModelLoadError

class OCRXModuleBase(ABC):
    """
    Abstract Base Class for an OCR-X pipeline module.
    """
    def __init__(self, module_id: str, config: Dict[str, Any]):
        self.module_id = module_id
        self.config = config if isinstance(config, dict) else {}
        # Logger name will include the actual subclass name later if possible,
        # or can be passed in. For now, use a generic module logger.
        # A more advanced setup might involve a logger factory.
        self.logger = logging.getLogger(f"ocrx.module.{self.module_id}")

        self._validate_config()
        self._initialize_resources()
        self.logger.info(f"Module '{self.module_id}' of type {self.__class__.__name__} initialized.")

    def _validate_config(self) -> None:
        self.logger.debug(f"Validating config for {self.module_id}")
        if not isinstance(self.config, dict):
            # This case is handled by the default in __init__, but good for clarity
            msg = f"Configuration for module '{self.module_id}' must be a dictionary."
            self.logger.error(msg)
            raise OCRXConfigurationError(msg)

        if not self.config.get("enabled", True):
             self.logger.warning(f"Module '{self.module_id}' is disabled via configuration.")
        # Subclasses will add more specific checks here.

    def _initialize_resources(self) -> None:
        self.logger.debug(f"Initializing resources for {self.module_id}.")
        # Subclasses will load models, connect to services, etc.
        pass

    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        if not model_path:
            raise OCRXConfigurationError(f"Model path not provided for {self.module_id}")
        self.logger.info(f"Loading model from '{model_path}' for {self.module_id}...")
        # Actual model loading logic by subclasses
        # This base implementation is a placeholder and should be overridden
        raise NotImplementedError(f"load_model() not implemented in {self.__class__.__name__}")

    @abstractmethod
    def process(self, data: Any, **kwargs: Any) -> Any:
        pass

    def is_enabled(self) -> bool:
        return self.config.get("enabled", True)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(module_id='{self.module_id}')>"
