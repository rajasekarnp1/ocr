# stubs/dummy_engine.py
import logging
from typing import Any, Dict, List, Optional # Ensure all necessary typing imports
from ocr_engine_interface import OCREngine # Adjust path if needed during actual execution

class DummyLocalEngine(OCREngine):
    """
    A dummy implementation of an OCR engine for testing purposes.
    """
    def __init__(self, engine_config: Dict[str, Any], logger: logging.Logger):
        super().__init__(engine_config, logger)
        self.model_path: Optional[str] = engine_config.get("model_path")
        self.logger.debug(f"DummyLocalEngine config: {engine_config}")

    def initialize(self) -> None:
        """Initializes the dummy engine."""
        if not self.model_path:
            self.logger.warning(f"No model_path configured for {self.get_engine_name()}.")
            # For a dummy engine, we might still consider it "initialized"
            # or raise an error if model_path is essential for its dummy logic.
            # For this example, let's assume it can initialize without a real model.
        self.logger.info(f"DummyLocalEngine '{self.get_engine_name()}' initialized. Model path: {self.model_path}")
        self._is_initialized = True

    def recognize(self, image_data: Any, language_hint: Optional[str] = None) -> Dict[str, Any]:
        """Simulates OCR recognition."""
        if not self._is_initialized:
            self.logger.error(f"{self.get_engine_name()} called before initialization.")
            raise RuntimeError(f"Engine {self.get_engine_name()} not initialized.")
        
        self.logger.info(f"DummyLocalEngine recognizing image (type: {type(image_data)}). Language hint: {language_hint}")
        
        # Simulate some processing based on image_data type if needed
        image_info = "simulated_image_processed"
        if isinstance(image_data, str): # if it's a path
            image_info = f"simulated_image_from_path_{image_data}"


        return {
            "text": f"Dummy OCR result from {self.get_engine_name()} for {image_info}",
            "confidence": 0.99,
            "segments": [{
                "text": "Dummy segment", 
                "bounding_box": [0,0,10,10], # x1, y1, x2, y2
                "confidence": 0.99
            }],
            "engine_name": self.get_engine_name()
        }

    def get_engine_name(self) -> str:
        """Returns the name of this dummy engine."""
        return self.engine_config.get("name", "Dummy_Local_Engine_v1")

    def is_available(self) -> bool:
        """Checks if the dummy engine is initialized."""
        return self._is_initialized
