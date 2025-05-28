"""
OCR Workflow Orchestrator for the OCR-X Project.

This module defines the main orchestrator class responsible for managing the
OCR pipeline, including image loading, preprocessing, engine management,
recognition, and postprocessing.
"""

import logging
import importlib
import os
from typing import Any, Dict, List, Optional

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from ocr_engine_interface import OCREngine
from config_loader import load_config, DEFAULT_LOGGING_CONFIG

# For image loading - install Pillow: pip install Pillow
from PIL import Image, UnidentifiedImageError
import numpy as np


class OCREngineManager:
    """
    Manages the discovery, loading, and access of available OCR engines.
    """
    def __init__(self, app_config: Dict[str, Any], parent_logger: logging.Logger):
        """
        Initializes the OCREngineManager.

        :param app_config: The global application configuration dictionary.
        :param parent_logger: The parent logger instance (typically from OCRWorkflowOrchestrator).
        """
        self.engines: Dict[str, OCREngine] = {}
        self.engine_configs: Dict[str, Any] = app_config.get('ocr_engines', {})
        self.logger = parent_logger.getChild(self.__class__.__name__)
        self.logger.info(f"OCREngineManager initialized with {len(self.engine_configs)} engine configurations.")

    def discover_and_load_engines(self) -> None:
        """
        Discovers and loads OCR engines based on the application configuration.
        """
        self.logger.info("Starting OCR engine discovery and loading process...")
        for engine_name, engine_config_data in self.engine_configs.items():
            if not engine_config_data.get('enabled', False):
                self.logger.info(f"Engine '{engine_name}' is disabled in configuration. Skipping.")
                continue

            module_path = engine_config_data.get('module')
            class_name = engine_config_data.get('class')

            if not module_path or not class_name:
                self.logger.error(f"Engine '{engine_name}' configuration is missing 'module' or 'class' path. Skipping.")
                continue

            try:
                self.logger.debug(f"Attempting to load engine '{engine_name}' from module '{module_path}' class '{class_name}'.")
                engine_module = importlib.import_module(module_path)
                engine_class = getattr(engine_module, class_name)
                
                # Pass only the specific engine's config section, not the whole ocr_engines dict
                engine_instance_config = engine_config_data.get('config', {}) # Assuming engine-specific params are under 'config'
                if not engine_instance_config and engine_config_data: # Fallback for flat config structure
                    # Filter out common keys like 'enabled', 'module', 'class' to pass only specific params
                    engine_instance_config = {k: v for k, v in engine_config_data.items() if k not in ['enabled', 'module', 'class']}


                child_logger = self.logger.getChild(engine_name) # Create a child logger for the engine instance
                engine_instance: OCREngine = engine_class(engine_config=engine_instance_config, logger=child_logger)
                
                engine_instance.initialize() # Can raise exceptions

                if engine_instance.is_available():
                    self.engines[engine_name] = engine_instance
                    self.logger.info(f"Successfully loaded and initialized engine: '{engine_name}' ({engine_instance.get_engine_name()}).")
                else:
                    self.logger.warning(f"Engine '{engine_name}' loaded but reported not available after initialization.")

            except ImportError as e:
                self.logger.error(f"Failed to import module '{module_path}' for engine '{engine_name}': {e}", exc_info=True)
            except AttributeError as e:
                self.logger.error(f"Failed to get class '{class_name}' from module '{module_path}' for engine '{engine_name}': {e}", exc_info=True)
            except TypeError as e: # Catches errors in engine constructor call
                self.logger.error(f"TypeError instantiating engine '{engine_name}'. Check constructor arguments: {e}", exc_info=True)
            except Exception as e: # Catch exceptions from engine's initialize() or other instantiation issues
                self.logger.error(f"Failed to initialize engine '{engine_name}': {e}", exc_info=True)
        
        self.logger.info(f"Engine discovery complete. {len(self.engines)} engines are available.")

    def get_engine(self, engine_name: str) -> Optional[OCREngine]:
        """
        Returns the instantiated engine if it exists and is available.

        :param engine_name: The name of the engine to retrieve.
        :return: An instance of the OCREngine, or None if not found or unavailable.
        """
        engine = self.engines.get(engine_name)
        if engine and engine.is_available():
            return engine
        elif engine: # Exists but not available
            self.logger.warning(f"Engine '{engine_name}' was requested but is not currently available.")
        else: # Does not exist
            self.logger.warning(f"Engine '{engine_name}' was requested but not found (or not loaded).")
        return None

    def get_available_engines(self) -> List[str]:
        """
        Returns a list of names of successfully loaded and available engines.

        :return: A list of engine names.
        """
        return [name for name, engine in self.engines.items() if engine.is_available()]


class OCRWorkflowOrchestrator:
    """
    Orchestrates the entire OCR workflow, from loading images to returning processed text.
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the OCRWorkflowOrchestrator.

        :param config_path: Path to the main application configuration file.
        """
        # Load configuration first, which also sets up logging
        self.config: Dict[str, Any] = load_config(config_path)
        
        # Now that logging is configured by load_config, get the logger for this class
        self.logger = logging.getLogger(__name__) # Or a specific name like 'OCRWorkflowOrchestrator'
        self.logger.info("OCRWorkflowOrchestrator initializing...")

        # Initialize Engine Manager
        self.engine_manager = OCREngineManager(app_config=self.config, parent_logger=self.logger)
        self.engine_manager.discover_and_load_engines()

        # Placeholder for Preprocessor and Postprocessor
        self.preprocessor: Optional[Any] = None # Replace Any with actual preprocessor type later
        self.postprocessor: Optional[Any] = None # Replace Any with actual postprocessor type later
        self.logger.info("Preprocessor and Postprocessor are placeholders.")

        available_engines = self.engine_manager.get_available_engines()
        if available_engines:
            self.logger.info(f"Available OCR engines: {', '.join(available_engines)}")
        else:
            self.logger.warning("No OCR engines available after discovery and loading.")
        
        self.logger.info("OCRWorkflowOrchestrator initialized.")

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Loads an image from the specified path and converts it to an RGB NumPy array.

        :param image_path: The path to the image file.
        :return: An RGB NumPy array representing the image, or None if loading fails.
        """
        self.logger.info(f"Loading image from: {image_path}")
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found at: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            img = Image.open(image_path)
            img_rgb = img.convert("RGB") # Ensure 3 channels (RGB)
            numpy_image = np.array(img_rgb)
            self.logger.debug(f"Image '{image_path}' loaded successfully. Shape: {numpy_image.shape}, Type: {numpy_image.dtype}")
            return numpy_image
        except FileNotFoundError: # Should be caught by os.path.exists, but good to have
            self.logger.error(f"Image file not found (PIL): {image_path}", exc_info=True)
            raise
        except UnidentifiedImageError:
            self.logger.error(f"Cannot identify image file (unsupported format or corrupted): {image_path}", exc_info=True)
            raise ValueError(f"Unsupported image format or corrupted file: {image_path}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading image '{image_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load image '{image_path}': {e}")


    def preprocess_image(self, image_data: Any) -> Any:
        """
        Placeholder for image preprocessing.
        In a real implementation, this would involve steps like binarization,
        noise reduction, deskewing, etc.

        :param image_data: The image data to preprocess (e.g., NumPy array).
        :return: The preprocessed image data.
        """
        self.logger.info(f"Preprocessing image (type: {type(image_data)})... (Placeholder - returning as is)")
        if self.preprocessor:
            # return self.preprocessor.process(image_data) # Conceptual
            pass
        return image_data

    def run_recognition(self, image_data: Any, engine_name: str, language_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Runs OCR on the image data using the specified engine.

        :param image_data: The image data to process (e.g., NumPy array from preprocessing).
        :param engine_name: The name of the OCR engine to use.
        :param language_hint: Optional language hint for the OCR engine.
        :return: A dictionary containing the OCR results, or None if the engine is not found or fails.
        """
        self.logger.info(f"Attempting OCR with engine: '{engine_name}'")
        engine = self.engine_manager.get_engine(engine_name)
        if not engine:
            self.logger.error(f"Engine '{engine_name}' not available or not found for recognition.")
            return None
        
        try:
            self.logger.debug(f"Using engine '{engine.get_engine_name()}' for recognition.")
            # Pass language hint if provided by the orchestrator's process_document method
            result = engine.recognize(image_data, language_hint=language_hint)
            self.logger.info(f"Recognition by '{engine_name}' completed.")
            self.logger.debug(f"Raw result from '{engine_name}': {str(result)[:200]}...") # Log snippet
            return result
        except Exception as e:
            self.logger.error(f"Error during recognition with engine '{engine_name}': {e}", exc_info=True)
            return None # Or re-raise a specific OrchestratorProcessingError

    def postprocess_text(self, ocr_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder for text postprocessing.
        In a real implementation, this could involve spell checking,
        Natural Language Processing (NLP) for error correction, formatting, etc.

        :param ocr_output: The raw OCR output dictionary.
        :return: The postprocessed OCR output dictionary.
        """
        self.logger.info("Postprocessing OCR output... (Placeholder - returning as is)")
        if self.postprocessor:
            # return self.postprocessor.process(ocr_output) # Conceptual
            pass
        return ocr_output

    def process_document(self, image_path: str, requested_engine_name: Optional[str] = None, language_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the full OCR pipeline for a single document.

        :param image_path: Path to the image file to process.
        :param requested_engine_name: Optional name of the specific engine to use.
        :param language_hint: Optional language hint for the OCR engine.
        :return: A dictionary containing the final OCR results, or None if processing fails at any critical stage.
        """
        self.logger.info(f"Starting document processing for: '{image_path}'. Requested engine: {requested_engine_name}, Lang hint: {language_hint}")
        try:
            image_data = self.load_image(image_path)
            if image_data is None: # Should raise error in load_image
                return None 
            
            preprocessed_data = self.preprocess_image(image_data)
            if preprocessed_data is None:
                self.logger.error("Preprocessing failed to return data.")
                return None

            engine_name_to_use: Optional[str] = None
            available_engines = self.engine_manager.get_available_engines()

            if requested_engine_name:
                if requested_engine_name in available_engines:
                    engine_name_to_use = requested_engine_name
                else:
                    self.logger.warning(f"Requested engine '{requested_engine_name}' is not available. Trying default or first available.")
            
            if not engine_name_to_use:
                default_engine = self.config.get('app_settings', {}).get('default_ocr_engine')
                if default_engine and default_engine in available_engines:
                    engine_name_to_use = default_engine
                    self.logger.info(f"Using default engine: '{engine_name_to_use}'.")
                elif available_engines:
                    engine_name_to_use = available_engines[0]
                    self.logger.info(f"Using first available engine: '{engine_name_to_use}'.")
            
            if not engine_name_to_use:
                self.logger.error("No OCR engine available to process the document.")
                return None
            
            self.logger.info(f"Selected engine for recognition: '{engine_name_to_use}'.")
            raw_ocr_output = self.run_recognition(preprocessed_data, engine_name_to_use, language_hint=language_hint)
            if raw_ocr_output is None:
                self.logger.error(f"Recognition failed with engine '{engine_name_to_use}'.")
                return None

            final_output = self.postprocess_text(raw_ocr_output)
            if final_output is None:
                self.logger.error("Postprocessing failed to return data.")
                return None
                
            self.logger.info(f"Document processing completed successfully for: '{image_path}' using engine '{engine_name_to_use}'.")
            return final_output

        except FileNotFoundError as e:
            self.logger.error(f"Processing aborted: {e}")
            return {"error": str(e), "source": "load_image"}
        except ValueError as e: # E.g. unsupported image format
            self.logger.error(f"Processing aborted due to value error: {e}", exc_info=True)
            return {"error": str(e), "source": "load_image_or_engine_config"}
        except RuntimeError as e: # E.g. engine not initialized or other runtime issues
            self.logger.error(f"Processing aborted due to runtime error: {e}", exc_info=True)
            return {"error": str(e), "source": "processing_pipeline"}
        except Exception as e:
            self.logger.critical(f"An unexpected critical error occurred in process_document for '{image_path}': {e}", exc_info=True)
            return {"error": f"Unexpected critical error: {e}", "source": "orchestrator"}


if __name__ == '__main__':
    # This block is for conceptual demonstration and testing of the orchestrator.
    # It requires a dummy config.yaml and the dummy_engine.py stub.

    # 1. Create a dummy config.yaml
    dummy_config_content = """
app_settings:
  default_ocr_engine: "dummy_local" # Optional: specify a default
  # other_app_setting: "value"

ocr_engines:
  dummy_local:
    enabled: true
    module: "stubs.dummy_engine"  # Relative to where python is run, or adjust PYTHONPATH
    class: "DummyLocalEngine"
    config: # This is the 'engine_config_data' passed to the engine's __init__
      name: "MySuperDummy V1" # Engine can use this to override its default name
      model_path: "models/dummy_model.onnx" # Example specific param
      another_param: 123
  
  # Example of a disabled engine
  disabled_dummy_engine:
    enabled: false
    module: "stubs.dummy_engine"
    class: "DummyLocalEngine"
    config:
      model_path: "models/another_dummy.onnx"

  # Example of a misconfigured engine (e.g., wrong class name)
  misconfigured_engine:
    enabled: true
    module: "stubs.dummy_engine"
    class: "NonExistentEngineClass" 
    config:
      model_path: "models/misconfigured.onnx"
"""
    config_file = "temp_config_orchestrator.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(dummy_config_content)

    # Create a dummy image file for testing load_image
    dummy_image_path = "dummy_test_image.png"
    try:
        img = Image.new('RGB', (60, 30), color = 'red')
        img.save(dummy_image_path)
    except Exception as e:
        # If Pillow is not installed, this will fail.
        # The orchestrator test will then likely fail at load_image.
        # For now, we'll just log it. The main test is orchestrator logic.
        logging.basicConfig(level=logging.INFO) # Basic config if load_config hasn't run
        logging.warning(f"Could not create dummy image, Pillow might not be installed or other issue: {e}")
        # Create an empty file so FileNotFoundError is not the first issue
        if not os.path.exists(dummy_image_path):
            open(dummy_image_path, 'a').close()


    # 2. Instantiate and use the Orchestrator
    # Logging will be configured by load_config inside OCRWorkflowOrchestrator
    orchestrator = OCRWorkflowOrchestrator(config_path=config_file)
    
    main_logger = logging.getLogger("OrchestratorExample") # Get logger after config is loaded

    main_logger.info("\n--- Testing Orchestrator ---")
    main_logger.info(f"Available engines in manager: {orchestrator.engine_manager.get_available_engines()}")

    # Test processing with the successfully loaded dummy engine
    if orchestrator.engine_manager.get_available_engines():
        main_logger.info(f"\n--- Processing document with specific engine '{orchestrator.engine_manager.get_available_engines()[0]}' ---")
        result = orchestrator.process_document(dummy_image_path, requested_engine_name=orchestrator.engine_manager.get_available_engines()[0], language_hint="en")
        main_logger.info(f"Result from process_document: {result}")
        
        main_logger.info("\n--- Processing document with default engine selection logic ---")
        result_default = orchestrator.process_document(dummy_image_path, language_hint="fr")
        main_logger.info(f"Result (default engine): {result_default}")

    else:
        main_logger.warning("No engines were loaded successfully, cannot run process_document test.")

    # Test with a non-existent image
    main_logger.info("\n--- Processing non-existent document ---")
    result_non_existent = orchestrator.process_document("non_existent_image.png")
    main_logger.info(f"Result (non-existent file): {result_non_existent}")
    
    # Clean up dummy files
    if os.path.exists(config_file):
        os.remove(config_file)
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
    main_logger.info("\nDummy config and image files removed.")

```
The `ocr_workflow_orchestrator.py` file has been created with the `OCREngineManager` and `OCRWorkflowOrchestrator` classes.

**Key features implemented:**

*   **`OCREngineManager`:**
    *   `__init__`: Initializes engine configurations and logger.
    *   `discover_and_load_engines`: Dynamically imports and instantiates engine classes from modules specified in the configuration, calls their `initialize()` method, and stores them if available. Handles various errors during this process.
    *   `get_engine`: Retrieves an available engine by name.
    *   `get_available_engines`: Lists names of available engines.
*   **`OCRWorkflowOrchestrator`:**
    *   `__init__`: Loads configuration using `config_loader.py`, sets up its logger, instantiates `OCREngineManager`, and triggers engine discovery.
    *   `load_image`: Uses Pillow to load images, converts to RGB NumPy array, and handles file/format errors.
    *   `preprocess_image` and `postprocess_text`: Implemented as placeholders.
    *   `run_recognition`: Gets an engine from the manager and calls its `recognize` method.
    *   `process_document`: Orchestrates the full pipeline, including logic to select an engine (requested, default, or first available) and comprehensive error handling.
*   **Type Hinting and Docstrings:** Included for all classes and methods.
*   **`if __name__ == '__main__':` Block:**
    *   Creates a temporary `temp_config_orchestrator.yaml` defining a `dummy_local` engine (referencing `stubs.dummy_engine.DummyLocalEngine`), a disabled engine, and a misconfigured engine.
    *   Creates a dummy PNG image using Pillow for testing.
    *   Instantiates `OCRWorkflowOrchestrator` with the temporary config.
    *   Tests `process_document` with a specific engine and default engine selection.
    *   Tests `process_document` with a non-existent image file.
    *   Cleans up the temporary config and image files.

This setup allows for testing the dynamic loading of engines and the basic orchestration flow. The next steps would involve creating concrete engine implementations and actual pre/post-processing modules.
