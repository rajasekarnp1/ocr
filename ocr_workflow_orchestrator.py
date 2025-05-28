import logging
import os
# Correctly import the placeholder classes from their respective files
from preprocessing_module import GeometricCorrector # Using actual class, but it has placeholder logic
from recognition_module import ONNXRecognizer # Using actual class, but it has placeholder logic
from postprocessing_module import PostprocessingModulePlaceholder # Using the new placeholder
from config_loader import load_config, create_default_config_if_not_exists

# Setup basic logging if no config is loaded yet or if run standalone
# This will be overridden if load_config is successful and provides its own logging config.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class OCRWorkflowOrchestrator:
    def __init__(self, config_path="config.yaml"):
        """
        Initializes the OCR workflow orchestrator.
        Loads configuration and initializes processing modules.
        """
        self.logger = logging.getLogger(__name__)
        try:
            # Load configuration using the imported load_config
            # This will also set up logging as per the config file or defaults
            self.config = load_config(config_path) 
            
            # Configure logging based on loaded config (config_loader ideally handles this,
            # but we can re-apply or ensure orchestrator's logger level if needed)
            log_level_str = self.config.get('logging', {}).get('loggers', {}).get(__name__, {}).get('level')
            if not log_level_str: # Fallback to root logger level or a default
                log_level_str = self.config.get('logging', {}).get('root', {}).get('level', 'INFO').upper()
            
            # Ensure logger level is set if it was configured
            # This might be redundant if logging.config.dictConfig in load_config fully handles it
            # but serves as a direct check for the orchestrator's own logger.
            current_effective_level = self.logger.getEffectiveLevel()
            desired_level = getattr(logging, log_level_str, logging.INFO)
            if current_effective_level > desired_level : # Only set if more verbose level is desired
                 self.logger.setLevel(desired_level)


            self.logger.info("Initializing OCR Workflow Orchestrator...")
            
            # Initialize modules with their respective settings from the config
            # Using the actual GeometricCorrector and ONNXRecognizer which have placeholder logic,
            # and the new PostprocessingModulePlaceholder.
            self.preprocessor = GeometricCorrector(
                model_path=self.config.get('preprocessing_settings', {}).get('model_path', 'dummy_geom_model.onnx')
            )
            self.recognizer = ONNXRecognizer(
                model_path=self.config.get('recognition_settings', {}).get('model_path', 'dummy_ocr_model.onnx'),
                use_directml=self.config.get('recognition_settings', {}).get('use_directml', True)
            )
            self.postprocessor = PostprocessingModulePlaceholder(
                settings=self.config.get('postprocessing_settings', {})
            )

            self.logger.info("OCR Workflow Orchestrator initialized successfully.")

        except FileNotFoundError as e: # Specifically for config file not found by load_config if it raises it
            self.logger.critical(f"Configuration file {config_path} not found. Orchestrator cannot start. Error: {e}", exc_info=True)
            raise
        except ValueError as e: # Config parsing errors
            self.logger.critical(f"Error parsing configuration file {config_path}. Orchestrator cannot start. Error: {e}", exc_info=True)
            raise
        except RuntimeError as e: # Issues from module initializations (e.g. model loading failures)
            self.logger.critical(f"Runtime error during OCR Workflow Orchestrator initialization: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.critical(f"Failed to initialize OCR Workflow Orchestrator: {e}", exc_info=True)
            raise

    def load_image(self, image_path):
        """
        Loads an image from the given path.
        Placeholder for actual image loading logic (e.g., using Pillow or OpenCV).
        """
        self.logger.debug(f"Attempting to load image from: {image_path}")
        if not image_path or not isinstance(image_path, str):
            self.logger.error("Invalid image path provided for loading.")
            raise ValueError("Image path must be a non-empty string.")
        if not os.path.exists(image_path): 
             self.logger.error(f"Image file not found: {image_path}")
             raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.logger.info(f"Image loaded successfully from {image_path}")
        return f"MockImageData_for_{os.path.basename(image_path)}"

    def process_document(self, image_path):
        """
        Processes a single document through the full OCR pipeline.
        """
        self.logger.info(f"Starting OCR process for document: {image_path}")
        try:
            image_data = self.load_image(image_path)
            # load_image raises error, so no need to check for None if it's well-behaved

            self.logger.debug(f"Preprocessing image: {image_path}")
            # GeometricCorrector's correct method is used.
            preprocessed_image = self.preprocessor.correct(image_data)
            if preprocessed_image is None: # Should not happen if .correct() returns original on error
                self.logger.error(f"Preprocessing returned None for {image_path}, aborting process.")
                return "Error: Preprocessing failed and returned None."

            self.logger.debug(f"Running recognition on preprocessed image: {image_path}")
            # ONNXRecognizer's predict method is used.
            raw_text, confidence = self.recognizer.predict(preprocessed_image)
            if raw_text is None: # predict returns (None, 0.0) on error
                self.logger.error(f"Recognition failed for {image_path}, aborting process.")
                return "Error: Recognition failed and returned None."
            
            # Prepare data for postprocessor, which expects a dict
            raw_ocr_data = {"text": raw_text, "confidence": confidence}

            self.logger.debug(f"Post-processing OCR data for: {image_path}")
            # PostprocessingModulePlaceholder's run_all method is used.
            final_text_results = self.postprocessor.run_all(raw_ocr_data)
            if final_text_results is None: # run_all might return None on error
                self.logger.error(f"Post-processing failed for {image_path}, aborting process.")
                return "Error: Post-processing failed and returned None."
            
            self.logger.info(f"Successfully processed document: {image_path}")
            return final_text_results

        except FileNotFoundError as fnf_err:
            self.logger.error(f"File not found during processing of {image_path}: {fnf_err}", exc_info=False)
            return f"Error: File not found - {image_path}"
        except ValueError as val_err:
            self.logger.error(f"Value error during processing of {image_path}: {val_err}", exc_info=False)
            return f"Error: Invalid input or value - {str(val_err)}"
        except Exception as e:
            self.logger.error(f"An unexpected error occurred processing {image_path}: {e}", exc_info=True)
            return "Error: An unexpected error occurred."

    def get_results(self, processed_data):
        """
        Formats and returns the final results.
        """
        self.logger.debug("Formatting final results.")
        if processed_data is None or processed_data.startswith("Error:"):
            return f"No valid data to format. Processing status: {processed_data}"
        return f"Formatted Results: {str(processed_data)}"

if __name__ == '__main__':
    # Ensure a default config_dev.yaml exists for the orchestrator's main execution
    # This uses the function imported from config_loader.py
    config_file_path = "config_dev.yaml"
    create_default_config_if_not_exists(config_file_path)

    # Now, load_config will either load the existing or the newly created default config.
    # This also sets up logging.
    loaded_config_for_main = load_config(config_file_path)
    
    # Get a logger for this __main__ block, now that logging is configured.
    main_logger = logging.getLogger(__name__ + "_main") # Unique name for this logger
    main_logger.info(f"Logging for orchestrator __main__ is active. Config level: {loaded_config_for_main.get('logging',{}).get('root',{}).get('level')}")

    # Create dummy files and directories needed for the orchestrator based on default config values
    # These paths should align with what's in the default config_dev.yaml
    
    # Preprocessing model path from default config
    default_geom_model_path = loaded_config_for_main.get('preprocessing_settings', {}).get('model_path', "dummy_geom_model.onnx")
    geom_model_dir = os.path.dirname(default_geom_model_path)
    if geom_model_dir and not os.path.exists(geom_model_dir):
        os.makedirs(geom_model_dir, exist_ok=True)
        main_logger.info(f"Created directory for geometric model: {geom_model_dir}")
    if not os.path.exists(default_geom_model_path):
        with open(default_geom_model_path, 'a') as f: f.write("dummy onnx geom model")
        main_logger.info(f"Created dummy geometric model: {default_geom_model_path}")

    # Recognition model path from default config
    default_ocr_model_path = loaded_config_for_main.get('recognition_settings', {}).get('model_path', "dummy_ocr_model.onnx")
    ocr_model_dir = os.path.dirname(default_ocr_model_path)
    if ocr_model_dir and not os.path.exists(ocr_model_dir):
        os.makedirs(ocr_model_dir, exist_ok=True)
        main_logger.info(f"Created directory for OCR model: {ocr_model_dir}")
    if not os.path.exists(default_ocr_model_path):
        with open(default_ocr_model_path, 'a') as f: f.write("dummy onnx ocr model")
        main_logger.info(f"Created dummy OCR model: {default_ocr_model_path}")
    
    # Postprocessing model path from default config (if any, e.g. nlp_model_path)
    default_nlp_model_path = loaded_config_for_main.get('postprocessing_settings', {}).get('nlp_model_path', "dummy_nlp_model.onnx")
    if default_nlp_model_path: # Only create if a path is defined
        nlp_model_dir = os.path.dirname(default_nlp_model_path)
        if nlp_model_dir and not os.path.exists(nlp_model_dir):
            os.makedirs(nlp_model_dir, exist_ok=True)
            main_logger.info(f"Created directory for NLP model: {nlp_model_dir}")
        if not os.path.exists(default_nlp_model_path):
            with open(default_nlp_model_path, 'a') as f: f.write("dummy nlp model")
            main_logger.info(f"Created dummy NLP model: {default_nlp_model_path}")

    # Dummy image file
    dummy_image = "dummy_image.png"
    if not os.path.exists(dummy_image):
        with open(dummy_image, "w") as f: f.write("dummy image data")
        main_logger.info(f"Created dummy image file: {dummy_image}")

    try:
        orchestrator = OCRWorkflowOrchestrator(config_path=config_file_path)
        result = orchestrator.process_document(dummy_image)
        main_logger.info(f"Orchestrator Result: {orchestrator.get_results(result)}")

        main_logger.info("Testing error handling with non-existent image:")
        result_error = orchestrator.process_document("non_existent_image.png")
        main_logger.info(f"Orchestrator Error Result: {orchestrator.get_results(result_error)}")
    except Exception as e:
        main_logger.critical(f"Error in OCRWorkflowOrchestrator example: {e}", exc_info=True)

    # Clean up dummy files (optional, good for testing but can be commented out)
    # main_logger.info("Cleaning up dummy files...")
    # if os.path.exists(dummy_image): os.remove(dummy_image)
    # if os.path.exists(default_geom_model_path): os.remove(default_geom_model_path)
    # if os.path.exists(default_ocr_model_path): os.remove(default_ocr_model_path)
    # if default_nlp_model_path and os.path.exists(default_nlp_model_path): os.remove(default_nlp_model_path)
    # if os.path.exists(config_file_path): os.remove(config_file_path) # Be careful with this one
    # main_logger.info("Cleanup complete.")
