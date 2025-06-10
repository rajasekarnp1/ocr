import logging
import os
# Correctly import the classes from their respective files
from preprocessing_module import GeometricCorrector, ImageBinarizer, ImageDeskewer
from recognition_module import ONNXRecognizer
from postprocessing_module import TextCleaner, SpellCorrector
from config_loader import load_config, create_default_config_if_not_exists # create_default_config_if_not_exists for __main__

import numpy as np

# Assuming custom_exceptions.py is in the same directory or PYTHONPATH
try:
    from custom_exceptions import OCRFileNotFoundError, OCRConfigurationError, OCRPipelineError, OCRImageProcessingError, OCRModelError
except ImportError:
    # Basic fallback if custom_exceptions is not found
    OCRFileNotFoundError = FileNotFoundError
    OCRConfigurationError = RuntimeError
    OCRPipelineError = RuntimeError
    OCRImageProcessingError = RuntimeError
    OCRModelError = RuntimeError

# For Mojo interop
try:
    from mojo.mojo.python import Python
    mojo_python_module_available = True
    # Attempt to import the specific Mojo utility module for normalization
    # This helps determine if the .mojo file itself is found and syntactically valid for early check
    try:
        mojo_image_utils = Python.import_module("mojo_image_utils")
        mojo_normalize_function = mojo_image_utils.normalize_flat_u8_to_float32_mojo
        mojo_normalization_available = True
        logging.info("Mojo 'normalize_flat_u8_to_float32_mojo' function loaded successfully.")
    except Exception as e:
        logging.warning(f"Mojo 'mojo_image_utils.mojo' or 'normalize_flat_u8_to_float32_mojo' function not found or error during import: {e}. Will use NumPy fallback for normalization.")
        mojo_normalization_available = False
        mojo_image_utils = None # Ensure it's None if import failed
        mojo_normalize_function = None
except (ImportError, ModuleNotFoundError):
    logging.warning("Mojo SDK (mojo.mojo.python) not found. Will use NumPy fallback for all Mojo operations.")
    mojo_python_module_available = False
    mojo_normalization_available = False
    mojo_image_utils = None
    mojo_normalize_function = None

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
                model_path=self.config.get('preprocessing_settings', {}).get('model_path', 'models/dummy_geometric_model.onnx')
            )
            self.recognizer = ONNXRecognizer(
                # Ensure the key 'svtr_recognizer' is used as defined in config_loader's default.
                model_path=self.config.get('recognition_settings', {}).get('svtr_recognizer', 'models/dummy_recognition_model.onnx'),
                use_directml=self.config.get('recognition_settings', {}).get('use_directml', True)
            )
            # Instantiate TextCleaner instead of PostprocessingModulePlaceholder
            # Whitelist can be customized via config if needed, e.g., self.config.get('postprocessing_settings', {}).get('whitelist_chars')
            custom_whitelist = self.config.get('postprocessing_settings', {}).get('whitelist_chars')
            self.text_cleaner = TextCleaner(whitelist_chars=custom_whitelist) # Using custom_whitelist which might be None (for default)

            # Initialize the ImageBinarizer
            self.binarizer = ImageBinarizer()
            # Initialize the ImageDeskewer
            deskewer_settings = self.config.get('deskewer_settings', {})
            self.deskewer = ImageDeskewer(**deskewer_settings)

            # Initialize SpellCorrector
            spell_corrector_dict_path = self.config.get('postprocessing_settings', {}).get('dictionary_path', 'default_dict.txt')
            self.spell_corrector = SpellCorrector(dictionary_path=spell_corrector_dict_path)

            self.logger.info("OCR Workflow Orchestrator initialized successfully with TextCleaner, ImageDeskewer, and SpellCorrector.")

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

        self.logger.info(f"Image loaded successfully from {image_path}. For binarization, this will be a uint8 NumPy array.")
        # Modify to return a dummy uint8 NumPy array simulating a grayscale or color image.
        # For testing binarization, let's make it a 2D grayscale-like image.
        # Values should be in 0-255 range, dtype uint8.
        # Example: a 50x50 image with varying intensity
        dummy_image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
        # To test color conversion, uncomment below to make it 3-channel:
        # dummy_image = np.dstack([dummy_image]*3)
        return dummy_image


    def process_document(self, image_path):
        """
        Processes a single document through the full OCR pipeline.
        """
        self.logger.info(f"Starting OCR process for document: {image_path}")
        try:
            # image_data_np will be a NumPy array (uint8) from load_image
            image_data_np = self.load_image(image_path)
            if not isinstance(image_data_np, np.ndarray):
                 self.logger.error(f"Image loading did not return a NumPy array for {image_path}, aborting.")
                 return "Error: Image loading failed to produce NumPy array."

            # 1. Binarization
            self.logger.debug(f"Binarizing image: {image_path} (original shape: {image_data_np.shape})")
            binarized_image_np = self.binarizer.binarize(image_data_np)
            if binarized_image_np is None:
                self.logger.error(f"Binarization failed for {image_path}, aborting process.")
                return "Error: Binarization failed."
            if not isinstance(binarized_image_np, np.ndarray): # Should be caught by binarizer, but good check
                 self.logger.error(f"Binarization did not return a NumPy array for {image_path}, aborting.")
                 return "Error: Binarization failed to produce NumPy array."
            self.logger.info(f"Image binarized successfully. Shape after binarization: {binarized_image_np.shape}")

            # 2. Deskewing
            # ImageDeskewer expects a binarized uint8 image.
            self.logger.debug(f"Deskewing image: {image_path} (shape: {binarized_image_np.shape})")
            deskewed_image_np = self.deskewer.deskew(binarized_image_np)
            if deskewed_image_np is None: # Deskewer might return None or original on error/no skew
                self.logger.warning(f"Deskewing returned None for {image_path}. Using binarized image instead.")
                deskewed_image_np = binarized_image_np # Fallback to binarized
            elif not isinstance(deskewed_image_np, np.ndarray):
                 self.logger.error(f"Deskewing did not return a NumPy array for {image_path}, aborting.")
                 return "Error: Deskewing failed to produce NumPy array."
            self.logger.info(f"Image deskewed successfully. Shape after deskewing: {deskewed_image_np.shape}")

            # 3. Normalization (uint8 to float32 for ONNX)
            # This step now processes the deskewed_image_np (which is uint8).
            normalized_float32_image_np: np.ndarray | None = None

            if mojo_normalization_available and mojo_normalize_function is not None:
                self.logger.info(f"Attempting normalization with Mojo: {image_path} (shape: {deskewed_image_np.shape})")
                try:
                    height, width = deskewed_image_np.shape[0], deskewed_image_np.shape[1]
                    flat_u8_list = deskewed_image_np.flatten().tolist()
                    mojo_processed_flat_list_obj = mojo_normalize_function(flat_u8_list, height, width)
                    mojo_processed_flat_list = list(mojo_processed_flat_list_obj)

                    if not mojo_processed_flat_list:
                        self.logger.error("Mojo normalization returned an empty list. Falling back to NumPy.")
                        raise RuntimeError("Mojo normalization returned empty list")

                    normalized_float32_image_np = np.array(mojo_processed_flat_list, dtype=np.float32).reshape(height, width)
                    self.logger.info(f"Normalization with Mojo successful. Output shape: {normalized_float32_image_np.shape}")

                except Exception as e_mojo:
                    self.logger.error(f"Error during Mojo normalization: {e_mojo}. Falling back to NumPy for this call.", exc_info=True)
                    normalized_float32_image_np = None

            if normalized_float32_image_np is None:
                self.logger.info(f"Using NumPy for uint8 to float32 normalization: {image_path} (shape: {deskewed_image_np.shape})")
                normalized_float32_image_np = deskewed_image_np.astype(np.float32) / 255.0

            # 4. Geometric Correction (ONNX)
            # Takes the normalized float32 image.
            self.logger.debug(f"Applying geometric correction to normalized image: {image_path} (shape: {normalized_float32_image_np.shape})")
            corrected_image_np = self.preprocessor.correct(normalized_float32_image_np)

            if corrected_image_np is None:
                self.logger.error(f"Geometric correction returned None for {image_path}, aborting process.")
                return "Error: Geometric correction failed and returned None."
            if not isinstance(corrected_image_np, np.ndarray):
                 self.logger.error(f"Geometric correction did not return a NumPy array for {image_path}, aborting.")
                 return "Error: Geometric correction failed to produce NumPy array."

            # 3. Recognition
            # ONNXRecognizer also expects float32 for its dummy model.
            # The output from geometric corrector is already float32.
            self.logger.debug(f"Running recognition on corrected image: {image_path} (shape: {corrected_image_np.shape})")
            raw_text, confidence = self.recognizer.predict(corrected_image_np) # Pass float32 array
            if raw_text is None:
                self.logger.error(f"Recognition failed for {image_path}, aborting process.")
                return "Error: Recognition failed and returned None."

            # Prepare data for postprocessor, which expects a dict
            raw_ocr_data = {"text": raw_text, "confidence": confidence}

            self.logger.debug(f"Cleaning OCR text for: {image_path}")
            # Use TextCleaner's clean_text method
            cleaned_data_dict = self.text_cleaner.clean_text(raw_ocr_data)

            if "error" in cleaned_data_dict:
                self.logger.error(f"Text cleaning failed for {image_path}: {cleaned_data_dict['error']}")
                return f"Error: Text cleaning failed - {cleaned_data_dict['error']}"

            # 5. Spell Correction
            self.logger.debug(f"Spell correcting text for: {image_path}")
            spell_checked_data_dict = self.spell_corrector.correct_text(cleaned_data_dict)

            if "spell_correction_error" in spell_checked_data_dict:
                self.logger.error(f"Spell correction failed for {image_path}: {spell_checked_data_dict['spell_correction_error']}")
                # Return previous state (cleaned_data_dict) or an error indicating spell check failure
                return {**cleaned_data_dict, "final_error_stage": "spell_correction"}


            self.logger.info(f"Successfully processed, cleaned, and spell-checked document: {image_path}. Spell-checked text (first 100): '{spell_checked_data_dict.get('spell_checked_text', '')[:100]}'")
            return spell_checked_data_dict

        except OCRFileNotFoundError as e_fnf: # Custom file not found (e.g., from load_image or module init)
            self.logger.error(f"OCR Resource File Not Found Error during processing of '{image_path}': {e_fnf}", exc_info=True)
            raise OCRPipelineError(f"A required resource file was not found processing '{image_path}'. Details: {e_fnf}") from e_fnf
        except (OCRModelError, OCRImageProcessingError, OCRPipelineError) as e_pipe: # Custom operational errors from modules
            self.logger.error(f"OCR Pipeline Error for '{image_path}': {e_pipe}", exc_info=True)
            if isinstance(e_pipe, OCRPipelineError): # If it's already our specific pipeline error, just re-raise
                raise
            raise OCRPipelineError(f"A core OCR processing step failed for '{image_path}'. Details: {e_pipe}") from e_pipe
        except (TypeError, ValueError) as e_data:
            self.logger.error(f"Data integrity or type error processing '{image_path}': {e_data}", exc_info=True)
            raise OCRPipelineError(f"Data error encountered while processing '{image_path}'. Ensure image format and content are compatible.") from e_data
        except FileNotFoundError as e_os_fnf: # Catch standard FileNotFoundError (e.g. if a temp file operation failed, not caught by OCRFileNotFoundError)
            self.logger.error(f"OS File Not Found Error during processing of '{image_path}': {e_os_fnf}", exc_info=True)
            raise OCRPipelineError(f"A required system file operation failed for '{image_path}'. Details: {e_os_fnf}") from e_os_fnf
        except Exception as e_unexpected:
            self.logger.error(f"An unexpected error occurred while processing '{image_path}': {e_unexpected}", exc_info=True)
            raise OCRPipelineError(f"An unexpected critical error occurred processing '{image_path}'.") from e_unexpected

    def get_results(self, processed_data: dict | str) -> str: # Can receive error string for backward compatibility or direct error message
        self.logger.debug(f"Formatting final results. Input type: {type(processed_data)}")

        # If process_document now always raises exceptions, this path is less likely for pipeline errors,
        # but might be used if an error string is directly passed for some other reason.
        if isinstance(processed_data, str) and processed_data.startswith("Error:"):
             self.logger.warning(f"Formatting an error message passed as string: {processed_data}")
             return f"Processing Failed. Status: {processed_data}"

        if not isinstance(processed_data, dict):
            self.logger.error(f"get_results expected a dictionary, got {type(processed_data)}. Data: {processed_data}")
            # This indicates an issue if process_document was expected to succeed or raise a proper OCRPipelineError
            return "Error: Invalid data received by get_results for formatting. Expected a results dictionary."

        spell_checked_text = processed_data.get('spell_checked_text', '[Spell check not performed or failed]')
        cleaned_text = processed_data.get('cleaned_text', '[Text cleaning not performed or failed]')
        original_text = processed_data.get('original_text', '[Original text not available]')
        confidence = processed_data.get('confidence', 0.0)

        # Collect other metadata, excluding the main text fields and potentially large image shape tuples if too verbose
        excluded_keys = {'spell_checked_text', 'cleaned_text', 'original_text', 'confidence',
                         'original_image_shape','binarized_image_shape', 'deskewed_image_shape', 'corrected_image_shape'}
        other_metadata = {k:v for k,v in processed_data.items() if k not in excluded_keys}

        formatted_str = (f"--- OCR Processed Results ---\n"
                         f"  Spell-Checked Text: '{spell_checked_text}'\n"
                         f"  Cleaned Text      : '{cleaned_text}'\n"
                         f"  Original OCR Text : '{original_text[:100]}{'...' if len(original_text) > 100 else ''}'\n"
                         f"  Confidence        : {confidence:.2f}\n"
                         f"  Additional Info   : {other_metadata if other_metadata else 'None'}")
        self.logger.debug("Results formatted successfully.")
        return formatted_str


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

    # --- Ensure ONNX models are generated for the __main__ block ---
    # (config_loader.create_default_config_if_not_exists already creates placeholders if missing,
    # but here we try to generate the actual dummy ONNX models if the scripts are available)

    models_dir = "models" # Should match config
    os.makedirs(models_dir, exist_ok=True)

    geom_model_main_path = loaded_config_for_main.get('preprocessing_settings', {}).get('model_path')
    rec_model_main_path = loaded_config_for_main.get('recognition_settings', {}).get('model_path') # svtr_recognizer key in config

    if geom_model_main_path and not os.path.exists(geom_model_main_path):
        main_logger.warning(f"Main: Geometric model {geom_model_main_path} not found. Attempting to generate...")
        try:
            from generate_dummy_geometric_model import generate_model as gen_geom
            gen_geom(geom_model_main_path)
            main_logger.info(f"Main: Generated {geom_model_main_path}")
        except Exception as e_gen:
            main_logger.error(f"Main: Failed to generate geometric model: {e_gen}")
            # Create a simple placeholder if generation fails, so orchestrator init doesn't immediately fail on FileNotFoud
            with open(geom_model_main_path, "w") as f: f.write("dummy content for geom model if generator failed")


    if rec_model_main_path and not os.path.exists(rec_model_main_path):
        main_logger.warning(f"Main: Recognition model {rec_model_main_path} not found. Attempting to generate...")
        try:
            from generate_dummy_recognition_model import generate_model as gen_rec
            gen_rec(rec_model_main_path)
            main_logger.info(f"Main: Generated {rec_model_main_path}")
        except Exception as e_gen:
            main_logger.error(f"Main: Failed to generate recognition model: {e_gen}")
            with open(rec_model_main_path, "w") as f: f.write("dummy content for rec model if generator failed")

    # Postprocessing model path (dummy file, as it's not ONNX based yet)
    default_nlp_model_path = loaded_config_for_main.get('postprocessing_settings', {}).get('nlp_model_path', os.path.join(models_dir, "dummy_nlp_model.onnx"))
    if default_nlp_model_path:
        nlp_model_dir = os.path.dirname(default_nlp_model_path)
        if nlp_model_dir and not os.path.exists(nlp_model_dir): os.makedirs(nlp_model_dir, exist_ok=True)
        if not os.path.exists(default_nlp_model_path):
            with open(default_nlp_model_path, 'a') as f: f.write("dummy nlp model data")
            main_logger.info(f"Main: Created dummy NLP model: {default_nlp_model_path}")


    # Dummy image file - this is just a path, load_image will produce the NumPy array
    dummy_image_path_for_main = "dummy_image_for_orchestrator.png"
    if not os.path.exists(dummy_image_path_for_main):
        with open(dummy_image_path_for_main, "w") as f: f.write("dummy image file content (not actual image data)")
        main_logger.info(f"Main: Created dummy image file: {dummy_image_path_for_main}")

    try:
        orchestrator = OCRWorkflowOrchestrator(config_path=config_file_path)
        # process_document now uses a NumPy array from load_image
        result = orchestrator.process_document(dummy_image_path_for_main)
        main_logger.info(f"Orchestrator Result: {orchestrator.get_results(result)}")

        main_logger.info("Testing error handling with non-existent image:")
        result_error = orchestrator.process_document("non_existent_image.png") # This will fail at load_image
        main_logger.info(f"Orchestrator Error Result: {orchestrator.get_results(result_error)}")
    except Exception as e:
        main_logger.critical(f"Error in OCRWorkflowOrchestrator example: {e}", exc_info=True)

    # Clean up dummy files (optional)
    # ... (cleanup logic can be added if desired, similar to original but considering new model paths)
