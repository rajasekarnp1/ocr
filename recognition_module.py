import logging
import os
import onnxruntime as ort
import numpy as np

# Assuming custom_exceptions.py is in the same directory or PYTHONPATH
try:
    from custom_exceptions import OCRFileNotFoundError, OCRModelError, OCRPipelineError
except ImportError:
    # Basic fallback if custom_exceptions is not found
    OCRFileNotFoundError = FileNotFoundError
    OCRModelError = RuntimeError
    OCRPipelineError = RuntimeError # General pipeline error

# Import for Mojo interop - This should be handled by the orchestrator or main entry point
# For this module, we assume 'mojo_available' might be passed or determined differently if used directly.
# However, the current implementation has it as a global, which is fine for this project structure.
try:
    from mojo.mojo.python import Python
    mojo_available = True # Global flag for this module
except (ImportError, ModuleNotFoundError):
    mojo_available = False

class ONNXRecognizer:
    def __init__(self, model_path: str, use_directml: bool = True, preferred_provider_only: bool = False):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.logger.debug(f"Initializing ONNXRecognizer with model: {model_path}, use_directml: {use_directml}, preferred_provider_only: {preferred_provider_only}")

        if not model_path or not isinstance(model_path, str):
            self.logger.error("Invalid model_path type for ONNXRecognizer: Must be a non-empty string.")
            raise ValueError("Model path must be a non-empty string.")

        try:
            self.logger.info(f"Attempting to load ONNX recognition model from: {model_path}")
            if not os.path.exists(model_path):
                 self.logger.error(f"ONNX recognition model file not found at path: {model_path}")
                 raise OCRFileNotFoundError(f"ONNX recognition model file not found: {model_path}")

            providers = []
            if use_directml:
                available_providers = ort.get_available_providers()
                if 'DmlExecutionProvider' in available_providers:
                    providers.append('DmlExecutionProvider')
                    self.logger.debug("DirectMLExecutionProvider selected.")
                elif preferred_provider_only:
                    self.logger.error("DirectMLExecutionProvider requested (preferred_provider_only=True) but not available.")
                    raise OCRModelError("DirectML provider not available as configured (preferred_provider_only=True).")
                else:
                    self.logger.warning("DirectMLExecutionProvider requested but not available. Falling back to CPU.")

            providers.append('CPUExecutionProvider')
            self.logger.debug(f"Attempting to load ONNX session with providers: {providers}")

            sess_options = ort.SessionOptions()
            # Consider making log severity configurable, e.g., via config file
            # sess_options.log_severity_level = 2 # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal

            self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)

            self.input_name = self.session.get_inputs()[0].name
            # Assuming single output for dummy model, real models might have multiple.
            self.output_name = self.session.get_outputs()[0].name

            current_provider = self.session.get_providers()[0]
            self.logger.info(f"ONNX recognition model '{os.path.basename(model_path)}' loaded successfully. Effective provider: {current_provider}")
            self.logger.debug(f"Model Input: '{self.input_name}', Output: '{self.output_name}'")

        except OCRFileNotFoundError:
            raise # Re-raise specific custom error
        except ort.capi.onnxruntime_pybind11_state.Fail as ort_load_error:
            self.logger.error(f"ONNXRuntime model loading failed for {model_path}: {ort_load_error}", exc_info=True)
            raise OCRModelError(f"Failed to load ONNX recognition model '{model_path}'.") from ort_load_error
        except Exception as e:
            self.logger.error(f"Unexpected error loading ONNX recognition model {model_path}: {e}", exc_info=True)
            raise OCRModelError(f"Unexpected error initializing ONNXRecognizer with model '{model_path}'.") from e

    def _prepare_input(self, processed_image_np: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Preparing input for ONNX model. Shape: {processed_image_np.shape}, dtype: {processed_image_np.dtype}")
        if not isinstance(processed_image_np, np.ndarray):
            self.logger.error(f"Invalid input type for _prepare_input: Expected NumPy array, got {type(processed_image_np)}.")
            raise TypeError("Input for ONNX model preparation must be a NumPy array.")

        # Dummy model expects float32. Real models might need specific normalization, resizing, channel ordering.
        if processed_image_np.dtype != np.float32:
            self.logger.warning(f"Input for _prepare_input is {processed_image_np.dtype}, converting to float32.")
            try:
                return processed_image_np.astype(np.float32)
            except Exception as e_conv:
                self.logger.error(f"Failed to convert input to float32 in _prepare_input: {e_conv}", exc_info=True)
                raise OCRImageProcessingError("Failed to convert input to float32 for ONNX model.") from e_conv
        return processed_image_np


    def _parse_output(self, model_output_tensor: np.ndarray) -> tuple[str, float]:
        self.logger.debug(f"Parsing ONNX model output. Shape: {model_output_tensor.shape}, dtype: {model_output_tensor.dtype}")
        try:
            # For dummy model (input * 2.0), output is a float tensor.
            # Artificial parsing: convert sum to string, fixed confidence.
            text_representation = f"DummyTextOutput_from_shape_{model_output_tensor.shape}_sum_{np.sum(model_output_tensor):.1f}"
            confidence = 0.95
            self.logger.debug(f"Parsed output to text: '{text_representation[:50]}...', confidence: {confidence:.2f}")
            return text_representation, confidence
        except Exception as e:
            self.logger.error(f"Error parsing ONNX model output tensor: {e}", exc_info=True)
            # Return error indication instead of raising exception here, predict() will handle overall failure.
            return "Error parsing ONNX output", 0.0


    def predict(self, processed_image_np: np.ndarray) -> tuple[str | None, float]:
        self.logger.info(f"Starting ONNX prediction. Input shape: {getattr(processed_image_np, 'shape', 'N/A')}")
        if processed_image_np is None: # Should ideally be caught earlier by type hint, but defensive check
            self.logger.warning("Input image_data_np is None for ONNX recognition. Returning None.")
            return None, 0.0

        if not isinstance(processed_image_np, np.ndarray):
             self.logger.error(f"Invalid data type for recognition. Expected NumPy array, got {type(processed_image_np)}.")
             # Consider raising OCRPipelineError or TypeError
             raise TypeError("Invalid data type for recognition. Input must be a NumPy array.")

        try:
            prepared_input_np = self._prepare_input(processed_image_np)
            input_feed = {self.input_name: prepared_input_np}

            self.logger.debug(f"Running ONNX session for recognition with input: '{self.input_name}'")
            raw_output_tensors = self.session.run([self.output_name], input_feed)
            model_output_tensor = raw_output_tensors[0] # Assuming single output

            text, confidence = self._parse_output(model_output_tensor)
            if text == "Error parsing ONNX output": # Check for parsing failure indication
                 self.logger.error("Failed to parse ONNX output during prediction.")
                 # Return None, 0.0 to indicate failure to the orchestrator
                 return None, 0.0

            self.logger.info(f"ONNX recognition successful. Text: '{text[:50]}...', Confidence: {confidence:.2f}")

            # --- Mojo function call ---
            if mojo_available: # This global flag is set at module import time
                self.logger.info("Attempting to call Mojo function 'example_mojo_tensor_operation'...")
                dummy_mojo_input = [int(s) for s in model_output_tensor.shape if s is not None]
                if not dummy_mojo_input: dummy_mojo_input = [1,2,3]

                try:
                    # This assumes mojo_recognizer_utils.mojo is in PYTHONPATH/MODULE_PATH
                    mojo_utils_module = Python.import_module("mojo_recognizer_utils")
                    processed_data_from_mojo = mojo_utils_module.example_mojo_tensor_operation(dummy_mojo_input)
                    self.logger.info(f"Mojo 'example_mojo_tensor_operation' successful. Result: {processed_data_from_mojo}")
                except Exception as e_mojo: # Catching broad exception as Mojo interop can have various errors
                    self.logger.error(f"Error calling Mojo function 'example_mojo_tensor_operation': {e_mojo}", exc_info=True)
                    # Non-critical, so just log and continue.
            else:
                self.logger.debug("Mojo SDK not detected or not available. Skipping 'example_mojo_tensor_operation' call.")
            # --- End of Mojo function call ---

            return text, confidence
        except ort.OrtException as ort_infer_error:
            self.logger.error(f"ONNXRuntime error during prediction: {ort_infer_error}", exc_info=True)
            raise OCRModelError("Error during ONNX inference in ONNXRecognizer.") from ort_infer_error
        except Exception as e:
            self.logger.error(f"Unexpected error during ONNX prediction or Mojo call: {e}", exc_info=True)
            raise OCRPipelineError("Unexpected error during ONNXRecognizer prediction.") from e

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        from config_loader import load_config, create_default_config_if_not_exists
        create_default_config_if_not_exists("config_dev.yaml")
        config = load_config("config_dev.yaml")
    except ImportError:
        logging.warning("config_loader not found, using basic logging configuration for recognition_module.py.")
        config = {"recognition_settings": {"model_path": "models/dummy_recognition_model.onnx"}} # Fallback
    except Exception as e:
        logging.error(f"Error loading config via config_loader in recognition_module.py: {e}. Using basic config.", exc_info=True)
        config = {"recognition_settings": {"model_path": "models/dummy_recognition_model.onnx"}} # Fallback

    # Use model path from config
    module_model_path = config.get("recognition_settings", {}).get("model_path", "models/dummy_recognition_model.onnx")

    # Ensure model directory and a placeholder model exist if running standalone and they are missing
    model_dir = os.path.dirname(module_model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(module_model_path):
        logging.warning(f"Recognition model {module_model_path} not found for standalone run. Attempting to generate.")
        try:
            from generate_dummy_recognition_model import generate_model as generate_rec_model
            generate_rec_model(module_model_path)
            logging.info(f"Generated dummy recognition model at: {module_model_path}")
        except ImportError:
            logging.error("generate_dummy_recognition_model.py not found. Cannot generate model for __main__.")
        except Exception as gen_e:
            logging.error(f"Failed to generate dummy recognition model for __main__: {gen_e}")


    if os.path.exists(module_model_path):
        try:
            recognizer = ONNXRecognizer(model_path=module_model_path, use_directml=False) # use_directml=False for CI/simpler env

            # Create a dummy NumPy array. The dummy_recognition_model.onnx expects float32 tensor.
            mock_preprocessed_image_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
            logging.info(f"Mock input image (NumPy array) for recognition: \n{mock_preprocessed_image_np}")

            text, conf = recognizer.predict(mock_preprocessed_image_np)
            logging.info(f"ONNX Recognizer test result - Text: '{text}', Confidence: {conf}")

            # Expected output from dummy recognition model (input * 2.0) parsed by _parse_output:
            # model_output = [[20., 40.], [60., 80.]]
            # text = "DummyTextOutput_from_shape_(2, 2)_sum_200.0"
            # confidence = 0.95
            expected_sum = np.sum(mock_preprocessed_image_np * 2.0)
            expected_shape_str = "_".join(map(str, mock_preprocessed_image_np.shape)) # e.g., "2_2"
            expected_text = f"DummyTextOutput_from_shape_{expected_shape_str}_sum_{expected_sum:.1f}"
            assert text == expected_text, f"Text output '{text}' did not match expected '{expected_text}'"
            assert conf == 0.95, "Confidence output did not match expected 0.95"

        except Exception as e:
            logging.error(f"Error in ONNXRecognizer example: {e}", exc_info=True)
    else:
        logging.error(f"Cannot run ONNXRecognizer example because model file is missing: {module_model_path}")
