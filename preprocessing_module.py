import logging
import os
import onnxruntime as ort
import numpy as np
import cv2
import math

# Assuming custom_exceptions.py is in the same directory or PYTHONPATH
try:
    from custom_exceptions import OCRFileNotFoundError, OCRModelError, OCRImageProcessingError
except ImportError:
    # Basic fallback if custom_exceptions is not found (e.g. during isolated testing)
    class OCRFileNotFoundError(FileNotFoundError): pass
    class OCRModelError(RuntimeError): pass
    class OCRImageProcessingError(RuntimeError): pass

# Module-level logger (Classes will get their own specific loggers via __name__)
# This logger is used by functions if any, or for module-level info.
# Classes should use self.logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__) # Fallback for module-level logging if needed by test setup

class GeometricCorrector:
    def __init__(self, model_path: str, onnx_providers: list[str] | None = None):
        self.logger = logging.getLogger(__name__)
        if onnx_providers is None:
            onnx_providers = ['CPUExecutionProvider']

        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.logger.debug(f"Initializing GeometricCorrector with model: {model_path} and providers: {onnx_providers}")

        if not model_path or not isinstance(model_path, str):
            self.logger.error("Invalid model path provided for GeometricCorrector: Must be a non-empty string.")
            raise ValueError("Model path must be a non-empty string.")

        try:
            self.logger.info(f"Attempting to load geometric correction ONNX model from: {model_path}")
            if not os.path.exists(model_path):
                 self.logger.error(f"Geometric correction model file not found at path: {model_path}")
                 raise OCRFileNotFoundError(f"Geometric correction model file not found: {model_path}")

            self.session = ort.InferenceSession(self.model_path, providers=onnx_providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.logger.info(f"Geometric correction model '{os.path.basename(model_path)}' loaded successfully.")
            self.logger.debug(f"Model Input: {self.input_name}, Output: {self.output_name}, Provider: {self.session.get_providers()}")

        except OCRFileNotFoundError:
            raise
        except ort.capi.onnxruntime_pybind11_state.Fail as ort_fail_error:
            self.logger.error(f"ONNXRuntime model loading failed for geometric model {model_path}: {ort_fail_error}", exc_info=True)
            raise OCRModelError(f"Failed to load geometric ONNX model '{model_path}'.") from ort_fail_error
        except Exception as e:
            self.logger.error(f"Unexpected error loading geometric model {model_path}: {e}", exc_info=True)
            raise OCRModelError(f"Unexpected error initializing GeometricCorrector with model '{model_path}'.") from e

    def correct(self, image_data_np: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Applying geometric correction. Input shape: {image_data_np.shape}, dtype: {image_data_np.dtype}")
        if not isinstance(image_data_np, np.ndarray):
            self.logger.error(f"Invalid data type for geometric correction. Expected NumPy array, got {type(image_data_np)}.")
            raise TypeError("Input for geometric correction must be a NumPy array.")

        if image_data_np.dtype != np.float32:
            self.logger.warning(f"GeometricCorrector input not float32, is {image_data_np.dtype}. Converting.")
            try:
                image_data_np = image_data_np.astype(np.float32)
            except Exception as e_conv:
                self.logger.error(f"Failed to convert input to float32 for GeometricCorrector: {e_conv}", exc_info=True)
                raise OCRImageProcessingError("Failed to convert input to float32 for geometric correction.") from e_conv

        try:
            input_feed = {self.input_name: image_data_np}
            self.logger.debug(f"Running ONNX session for GeometricCorrector with input: '{self.input_name}'")
            result_tensors = self.session.run([self.output_name], input_feed)
            corrected_image_np = result_tensors[0]
            self.logger.info(f"Geometric correction successful. Output shape: {corrected_image_np.shape}, dtype: {corrected_image_np.dtype}")
            return corrected_image_np
        except ort.OrtException as ort_e:
            self.logger.error(f"ONNXRuntime error during geometric correction inference: {ort_e}", exc_info=True)
            raise OCRModelError("Error during geometric correction ONNX inference.") from ort_e
        except Exception as e:
            self.logger.error(f"Unexpected error during geometric correction: {e}", exc_info=True)
            raise OCRImageProcessingError("Unexpected error during geometric correction.") from e


class ImageDeskewer:
    def __init__(self, angle_threshold_degrees: float = 0.5, min_contour_area_ratio: float = 0.01, max_contour_area_ratio: float = 0.95):
        self.logger = logging.getLogger(__name__)
        self.angle_threshold = angle_threshold_degrees
        self.min_contour_area_ratio = min_contour_area_ratio
        self.max_contour_area_ratio = max_contour_area_ratio
        self.logger.info(f"ImageDeskewer initialized. Angle threshold: {angle_threshold_degrees:.2f}°, Contour area ratio: {min_contour_area_ratio:.3f}-{max_contour_area_ratio:.3f}.")

    def deskew(self, image_numpy_array: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Starting deskew operation. Input shape: {image_numpy_array.shape}, dtype: {image_numpy_array.dtype}")

        if not isinstance(image_numpy_array, np.ndarray):
            self.logger.error("ImageDeskewer: Input is not a NumPy array.")
            raise TypeError("Input image for deskewing must be a NumPy array.")

        original_shape = image_numpy_array.shape
        img_to_process = image_numpy_array

        if len(original_shape) == 3 and original_shape[2] == 1:
            img_to_process = np.squeeze(image_numpy_array, axis=2)
            self.logger.debug(f"Squeezed HxWx1 image to 2D: {img_to_process.shape}")
        elif len(original_shape) != 2:
            self.logger.error(f"ImageDeskewer: Unsupported image shape {original_shape}. Must be 2D (HxW) or HxWx1 grayscale.")
            raise OCRImageProcessingError(f"Unsupported image shape for deskewing: {original_shape}")

        if img_to_process.dtype != np.uint8:
            self.logger.warning(f"ImageDeskewer: Input image dtype is {img_to_process.dtype}, not uint8. Clipping and converting.")
            try:
                img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)
            except Exception as e_conv:
                self.logger.error(f"Failed to convert input to uint8 for ImageDeskewer: {e_conv}", exc_info=True)
                raise OCRImageProcessingError("Failed to convert input to uint8 for deskewing.") from e_conv

        img_for_contours = img_to_process
        if np.mean(img_for_contours) > 127:
             self.logger.debug("Deskewer: Image has high mean value, possibly black text on white. Inverting for contour detection.")
             img_for_contours = cv2.bitwise_not(img_for_contours)

        self.logger.debug("Finding contours for deskewing...")
        try:
            contours, _ = cv2.findContours(img_for_contours, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except cv2.error as e_cv_contour:
            self.logger.error(f"OpenCV error in findContours during deskew: {e_cv_contour}", exc_info=True)
            raise OCRImageProcessingError("OpenCV error in findContours during deskew.") from e_cv_contour

        if not contours:
            self.logger.warning("ImageDeskewer: No contours found. Skipping deskew, returning original image.")
            return image_numpy_array

        img_area = img_to_process.shape[0] * img_to_process.shape[1]
        min_area_px = img_area * self.min_contour_area_ratio
        max_area_px = img_area * self.max_contour_area_ratio

        suitable_contours = [c for c in contours if min_area_px < cv2.contourArea(c) < max_area_px]

        if not suitable_contours:
            self.logger.warning(f"ImageDeskewer: No suitable contours found within area range ({min_area_px:.1f}-{max_area_px:.1f} px). Skipping deskew.")
            return image_numpy_array

        largest_contour = max(suitable_contours, key=cv2.contourArea)
        self.logger.debug(f"Largest suitable contour area for deskew: {cv2.contourArea(largest_contour):.2f} px.")

        rect = cv2.minAreaRect(largest_contour)
        angle_from_cv = rect[2]
        box_w_cv, box_h_cv = rect[1]

        # Determine the actual orientation of the text block's longer dimension
        text_orientation_angle = angle_from_cv
        if box_w_cv < box_h_cv:
            text_orientation_angle = angle_from_cv + 90

        # Normalize this orientation to be the smallest angle to the horizontal axis
        # Iteratively normalize until it's in the [-45, 45] range.
        while text_orientation_angle > 45.0: # Handles angles like 90, 180, etc.
            text_orientation_angle -= 90.0
        while text_orientation_angle < -45.0: # Handles angles like -90, -180, etc.
            text_orientation_angle += 90.0

        rotation_angle_for_matrix = -text_orientation_angle

        self.logger.info(f"ImageDeskewer: cv2_angle={angle_from_cv:.2f}, cv_dims=({box_w_cv:.1f},{box_h_cv:.1f}) => final_text_angle={text_orientation_angle:.2f} => rotation_to_apply={rotation_angle_for_matrix:.2f}°.")

        if abs(rotation_angle_for_matrix) < self.angle_threshold:
            self.logger.info(f"ImageDeskewer: Calculated rotation {rotation_angle_for_matrix:.2f}° is below threshold {self.angle_threshold}°. No deskew rotation applied.")
            return image_numpy_array

        self.logger.info(f"ImageDeskewer: Applying rotation of {rotation_angle_for_matrix:.2f}° to deskew image.")
        (h, w) = img_to_process.shape[:2]
        center = (w // 2, h // 2)

        padding_w = int(abs(w * math.sin(math.radians(rotation_angle_for_matrix))) * 0.5 + abs(h * math.cos(math.radians(rotation_angle_for_matrix))) * 0.1)
        padding_h = int(abs(h * math.sin(math.radians(rotation_angle_for_matrix))) * 0.5 + abs(w * math.cos(math.radians(rotation_angle_for_matrix))) * 0.1)
        border_val = 255

        padded_img = cv2.copyMakeBorder(img_to_process, padding_h, padding_h, padding_w, padding_w, cv2.BORDER_CONSTANT, value=border_val)
        padded_center_x, padded_center_y = center[0] + padding_w, center[1] + padding_h

        try:
            M = cv2.getRotationMatrix2D((padded_center_x, padded_center_y), rotation_angle_for_matrix, 1.0)
            deskewed_padded_img = cv2.warpAffine(padded_img, M, (padded_img.shape[1], padded_img.shape[0]),
                                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
        except cv2.error as e_cv_affine:
            self.logger.error(f"OpenCV error during warpAffine in deskew: {e_cv_affine}", exc_info=True)
            raise OCRImageProcessingError("OpenCV error during image rotation for deskew.") from e_cv_affine

        deskewed_img = deskewed_padded_img[padding_h : padding_h + h, padding_w : padding_w + w]

        self.logger.info(f"Image deskewed. Applied rotation: {rotation_angle_for_matrix:.2f}°. Output shape: {deskewed_img.shape}")
        if len(original_shape) == 3 and original_shape[2] == 1:
            deskewed_img = np.expand_dims(deskewed_img, axis=2)
            self.logger.debug(f"Reshaped deskewed image to HxWx1: {deskewed_img.shape}")

        return deskewed_img

class ImageBinarizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("ImageBinarizer initialized.")

    def binarize(self, image_numpy_array: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Starting binarization. Input shape: {image_numpy_array.shape}, dtype: {image_numpy_array.dtype}")

        if not isinstance(image_numpy_array, np.ndarray):
            self.logger.error("ImageBinarizer: Input is not a NumPy array.")
            raise TypeError("Input image for binarization must be a NumPy array.")

        if image_numpy_array.size == 0:
            self.logger.warning("ImageBinarizer: Input image array is empty. Returning as is.")
            return image_numpy_array

        img_to_process = image_numpy_array
        if img_to_process.dtype != np.uint8:
            self.logger.warning(f"ImageBinarizer: Input image dtype is {img_to_process.dtype}, not uint8. Attempting conversion.")
            if np.max(img_to_process) <= 1.0 and img_to_process.dtype == np.float32:
                self.logger.info("ImageBinarizer: Input seems to be float32 normalized (0-1). Scaling to 0-255.")
                img_to_process = (img_to_process * 255.0)
            try:
                img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)
                self.logger.debug(f"Converted to uint8. New min/max: {img_to_process.min()}/{img_to_process.max()}")
            except Exception as e_conv:
                self.logger.error(f"Failed to convert input to uint8 for ImageBinarizer: {e_conv}", exc_info=True)
                raise OCRImageProcessingError("Failed to convert input to uint8 for binarization.") from e_conv

        gray_image: np.ndarray
        if len(img_to_process.shape) == 3 and img_to_process.shape[2] == 3:
            self.logger.debug("ImageBinarizer: Input is 3-channel color. Converting to grayscale.")
            try:
                gray_image = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
            except cv2.error as e_bgr:
                self.logger.warning(f"ImageBinarizer: OpenCV BGR2GRAY conversion failed ({e_bgr}). Trying RGB2GRAY.")
                try:
                    gray_image = cv2.cvtColor(img_to_process, cv2.COLOR_RGB2GRAY)
                except cv2.error as e_rgb:
                    self.logger.error(f"ImageBinarizer: OpenCV color conversion to gray failed for both BGR and RGB: {e_rgb}", exc_info=True)
                    raise OCRImageProcessingError("Failed to convert color image to grayscale for binarization.") from e_rgb
        elif len(img_to_process.shape) == 3 and img_to_process.shape[2] == 1:
            self.logger.debug("ImageBinarizer: Input is HxWx1 grayscale. Squeezing to HxW.")
            gray_image = np.squeeze(img_to_process, axis=2)
        elif len(img_to_process.shape) == 2:
            self.logger.debug("ImageBinarizer: Input is 2-channel (HxW) grayscale.")
            gray_image = img_to_process
        else:
            self.logger.error(f"ImageBinarizer: Unsupported image shape {img_to_process.shape} for binarization.")
            raise OCRImageProcessingError(f"Unsupported image shape for binarization: {img_to_process.shape}")

        self.logger.debug(f"Applying Otsu's binarization to grayscale image of shape {gray_image.shape}")
        try:
            _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.logger.info(f"Binarization successful. Output shape: {binarized_image.shape}, dtype: {binarized_image.dtype}")
            return binarized_image
        except cv2.error as e_otsu:
            self.logger.error(f"OpenCV error during Otsu's thresholding: {e_otsu}", exc_info=True)
            raise OCRImageProcessingError("OpenCV error during Otsu's thresholding for binarization.") from e_otsu
        except Exception as e:
            self.logger.error(f"Unexpected error during binarization: {e}", exc_info=True)
            raise OCRImageProcessingError("Unexpected error during binarization.") from e

if __name__ == '__main__':
    # Setup basic logging for the script to run standalone
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        from config_loader import load_config, create_default_config_if_not_exists
        create_default_config_if_not_exists("config_dev.yaml")
        config = load_config("config_dev.yaml")
    except ImportError:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.warning("config_loader not found or failed, using basic logging configuration for preprocessing_module.py.")
        config = {"preprocessing_settings": {"model_path": "models/dummy_geometric_model.onnx"}}
    except Exception as e:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.error(f"Error loading config via config_loader in preprocessing_module.py: {e}. Using basic config.", exc_info=True)
        config = {"preprocessing_settings": {"model_path": "models/dummy_geometric_model.onnx"}}

    config_settings = config.get("preprocessing_settings", {})
    if not isinstance(config_settings, dict):
        config_settings = {}
    module_model_path = config_settings.get("model_path", "models/dummy_geometric_model.onnx")

    logging.info("\n--- Testing GeometricCorrector ---")
    model_dir = os.path.dirname(module_model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        logging.info(f"Created model directory for standalone run: {model_dir}")

    if not os.path.exists(module_model_path):
        logging.warning(f"Model {module_model_path} not found for standalone run. Attempting to generate.")
        try:
            from generate_dummy_geometric_model import generate_model as generate_geom_model
            generate_geom_model(module_model_path)
            logging.info(f"Generated dummy geometric model at: {module_model_path}")
        except ImportError:
            logging.error("generate_dummy_geometric_model.py not found. Cannot generate model for __main__.")
        except Exception as gen_e:
            logging.error(f"Failed to generate dummy geometric model for __main__: {gen_e}")

    if os.path.exists(module_model_path):
        try:
            corrector = GeometricCorrector(model_path=module_model_path)
            mock_image_np_geom = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            logging.info(f"Mock input for GeometricCorrector: \n{mock_image_np_geom}")
            corrected_image = corrector.correct(mock_image_np_geom)
            logging.info(f"GeometricCorrector test result: \n{corrected_image}")
            expected_output_geom = mock_image_np_geom + 1.0
            assert np.array_equal(corrected_image, expected_output_geom), "GeometricCorrector output not as expected."
        except Exception as e:
            logging.error(f"Error in GeometricCorrector example: {e}", exc_info=True)
    else:
        logging.error(f"Cannot run GeometricCorrector example: model file missing at {module_model_path}")

    logging.info("\n--- Testing ImageBinarizer ---")
    binarizer = ImageBinarizer()
    mock_gray_image_uint8 = np.array([[50, 100, 150], [200, 120, 30]], dtype=np.uint8)
    logging.info(f"Mock grayscale input for Binarizer (uint8): \n{mock_gray_image_uint8}")
    binarized_gray = binarizer.binarize(mock_gray_image_uint8.copy())
    if binarized_gray is not None:
        logging.info(f"Binarizer output for grayscale (uint8): \n{binarized_gray}")
        assert np.all(np.logical_or(binarized_gray == 0, binarized_gray == 255)), "Binarized image should only contain 0 or 255."

    mock_color_image_uint8 = np.array([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]]
    ], dtype=np.uint8)
    logging.info(f"Mock color input for Binarizer (uint8, 3-channel): \n{mock_color_image_uint8}")
    binarized_color = binarizer.binarize(mock_color_image_uint8.copy())
    if binarized_color is not None:
        logging.info(f"Binarizer output for color (uint8, 3-channel): \n{binarized_color}")
        assert np.all(np.logical_or(binarized_color == 0, binarized_color == 255)), "Binarized image from color input should only contain 0 or 255."
        assert len(binarized_color.shape) == 2, "Binarized image from color input should be 2D."

    mock_float_image = np.array([[0.1, 0.4, 0.6], [0.9, 0.5, 0.2]], dtype=np.float32) * 255.0
    logging.info(f"Mock float input for Binarizer (scaled to 0-255): \n{mock_float_image}")
    binarized_float = binarizer.binarize(mock_float_image.copy())
    if binarized_float is not None:
        logging.info(f"Binarizer output for float input: \n{binarized_float}")
        assert np.all(np.logical_or(binarized_float == 0, binarized_float == 255)), "Binarized image from float input should only contain 0 or 255."

    mock_hw1_image_uint8 = np.expand_dims(mock_gray_image_uint8, axis=2)
    logging.info(f"Mock HxWx1 input for Binarizer (uint8): \n{mock_hw1_image_uint8}")
    binarized_hw1 = binarizer.binarize(mock_hw1_image_uint8.copy())
    if binarized_hw1 is not None:
        logging.info(f"Binarizer output for HxWx1 (uint8): \n{binarized_hw1}")
        assert np.all(np.logical_or(binarized_hw1 == 0, binarized_hw1 == 255)), "Binarized image from HxWx1 input should only contain 0 or 255."
        assert len(binarized_hw1.shape) == 2, "Binarized image from HxWx1 input should be 2D."

    logging.info("\n--- Testing ImageDeskewer ---")
    deskewer = ImageDeskewer(angle_threshold_degrees=0.1)
    test_img_h, test_img_w = 200, 300
    base_img = np.full((test_img_h, test_img_w), 255, dtype=np.uint8)
    rect_h, rect_w = 100, 200
    start_x, start_y = (test_img_w - rect_w) // 2, (test_img_h - rect_h) // 2
    base_img[start_y : start_y + rect_h, start_x : start_x + rect_w] = 0

    skew_angle_test = 15.0
    center_test = (test_img_w // 2, test_img_h // 2)
    M_skew = cv2.getRotationMatrix2D(center_test, skew_angle_test, 1.0)
    skewed_image_for_test = cv2.warpAffine(base_img, M_skew, (test_img_w, test_img_h), borderValue=(255,255,255))

    logging.info(f"Created synthetic skewed image for deskew test (angle: {skew_angle_test}°).")
    deskewed_output = deskewer.deskew(skewed_image_for_test.copy())
    if deskewed_output is not None:
        logging.info(f"Deskewer output shape: {deskewed_output.shape}")
        if abs(skew_angle_test) > deskewer.angle_threshold :
            assert not np.array_equal(deskewed_output, skewed_image_for_test), "Deskewed image should not be identical to skewed if angle was significant."

    no_skew_image = base_img.copy()
    logging.info("Testing deskewer with non-skewed image.")
    deskewed_no_skew_output = deskewer.deskew(no_skew_image.copy())
    if deskewed_no_skew_output is not None:
        assert np.array_equal(deskewed_no_skew_output, no_skew_image), "Deskewing a non-skewed image resulted in unexpected changes."
        logging.info("Deskewer correctly handled non-skewed image.")

    logging.info("\n--- Testing Mojo Histogram Calculation ---")
    sample_hist_image_np = np.array([
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
        [10, 20, 10, 20, 10]
    ], dtype=np.uint8)
    logging.info(f"Sample image for histogram (shape: {sample_hist_image_np.shape}):\n{sample_hist_image_np}")

    hist_mojo = None
    try:
        from mojo.mojo.python import Python
        mojo_image_utils_hist = Python.import_module("mojo_image_utils")
        calculate_histogram_mojo_fn = mojo_image_utils_hist.calculate_histogram_mojo

        height_hist, width_hist = sample_hist_image_np.shape[0], sample_hist_image_np.shape[1]
        flat_u8_list_hist = sample_hist_image_np.flatten().tolist()

        logging.info("Attempting histogram calculation with Mojo...")
        hist_obj_mojo = calculate_histogram_mojo_fn(flat_u8_list_hist, height_hist, width_hist)
        hist_mojo = list(hist_obj_mojo)

        if not hist_mojo or len(hist_mojo) != 256:
            logging.error("Mojo histogram calculation returned an invalid list (empty or not 256 bins). Will use NumPy fallback.")
            hist_mojo = None
        else:
            logging.info("Mojo histogram calculation successful.")
            logging.debug("Mojo Histogram (non-zero bins):")
            for i, count in enumerate(hist_mojo):
                if count > 0:
                    logging.debug(f"  Bin {i}: {count}")

    except Exception as e_mojo_hist:
        logging.error(f"Error during Mojo histogram calculation: {e_mojo_hist}. Will use NumPy fallback.", exc_info=True)
        hist_mojo = None

    if hist_mojo is None:
        logging.info("Using NumPy for histogram calculation as fallback or for comparison.")
        hist_numpy, _ = np.histogram(sample_hist_image_np.ravel(), bins=256, range=(0,256))
        logging.info("NumPy histogram calculation successful.")
        logging.debug("NumPy Histogram (non-zero bins):")
        for i, count in enumerate(hist_numpy):
            if count > 0:
                logging.debug(f"  Bin {i}: {count}")

        if hist_mojo is not None:
             are_equal = np.array_equal(hist_mojo, hist_numpy)
             logging.info(f"Comparison: Mojo histogram == NumPy histogram? {are_equal}")
             if not are_equal:
                 for i in range(256):
                     if hist_mojo[i] != hist_numpy[i]:
                         logging.warning(f"Diff at bin {i}: Mojo={hist_mojo[i]}, NumPy={hist_numpy[i]}")
    logging.info("Histogram calculation test finished.")

    logging.info("\n--- Testing Mojo Bounding Box Calculation ---")
    sample_bbox_image_np = np.array([
        [0,   0,   0,   0,   0],
        [0, 255,   0,   0,   0],
        [0,   0, 128,   0,   0],
        [0,  50,   0,  80,   0],
        [0,   0,   0,   0,   0]
    ], dtype=np.uint8)
    logging.info(f"Sample image for BBox (shape: {sample_bbox_image_np.shape}):\n{sample_bbox_image_np}")

    bbox_mojo = None
    try:
        from mojo.mojo.python import Python
        mojo_image_utils_bbox = Python.import_module("mojo_image_utils")
        calculate_bounding_box_mojo_fn = mojo_image_utils_bbox.calculate_bounding_box_mojo

        height_bbox, width_bbox = sample_bbox_image_np.shape[0], sample_bbox_image_np.shape[1]
        flat_u8_list_bbox = sample_bbox_image_np.flatten().tolist()

        logging.info("Attempting bounding box calculation with Mojo...")
        bbox_obj_mojo = calculate_bounding_box_mojo_fn(flat_u8_list_bbox, height_bbox, width_bbox)

        if bbox_obj_mojo is not None:
            try:
                bbox_mojo_tuple = tuple(bbox_obj_mojo)
                if len(bbox_mojo_tuple) == 4:
                    bbox_mojo = bbox_mojo_tuple
                    logging.info(f"Mojo bounding box calculation successful: {bbox_mojo}")
                else:
                    logging.error(f"Mojo bounding box calculation returned invalid tuple structure: {bbox_mojo_tuple}. Will use NumPy fallback.")
                    bbox_mojo = "Error"
            except Exception as e_conversion:
                logging.error(f"Error converting Mojo BBox result to Python tuple: {e_conversion}. Will use NumPy fallback.", exc_info=True)
                bbox_mojo = "Error"
        else:
            bbox_mojo = None
            logging.info("Mojo bounding box calculation: No foreground pixels found (Mojo returned None).")

    except Exception as e_mojo_bbox:
        logging.error(f"Error during Mojo bounding box calculation: {e_mojo_bbox}. Will use NumPy fallback.", exc_info=True)
        bbox_mojo = "Error"

    logging.info("Calculating bounding box with NumPy...")
    rows, cols = np.where(sample_bbox_image_np > 0)
    numpy_bbox = None
    if rows.size > 0:
        min_r, min_c = np.min(rows), np.min(cols)
        max_r, max_c = np.max(rows), np.max(cols)
        numpy_bbox = (int(min_r), int(min_c), int(max_r), int(max_c))
        logging.info(f"NumPy bounding box: {numpy_bbox}")
    else:
        logging.info("NumPy bounding box: No foreground pixels found.")

    if bbox_mojo != "Error" and bbox_mojo is not None and numpy_bbox is not None:
        are_bboxes_equal = (bbox_mojo == numpy_bbox)
        logging.info(f"Comparison: Mojo BBox == NumPy BBox? {are_bboxes_equal}")
        if not are_bboxes_equal:
            logging.warning(f"BBox Mismatch: Mojo={bbox_mojo}, NumPy={numpy_bbox}")
    elif bbox_mojo == "Error":
        logging.warning("Mojo BBox calculation resulted in an error, comparison with NumPy result not fully valid.")
    elif bbox_mojo is None and numpy_bbox is not None:
        logging.warning(f"Mojo found no BBox, NumPy found: {numpy_bbox}. Mismatch.")
    elif numpy_bbox is None and bbox_mojo is not None:
        logging.warning(f"NumPy found no BBox, Mojo found: {bbox_mojo}. Mismatch.")

    logging.info("Bounding Box calculation test finished.")
