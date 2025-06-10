import logging
from typing import Any, Dict, Optional
import numpy as np
import cv2
# Attempt to import ximgproc for Sauvola, but make it optional
try:
    from cv2 import ximgproc # type: ignore
    XIMGPROC_AVAILABLE = True
except ImportError:
    XIMGPROC_AVAILABLE = False
    # logging.getLogger(__name__).warning( # Logger not defined at module level yet
    #    "cv2.ximgproc not found. Sauvola binarization will not be available. "
    #    "Consider installing opencv-contrib-python."
    # )


from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.exceptions import OCRXConfigurationError, OCRXProcessingError

class AdaptivePreprocessor(OCRXModuleBase):
    """
    Performs adaptive preprocessing on an image, including orientation,
    skew correction, blur, and binarization.
    """

    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        # Initialize logger at instance level after super().__init__
        if not XIMGPROC_AVAILABLE and self.config.get("binarization", {}).get("method") == "sauvola":
            self.logger.warning(
               "cv2.ximgproc not found. Sauvola binarization specified in config but will not be available. "
               "Consider installing opencv-contrib-python. Otsu will be used as fallback if enabled, or no binarization."
            )

    def _validate_config(self) -> None:
        super()._validate_config()
        # Example: check for specific preprocessing step configurations
        self.config.setdefault("osd_skew_correction", {"enabled": True}) # Default to enabled
        self.config.setdefault("blur", {"enabled": False, "kernel_size": (5,5)})
        self.config.setdefault("binarization", {"enabled": True, "method": "otsu", "threshold": 0, "max_value": 255})

        if not isinstance(self.config["osd_skew_correction"], dict):
            raise OCRXConfigurationError(f"{self.module_id}: 'osd_skew_correction' config must be a dict.")
        if not isinstance(self.config["blur"], dict):
            raise OCRXConfigurationError(f"{self.module_id}: 'blur' config must be a dict.")
        if not isinstance(self.config["binarization"], dict):
            raise OCRXConfigurationError(f"{self.module_id}: 'binarization' config must be a dict.")

        bin_method = self.config["binarization"].get("method", "otsu")
        if bin_method not in ["otsu", "sauvola"]:
            raise OCRXConfigurationError(f"{self.module_id}: Unknown binarization method '{bin_method}'. Supported: otsu, sauvola.")

        self.logger.info(f"{self.module_id} validated config: {self.config}")


    def _orient_and_skew_correct(self, image: np.ndarray, step_config: Dict[str, Any]) -> np.ndarray:
        self.logger.debug(f"Applying OSD & Skew Correction with config: {step_config}")

        # 1. OSD (Orientation Detection) - Placeholder for MVP
        # In a full implementation, this would use Tesseract's OSD or a dedicated model.
        # For now, assume standard orientation or rely on skew correction for minor adjustments.
        self.logger.info("OSD (Orientation Detection) is a placeholder for MVP.")

        # 2. Skew Correction
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert if text is dark on light background (common for scanned documents)
        # This helps contour detection. A more robust method might analyze pixel intensity distribution.
        if np.mean(gray) > 127: # Heuristic: if mostly light, invert
             inverted_gray = cv2.bitwise_not(gray)
        else:
             inverted_gray = gray

        # Threshold to get binary image for contour finding
        _, thresh = cv2.threshold(inverted_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.logger.warning("No contours found for skew correction. Skipping.")
            return image

        # Find the largest contour by area (heuristic: main text block)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the minimum area bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2] # Angle is in [-90, 0)

        # Adjust angle: if angle is closer to -90, it means 90 degrees rotation in the other direction
        if angle < -45:
            angle = 90 + angle

        self.logger.info(f"Detected skew angle: {angle:.2f} degrees.")

        if abs(angle) < step_config.get("min_skew_angle_to_correct", 0.5): # Don't correct very small angles
            self.logger.info(f"Skew angle {angle:.2f} is below threshold, no correction applied.")
            return image

        # Rotate the image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Use white padding to avoid black borders after rotation
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

        self.logger.info("Skew correction applied.")
        return corrected_image

    def _apply_blur(self, image: np.ndarray, step_config: Dict[str, Any]) -> np.ndarray:
        self.logger.debug(f"Applying Gaussian Blur with config: {step_config}")
        kernel_size_tuple = tuple(step_config.get("kernel_size", (5,5)))
        if not (isinstance(kernel_size_tuple, tuple) and len(kernel_size_tuple) == 2 and
                all(isinstance(x, int) and x > 0 and x % 2 == 1 for x in kernel_size_tuple)):
            self.logger.warning(f"Invalid kernel_size {kernel_size_tuple} for GaussianBlur. Must be tuple of 2 odd positive ints. Using (5,5).")
            kernel_size_tuple = (5,5)

        sigma_x = step_config.get("sigma_x", 0) # Default sigmaX for GaussianBlur
        blurred_image = cv2.GaussianBlur(image, kernel_size_tuple, sigma_x)
        self.logger.info(f"Gaussian blur applied with kernel {kernel_size_tuple}.")
        return blurred_image

    def _apply_binarization(self, image: np.ndarray, step_config: Dict[str, Any]) -> np.ndarray:
        self.logger.debug(f"Applying Binarization with config: {step_config}")
        method = step_config.get("method", "otsu").lower()

        # Ensure image is grayscale for binarization
        if len(image.shape) == 3 and image.shape[2] == 3: # BGR
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2: # Already grayscale
            gray_image = image
        else:
            raise OCRXProcessingError(f"Unsupported image format for binarization: shape {image.shape}")

        binary_image: Optional[np.ndarray] = None
        if method == "sauvola":
            if XIMGPROC_AVAILABLE:
                window_size = step_config.get("sauvola_window_size", 25) # Example param
                k = step_config.get("sauvola_k", 0.2) # Example param
                if not (isinstance(window_size, int) and window_size > 0 and window_size % 2 == 1):
                    self.logger.warning(f"Invalid Sauvola window_size {window_size}. Must be an odd positive int. Using 25.")
                    window_size = 25
                if not (isinstance(k, (float, int)) and 0.01 <= k <= 1.0): # k is typically small
                    self.logger.warning(f"Sauvola k value {k} out of typical range (0.01-1.0). Using 0.2.")
                    k = 0.2

                self.logger.info(f"Applying Sauvola binarization (window: {window_size}, k: {k})...")
                # cv2.ximgproc.niBlackThreshold with NIBLACK_SAUVOLA
                # Note: niBlackThreshold with NIBLACK_SAUVOLA is Sauvola's method.
                binary_image = ximgproc.niBlackThreshold(
                    gray_image,
                    maxValue=step_config.get("max_value", 255),
                    type=cv2.THRESH_BINARY, # type: ignore
                    blockSize=window_size, # type: ignore
                    k=k, # type: ignore
                    binarizationMethod=ximgproc.BINARIZATION_SAUVOLA # type: ignore
                )
            else:
                self.logger.warning("Sauvola binarization requested but cv2.ximgproc is not available. Falling back to Otsu.")
                method = "otsu" # Fallback

        if method == "otsu": # Also handles fallback from Sauvola
            threshold_val = step_config.get("threshold", 0) # Otsu ignores this if THRESH_OTSU is used
            max_val = step_config.get("max_value", 255)
            self.logger.info("Applying Otsu's binarization...")
            _, binary_image = cv2.threshold(gray_image, threshold_val, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if binary_image is None:
            raise OCRXProcessingError(f"Binarization failed for method '{step_config.get('method', 'otsu')}'")

        self.logger.info(f"Binarization applied using method: {method}.")
        # Convert binary image back to BGR for consistency if other modules expect 3 channels
        return cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)


    def process(self, image: np.ndarray, config_override: Optional[Dict] = None) -> np.ndarray:
        """
        Applies enabled preprocessing steps to the input image.

        Args:
            image: The input image as a NumPy BGR array.
            config_override: Optional dictionary to override specific step configurations for this call.

        Returns:
            The processed image as a NumPy BGR array.
        """
        if not self.is_enabled():
            self.logger.info(f"Module {self.module_id} is disabled. Skipping preprocessing.")
            return image

        if not isinstance(image, np.ndarray):
            raise OCRXProcessingError(f"{self.module_id} received invalid image data type: {type(image)}")

        self.logger.info(f"Starting preprocessing for image (shape: {image.shape}) with {self.module_id}.")
        processed_image = image.copy()

        # Merge base config with runtime override
        current_config = {**self.config, **(config_override or {})}

        # 1. OSD and Skew Correction
        osd_skew_config = current_config.get("osd_skew_correction", {})
        if osd_skew_config.get("enabled", False): # Default to False if not specified in merged
            try:
                processed_image = self._orient_and_skew_correct(processed_image, osd_skew_config)
            except Exception as e:
                self.logger.error(f"Error during OSD/Skew Correction: {e}", exc_info=True)
                # Decide: re-raise, or continue with original/partially processed image?
                # For now, log and continue with current state of processed_image

        # 2. Blur
        blur_config = current_config.get("blur", {})
        if blur_config.get("enabled", False):
            try:
                processed_image = self._apply_blur(processed_image, blur_config)
            except Exception as e:
                self.logger.error(f"Error during Blur: {e}", exc_info=True)

        # 3. Binarization
        binarization_config = current_config.get("binarization", {})
        if binarization_config.get("enabled", False):
            try:
                processed_image = self._apply_binarization(processed_image, binarization_config)
            except Exception as e:
                self.logger.error(f"Error during Binarization: {e}", exc_info=True)

        self.logger.info(f"Preprocessing completed for image with {self.module_id}.")
        return processed_image
