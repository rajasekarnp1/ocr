import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple

# Configure basic logging for the module
# This will be overridden if the main application configures logging.
# Using a specific name for the logger rather than root.
module_logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

class ImagePreprocessor:
    """
    Handles preprocessing of images for OCR, including skew correction and binarization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the ImagePreprocessor.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary.
                                               Expected keys:
                                               - 'binarization_block_size' (int, odd, default: 11)
                                               - 'binarization_c_value' (int, default: 7)
            logger (Optional[logging.Logger]): Logger instance. If None, a new logger
                                               for this class will be created.
        """
        self.config = config if config is not None else {}
        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self.logger.info("ImagePreprocessor initialized with config: %s", self.config)

    def _calculate_skew_angle(self, image_gray: np.ndarray) -> float:
        """
        Calculates the skew angle of the text in a grayscale image.
        This implementation uses the minimum area rectangle method on contours
        of the inverted and thresholded image.

        Args:
            image_gray (np.ndarray): Grayscale input image.

        Returns:
            float: The estimated skew angle in degrees. Returns 0.0 if angle detection fails
                   or if the detected angle is considered unreliable.
        """
        self.logger.debug("Calculating skew angle.")

        # Invert the image (assuming dark text on light background)
        inverted_image = cv2.bitwise_not(image_gray)

        # Threshold the image to get a binary image. Otsu's method is good for finding foreground.
        _, thresh = cv2.threshold(inverted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find coordinates of all non-zero (white) pixels, which represent the text.
        # In OpenCV 3.x and earlier, findContours modified the source image.
        # In OpenCV 4.x, it doesn't. Assuming OpenCV 4.x or handling it if needed.
        # For robustness, we can use np.where as before, or find contours.
        # Using np.where is simpler if we just need points for minAreaRect.
        coords = np.column_stack(np.where(thresh > 0))

        if coords.shape[0] < 50: # Heuristic: need a certain number of points for a reliable rectangle
            self.logger.warning(f"Not enough non-zero pixels ({coords.shape[0]}) to reliably detect skew. Returning 0.0 angle.")
            return 0.0

        # Get the minimum area bounding rectangle for the non-zero pixels
        try:
            # rect is ((center_x, center_y), (width, height), angle_of_rotation)
            # The angle is in degrees, from -90 (exclusive) to 0 (inclusive).
            rect = cv2.minAreaRect(coords) # Note: expects points in (y,x) format from np.where
            angle = rect[-1]
        except Exception as e:
            self.logger.warning(f"cv2.minAreaRect failed: {e}. Returning 0.0 angle.")
            return 0.0

        # Adjust angle: cv2.minAreaRect returns angles in [-90, 0).
        # A horizontal rectangle has an angle of -90 or 0.
        # If width < height, angle is the one that needs to be corrected.
        # If width > height, angle is correct, but might be close to -90.
        # We want the angle relative to the horizontal axis.
        # If angle is -90 (perfectly vertical box), this means text is vertical, which is unusual for skew.
        # If angle is 0 (perfectly horizontal box), no skew.

        if rect[1][0] < rect[1][1]: # width < height, box is "standing up"
            angle = angle + 90 # Rotate angle to be relative to horizontal text line

        # Now, 'angle' represents the deviation from horizontal.
        # If angle is, e.g., -3 degrees, it means text is tilted 3 degrees clockwise.
        # We want to rotate counter-clockwise by 3 degrees. So, use -angle for rotation.
        # If angle is, e.g., 3 degrees (after +90 adjustment), text is tilted 3 deg counter-clockwise.
        # We want to rotate clockwise by 3 degrees. So, use -angle for rotation.

        # Clamp angle to a practical range, e.g., +/- 30 or +/- 45 degrees.
        # Angles outside this are likely errors or extreme cases not handled by this basic method.
        if abs(angle) > 45:
            self.logger.warning(f"Detected skew angle {angle:.2f} is outside +/- 45 degree range. Clamping to 0.0 for safety.")
            return 0.0

        # If the angle is very small, consider it as no skew.
        if abs(angle) < 0.1: # Threshold for negligible skew
             return 0.0

        self.logger.info(f"Calculated skew angle for correction: {angle:.2f} degrees.")
        return angle # This is the angle by which the image is skewed. Rotation should be by -angle.

    def correct_skew(self, image_data: np.ndarray) -> np.ndarray:
        """
        Corrects the skew of an image. Assumes input is already grayscale.

        Args:
            image_data (np.ndarray): Grayscale input image data.

        Returns:
            np.ndarray: Deskewed grayscale image data. Returns original if processing fails.
        """
        self.logger.info("Attempting skew correction on grayscale image.")

        # Ensure input is grayscale
        if not (len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[2] == 1)):
            self.logger.warning("Input image for skew correction is not grayscale. Returning original.")
            return image_data

        # If it's a 3D grayscale image, squeeze it to 2D
        current_image_gray = np.squeeze(image_data)


        skew_angle = self._calculate_skew_angle(current_image_gray)

        if abs(skew_angle) < 0.1: # Negligible skew
            self.logger.info("Skew angle is negligible. No rotation applied.")
            return current_image_gray

        (h, w) = current_image_gray.shape[:2]
        center = (w // 2, h // 2)

        try:
            # We want to rotate by the negative of the detected skew angle
            # e.g. if text is skewed 5 degrees clockwise (angle = 5), rotate by -5.
            rotation_matrix = cv2.getRotationMatrix2D(center, -skew_angle, 1.0)

            # Determine border color: if most of the border is white, fill with white. Otherwise black.
            # This helps avoid black borders on white paper after rotation.
            # Simple heuristic: check corners
            corners = [current_image_gray[0,0], current_image_gray[0,w-1], current_image_gray[h-1,0], current_image_gray[h-1,w-1]]
            avg_corner_intensity = np.mean(corners)
            border_value = (255,255,255) if avg_corner_intensity > 128 else (0,0,0)

            deskewed_image = cv2.warpAffine(current_image_gray, rotation_matrix, (w, h),
                                            flags=cv2.INTER_CUBIC, # Smoother interpolation
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=border_value) # Fill with white or black
            self.logger.info(f"Applied skew correction with rotation angle: {-skew_angle:.2f} degrees.")
            return deskewed_image
        except cv2.error as e:
            self.logger.error(f"OpenCV error during cv2.warpAffine for skew correction: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during skew correction: {e}")
        return current_image_gray # Return original grayscale image on error

    def binarize_image(self, image_data: np.ndarray) -> np.ndarray:
        """
        Binarizes an image using adaptive thresholding. Assumes input is grayscale.

        Args:
            image_data (np.ndarray): Grayscale input image data.

        Returns:
            np.ndarray: Binarized image data. Returns original if processing fails.
        """
        self.logger.info("Applying adaptive threshold binarization to grayscale image.")

        # Ensure input is grayscale
        if not (len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[2] == 1)):
            self.logger.warning("Input image for binarization is not grayscale. Returning original.")
            return image_data

        current_image_gray = np.squeeze(image_data)

        try:
            block_size = self.config.get('binarization_block_size', 11)
            c_value = self.config.get('binarization_c_value', 7)

            if not isinstance(block_size, int) or block_size <= 1:
                self.logger.warning(f"Invalid binarization_block_size: {block_size}. Using default 11.")
                block_size = 11
            if block_size % 2 == 0: # Ensure block_size is odd
                block_size +=1
                self.logger.info(f"Binarization block_size was even, adjusted to {block_size}")

            if not isinstance(c_value, (int, float)):
                self.logger.warning(f"Invalid binarization_c_value: {c_value}. Using default 7.")
                c_value = 7

            binarized_image = cv2.adaptiveThreshold(current_image_gray, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, # Standard binary: foreground white, background black
                                                    block_size, c_value)
            self.logger.info(f"Applied adaptive thresholding with block_size={block_size}, C={c_value}.")
            return binarized_image
        except cv2.error as e:
            self.logger.error(f"OpenCV error during adaptive thresholding: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during adaptive thresholding: {e}")
        return current_image_gray # Return original grayscale image on error


    def process_image(self, image_data: np.ndarray) -> np.ndarray:
        """
        Main method to preprocess an image.
        Converts to grayscale (if needed), corrects skew, and binarizes.

        Args:
            image_data (np.ndarray): Input image data (assumed to be RGB, BGR, or Grayscale NumPy array).

        Returns:
            np.ndarray: Processed image data (grayscale, deskewed, binarized).
                        The output is always a 2D grayscale (binary) image.

        Raises:
            ValueError: If input image_data is not a NumPy array or has unsupported dimensions.
        """
        self.logger.info("Starting image processing workflow.")
        if not isinstance(image_data, np.ndarray):
            self.logger.error("Invalid input: image_data must be a NumPy array.")
            raise ValueError("Input image_data must be a NumPy array.")

        if image_data.ndim not in [2, 3] or \
           (image_data.ndim == 3 and image_data.shape[2] not in [1, 3]):
            self.logger.error(f"Invalid image dimensions: shape {image_data.shape}. Must be 2D, 3D (1 channel), or 3D (3 channels).")
            raise ValueError("Input image must be a 2D (grayscale) or 3D (1 or 3 channels) NumPy array.")

        # 1. Convert to Grayscale if it's a color image
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            self.logger.info("Converting color image to grayscale.")
            try:
                # Assume BGR as it's common with cv2.imread, but RGB is also possible.
                # cvtColor handles both if the source image is loaded correctly by OpenCV.
                gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                self.logger.error(f"OpenCV error during grayscale conversion: {e}")
                raise  # Re-raise critical error
        elif image_data.ndim == 3 and image_data.shape[2] == 1: # Grayscale but 3D
             self.logger.info("Image is 3D grayscale. Squeezing to 2D.")
             gray_image = np.squeeze(image_data)
        elif image_data.ndim == 2: # Already 2D grayscale
            self.logger.info("Image is already 2D grayscale.")
            gray_image = image_data.copy() # Work on a copy
        else: # Should be caught by initial checks, but as a safeguard
            self.logger.error(f"Unsupported image format for grayscale conversion: shape {image_data.shape}")
            raise ValueError(f"Unsupported image format for grayscale conversion: shape {image_data.shape}")

        # 2. Correct Skew
        # Skew correction should operate on grayscale and return grayscale
        deskewed_image = self.correct_skew(gray_image)

        # 3. Binarize Image
        # Binarization should operate on grayscale (deskewed) and return binary (still grayscale format)
        processed_image = self.binarize_image(deskewed_image)

        self.logger.info("Image processing workflow completed.")
        return processed_image

if __name__ == '__main__':
    # Configure basic logging for the test script
    # This setup is specific to this __main__ block for testing.
    # The application itself should configure logging.
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Ensure logs go to console

    main_logger = logging.getLogger(__file__) # Logger for this test script
    main_logger.info("Running ImagePreprocessor self-test from __main__...")

    # Create a dummy RGB image (e.g., 300x400 pixels, 3 channels)
    dummy_height, dummy_width = 300, 400
    # Start with a white background
    dummy_rgb_image = np.full((dummy_height, dummy_width, 3), (220, 220, 220), dtype=np.uint8)

    # Add some "text" (a dark rectangle)
    # Coordinates: top-left (x1,y1), bottom-right (x2,y2)
    cv2.putText(dummy_rgb_image, "Test Text Line 1", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
    cv2.putText(dummy_rgb_image, "Another Line ABC", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)

    # Introduce a slight rotation to test skew correction
    angle_to_skew = 3.0 # degrees clockwise
    center = (dummy_width // 2, dummy_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_to_skew, 1.0)

    # Fill with a light gray color similar to background after rotation
    skewed_dummy_image = cv2.warpAffine(dummy_rgb_image, rotation_matrix, (dummy_width, dummy_height),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(210,210,210))

    main_logger.info(f"Created a dummy skewed RGB image with shape: {skewed_dummy_image.shape}, dtype: {skewed_dummy_image.dtype}")

    # Save the dummy images for inspection (optional)
    output_dir = "ocr_components_test_output"
    try:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            main_logger.info(f"Created directory {output_dir} for test images.")

        cv2.imwrite(os.path.join(output_dir, "0_dummy_original_image.png"), dummy_rgb_image)
        cv2.imwrite(os.path.join(output_dir, "1_dummy_skewed_image.png"), skewed_dummy_image)
        main_logger.info(f"Saved original and skewed dummy images in '{output_dir}/'.")
    except Exception as e:
        main_logger.error(f"Error saving dummy images: {e}. This might happen in environments without GUI/write access or restrictive permissions.")

    # Instantiate the preprocessor
    # Test with default config and then with custom
    preprocessor_default_config = ImagePreprocessor(logger=main_logger) # Use main_logger for its output
    preprocessor_custom_config = ImagePreprocessor(
        config={'binarization_block_size': 15, 'binarization_c_value': 5},
        logger=main_logger
    )

    for i, preprocessor_instance in enumerate([preprocessor_default_config, preprocessor_custom_config]):
        main_logger.info(f"--- Testing with Preprocessor instance {i+1} ---")
        try:
            # Convert BGR (OpenCV default) to RGB if needed by a component,
            # but our preprocessor expects BGR or Grayscale and handles it.
            # Here, skewed_dummy_image is already BGR-like.

            # First, convert to grayscale (as orchestrator might do)
            gray_before_processing = cv2.cvtColor(skewed_dummy_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(output_dir, f"{i+2}_dummy_skewed_gray.png"), gray_before_processing)
            main_logger.info(f"Saved initial grayscale version: {i+2}_dummy_skewed_gray.png")

            # Test individual steps (optional, for debugging)
            deskewed_img = preprocessor_instance.correct_skew(gray_before_processing.copy()) # Pass a copy
            cv2.imwrite(os.path.join(output_dir, f"{i+2}_dummy_deskewed.png"), deskewed_img)
            main_logger.info(f"Saved deskewed version: {i+2}_dummy_deskewed.png")

            binarized_from_deskewed = preprocessor_instance.binarize_image(deskewed_img.copy()) # Pass a copy
            cv2.imwrite(os.path.join(output_dir, f"{i+2}_dummy_binarized_from_deskewed.png"), binarized_from_deskewed)
            main_logger.info(f"Saved binarized (from deskewed) version: {i+2}_dummy_binarized_from_deskewed.png")

            # Test the full process_image
            main_logger.info("Testing full process_image()...")
            processed_image = preprocessor_instance.process_image(skewed_dummy_image.copy()) # Pass a copy

            main_logger.info(f"Processed image characteristics - Shape: {processed_image.shape}, Dtype: {processed_image.dtype}")
            main_logger.info(f"Processed image min value: {processed_image.min()}, max value: {processed_image.max()}")

            cv2.imwrite(os.path.join(output_dir, f"{i+2}_dummy_final_processed_image.png"), processed_image)
            main_logger.info(f"Saved final processed image: {i+2}_dummy_final_processed_image.png")

            if len(processed_image.shape) == 2:
                main_logger.info("Output is grayscale (2D array), as expected.")
            else:
                main_logger.error(f"Output image is not grayscale. Shape: {processed_image.shape}")

            unique_values = np.unique(processed_image)
            if len(unique_values) <= 2 and (0 in unique_values or 255 in unique_values): # Allows for all black or all white
                 main_logger.info(f"Output image is binary or near-binary. Unique values: {unique_values}")
            else:
                main_logger.warning(f"Output image may not be strictly binary. Unique values: {unique_values}.")

        except ValueError as ve:
            main_logger.error(f"ValueError during processing with instance {i+1}: {ve}")
        except Exception as e:
            main_logger.error(f"An unexpected error occurred during self-test with instance {i+1}: {e}", exc_info=True)

    main_logger.info(f"ImagePreprocessor self-test finished. Check images in '{output_dir}/'.")
