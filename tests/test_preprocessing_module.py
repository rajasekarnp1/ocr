import unittest
import os
import logging
import shutil
import numpy as np
import cv2 # For ImageDeskewer and synthetic image creation
import onnxruntime as ort # For checking session type

# Assuming modules are in parent directory or PYTHONPATH is set
# This allows 'python -m unittest discover tests' from /app, or direct run if /app is in path
try:
    from preprocessing_module import GeometricCorrector, ImageBinarizer, ImageDeskewer
    from custom_exceptions import OCRFileNotFoundError, OCRModelError, OCRImageProcessingError
    # For generating dummy ONNX model for GeometricCorrector tests
    from generate_dummy_geometric_model import generate_model as generate_geometric_onnx
except ImportError:
    # Fallback for direct execution if path not set, less ideal
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing_module import GeometricCorrector, ImageBinarizer, ImageDeskewer
    from custom_exceptions import OCRFileNotFoundError, OCRModelError, OCRImageProcessingError
    from generate_dummy_geometric_model import generate_model as generate_geometric_onnx


# Configure logging for tests
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger for the test module itself
if not logger.handlers: # Avoid adding handlers multiple times if tests are run in a suite
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO) # Set to DEBUG for more verbose test logging


class TestGeometricCorrector(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_geom_corrector_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.valid_model_path = os.path.join(self.test_dir, "dummy_geometric_model.onnx")
        # Generate a valid dummy ONNX model for testing
        try:
            generate_geometric_onnx(self.valid_model_path)
            logger.debug(f"Dummy ONNX model generated at {self.valid_model_path} for TestGeometricCorrector.")
        except Exception as e:
            logger.error(f"Failed to generate dummy ONNX model for tests: {e}", exc_info=True)
            raise # Fail setup if model can't be created

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logger.debug(f"Cleaned up test directory: {self.test_dir}")

    def test_successful_initialization(self):
        logger.info("TestGeometricCorrector: test_successful_initialization")
        corrector = GeometricCorrector(model_path=self.valid_model_path)
        self.assertIsNotNone(corrector.session, "ONNX session should be initialized.")
        self.assertIsInstance(corrector.session, ort.InferenceSession, "Session is not an ONNX InferenceSession.")
        self.assertEqual(corrector.model_path, self.valid_model_path)

    def test_initialization_model_not_found(self):
        logger.info("TestGeometricCorrector: test_initialization_model_not_found")
        non_existent_path = os.path.join(self.test_dir, "non_existent.onnx")
        with self.assertRaises(OCRFileNotFoundError):
            GeometricCorrector(model_path=non_existent_path)

    def test_initialization_invalid_model_path_type(self):
        logger.info("TestGeometricCorrector: test_initialization_invalid_model_path_type")
        with self.assertRaises(ValueError):
            GeometricCorrector(model_path=None)
        with self.assertRaises(ValueError):
            GeometricCorrector(model_path=123)

    def test_correct_method_with_onnx(self):
        logger.info("TestGeometricCorrector: test_correct_method_with_onnx")
        corrector = GeometricCorrector(model_path=self.valid_model_path)
        # Dummy ONNX model adds 1.0 to each element
        input_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected_output = input_array + 1.0

        corrected_array = corrector.correct(input_array)
        self.assertIsInstance(corrected_array, np.ndarray)
        np.testing.assert_array_almost_equal(corrected_array, expected_output, decimal=5)

    def test_correct_method_invalid_input_type(self):
        logger.info("TestGeometricCorrector: test_correct_method_invalid_input_type")
        corrector = GeometricCorrector(model_path=self.valid_model_path)
        with self.assertRaises(TypeError):
            corrector.correct("not_a_numpy_array")
        with self.assertRaises(TypeError):
            corrector.correct(None) # Type check should catch this before None specific logic

    def test_correct_method_input_dtype_conversion(self):
        logger.info("TestGeometricCorrector: test_correct_method_input_dtype_conversion")
        corrector = GeometricCorrector(model_path=self.valid_model_path)
        input_array_uint8 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        # Expected output if uint8 is converted to float32 then 1.0 is added
        expected_output = input_array_uint8.astype(np.float32) + 1.0

        corrected_array = corrector.correct(input_array_uint8)
        self.assertIsInstance(corrected_array, np.ndarray)
        self.assertEqual(corrected_array.dtype, np.float32)
        np.testing.assert_array_almost_equal(corrected_array, expected_output, decimal=5)

class TestImageBinarizer(unittest.TestCase):
    def setUp(self):
        self.binarizer = ImageBinarizer()
        logger.info("TestImageBinarizer: setUp completed.")

    def test_binarize_grayscale_uint8(self):
        logger.info("TestImageBinarizer: test_binarize_grayscale_uint8")
        img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        bin_img = self.binarizer.binarize(img)
        self.assertIsInstance(bin_img, np.ndarray)
        self.assertEqual(bin_img.dtype, np.uint8)
        self.assertTrue(np.all(np.logical_or(bin_img == 0, bin_img == 255)))
        # Otsu threshold for [[50,100],[150,200]] is 125. So expected: [[0,0],[255,255]]
        expected = np.array([[0,0],[255,255]], dtype=np.uint8)
        np.testing.assert_array_equal(bin_img, expected)


    def test_binarize_color_uint8(self):
        logger.info("TestImageBinarizer: test_binarize_color_uint8")
        # Create a BGR-like image where green channel is dominant for grayscale conversion
        img_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        img_bgr[0,0] = [10, 50, 10] # Becomes ~38 gray
        img_bgr[0,1] = [20, 150, 20] # Becomes ~116 gray
        img_bgr[1,0] = [30, 200, 30] # Becomes ~158 gray
        img_bgr[1,1] = [40, 250, 40] # Becomes ~200 gray
        # Expected grayscale (approx): [[38, 116], [158, 200]]
        # Otsu threshold for this gray image is around 137. Expected binarized: [[0,0],[255,255]]

        bin_img = self.binarizer.binarize(img_bgr)
        self.assertIsInstance(bin_img, np.ndarray)
        self.assertEqual(bin_img.dtype, np.uint8)
        self.assertEqual(bin_img.ndim, 2) # Should be 2D
        self.assertTrue(np.all(np.logical_or(bin_img == 0, bin_img == 255)))

    def test_binarize_float32_normalized(self):
        logger.info("TestImageBinarizer: test_binarize_float32_normalized")
        img = np.array([[0.1, 0.4], [0.6, 0.9]], dtype=np.float32)
        # Expected uint8: [[25, 102], [153, 229]]
        # Otsu threshold for this is around 127. Expected binarized: [[0,0],[255,255]]
        bin_img = self.binarizer.binarize(img)
        self.assertIsInstance(bin_img, np.ndarray)
        self.assertEqual(bin_img.dtype, np.uint8)
        expected = np.array([[0,0],[255,255]], dtype=np.uint8)
        np.testing.assert_array_equal(bin_img, expected)

    def test_binarize_hw1_uint8(self):
        logger.info("TestImageBinarizer: test_binarize_hw1_uint8")
        img_hw1 = np.array([[[50]], [[100]], [[150]], [[200]]], dtype=np.uint8).reshape(2,2,1)
        bin_img = self.binarizer.binarize(img_hw1)
        self.assertIsInstance(bin_img, np.ndarray)
        self.assertEqual(bin_img.ndim, 2)
        self.assertEqual(bin_img.dtype, np.uint8)
        expected = np.array([[0,0],[255,255]], dtype=np.uint8) # Based on [[50,100],[150,200]]
        np.testing.assert_array_equal(bin_img, expected)

    def test_binarize_empty_image(self):
        logger.info("TestImageBinarizer: test_binarize_empty_image")
        img = np.array([], dtype=np.uint8)
        bin_img = self.binarizer.binarize(img)
        self.assertEqual(bin_img.size, 0)

    def test_binarize_invalid_input_type(self):
        logger.info("TestImageBinarizer: test_binarize_invalid_input_type")
        with self.assertRaises(TypeError):
            self.binarizer.binarize("not_an_array")

    def test_binarize_unsupported_shape(self):
        logger.info("TestImageBinarizer: test_binarize_unsupported_shape")
        img_4d = np.zeros((1,1,1,1), dtype=np.uint8)
        with self.assertRaises(OCRImageProcessingError): # Expecting custom error
            self.binarizer.binarize(img_4d)
        img_1channel_color = np.zeros((10,10,4), dtype=np.uint8) # e.g. RGBA
        with self.assertRaises(OCRImageProcessingError):
            self.binarizer.binarize(img_1channel_color)


class TestImageDeskewer(unittest.TestCase):
    def setUp(self):
        # Low threshold for tests to ensure rotation happens if any angle is detected
        self.deskewer = ImageDeskewer(angle_threshold_degrees=0.01)
        logger.info("TestImageDeskewer: setUp completed.")

    def _create_skewed_image(self, angle_degrees, text_is_black=True, size=(200, 400)):
        h, w = size
        img = np.full((h, w), 255, dtype=np.uint8) # White background

        # Draw a rectangle representing a text block
        rect_h, rect_w = int(h*0.4), int(w*0.7)
        start_y, start_x = (h - rect_h) // 2, (w - rect_w) // 2

        text_color = 0 if text_is_black else 255
        img[start_y : start_y + rect_h, start_x : start_x + rect_w] = text_color

        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        # Use white (255) for border if background is white, black (0) if background is black
        border_color = 255 if text_is_black else 0
        skewed_img = cv2.warpAffine(img, rot_matrix, (w, h), borderValue=border_color)

        # Binarize: Deskewer expects binarized image (white text on black bg, or vice-versa)
        # If text was black (0) on white (255), after rotation, it remains so.
        # If ImageBinarizer output is white text (255) on black (0), we should simulate that.
        if text_is_black: # text is 0, background 255
            # If deskewer expects white text on black, invert
            # current deskewer inverts if mean > 127 (black text on white)
            final_img = skewed_img
        else: # text is 255, background 0
            final_img = skewed_img
        return final_img

    def test_deskew_straight_image(self):
        logger.info("TestImageDeskewer: test_deskew_straight_image")
        straight_img = self._create_skewed_image(0) # 0 degrees skew
        deskewed_img = self.deskewer.deskew(straight_img.copy())
        # Expect original image to be returned if angle is below threshold
        np.testing.assert_array_equal(deskewed_img, straight_img,
                                      "Deskewing a straight image changed it unexpectedly.")

    def test_deskew_slightly_skewed_image(self):
        logger.info("TestImageDeskewer: test_deskew_slightly_skewed_image")
        skew_angle = 5.0
        skewed_img = self._create_skewed_image(skew_angle)

        # To verify, we can't just check if output is identical to unskewed original
        # due to rotation artifacts. Instead, check if the angle is corrected.
        # This requires getting the angle from the deskewed image.
        deskewed_img = self.deskewer.deskew(skewed_img.copy())

        # Re-check angle on deskewed_img (should be close to 0)
        img_for_contours = deskewed_img
        if np.mean(deskewed_img) > 127: img_for_contours = cv2.bitwise_not(deskewed_img)
        contours, _ = cv2.findContours(img_for_contours, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(len(contours) > 0, "No contours found in deskewed image for verification.")

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        final_angle_cv2 = rect[2]

        if final_angle_cv2 < -45: final_corrected_angle = -(90 + final_angle_cv2)
        else: final_corrected_angle = -final_angle_cv2

        logger.info(f"Deskew test (skew: {skew_angle}°): Angle after deskewing: {final_corrected_angle:.2f}°")
        self.assertAlmostEqual(final_corrected_angle, 0.0, delta=self.deskewer.angle_threshold + 0.5, # Allow small delta
                               msg=f"Deskewed angle {final_corrected_angle} not close to 0 for initial skew {skew_angle}°")

    def test_deskew_more_skewed_image(self):
        logger.info("TestImageDeskewer: test_deskew_more_skewed_image")
        skew_angle = -30.0 # Clockwise skew
        skewed_img = self._create_skewed_image(skew_angle)
        deskewed_img = self.deskewer.deskew(skewed_img.copy())

        img_for_contours = deskewed_img
        if np.mean(deskewed_img) > 127: img_for_contours = cv2.bitwise_not(deskewed_img)
        contours, _ = cv2.findContours(img_for_contours, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(len(contours) > 0, "No contours found in deskewed image for verification.")

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        final_angle_cv2 = rect[2]

        if final_angle_cv2 < -45: final_corrected_angle = -(90 + final_angle_cv2)
        else: final_corrected_angle = -final_angle_cv2

        logger.info(f"Deskew test (skew: {skew_angle}°): Angle after deskewing: {final_corrected_angle:.2f}°")
        self.assertAlmostEqual(final_corrected_angle, 0.0, delta=self.deskewer.angle_threshold + 1.0, # Larger delta for larger skews
                               msg=f"Deskewed angle {final_corrected_angle} not close to 0 for initial skew {skew_angle}°")

    def test_deskew_no_contours(self):
        logger.info("TestImageDeskewer: test_deskew_no_contours")
        # ImageBinarizer output is white text (255) on black background (0).
        # So, an all-black image will have no contours for findContours to pick up directly.
        all_black_img = np.zeros((100, 100), dtype=np.uint8)
        deskewed_img = self.deskewer.deskew(all_black_img.copy())
        np.testing.assert_array_equal(deskewed_img, all_black_img,
                                      "Deskewing an all-black image should return it unchanged.")

    def test_deskew_unsuitable_contours(self):
        logger.info("TestImageDeskewer: test_deskew_unsuitable_contours")
        img = np.full((200,200), 0, dtype=np.uint8) # Black background
        # Add a very small white dot (contour area too small)
        img[100,100] = 255
        deskewed_img = self.deskewer.deskew(img.copy())
        np.testing.assert_array_equal(deskewed_img, img,
                                      "Deskewing image with only tiny contour should return it unchanged.")

    def test_deskew_invalid_input_type(self):
        logger.info("TestImageDeskewer: test_deskew_invalid_input_type")
        with self.assertRaises(TypeError):
            self.deskewer.deskew("not_an_array")

    def test_deskew_unsupported_shape(self):
        logger.info("TestImageDeskewer: test_deskew_unsupported_shape")
        img_color = np.zeros((100,100,3), dtype=np.uint8) # Color image
        with self.assertRaises(OCRImageProcessingError): # Deskewer now expects 2D or HxWx1
            self.deskewer.deskew(img_color)

        img_4d = np.zeros((1,1,1,1), dtype=np.uint8)
        with self.assertRaises(OCRImageProcessingError):
            self.deskewer.deskew(img_4d)


if __name__ == '__main__':
    # Ensure logging is configured if running file directly
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()
