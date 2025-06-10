import pytest
import numpy as np
import cv2
from ocrx.modules.preprocessor import AdaptivePreprocessor, XIMGPROC_AVAILABLE
from ocrx.core.exceptions import OCRXConfigurationError

# --- Helper Functions ---
def create_dummy_bgr_image(height: int, width: int, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Creates a dummy BGR image (NumPy array)."""
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image

def create_skewed_text_image(height: int, width: int, angle: float) -> np.ndarray:
    """Creates a simple image with a black line, then skews it."""
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    # Draw a long horizontal line that will show skew clearly
    cv2.line(img, (width // 10, height // 2), (width * 9 // 10, height // 2), (0,0,0), 3)

    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    skewed_img = cv2.warpAffine(img, M, (width, height), borderValue=(255,255,255))
    return skewed_img

# --- Fixtures ---
@pytest.fixture
def default_preprocessor() -> AdaptivePreprocessor:
    """Returns an AdaptivePreprocessor instance with default config."""
    return AdaptivePreprocessor(module_id="test_prep", config={})

@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """A sample RGB image (BGR for OpenCV)."""
    return create_dummy_bgr_image(100, 150, color=(200, 200, 200))

# --- Test Cases ---

def test_preprocessor_init_default(default_preprocessor: AdaptivePreprocessor):
    assert default_preprocessor.is_enabled()
    assert default_preprocessor.config["osd_skew_correction"]["enabled"] is True
    assert default_preprocessor.config["blur"]["enabled"] is False
    assert default_preprocessor.config["binarization"]["enabled"] is True
    assert default_preprocessor.config["binarization"]["method"] == "otsu"

def test_preprocessor_init_custom_config():
    custom_config = {
        "osd_skew_correction": {"enabled": False},
        "blur": {"enabled": True, "kernel_size": (3,3)},
        "binarization": {"enabled": True, "method": "sauvola", "sauvola_window_size": 15}
    }
    preprocessor = AdaptivePreprocessor(module_id="custom_prep", config=custom_config)
    assert not preprocessor.config["osd_skew_correction"]["enabled"]
    assert preprocessor.config["blur"]["enabled"]
    assert preprocessor.config["blur"]["kernel_size"] == (3,3)
    assert preprocessor.config["binarization"]["method"] == "sauvola"

def test_validate_config_errors():
    with pytest.raises(OCRXConfigurationError, match="'osd_skew_correction' config must be a dict"):
        AdaptivePreprocessor(module_id="err_prep", config={"osd_skew_correction": "not_a_dict"})
    with pytest.raises(OCRXConfigurationError, match="Unknown binarization method 'invalid_method'"):
        AdaptivePreprocessor(module_id="err_prep_bin", config={"binarization": {"method": "invalid_method"}})

def test_process_disabled_module(default_preprocessor: AdaptivePreprocessor, sample_image_rgb: np.ndarray):
    default_preprocessor.config["enabled"] = False
    processed_image = default_preprocessor.process(sample_image_rgb.copy())
    assert np.array_equal(processed_image, sample_image_rgb) # Should return original

# --- Test individual steps (internal methods) ---

def test_apply_blur(default_preprocessor: AdaptivePreprocessor, sample_image_rgb: np.ndarray):
    config = {"enabled": True, "kernel_size": (5,5), "sigma_x": 0}
    blurred_image = default_preprocessor._apply_blur(sample_image_rgb.copy(), config)
    assert not np.array_equal(blurred_image, sample_image_rgb) # Should be different
    assert blurred_image.shape == sample_image_rgb.shape

def test_apply_binarization_otsu(default_preprocessor: AdaptivePreprocessor, sample_image_rgb: np.ndarray):
    config = {"enabled": True, "method": "otsu", "max_value": 255}
    # Create an image that's not all one color for Otsu to work
    img_for_otsu = sample_image_rgb.copy()
    cv2.rectangle(img_for_otsu, (10,10), (50,50), (50,50,50), -1) # Darker patch

    binary_image = default_preprocessor._apply_binarization(img_for_otsu, config)
    assert binary_image.shape == img_for_otsu.shape
    assert len(np.unique(binary_image)) <= 2 # Should be binary (0 and 255) or just one if fully black/white
    # Check if it's BGR
    assert len(binary_image.shape) == 3 and binary_image.shape[2] == 3


@pytest.mark.skipif(not XIMGPROC_AVAILABLE, reason="cv2.ximgproc (opencv-contrib) not available for Sauvola test.")
def test_apply_binarization_sauvola(default_preprocessor: AdaptivePreprocessor, sample_image_rgb: np.ndarray):
    config = {"enabled": True, "method": "sauvola", "sauvola_window_size": 25, "sauvola_k": 0.2, "max_value": 255}
    img_for_sauvola = sample_image_rgb.copy()
    cv2.putText(img_for_sauvola, "Text", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2) # Add some text

    binary_image = default_preprocessor._apply_binarization(img_for_sauvola, config)
    assert binary_image.shape == img_for_sauvola.shape
    unique_values = np.unique(binary_image)
    assert len(unique_values) <= 2
    assert (0 in unique_values and 255 in unique_values) or \
           (len(unique_values) == 1 and (unique_values[0] == 0 or unique_values[0] == 255))


def test_skew_correction(default_preprocessor: AdaptivePreprocessor):
    skewed_img = create_skewed_text_image(200, 300, angle=5.0)
    # Convert to BGR if needed (create_skewed_text_image already returns BGR)

    config = {"enabled": True, "min_skew_angle_to_correct": 0.5}
    corrected_img = default_preprocessor._orient_and_skew_correct(skewed_img, config)

    # This is a basic check. True verification needs more advanced image analysis.
    # For now, we check if the image was processed (not identical) and shape is maintained.
    assert corrected_img.shape == skewed_img.shape
    # If angle was significant, images should not be identical
    if 5.0 >= config["min_skew_angle_to_correct"]:
         assert not np.array_equal(corrected_img, skewed_img), "Image should have changed after skew correction"
    else:
         assert np.array_equal(corrected_img, skewed_img), "Image should not change if skew angle is below threshold"

    # Test with very small angle that should be ignored
    small_skew_img = create_skewed_text_image(200, 300, angle=0.1)
    corrected_small_skew = default_preprocessor._orient_and_skew_correct(small_skew_img, config)
    assert np.array_equal(corrected_small_skew, small_skew_img), "Small skew should be ignored"


def test_process_method_full_pipeline(default_preprocessor: AdaptivePreprocessor, tmp_path: Path):
    """Test the main process method with a sequence of operations."""
    # Create a skewed image for testing
    test_img_path = tmp_path / "skewed_for_process.png"
    skewed_img = create_skewed_text_image(200, 300, angle=-3.0)
    cv2.imwrite(str(test_img_path), skewed_img)

    # Load it as if it's an input
    img_to_process = cv2.imread(str(test_img_path))
    assert img_to_process is not None

    # Configure preprocessor to enable all steps
    config = {
        "enabled": True,
        "osd_skew_correction": {"enabled": True, "min_skew_angle_to_correct": 0.5},
        "blur": {"enabled": True, "kernel_size": (3,3)}, # Use small kernel for less drastic change
        "binarization": {"enabled": True, "method": "otsu"}
    }
    preprocessor = AdaptivePreprocessor("full_pipe_prep", config)

    processed_image = preprocessor.process(img_to_process)

    assert processed_image.shape == img_to_process.shape
    # It's hard to assert specific pixel values after a full pipeline.
    # We primarily check that it runs without error and output is different from input.
    assert not np.array_equal(processed_image, img_to_process)

    # Check if binarized (mostly 2 unique values, 0 and 255, for each channel if BGR)
    # The binarization step converts back to BGR, so each channel should be binary.
    unique_vals_ch1 = np.unique(processed_image[:,:,0])
    assert len(unique_vals_ch1) <= 2 or (len(unique_vals_ch1) == 1 and (unique_vals_ch1[0] == 0 or unique_vals_ch1[0] == 255))

def test_process_with_config_override(default_preprocessor: AdaptivePreprocessor, sample_image_rgb: np.ndarray):
    # Default config has blur disabled
    assert default_preprocessor.config["blur"]["enabled"] is False

    # Override to enable blur for this call
    override_config = {"blur": {"enabled": True, "kernel_size": (5,5)}}

    # Make a copy to ensure original sample_image_rgb is not modified by other tests if it were mutable
    img_copy = sample_image_rgb.copy()
    processed_image = default_preprocessor.process(img_copy, config_override=override_config)

    # Check that blur was applied (image is different)
    assert not np.array_equal(processed_image, img_copy)

    # Check that the original module config is unchanged
    assert default_preprocessor.config["blur"]["enabled"] is False
