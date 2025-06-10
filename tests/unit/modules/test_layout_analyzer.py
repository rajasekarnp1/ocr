import pytest
import numpy as np
import cv2

from ocrx.modules.layout_analyzer import LayoutAnalyzer
from ocrx.core.data_objects import PageContext, TextRegion
from ocrx.core.exceptions import OCRXConfigurationError

# --- Helper Functions ---
def create_test_image_with_rectangles(height: int, width: int, rects: list, color_mode: str = "BGR") -> np.ndarray:
    """Creates an image with specified black rectangles on a white background."""
    if color_mode == "BGR":
        img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
        rect_color = (0, 0, 0)
    else: # Grayscale
        img = np.full((height, width), 255, dtype=np.uint8)
        rect_color = 0

    for r in rects: # r is (x, y, w, h)
        cv2.rectangle(img, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), rect_color, -1)
    return img

# --- Fixtures ---
@pytest.fixture
def default_layout_analyzer() -> LayoutAnalyzer:
    return LayoutAnalyzer(module_id="test_layout", config={})

@pytest.fixture
def sample_page_context() -> PageContext:
    # Preprocessed image is used by LayoutAnalyzer, original_image for cropping
    img_bgr = create_test_image_with_rectangles(200, 300, [(20, 30, 100, 40), (150, 80, 80, 50)])
    return PageContext(page_number=0, original_image=img_bgr.copy(), preprocessed_image=img_bgr)


# --- Test Cases ---
def test_layout_analyzer_init_default(default_layout_analyzer: LayoutAnalyzer):
    assert default_layout_analyzer.is_enabled()
    assert default_layout_analyzer.config["min_contour_area"] == 100
    assert default_layout_analyzer.config["threshold_method"] == "adaptive_gaussian"

def test_layout_analyzer_init_custom_config():
    custom_config = {
        "min_contour_area": 50,
        "aspect_ratio_range": (0.2, 8.0),
        "threshold_method": "otsu",
        "morph_kernel_size": (3,3),
        "morph_op": "dilate"
    }
    analyzer = LayoutAnalyzer(module_id="custom_layout", config=custom_config)
    assert analyzer.config["min_contour_area"] == 50
    assert analyzer.config["threshold_method"] == "otsu"
    assert analyzer.config["morph_op"] == "dilate"

def test_layout_analyzer_validate_config_errors():
    with pytest.raises(OCRXConfigurationError, match="'min_contour_area' must be a non-negative number"):
        LayoutAnalyzer(module_id="err_layout", config={"min_contour_area": -1})
    with pytest.raises(OCRXConfigurationError, match="'aspect_ratio_range' must be a tuple of two positive numbers"):
        LayoutAnalyzer(module_id="err_layout2", config={"aspect_ratio_range": (0, 10)})
    with pytest.raises(OCRXConfigurationError, match="Invalid 'threshold_method'"):
        LayoutAnalyzer(module_id="err_layout3", config={"threshold_method": "invalid"})
    with pytest.raises(OCRXConfigurationError, match="'adaptive_block_size' must be an odd integer > 1"):
        LayoutAnalyzer(module_id="err_layout4", config={"adaptive_block_size": 2})
    with pytest.raises(OCRXConfigurationError, match="Invalid 'morph_op'"):
        LayoutAnalyzer(module_id="err_layout5", config={"morph_op": "unknown"})


def test_process_disabled(default_layout_analyzer: LayoutAnalyzer, sample_page_context: PageContext):
    default_layout_analyzer.config["enabled"] = False
    image_copy = sample_page_context.preprocessed_image.copy() # type: ignore

    default_layout_analyzer.process(image_copy, sample_page_context) # type: ignore

    assert not sample_page_context.layout_regions # No regions should be added
    # Ensure image was not modified (though LayoutAnalyzer doesn't modify input image)
    assert np.array_equal(sample_page_context.preprocessed_image, image_copy)


def test_process_finds_rectangles(default_layout_analyzer: LayoutAnalyzer, tmp_path: Path):
    # Create an image with known rectangles
    rects_coords = [(20, 30, 100, 40), (150, 80, 80, 50)] # (x, y, w, h)
    img_height, img_width = 200, 300
    test_image = create_test_image_with_rectangles(img_height, img_width, rects_coords)

    # Save and reload to simulate a more realistic input that might have BGR ordering issues if not handled
    # test_img_path = tmp_path / "layout_test.png"
    # cv2.imwrite(str(test_img_path), test_image)
    # loaded_image = cv2.imread(str(test_img_path)) # This will be BGR
    # assert loaded_image is not None

    page_ctx = PageContext(page_number=0, original_image=test_image.copy(), preprocessed_image=test_image)

    # Use default config, which should find these rectangles
    default_layout_analyzer.process(page_ctx.preprocessed_image, page_ctx) # type: ignore

    assert len(page_ctx.layout_regions) == len(rects_coords)

    # Check bounding boxes (allow for minor differences if any due to contour finding)
    # Sort both expected and actual by (y,x) for consistent comparison
    expected_bboxes_x1y1x2y2 = sorted([(r[0], r[1], r[0]+r[2], r[1]+r[3]) for r in rects_coords], key=lambda b: (b[1], b[0]))
    actual_bboxes_x1y1x2y2 = sorted([tr.bounding_box for tr in page_ctx.layout_regions], key=lambda b: (b[1], b[0]))

    for actual, expected in zip(actual_bboxes_x1y1x2y2, expected_bboxes_x1y1x2y2):
        assert actual == expected, f"Expected {expected}, got {actual}"

        # Verify image_crop content (basic check: non-empty and correct shape)
        region = next(tr for tr in page_ctx.layout_regions if tr.bounding_box == actual)
        assert region.image_crop is not None
        expected_h, expected_w = expected[3] - expected[1], expected[2] - expected[0]
        assert region.image_crop.shape == (expected_h, expected_w, 3) # H, W, C (BGR)

def test_process_filters_small_contours(default_layout_analyzer: LayoutAnalyzer):
    small_rect_area = default_layout_analyzer.config["min_contour_area"] // 2
    small_rect_dim = int(np.sqrt(small_rect_area))

    rects_coords = [
        (20, 30, 100, 40),                           # Valid
        (5, 5, small_rect_dim, small_rect_dim)       # Too small
    ]
    test_image = create_test_image_with_rectangles(200, 300, rects_coords)
    page_ctx = PageContext(page_number=0, original_image=test_image.copy(), preprocessed_image=test_image)

    default_layout_analyzer.process(page_ctx.preprocessed_image, page_ctx) # type: ignore
    assert len(page_ctx.layout_regions) == 1
    assert page_ctx.layout_regions[0].bounding_box == (20, 30, 20+100, 30+40)


def test_process_filters_bad_aspect_ratio(default_layout_analyzer: LayoutAnalyzer):
    # Default AR range is (0.1, 10.0)
    rects_coords = [
        (20, 30, 100, 40),  # AR = 100/40 = 2.5 (Valid)
        (50, 80, 5, 100),   # AR = 5/100 = 0.05 (Too thin, invalid)
        (70, 100, 100, 5)   # AR = 100/5 = 20 (Too wide, invalid)
    ]
    test_image = create_test_image_with_rectangles(200, 200, rects_coords)
    page_ctx = PageContext(page_number=0, original_image=test_image.copy(), preprocessed_image=test_image)

    default_layout_analyzer.process(page_ctx.preprocessed_image, page_ctx) # type: ignore
    assert len(page_ctx.layout_regions) == 1
    assert page_ctx.layout_regions[0].bounding_box == (20, 30, 20+100, 30+40)

def test_process_empty_image(default_layout_analyzer: LayoutAnalyzer):
    empty_image = create_test_image_with_rectangles(100, 100, []) # White image
    page_ctx = PageContext(page_number=0, original_image=empty_image.copy(), preprocessed_image=empty_image)

    default_layout_analyzer.process(page_ctx.preprocessed_image, page_ctx) # type: ignore
    assert len(page_ctx.layout_regions) == 0

def test_process_with_otsu_thresholding():
    config = {"threshold_method": "otsu"}
    analyzer = LayoutAnalyzer("otsu_test", config)

    rects_coords = [(20, 30, 100, 40)]
    test_image = create_test_image_with_rectangles(200, 300, rects_coords)
    page_ctx = PageContext(page_number=0, original_image=test_image.copy(), preprocessed_image=test_image)

    analyzer.process(page_ctx.preprocessed_image, page_ctx) # type: ignore
    assert len(page_ctx.layout_regions) == 1 # Check it ran without error and found the region

def test_process_with_dilate_morph_op():
    config = {"morph_op": "dilate", "morph_kernel_size": (3,3)}
    analyzer = LayoutAnalyzer("dilate_test", config)

    rects_coords = [(20, 30, 100, 40)]
    test_image = create_test_image_with_rectangles(200, 300, rects_coords)
    page_ctx = PageContext(page_number=0, original_image=test_image.copy(), preprocessed_image=test_image)

    analyzer.process(page_ctx.preprocessed_image, page_ctx) # type: ignore
    # Exact number of regions might change with dilation, so just check it runs
    # and maybe that a region is found if the dilation isn't too aggressive.
    # For this test, primarily ensuring it runs with this config.
    assert len(page_ctx.layout_regions) >= 0 # Can be 0 or more depending on dilation effect
                                           # With a simple rect, dilation should still yield 1 region.
    if len(rects_coords) > 0 : assert len(page_ctx.layout_regions) > 0


def test_region_crop_from_original_image(default_layout_analyzer: LayoutAnalyzer):
    """Verifies that image_crop is from original_image, not preprocessed_image."""
    original_color = (0, 0, 255) # Blue
    preprocessed_color = (128, 128, 128) # Gray

    original_img = create_dummy_bgr_image(100,100, original_color)
    cv2.rectangle(original_img, (10,10), (90,90), (0,0,0), -1) # Black rectangle to find

    # Simulate preprocessed image being different (e.g. grayscale)
    preprocessed_img = create_dummy_bgr_image(100,100, preprocessed_color)
    cv2.rectangle(preprocessed_img, (10,10), (90,90), (0,0,0), -1) # Same black rectangle

    page_ctx = PageContext(page_number=0, original_image=original_img, preprocessed_image=preprocessed_img)
    default_layout_analyzer.process(page_ctx.preprocessed_image, page_ctx) # type: ignore

    assert len(page_ctx.layout_regions) == 1
    region_crop = page_ctx.layout_regions[0].image_crop
    assert region_crop is not None

    # Check a pixel from the background of the crop. It should match original_color.
    # The crop will be of the black rectangle, but its background (if any padding or if original was used)
    # would come from original_img.
    # The current crop is tight to the bounding box of the contour.
    # If the contour is the black rectangle, the crop will be all black.
    # Let's make the original rectangle blue, and the preprocessed one black.
    # The layout analyzer uses preprocessed_image to find contours.

    original_img_v2 = create_dummy_bgr_image(100,100, (255,255,255)) # White bg
    cv2.rectangle(original_img_v2, (10,10), (90,90), (255,0,0), -1) # Blue rectangle (BGR)

    preprocessed_img_v2 = create_dummy_bgr_image(100,100, (255,255,255))
    cv2.rectangle(preprocessed_img_v2, (10,10), (90,90), (0,0,0), -1) # Black rectangle for contour finding

    page_ctx_v2 = PageContext(page_number=0, original_image=original_img_v2, preprocessed_image=preprocessed_img_v2)
    default_layout_analyzer.process(page_ctx_v2.preprocessed_image, page_ctx_v2) # type: ignore

    assert len(page_ctx_v2.layout_regions) == 1
    region_crop_v2 = page_ctx_v2.layout_regions[0].image_crop
    assert region_crop_v2 is not None
    # Check the color of the cropped region (should be blue from original)
    assert np.all(region_crop_v2[0,0] == [255,0,0]) # Check top-left pixel of the crop (BGR: Blue)
