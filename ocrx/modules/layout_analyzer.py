import logging
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import cv2

from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.data_objects import PageContext, TextRegion
from ocrx.core.exceptions import OCRXConfigurationError, OCRXProcessingError

class LayoutAnalyzer(OCRXModuleBase):
    """
    Performs layout analysis on a page image to identify text regions.
    """

    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)

    def _validate_config(self) -> None:
        super()._validate_config()
        self.config.setdefault("min_contour_area", 100)
        self.config.setdefault("aspect_ratio_range", (0.1, 10.0)) # min_ar, max_ar
        self.config.setdefault("threshold_method", "adaptive_gaussian") # "adaptive_gaussian", "otsu"
        self.config.setdefault("adaptive_block_size", 11)
        self.config.setdefault("adaptive_c_value", 2)
        self.config.setdefault("morph_kernel_size", (5,5)) # For closing/dilation
        self.config.setdefault("morph_op", "close") # "close", "dilate", "erode", "open"

        if not isinstance(self.config["min_contour_area"], (int, float)) or self.config["min_contour_area"] < 0:
            raise OCRXConfigurationError(f"{self.module_id}: 'min_contour_area' must be a non-negative number.")
        if not (isinstance(self.config["aspect_ratio_range"], tuple) and len(self.config["aspect_ratio_range"]) == 2 and
                all(isinstance(x, (int, float)) and x > 0 for x in self.config["aspect_ratio_range"])):
            raise OCRXConfigurationError(f"{self.module_id}: 'aspect_ratio_range' must be a tuple of two positive numbers.")
        if self.config["threshold_method"] not in ["adaptive_gaussian", "otsu"]:
            raise OCRXConfigurationError(f"{self.module_id}: Invalid 'threshold_method'. Must be 'adaptive_gaussian' or 'otsu'.")
        if not (isinstance(self.config["adaptive_block_size"], int) and self.config["adaptive_block_size"] > 1 and self.config["adaptive_block_size"] % 2 == 1):
            raise OCRXConfigurationError(f"{self.module_id}: 'adaptive_block_size' must be an odd integer > 1.")
        if self.config["morph_op"] not in ["close", "dilate", "erode", "open"]:
            raise OCRXConfigurationError(f"{self.module_id}: Invalid 'morph_op'.")

        self.logger.info(f"{self.module_id} validated config: {self.config}")


    def process(self, image_to_process: np.ndarray, page_ctx: PageContext, config_override: Optional[Dict] = None) -> None:
        """
        Analyzes the layout of the preprocessed page image and populates page_ctx.layout_regions.

        Args:
            image_to_process: The preprocessed image (BGR NumPy array) for layout analysis.
            page_ctx: The PageContext object to populate with TextRegion objects.
            config_override: Optional dictionary to override module configurations for this call.
        """
        if not self.is_enabled():
            self.logger.info(f"Module {self.module_id} is disabled. Skipping layout analysis.")
            return

        if not isinstance(image_to_process, np.ndarray):
            raise OCRXProcessingError(f"{self.module_id} received invalid image data type: {type(image_to_process)}")

        current_config = {**self.config, **(config_override or {})}
        self.logger.info(f"Starting layout analysis for page {page_ctx.page_number} using {self.module_id}")

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)

        # 2. Thresholding
        thresh_method = current_config["threshold_method"]
        if thresh_method == "adaptive_gaussian":
            block_size = current_config["adaptive_block_size"]
            c_value = current_config["adaptive_c_value"]
            binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, block_size, c_value)
            self.logger.debug(f"Applied adaptive Gaussian thresholding (block: {block_size}, C: {c_value}).")
        elif thresh_method == "otsu":
            _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.logger.debug("Applied Otsu thresholding.")
        else: # Should have been caught by _validate_config
            raise OCRXConfigurationError(f"Unsupported threshold method: {thresh_method}")

        # 3. Morphological Operations
        kernel_size = tuple(current_config["morph_kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        morph_op_str = current_config["morph_op"].lower()

        if morph_op_str == "close":
            morphed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        elif morph_op_str == "dilate":
            morphed_image = cv2.dilate(binary_image, kernel, iterations=1)
        elif morph_op_str == "erode":
             morphed_image = cv2.erode(binary_image, kernel, iterations=1)
        elif morph_op_str == "open":
             morphed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        else: # Should have been caught by _validate_config
            raise OCRXConfigurationError(f"Unsupported morphological operation: {morph_op_str}")
        self.logger.debug(f"Applied morphological '{morph_op_str}' with kernel {kernel_size}.")

        # 4. Find Contours
        contours, hierarchy = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.debug(f"Found {len(contours)} initial contours.")

        # 5. Filter and Sort Contours
        valid_regions_data: List[Tuple[int, int, int, int, np.ndarray]] = [] # Store (y, x, w, h, crop) for sorting
        min_area = current_config["min_contour_area"]
        min_ar, max_ar = current_config["aspect_ratio_range"]

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h # Using bounding box area, cv2.contourArea(contour) is also an option

            if area < min_area:
                self.logger.log(logging.NOTSET, f"Contour {i} too small (area: {area}). Skipping.") # NOTSET often not shown
                continue

            aspect_ratio = float(w) / h if h > 0 else 0
            if not (min_ar <= aspect_ratio <= max_ar):
                self.logger.log(logging.NOTSET, f"Contour {i} fails aspect ratio check (AR: {aspect_ratio:.2f}). Skipping.")
                continue

            # Crop from original color page image (if available) or the input image_to_process
            source_for_crop = page_ctx.original_image if page_ctx.original_image is not None else image_to_process
            if source_for_crop is None:
                 self.logger.warning(f"Cannot crop region {i} as no source image (original or preprocessed) is available.")
                 continue

            # Ensure crop coordinates are within image bounds
            crop_x_end = min(x + w, source_for_crop.shape[1])
            crop_y_end = min(y + h, source_for_crop.shape[0])
            crop_x_start = max(0, x)
            crop_y_start = max(0, y)

            if crop_y_end <= crop_y_start or crop_x_end <= crop_x_start:
                self.logger.log(logging.NOTSET, f"Contour {i} crop invalid due to coordinates. Skipping.")
                continue

            region_crop = source_for_crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            if region_crop.size == 0:
                self.logger.log(logging.NOTSET, f"Contour {i} crop resulted in empty image. Skipping.")
                continue

            valid_regions_data.append(((x, y, w, h), region_crop)) # Store tuple (bounding_box, crop)

        # Sort regions: primarily top-to-bottom (y), then left-to-right (x)
        # This is a simple heuristic for reading order. More advanced methods exist.
        valid_regions_data.sort(key=lambda item: (item[0][1], item[0][0])) # Sort by y, then x

        # 6. Create TextRegion Objects
        page_ctx.layout_regions = [] # Clear any previous regions
        for i, (bbox, crop) in enumerate(valid_regions_data):
            region_id = f"page{page_ctx.page_number}_reg{i}"
            text_region = TextRegion(
                region_id=region_id,
                image_crop=crop,
                bounding_box=bbox, # type: ignore # bbox is (x,y,w,h), TextRegion expects (x1,y1,x2,y2)
                                           # This needs to be fixed: (x, y, x+w, y+h)
                sequence_id=i
            )
            # Correcting bounding_box format:
            x, y, w, h = bbox
            text_region.bounding_box = (x, y, x + w, y + h)

            page_ctx.layout_regions.append(text_region)
            self.logger.debug(f"Created TextRegion: {region_id} with bbox {text_region.bounding_box}")

        self.logger.info(f"Layout analysis completed for page {page_ctx.page_number}. Found {len(page_ctx.layout_regions)} valid text regions.")
