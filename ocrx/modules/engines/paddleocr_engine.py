import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ocrx.core.ocr_engine_interface import OCREngine
from ocrx.core.data_objects import RecognitionResult
from ocrx.core.exceptions import OCRXModelLoadError, OCRXProcessingError

# Try to import PaddleOCR, but don't fail at import time if not installed.
# The actual check and error will happen in __init__ or is_available.
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None # type: ignore # Define PaddleOCR as None if not available


class PaddleOCREngineWrapper(OCREngine):
    """
    Wrapper for the PaddleOCR engine.
    """
    def __init__(self, engine_config: Dict[str, Any], logger: logging.Logger):
        super().__init__(engine_config, logger)
        self.ocr_instance: Optional[PaddleOCR] = None

        if not PADDLEOCR_AVAILABLE:
            msg = "PaddleOCR library not found. Please install paddleocr and paddlepaddle."
            self.logger.error(msg)
            # Don't raise here, allow is_available to report False.
            # Or, if it's critical and configured to be used, _validate_config in a module using this
            # might raise an error. For now, the engine itself will just be unavailable.
            self._is_initialized = False
            return

        try:
            # Filter config keys relevant to PaddleOCR instantiation
            # Common params: use_gpu, lang, det_model_dir, rec_model_dir, cls_model_dir, use_angle_cls etc.
            paddle_init_params = {
                k: v for k, v in self.engine_config.items()
                if k in ["use_gpu", "gpu_mem", "gpu_id",
                         "lang", "det", "rec", "cls", # More specific model paths
                         "det_model_dir", "rec_model_dir", "cls_model_dir", "ocr_version",
                         "use_angle_cls", "show_log", "use_mp", "total_process_num",
                         "max_batch_size", "enable_mkldnn", "use_tensorrt", "precision"]
            }
            # Ensure 'lang' is present, defaulting to 'en' if not specified.
            if 'lang' not in paddle_init_params:
                paddle_init_params['lang'] = self.engine_config.get('lang', 'en')

            self.logger.info(f"Initializing PaddleOCR with parameters: {paddle_init_params}")
            self.ocr_instance = PaddleOCR(**paddle_init_params)
            self._is_initialized = True
            self.logger.info(f"PaddleOCR engine '{self.get_engine_name()}' initialized successfully.")

        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR instance: {e}", exc_info=True)
            self._is_initialized = False
            # We don't re-raise here; is_available() will be false.
            # If this engine was mandatory, the orchestrator should handle it.

    def initialize(self) -> None:
        """
        Confirms initialization status. Actual initialization is done in __init__.
        """
        if self._is_initialized:
            self.logger.info(f"PaddleOCR engine '{self.get_engine_name()}' is already initialized.")
        else:
            self.logger.warning(f"PaddleOCR engine '{self.get_engine_name()}' failed to initialize or library not available.")
            # Attempt re-initialization if library is available but instance is not.
            if PADDLEOCR_AVAILABLE and self.ocr_instance is None:
                self.logger.info("Attempting re-initialization of PaddleOCR in initialize().")
                # This might be redundant if __init__ already tried and failed due to config.
                # However, if __init__ skipped due to PADDLEOCR_AVAILABLE=False then this is also skipped.
                # Consider if this re-init logic is truly needed or if __init__ failure is final.
                try:
                    paddle_init_params = { k: v for k, v in self.engine_config.items() if k in ["use_gpu", "lang", "use_angle_cls"] }
                    if 'lang' not in paddle_init_params: paddle_init_params['lang'] = 'en'
                    self.ocr_instance = PaddleOCR(**paddle_init_params)
                    self._is_initialized = True
                    self.logger.info(f"PaddleOCR re-initialized successfully during initialize().")
                except Exception as e:
                    self.logger.error(f"Re-initialization of PaddleOCR failed: {e}", exc_info=True)
                    self._is_initialized = False
                    # raise OCRXModelLoadError(f"PaddleOCR re-initialization failed: {e}") from e
            elif not PADDLEOCR_AVAILABLE:
                 raise OCRXModelLoadError("PaddleOCR library is not installed. Cannot initialize engine.")


    def recognize(self, image_region: np.ndarray, language_hint: Optional[str] = None) -> List[RecognitionResult]:
        """
        Performs OCR on the given image region using PaddleOCR.

        Args:
            image_region: The image region (BGR NumPy array) to process.
            language_hint: Optional language hint (currently ignored by this PaddleOCR wrapper as
                           language is set at initialization).

        Returns:
            A list of RecognitionResult objects, one for each detected line/block by PaddleOCR.

        Raises:
            OCRXProcessingError: If the engine is not available or OCR fails.
        """
        if not self.is_available() or self.ocr_instance is None:
            raise OCRXProcessingError(f"PaddleOCR engine '{self.get_engine_name()}' is not available.")

        self.logger.debug(f"Performing OCR on image region (shape: {image_region.shape}) with {self.get_engine_name()}. Language hint (ignored): {language_hint}")

        try:
            # PaddleOCR expects BGR numpy array.
            # The result is a list of lists, e.g., [[[box_coords], (text, confidence)], ...]
            # For text detection and recognition (ocr method):
            # result = [[[[[point1_x, point1_y], [p2_x,p2_y], [p3_x,p3_y], [p4_x,p4_y]]], ('text string', confidence_score)], ...]
            # For text recognition only (rec method, if passing cropped lines):
            # result = [('text string', confidence_score), ...]

            # Assuming `image_region` is a crop of a single text region that might contain multiple lines.
            # PaddleOCR's `ocr` method handles both detection and recognition within this region.
            ocr_results_raw = self.ocr_instance.ocr(image_region, cls=self.engine_config.get('use_angle_cls', True))

            processed_results: List[RecognitionResult] = []
            if ocr_results_raw:
                # PaddleOCR often returns a list containing one sublist of results for the input image.
                # [[[[box1_coords], (text1, conf1)], [[box2_coords], (text2, conf2)]]]
                # We need to handle this potential nesting.
                # If ocr_results_raw = [None], it means no text detected.
                if ocr_results_raw[0] is None and len(ocr_results_raw) == 1:
                    self.logger.info(f"No text detected by PaddleOCR in the given region.")
                    return []

                actual_lines = ocr_results_raw
                if isinstance(ocr_results_raw, list) and len(ocr_results_raw) == 1 and isinstance(ocr_results_raw[0], list):
                    actual_lines = ocr_results_raw[0]

                for line_data in actual_lines:
                    if not line_data: continue # Should not happen based on typical output

                    # line_data typically: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                    box_coords_quad = line_data[0] # This is a list of 4 points (quadrilateral)
                    text, confidence = line_data[1]

                    # Convert quadrilateral to a simple bounding box (min_x, min_y, max_x, max_y)
                    # This is an approximation if the quad is heavily rotated/distorted.
                    np_box_coords = np.array(box_coords_quad).astype(int)
                    min_x = np.min(np_box_coords[:, 0])
                    min_y = np.min(np_box_coords[:, 1])
                    max_x = np.max(np_box_coords[:, 0])
                    max_y = np.max(np_box_coords[:, 1])

                    # Bounding box relative to the image_region crop
                    char_box: Tuple[int,int,int,int] = (min_x, min_y, max_x, max_y)

                    recognition_obj = RecognitionResult(
                        text=str(text),
                        confidence=float(confidence),
                        char_boxes=[char_box], # Storing the line box as the primary char_box for now
                        engine_id=self.get_engine_name()
                        # PaddleOCR does not directly provide per-character confidences or detailed word boxes
                        # in its standard output format without further processing or custom models.
                    )
                    processed_results.append(recognition_obj)

            self.logger.info(f"PaddleOCR processed region, found {len(processed_results)} text lines.")
            return processed_results

        except Exception as e:
            self.logger.error(f"Error during PaddleOCR recognition: {e}", exc_info=True)
            raise OCRXProcessingError(f"PaddleOCR recognition failed: {e}") from e


    def get_engine_name(self) -> str:
        version = self.engine_config.get("ocr_version", "UnknownPaddleVersion")
        # Try to get actual PaddleOCR version if available, though paddleocr lib doesn't expose it easily.
        # For now, use config or a fixed string.
        return f"PaddleOCR_Engine_{version}"

    def is_available(self) -> bool:
        """Checks if the PaddleOCR instance is initialized and library is available."""
        return PADDLEOCR_AVAILABLE and self.ocr_instance is not None and self._is_initialized
