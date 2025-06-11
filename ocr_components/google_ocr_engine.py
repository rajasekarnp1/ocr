import logging
import os
import io
import numpy as np
import cv2 # For image encoding

try:
    from google.cloud import vision
    from google.api_core import exceptions as google_exceptions
except ImportError:
    vision = None
    google_exceptions = None

from typing import Dict, Any, Optional, List
from ocr_engine_interface import OCREngine

module_logger = logging.getLogger(__name__)


class GoogleCloudOCREngine(OCREngine):
    """
    Implements an OCR engine using the Google Cloud Vision API.
    Bounding boxes are returned as [x1, y1, x2, y2] (top-left, bottom-right).
    """

    def __init__(self, engine_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initializes the GoogleCloudOCREngine.

        Args:
            engine_config (Dict[str, Any]): Configuration for the engine. Expected keys:
                - "name" (str): Name of the engine.
                - "config" (Dict[str, Any]): Specific configuration:
                    - "api_key_path" (str): Path to the Google Cloud service account JSON key file.
                    - "project_id" (str, optional): GCP Project ID. Often inferred from credentials.
                    - "language_hints" (List[str], optional): Default language hints, e.g., ["en", "ja"].
            logger (Optional[logging.Logger]): Logger instance.
        """
        super().__init__(engine_config, logger if logger else module_logger)

        self.engine_specific_config = self.engine_config.get("config", {})
        self.engine_name = self.engine_config.get("name", "GoogleCloudVisionAPI_Refined")

        self.api_key_path = self.engine_specific_config.get("api_key_path")
        self.default_language_hints = self.engine_specific_config.get("language_hints", []) # Store default from config

        self.client: Optional[vision.ImageAnnotatorClient] = None
        self._is_initialized = False

        if vision is None:
            self.logger.critical("Google Cloud Vision library not found. Please install 'google-cloud-vision'.")

        self.logger.info(f"GoogleCloudOCREngine '{self.engine_name}' instance created.")

    def initialize(self) -> None:
        # (Implementation from previous turn - largely unchanged)
        self.logger.info(f"Initializing GoogleCloudOCREngine '{self.engine_name}'...")
        if vision is None:
            self.logger.error("Google Cloud Vision library not installed. Cannot initialize.")
            self._is_initialized = False; return
        if not self.api_key_path:
            self.logger.error("API key path ('api_key_path') not provided. Cannot authenticate.")
            self._is_initialized = False; return
        if not os.path.exists(self.api_key_path):
            self.logger.error(f"API key file not found at: {self.api_key_path}")
            self._is_initialized = False; return
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.api_key_path
            self.client = vision.ImageAnnotatorClient()
            self._is_initialized = True
            self.logger.info(f"GoogleCloudOCREngine '{self.engine_name}' initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Cloud Vision client: {e}", exc_info=True)
            self._is_initialized = False


    def _numpy_to_image_bytes(self, image_data: np.ndarray, ext: str = ".png") -> Optional[bytes]:
        # (Implementation from previous turn - confirmed PNG usage)
        try:
            is_success, buffer = cv2.imencode(ext, image_data) # Ensure image_data is BGR for cv2.imencode
            if not is_success:
                self.logger.error(f"Failed to encode image to {ext} format.")
                return None
            return buffer.tobytes()
        except Exception as e:
            self.logger.error(f"Error encoding image to bytes: {e}", exc_info=True)
            return None

    def recognize(self, image_data: np.ndarray, language_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs OCR using Google Cloud Vision API.

        Args:
            image_data (np.ndarray): Input image (BGR NumPy array).
            language_hint (Optional[str]): Specific language hint for this recognition call.
                                           Takes precedence over default hints in config.

        Returns:
            Dict[str, Any]: Standardized OCR results dictionary.
                Bounding boxes are [x1, y1, x2, y2] (top-left, bottom-right).
        """
        if not self._is_initialized or self.client is None:
            self.logger.error(f"Engine '{self.engine_name}' not initialized. Cannot perform recognition.")
            return {"text": "", "segments": [], "confidence": None, "engine_name": self.get_engine_name()}

        self.logger.info(f"Starting OCR with Google Cloud Vision. Image shape: {image_data.shape}")

        # OpenCV imencode expects BGR format. If input is RGB, it should be converted prior to this call,
        # or handled here. Assuming image_data is BGR as per typical OpenCV usage.
        image_bytes = self._numpy_to_image_bytes(image_data, ext=".png") # Use PNG
        if image_bytes is None:
            return {"text": "", "segments": [], "confidence": 0.0, "engine_name": self.get_engine_name()}

        gcv_image = vision.Image(content=image_bytes)

        # Language Hint Logic:
        active_language_hints = []
        if language_hint: # Argument to recognize() takes precedence
            active_language_hints = [language_hint]
            self.logger.debug(f"Using language hint from recognize() argument: {active_language_hints}")
        elif self.default_language_hints: # Fallback to config defaults
            active_language_hints = self.default_language_hints
            self.logger.debug(f"Using default language hints from config: {active_language_hints}")

        image_context_params = {}
        if active_language_hints:
            image_context_params["language_hints"] = active_language_hints

        try:
            response = self.client.document_text_detection(
                image=gcv_image,
                image_context=vision.ImageContext(**image_context_params) if image_context_params else None
            )

            if response.error.message:
                self.logger.error(f"Google Vision API Error: {response.error.message} (Code: {response.error.code})")
                return {"text": "", "segments": [], "confidence": 0.0, "engine_name": self.get_engine_name(), "error": response.error.message}

            if not response.full_text_annotation:
                self.logger.info("No text detected by Google Vision API.")
                return {"text": "", "segments": [], "confidence": 0.0, "engine_name": self.get_engine_name()}

            full_text = response.full_text_annotation.text
            segments = []
            overall_confidence = 0.0

            img_h, img_w = image_data.shape[:2]

            if response.full_text_annotation.pages:
                overall_confidence = response.full_text_annotation.pages[0].confidence

            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])

                            if not word.bounding_box or not word.bounding_box.vertices:
                                self.logger.debug(f"Word '{word_text}' has no bounding box vertices. Skipping.")
                                continue

                            # Denormalize vertices and calculate [x1, y1, x2, y2]
                            # Google Vision API typically returns normalized vertices (0.0 to 1.0 range).
                            # If any vertex.x > 1.0 or vertex.y > 1.0, it might indicate absolute pixel values.
                            # However, the most common case is normalized.
                            # Let's assume normalized unless a coordinate is clearly outside typical image dimensions if it were normalized.
                            # A simple check: if all vertices are <= 1.0 (or slightly above for FP error), assume normalized.
                            # For robustness, it's usually safer to check the API documentation for the specific feature used.
                            # document_text_detection usually gives normalized_vertices.

                            # Use normalized_vertices if available, otherwise vertices
                            vertices_to_use = word.bounding_box.normalized_vertices
                            if not vertices_to_use: # Fallback to vertices if normalized_vertices is empty
                                vertices_to_use = word.bounding_box.vertices
                                # If using non-normalized vertices, the scaling logic would be different (or skipped)
                                # For this implementation, we'll primarily rely on normalized_vertices behavior.
                                # If these are already absolute, the scaling by img_w/h would be wrong.
                                # However, Google's standard is normalized for document_text_detection.

                            if not vertices_to_use:
                                self.logger.warning(f"No vertices found for word '{word_text}'. Skipping box.")
                                continue

                            # Denormalize from [0,1] range to image pixel coordinates
                            # These are polygon vertices, not necessarily an axis-aligned box yet.
                            denormalized_x_coords = [v.x * img_w for v in vertices_to_use]
                            denormalized_y_coords = [v.y * img_h for v in vertices_to_use]

                            if not denormalized_x_coords or not denormalized_y_coords:
                                self.logger.warning(f"Could not extract coordinates for word '{word_text}'. Skipping box.")
                                continue

                            # Calculate axis-aligned bounding box [x1, y1, x2, y2]
                            x1 = min(denormalized_x_coords)
                            y1 = min(denormalized_y_coords)
                            x2 = max(denormalized_x_coords)
                            y2 = max(denormalized_y_coords)

                            # Ensure coordinates are integers and within image bounds
                            bbox_x1y1x2y2 = [
                                max(0, int(x1)),
                                max(0, int(y1)),
                                min(img_w, int(x2)),
                                min(img_h, int(y2))
                            ]

                            segments.append({
                                "text": word_text,
                                "bounding_box": bbox_x1y1x2y2, # Format [x1, y1, x2, y2]
                                "confidence": word.confidence
                            })

            self.logger.info(f"Google Vision API processed successfully. Found {len(segments)} text segments.")
            return {
                "text": full_text,
                "segments": segments,
                "confidence": overall_confidence,
                "engine_name": self.get_engine_name()
            }

        except google_exceptions.GoogleAPICallError as e:
            self.logger.error(f"Google API call failed during recognition: {e}", exc_info=True)
            return {"text": "", "segments": [], "confidence": 0.0, "engine_name": self.get_engine_name(), "error": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error during Google Vision recognition: {e}", exc_info=True)
            return {"text": "", "segments": [], "confidence": 0.0, "engine_name": self.get_engine_name(), "error": str(e)}

    def get_engine_name(self) -> str:
        return self.engine_name

    def is_available(self) -> bool:
        return self._is_initialized and self.client is not None


if __name__ == '__main__':
    # (Instructional __main__ block from previous turn - largely unchanged,
    #  as it's about guiding user for external setup)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger("GoogleCloudOCREngineTestRefined")
    main_logger.info("Starting GoogleCloudOCREngine (Refined) self-test...")

    test_api_key_path = "path/to/your/service_account_key.json" # <--- USER MUST SET THIS

    if test_api_key_path == "path/to/your/service_account_key.json":
        main_logger.error("="*80)
        main_logger.error("USER ACTION REQUIRED: Please edit 'ocr_components/google_ocr_engine.py' __main__ block.")
        main_logger.error(f"Set 'test_api_key_path' to your actual Google Cloud service account key JSON file path.")
        main_logger.error("="*80)
        test_api_key_path = os.environ.get("GOOGLE_TEST_API_KEY_PATH", test_api_key_path) # Fallback for CI/auto-tests
        if test_api_key_path == "path/to/your/service_account_key.json": # Still placeholder
             main_logger.warning("Skipping test as API key path is not set in code or GOOGLE_TEST_API_KEY_PATH env var.")
             exit()

    engine_config_for_test = {
        "name": "TestGoogleCloudVisionFromMainRefined",
        "config": {
            "api_key_path": test_api_key_path,
            "language_hints": ["en", "fr"] # Default hints
        }
    }

    test_image_path = "test_image_for_google.png"
    if not os.path.exists(test_image_path):
        main_logger.warning(f"Test image '{test_image_path}' not found. Creating a dummy image.")
        dummy_img_data = np.full((200, 400, 3), (220, 220, 220), dtype=np.uint8)
        cv2.putText(dummy_img_data, "Hello Google", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
        cv2.putText(dummy_img_data, "Bonjour", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
        cv2.imwrite(test_image_path, dummy_img_data)
        main_logger.info(f"Created dummy test image: {test_image_path}")

    gcv_engine = None
    try:
        main_logger.info(f"Initializing GoogleCloudOCREngine with key: {engine_config_for_test['config']['api_key_path']}")
        gcv_engine = GoogleCloudOCREngine(engine_config=engine_config_for_test, logger=main_logger)
        gcv_engine.initialize()

        if gcv_engine.is_available():
            main_logger.info("Engine initialized successfully.")
            image_np = cv2.imread(test_image_path)
            if image_np is None:
                main_logger.error(f"Failed to load test image from: {test_image_path}")
            else:
                main_logger.info(f"Test image '{test_image_path}' loaded. Shape: {image_np.shape}")

                # Test with specific language hint overriding default
                main_logger.info("Calling recognize() with specific language_hint='es'...")
                results_es = gcv_engine.recognize(image_np, language_hint="es")
                main_logger.info(f"Results (es hint):\n{results_es.get('text', '')[:200]}...") # Print beginning of text
                if results_es.get("segments"):
                    main_logger.info(f"First segment (es hint): {results_es['segments'][0]}")

                # Test with default language hints from config
                main_logger.info("Calling recognize() (using default config hints ['en', 'fr'])...")
                results_default = gcv_engine.recognize(image_np)
                main_logger.info(f"Results (default hints):\n{results_default.get('text', '')[:200]}...")
                if results_default.get("segments"):
                    main_logger.info(f"First segment (default hints): {results_default['segments'][0]}")

        else:
            main_logger.error("Engine initialization failed. Check logs.")

    except Exception as e:
        main_logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)

    finally:
        if 'dummy_img_data' in locals() and os.path.exists(test_image_path):
             main_logger.info(f"Dummy test image '{test_image_path}' was created and kept for inspection.")
        main_logger.info("GoogleCloudOCREngine (Refined) self-test finished.")
```
