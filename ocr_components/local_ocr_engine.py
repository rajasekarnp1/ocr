import logging
import os
import numpy as np
import cv2 # OpenCV for image manipulation
from typing import Dict, Any, Optional, List, Tuple

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

try:
    import pyclipper
except ImportError:
    pyclipper = None

module_logger = logging.getLogger(__name__)


class LocalOCREngine(OCREngine):
    """
    Implements an OCR engine that runs locally using ONNX models,
    with specific logic for PaddleOCR-style detection (DBNet) and recognition (CRNN).
    Refined for robustness and MVP completeness.
    """

    def __init__(self, engine_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(engine_config, logger if logger else module_logger)

        self.engine_specific_config = self.engine_config.get("config", {})
        self.engine_name = self.engine_config.get("name", "LocalPaddleOCREngine_Refined")

        self.use_gpu_directml = self.engine_specific_config.get("use_gpu_directml", False)
        self.detection_model_path = self.engine_specific_config.get("detection_model_path")
        self.recognition_model_path = self.engine_specific_config.get("recognition_model_path")
        self.character_dict_path = self.engine_specific_config.get("character_dict_path")

        self.det_input_size = np.array(self.engine_specific_config.get("det_input_size", [736, 736]))
        self.det_mean = np.array(self.engine_specific_config.get("det_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.det_std = np.array(self.engine_specific_config.get("det_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.det_db_thresh = self.engine_specific_config.get("det_db_thresh", 0.3)
        self.det_db_box_thresh = self.engine_specific_config.get("det_db_box_thresh", 0.6)
        self.det_db_unclip_ratio = self.engine_specific_config.get("det_db_unclip_ratio", 1.5)
        self.det_max_candidates = self.engine_specific_config.get("det_max_candidates", 1000)
        self.det_min_box_size = self.engine_specific_config.get("det_min_box_size", 3)

        self.rec_image_shape = self.engine_specific_config.get("rec_image_shape", [1, 48, 320]) # C, H, W
        self.rec_batch_size = self.engine_specific_config.get("rec_batch_size", 6)
        self.rec_norm_mean = self.engine_specific_config.get("rec_norm_mean", 0.5) # For (img/255.0 - mean) / std
        self.rec_norm_std = self.engine_specific_config.get("rec_norm_std", 0.5)

        self.detection_session: Optional[onnxruntime.InferenceSession] = None
        self.recognition_session: Optional[onnxruntime.InferenceSession] = None
        self.character_dict: Optional[List[str]] = None
        self.ctc_blank_idx: Optional[int] = None # To be determined from char_dict or model properties

        self._is_initialized = False
        self.logger.info(f"LocalOCREngine '{self.engine_name}' instance created.")
        if not pyclipper:
            self.logger.warning("Pyclipper library not found. DBNet unclip functionality will use simpler box expansion.")
        if not onnxruntime:
            self.logger.critical("ONNX Runtime library not found. Engine cannot function.")


    def initialize(self) -> None:
        self.logger.info(f"Initializing LocalOCREngine '{self.engine_name}'...")
        if not onnxruntime:
            self.logger.error("ONNX Runtime not available. Initialization failed.")
            self._is_initialized = False
            return

        # 1. Load Character Dictionary (Critical)
        if not self.character_dict_path or not os.path.exists(self.character_dict_path):
            self.logger.critical(f"Character dictionary path '{self.character_dict_path}' is invalid or file does not exist. Engine cannot initialize.")
            self._is_initialized = False; return

        try:
            self.character_dict = []
            with open(self.character_dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.character_dict.append(line.strip())
            if not self.character_dict:
                self.logger.critical(f"Character dictionary loaded from '{self.character_dict_path}' is empty. Engine cannot initialize.")
                self._is_initialized = False; return

            # Determine CTC blank index. Often it's the last class if model outputs num_chars+1 classes.
            # Or it could be a specific character like '-' or 'blank' if included in the dict.
            # For typical PaddleOCR models, the dictionary defines characters, and the model output layer has num_chars + 1 (for blank) units.
            # The blank index is typically `len(self.character_dict)`.
            self.ctc_blank_idx = len(self.character_dict)
            self.logger.info(f"Loaded character dictionary: {len(self.character_dict)} chars. CTC blank index assumed to be {self.ctc_blank_idx}.")

        except Exception as e:
            self.logger.critical(f"Failed to load or parse character dictionary from '{self.character_dict_path}': {e}", exc_info=True)
            self._is_initialized = False; return

        # 2. Setup ONNX Execution Providers
        available_providers = onnxruntime.get_available_providers()
        self.logger.debug(f"Available ONNX Execution Providers: {available_providers}")

        providers = []
        if self.use_gpu_directml:
            if 'DmlExecutionProvider' in available_providers:
                providers.append('DmlExecutionProvider')
                self.logger.info("Using DmlExecutionProvider (DirectML).")
            else:
                self.logger.warning("Configured to use DirectML, but DmlExecutionProvider is not available. Falling back to CPU.")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')
            self.logger.info("Using CPUExecutionProvider.")

        if 'CPUExecutionProvider' not in providers: # Ensure CPU is always an option as fallback
             providers.append('CPUExecutionProvider')


        # 3. Create ONNX Inference Sessions
        try:
            if not self.detection_model_path or not os.path.exists(self.detection_model_path):
                self.logger.error(f"Detection model path invalid: {self.detection_model_path}"); raise FileNotFoundError("Detection model file not found.")
            self.detection_session = onnxruntime.InferenceSession(self.detection_model_path, providers=providers)
            self.logger.info(f"Detection ONNX session created with providers: {self.detection_session.get_providers()}")
        except Exception as e:
            self.logger.error(f"Failed to create ONNX session for detection model '{self.detection_model_path}': {e}", exc_info=True)
            self._is_initialized = False

        try:
            if not self.recognition_model_path or not os.path.exists(self.recognition_model_path):
                self.logger.error(f"Recognition model path invalid: {self.recognition_model_path}"); raise FileNotFoundError("Recognition model file not found.")
            self.recognition_session = onnxruntime.InferenceSession(self.recognition_model_path, providers=providers)
            self.logger.info(f"Recognition ONNX session created with providers: {self.recognition_session.get_providers()}")
        except Exception as e:
            self.logger.error(f"Failed to create ONNX session for recognition model '{self.recognition_model_path}': {e}", exc_info=True)
            self._is_initialized = False

        if self.detection_session and self.recognition_session: # Char dict already checked
             self._is_initialized = True
             self.logger.info(f"LocalOCREngine '{self.engine_name}' initialized successfully.")
        else:
             # Ensure it's marked false if any critical part failed post char_dict check
             self._is_initialized = False
             self.logger.error(f"LocalOCREngine '{self.engine_name}' failed initialization due to ONNX session creation issues.")


    def _preprocess_image_for_detection(self, image_data: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
        # (Implementation from previous turn, assuming it's largely okay)
        # Minor check: ensure self.det_input_size is used correctly
        self.logger.debug(f"Preprocessing image for detection. Original shape: {image_data.shape}")
        try:
            original_h, original_w = image_data.shape[:2]
            # self.det_input_size should be [H, W]
            target_h, target_w = int(self.det_input_size[0]), int(self.det_input_size[1])


            ratio_h = target_h / original_h
            ratio_w = target_w / original_w
            resize_ratio = min(ratio_h, ratio_w)

            new_h = int(original_h * resize_ratio)
            new_w = int(original_w * resize_ratio)

            img_resized = cv2.resize(image_data, (new_w, new_h))

            pad_h = target_h - new_h
            pad_w = target_w - new_w

            img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w,
                                            cv2.BORDER_CONSTANT, value=[0,0,0])

            img_norm = (img_padded.astype(np.float32) / 255.0 - self.det_mean) / self.det_std
            img_chw = np.transpose(img_norm, (2, 0, 1))
            img_tensor = np.expand_dims(img_chw, axis=0)

            self.logger.debug(f"Detection preprocessed tensor shape: {img_tensor.shape}. Resize ratio used: {resize_ratio}")
            return img_tensor, resize_ratio, resize_ratio

        except Exception as e:
            self.logger.error(f"Error in _preprocess_image_for_detection: {e}", exc_info=True)
            return None, 1.0, 1.0

    def _run_detection(self, processed_image_tensor: np.ndarray) -> Optional[np.ndarray]:
        # (Implementation from previous turn)
        if not self.detection_session: self.logger.error("Det session not init."); return None
        self.logger.debug("Running text detection model...")
        try:
            input_name = self.detection_session.get_inputs()[0].name
            raw_output = self.detection_session.run(None, {input_name: processed_image_tensor})
            return raw_output[0]
        except Exception as e:
            self.logger.error(f"Error during _run_detection: {e}", exc_info=True)
            return None

    def _postprocess_detection(self, raw_detection_output: np.ndarray,
                               resize_ratio_h: float, resize_ratio_w: float,
                               original_shape: Tuple[int, int]) -> List[np.ndarray]:
        self.logger.debug("Postprocessing detection output...")
        if raw_detection_output is None: return []

        segmentation_map = np.squeeze(raw_detection_output)
        if segmentation_map.ndim == 3: segmentation_map = segmentation_map[0]

        binary_map = (segmentation_map > self.det_db_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        detected_boxes = []
        original_h, original_w = original_shape

        for i, contour in enumerate(contours[:self.det_max_candidates]):
            if len(contour) < 3: continue

            try:
                min_rect = cv2.minAreaRect(contour)
                points = cv2.boxPoints(min_rect)
            except Exception as e:
                self.logger.warning(f"Failed to get minAreaRect/boxPoints for contour {i}: {e}. Skipping.")
                continue

            mask = np.zeros(binary_map.shape, dtype=np.uint8) # Mask for score calculation
            cv2.drawContours(mask, [contour], -1, color=1, thickness=cv2.FILLED)
            box_score = np.sum(segmentation_map * mask) / (np.sum(mask) + 1e-6) # Add epsilon for stability

            if box_score < self.det_db_box_thresh: continue

            if pyclipper:
                try:
                    poly = points.reshape(-1, 2)
                    pco = pyclipper.PyclipperOffset()
                    distance = cv2.arcLength(poly.astype(np.int32), True) * self.det_db_unclip_ratio * 0.5 # Half of perimeter * ratio
                    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    expanded_polys = pco.Execute(distance)
                    if expanded_polys and len(expanded_polys) > 0:
                        expanded_poly_points = np.array(expanded_polys[0]).reshape(-1, 2)
                        # Optional: simplify if too many points, e.g., back to minAreaRect
                        if len(expanded_poly_points) > 4:
                             points = cv2.boxPoints(cv2.minAreaRect(expanded_poly_points))
                        else:
                             points = expanded_poly_points
                except Exception as e:
                    self.logger.warning(f"Pyclipper unclip failed for contour {i}: {e}. Using original box or simple expansion.")
                    # Fallback to simple expansion if pyclipper fails
                    center, (w_r, h_r), angle = min_rect
                    points = cv2.boxPoints(((center), (w_r * self.det_db_unclip_ratio, h_r * self.det_db_unclip_ratio), angle))
            else: # Simple expansion if no pyclipper
                center, (w_r, h_r), angle = min_rect
                points = cv2.boxPoints(((center), (w_r * self.det_db_unclip_ratio, h_r * self.det_db_unclip_ratio), angle))

            scaled_points = points / np.array([resize_ratio_w, resize_ratio_h])
            scaled_points[:, 0] = np.clip(scaled_points[:, 0], 0, original_w -1)
            scaled_points[:, 1] = np.clip(scaled_points[:, 1], 0, original_h -1)

            rect_width = np.linalg.norm(scaled_points[0] - scaled_points[1])
            rect_height = np.linalg.norm(scaled_points[1] - scaled_points[2])
            if min(rect_width, rect_height) < self.det_min_box_size: continue

            # Simple point ordering (TL, TR, BR, BL for upright rects, may need more robust for rotated)
            # For minAreaRect output, points are already ordered (though starting point can vary)
            # Re-order to ensure consistent TL start for perspective transform.
            rect = np.zeros((4, 2), dtype="float32")
            s = scaled_points.sum(axis=1)
            rect[0] = scaled_points[np.argmin(s)]
            rect[2] = scaled_points[np.argmax(s)]
            diff = np.diff(scaled_points, axis=1)
            rect[1] = scaled_points[np.argmin(diff)]
            rect[3] = scaled_points[np.argmax(diff)]

            detected_boxes.append(rect.astype(np.int32))
        self.logger.info(f"Postprocessing detection. Found {len(detected_boxes)} boxes after filtering.")
        return detected_boxes

    def _get_target_crop_dimensions(self, box_points: np.ndarray) -> Tuple[int, int]:
        # (Implementation from previous turn)
        rec_img_c, rec_img_h, rec_img_w_config = self.rec_image_shape
        tl, tr, br, bl = box_points[0], box_points[1], box_points[2], box_points[3]
        width1 = np.linalg.norm(tr - tl); width2 = np.linalg.norm(br - bl)
        avg_width = (width1 + width2) / 2
        height1 = np.linalg.norm(bl - tl); height2 = np.linalg.norm(br - tr)
        avg_height = (height1 + height2) / 2
        if avg_height == 0 or avg_width == 0: return rec_img_h, 10
        aspect_ratio = avg_width / avg_height
        target_rec_width = int(rec_img_h * aspect_ratio)
        return rec_img_h, max(1, min(target_rec_width, rec_img_w_config))

    def _preprocess_text_regions_for_recognition(self, original_image_data: np.ndarray,
                                               detected_boxes: List[np.ndarray]
                                               ) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        if not detected_boxes: self.logger.info("No boxes for recognition preprocessing."); return None
        batch_images_list, valid_boxes = [], []
        rec_img_c, rec_img_h, rec_img_w_config = self.rec_image_shape

        for i, box_points in enumerate(detected_boxes):
            try:
                src_points = box_points.astype(np.float32)
                # Basic check for degenerate box before perspective transform
                if cv2.contourArea(src_points) < self.det_min_box_size * self.det_min_box_size : # min area check
                    self.logger.warning(f"Box {i} is too small ({cv2.contourArea(src_points)} area). Skipping warp.")
                    continue

                target_h, target_w = self._get_target_crop_dimensions(src_points)
                if target_w <= 0 or target_h <=0:
                     self.logger.warning(f"Box {i} resulted in invalid target warp dims ({target_w}x{target_h}). Skipping.")
                     continue

                dst_points = np.array([[0,0], [target_w-1,0], [target_w-1,target_h-1], [0,target_h-1]], dtype=np.float32)
                transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                warped_region = cv2.warpPerspective(original_image_data, transform_matrix, (target_w, target_h))

                if warped_region.size == 0: self.logger.warning(f"Box {i} warp resulted in empty region."); continue

                # Color conversion and normalization (as in previous turn)
                if rec_img_c == 1:
                    img_for_norm = cv2.cvtColor(warped_region, cv2.COLOR_BGR2GRAY) if len(warped_region.shape) == 3 else warped_region
                    img_for_norm = np.expand_dims(img_for_norm, axis=-1)
                else: # Assume 3 channels
                    img_for_norm = cv2.cvtColor(warped_region, cv2.COLOR_GRAY2BGR) if len(warped_region.shape) == 2 else warped_region

                img_norm = (img_for_norm.astype(np.float32) / 255.0 - self.rec_norm_mean) / self.rec_norm_std
                img_chw = np.transpose(img_norm, (2, 0, 1))
                batch_images_list.append(img_chw)
                valid_boxes.append(box_points)
            except Exception as e:
                self.logger.error(f"Error preprocessing region {i} for recognition: {e}", exc_info=True)

        if not batch_images_list: self.logger.warning("No regions successfully preprocessed for rec."); return None

        # Padding (as in previous turn, ensure constant_values matches normalization if needed)
        padded_batch = []
        for img_tensor in batch_images_list:
            c, h, w = img_tensor.shape
            img_to_pad = img_tensor
            if w > rec_img_w_config: # Should be rare if _get_target_crop_dimensions is correct
                self.logger.warning(f"Rec region width {w} > config max {rec_img_w_config}. Resizing with data loss.")
                new_img_tensor = np.zeros((c, h, rec_img_w_config), dtype=img_tensor.dtype)
                for ch_idx in range(c): new_img_tensor[ch_idx,:,:] = cv2.resize(img_tensor[ch_idx,:,:], (rec_img_w_config, h))
                img_to_pad = new_img_tensor

            pad_width_val = rec_img_w_config - img_to_pad.shape[2]
            # Using 0 for padding. If normalization makes 0 a non-neutral value, this might need adjustment.
            # For (img/255 - 0.5) / 0.5, 0 becomes -1. If model expects -1 for padding, this is fine.
            padded_img = np.pad(img_to_pad, ((0,0), (0,0), (0, pad_width_val)), mode='constant', constant_values=-1 if self.rec_norm_mean==0.5 else 0)
            padded_batch.append(padded_img)

        return np.stack(padded_batch, axis=0), valid_boxes

    def _ctc_decode_with_confidence(self, preds_logits: np.ndarray) -> List[Tuple[str, float]]:
        if self.character_dict is None or self.ctc_blank_idx is None:
            self.logger.error("Cannot CTC decode without character_dict or ctc_blank_idx."); return []

        texts_with_conf = []
        # Apply softmax to logits to get probabilities
        exp_preds = np.exp(preds_logits - np.max(preds_logits, axis=2, keepdims=True))
        probs_preds = exp_preds / np.sum(exp_preds, axis=2, keepdims=True)

        for i in range(probs_preds.shape[0]): # Iterate over batch
            prob_sequence = probs_preds[i, :, :] # (sequence_length, num_characters)
            char_indices = np.argmax(prob_sequence, axis=1) # Greedy path

            decoded_chars = []
            char_confidences = []
            last_char_idx = -1

            for t in range(len(char_indices)):
                char_idx = char_indices[t]
                if char_idx == self.ctc_blank_idx:
                    last_char_idx = -1
                    continue
                if char_idx == last_char_idx: # Skip duplicate
                    continue

                # Ensure char_idx is within the valid range of self.character_dict
                if char_idx < len(self.character_dict): # Valid character, not the implicit blank
                    decoded_chars.append(self.character_dict[char_idx])
                    char_confidences.append(prob_sequence[t, char_idx])
                last_char_idx = char_idx

            text = "".join(decoded_chars)
            confidence = np.mean(char_confidences) if char_confidences else 0.0
            texts_with_conf.append((text, float(confidence)))

        return texts_with_conf

    def _run_recognition_batch(self, batch_image_tensors: np.ndarray) -> List[Tuple[str, float]]:
        # (Largely same as before, but calls _ctc_decode_with_confidence)
        if not self.recognition_session: self.logger.error("Rec session not init."); return []
        if batch_image_tensors is None or batch_image_tensors.shape[0] == 0: return []

        self.logger.debug(f"Running text recognition model on batch of size {batch_image_tensors.shape[0]}...")
        try:
            input_name = self.recognition_session.get_inputs()[0].name
            raw_output_logits = self.recognition_session.run(None, {input_name: batch_image_tensors})[0]

            recognized_texts_with_conf = self._ctc_decode_with_confidence(raw_output_logits)
            self.logger.debug(f"Recognition and decoding complete. Found {len(recognized_texts_with_conf)} texts for batch.")
            return recognized_texts_with_conf
        except Exception as e:
            self.logger.error(f"Error during _run_recognition_batch: {e}", exc_info=True)
            return [("Error: Rec Inference Failed", 0.0)] * batch_image_tensors.shape[0]

    def recognize(self, image_data: np.ndarray, language_hint: Optional[str] = None) -> Dict[str, Any]:
        # (Structure from previous turn, now using refined methods)
        if not self._is_initialized:
            self.logger.error(f"Engine '{self.engine_name}' not initialized. Cannot perform recognition.")
            return {"text": "", "segments": [], "confidence": None, "engine_name": self.get_engine_name()}

        self.logger.info(f"Starting OCR process. Image shape: {image_data.shape}")
        original_shape = image_data.shape[:2]

        det_input_tensor, h_ratio, w_ratio = self._preprocess_image_for_detection(image_data)
        if det_input_tensor is None: return {"text": "", "segments": [], "confidence": None, "engine_name": self.get_engine_name()}

        raw_det_output = self._run_detection(det_input_tensor)
        if raw_det_output is None: return {"text": "", "segments": [], "confidence": None, "engine_name": self.get_engine_name()}

        detected_boxes_quads = self._postprocess_detection(raw_det_output, h_ratio, w_ratio, original_shape)
        if not detected_boxes_quads: return {"text": "", "segments": [], "confidence": None, "engine_name": self.get_engine_name()}

        all_output_segments, all_text_parts = [], []
        total_confidence_sum, num_valid_segments = 0.0, 0

        num_boxes = len(detected_boxes_quads)
        for i in range(0, num_boxes, self.rec_batch_size):
            batch_boxes = detected_boxes_quads[i : min(i + self.rec_batch_size, num_boxes)]
            rec_input_tuple = self._preprocess_text_regions_for_recognition(image_data, batch_boxes)

            current_batch_texts_with_conf = []
            current_batch_original_boxes = batch_boxes # Default, used if rec_input_tuple is None or fails early

            if rec_input_tuple is not None:
                rec_input_batch_tensors, current_batch_original_boxes_from_prep = rec_input_tuple
                current_batch_original_boxes = current_batch_original_boxes_from_prep # Use boxes that survived prep
                if rec_input_batch_tensors is not None and rec_input_batch_tensors.shape[0] > 0 :
                    current_batch_texts_with_conf = self._run_recognition_batch(rec_input_batch_tensors)
                else: # Preprocessing returned empty tensor for this batch
                    self.logger.warning(f"Empty tensor from recognition preprocessing for batch starting at box index {i}.")
            else: # Preprocessing failed entirely for this batch
                self.logger.warning(f"Recognition preprocessing failed for batch starting at box index {i}.")

            # Ensure current_batch_texts_with_conf has an entry for each box in current_batch_original_boxes
            # Fill with error if recognition didn't produce enough results (e.g. prep failed for some items)
            # This alignment is crucial.
            processed_box_count_in_batch = len(current_batch_texts_with_conf)
            expected_box_count_in_batch = len(current_batch_original_boxes)

            for j in range(expected_box_count_in_batch):
                text, conf = ("REC_FAIL", 0.0) # Default for boxes that failed somewhere in rec pipeline for this batch
                if j < processed_box_count_in_batch: # We have a result for this box
                    text, conf = current_batch_texts_with_conf[j]

                if text.startswith("Error:") or text == "REC_FAIL": conf = 0.0

                all_text_parts.append(text)
                box_poly = current_batch_original_boxes[j]
                x_coords, y_coords = box_poly[:,0], box_poly[:,1]
                bbox_xywh = [int(np.min(x_coords)), int(np.min(y_coords)),
                             int(np.max(x_coords) - np.min(x_coords)), int(np.max(y_coords) - np.min(y_coords))]

                all_output_segments.append({
                    "text": text, "bounding_box": bbox_xywh, "confidence": conf, "word_details": []
                })
                if conf > 0.0 : # Consider valid if confidence is positive (not an explicit error string)
                    total_confidence_sum += conf
                    num_valid_segments +=1

        full_text = "\n".join(filter(lambda x: not (x.startswith("Error:") or x == "REC_FAIL"), all_text_parts))
        overall_confidence = (total_confidence_sum / num_valid_segments) if num_valid_segments > 0 else 0.0

        self.logger.info(f"OCR process completed. Recognized {len(all_output_segments)} segments. Overall confidence: {overall_confidence:.2f}")
        return {
            "text": full_text, "segments": all_output_segments, "confidence": overall_confidence,
            "engine_name": self.get_engine_name()
        }

    def get_engine_name(self) -> str: return self.engine_name
    def is_available(self) -> bool: return self._is_initialized


if __name__ == '__main__':
    # Main block remains for basic structural invocation.
    # Requires a char dict to be present at the specified path for initialize() to pass that stage.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_test_logger = logging.getLogger("LocalOCREngineRefinedTest")
    main_test_logger.info("Running LocalOCREngine (Refined) __main__ test...")

    dummy_root_dir = "temp_test_data_refined" # Changed dir name slightly
    dummy_model_dir = os.path.join(dummy_root_dir, "models")
    dummy_char_dict_file = os.path.join(dummy_root_dir, "dummy_refined_keys.txt") # Path for char dict
    dummy_det_model_file = os.path.join(dummy_model_dir, "dummy_refined_det.onnx")
    dummy_rec_model_file = os.path.join(dummy_model_dir, "dummy_refined_rec.onnx")

    try:
        os.makedirs(dummy_model_dir, exist_ok=True) # Create models subdir
        # Create the character dictionary that initialize() will try to load.
        test_chars = [' '] + [chr(ord('a')+i) for i in range(26)] + [chr(ord('0')+i) for i in range(10)]
        with open(dummy_char_dict_file, 'w', encoding='utf-8') as f:
            for char_val in test_chars: f.write(f"{char_val}\n")
        main_test_logger.info(f"Created dummy char dict: {dummy_char_dict_file}")

        # Create dummy model files (content doesn't matter, ONNX load will fail but test structure)
        with open(dummy_det_model_file, 'w') as f: f.write("dummy_det_model")
        with open(dummy_rec_model_file, 'w') as f: f.write("dummy_rec_model")
        main_test_logger.info(f"Created dummy model files in {dummy_model_dir}")
    except Exception as e:
        main_test_logger.error(f"Could not create dummy files/char_dict for testing: {e}")


    engine_config_for_test = {
        "name": "TestLocalPaddleRefined",
        "config": {
            "use_gpu_directml": False,
            "detection_model_path": dummy_det_model_file,
            "recognition_model_path": dummy_rec_model_file,
            "character_dict_path": dummy_char_dict_file, # Crucial for init
            # Other params can use defaults or be specified
             "rec_image_shape": [1, 32, 100], # C,H,W for rec model; smaller width for test
             "det_input_size": [320,320]
        }
    }
    local_engine = LocalOCREngine(engine_config=engine_config_for_test, logger=main_test_logger)
    local_engine.initialize()
    main_test_logger.info(f"Engine initialized state: {local_engine.is_available()}")

    if local_engine.is_available(): # Will be false if ONNX dummy models failed to load
        main_test_logger.info("Engine claims to be available (likely ONNX sessions are None or dummy).")
        dummy_image = np.full((200, 300, 3), (230, 230, 230), dtype=np.uint8)
        cv2.putText(dummy_image, "TEST", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50), 2)
        results = local_engine.recognize(dummy_image)
        main_test_logger.info(f"Results from recognize (dummy models will lead to errors in pipeline):\n{results}")
    else:
        main_test_logger.warning("Engine not available after init (expected with dummy models). Full recognize call skipped.")
        # Test a preprocessing step to see if it runs
        prep_tensor, _, _ = local_engine._preprocess_image_for_detection(np.full((100,100,3), 128, np.uint8))
        if prep_tensor is not None: main_test_logger.info(f"Det preprocessing test output shape: {prep_tensor.shape}")

    # Clean up (optional)
    try:
        if os.path.exists(dummy_char_dict_file): os.remove(dummy_char_dict_file)
        if os.path.exists(dummy_det_model_file): os.remove(dummy_det_model_file)
        if os.path.exists(dummy_rec_model_file): os.remove(dummy_rec_model_file)
        if os.path.exists(dummy_model_dir) and not os.listdir(dummy_model_dir) : os.rmdir(dummy_model_dir)
        if os.path.exists(dummy_root_dir) and not os.listdir(dummy_root_dir) : os.rmdir(dummy_root_dir) # remove root if empty
        main_test_logger.info("Cleaned up some dummy files/dirs.")
    except Exception as e: main_test_logger.warning(f"Error during cleanup: {e}")

    main_test_logger.info("LocalOCREngine (Refined) __main__ test finished.")
```
