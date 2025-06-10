import logging
import json
import csv
from typing import Any, Dict, Optional, List
from pathlib import Path

from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.data_objects import PageContext, TextRegion, MainCandidate, RecognitionResult
from ocrx.core.exceptions import OCRXConfigurationError

class AdvancedPostprocessor(OCRXModuleBase):
    """
    Performs post-processing on recognized text, including dictionary-based correction
    and placeholder for sequence-to-sequence model correction.
    """

    def __init__(self, module_id: str, config: Dict[str, Any]):
        self.correction_dict: Dict[str, str] = {}
        super().__init__(module_id, config)

    def _validate_config(self) -> None:
        super()._validate_config()
        self.config.setdefault("dictionary_path", None) # Optional
        self.config.setdefault("s2s_model_enabled", False)

        if self.config["dictionary_path"] and not isinstance(self.config["dictionary_path"], str):
            raise OCRXConfigurationError(f"{self.module_id}: 'dictionary_path' must be a string path if provided.")

        self.logger.info(f"{self.module_id} validated config: {self.config}")

    def _initialize_resources(self) -> None:
        super()._initialize_resources()
        dict_path_str = self.config.get("dictionary_path")
        if dict_path_str:
            dict_path = Path(dict_path_str)
            if not dict_path.exists() or not dict_path.is_file():
                self.logger.warning(f"Correction dictionary not found or not a file: {dict_path_str}. No dictionary corrections will be applied.")
                return
            try:
                if dict_path.suffix.lower() == ".json":
                    with open(dict_path, 'r', encoding='utf-8') as f:
                        self.correction_dict = json.load(f)
                elif dict_path.suffix.lower() == ".csv":
                    with open(dict_path, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.reader(f)
                        # Assuming CSV format: "misspelled_word","corrected_word"
                        for row in reader:
                            if len(row) == 2:
                                self.correction_dict[row[0]] = row[1]
                            else:
                                self.logger.warning(f"Skipping invalid CSV row in {dict_path_str}: {row}")
                else:
                    self.logger.warning(f"Unsupported dictionary file format: {dict_path.suffix}. Use .json or .csv.")

                if self.correction_dict:
                    self.logger.info(f"Loaded correction dictionary with {len(self.correction_dict)} entries from {dict_path_str}.")
                else:
                    self.logger.warning(f"Correction dictionary loaded from {dict_path_str} but is empty or failed to parse.")
            except Exception as e:
                self.logger.error(f"Error loading correction dictionary from {dict_path_str}: {e}", exc_info=True)

        if self.config.get("s2s_model_enabled", False):
            self.logger.info("S2S model correction is enabled in config, but it's a placeholder for MVP and will not be applied.")


    def _apply_dictionary_correction(self, text: str) -> str:
        if not self.correction_dict:
            return text

        # Simple word-based replacement. More complex logic might be needed for case sensitivity, partial words etc.
        # For MVP, let's do a case-sensitive whole word replacement.
        # A more robust way would be to split by spaces and punctuation, then replace.
        # This is a very basic example:
        corrected_words = []
        # Split text into words, preserving spaces somewhat crudely.
        # This won't handle punctuation attached to words well.
        # For MVP, a simple split() might be enough to demonstrate.
        words = text.split(' ')
        for word in words:
            # Preserve leading/trailing punctuation for the word if any (very basic)
            # A regex based split and join would be better.
            # For now, just direct lookup.
            corrected_words.append(self.correction_dict.get(word, word))

        return " ".join(corrected_words)

    def process(self, page_ctx: PageContext, config_override: Optional[Dict] = None) -> None:
        """
        Applies post-processing to text in each TextRegion of a PageContext.

        Args:
            page_ctx: The PageContext object containing TextRegions with OCR results.
            config_override: Optional runtime configuration (not used by this module for now).
        """
        if not self.is_enabled():
            self.logger.info(f"Module {self.module_id} is disabled. Skipping post-processing.")
            return

        current_config = {**self.config, **(config_override or {})}
        self.logger.info(f"Starting post-processing for page {page_ctx.page_number} using {self.module_id}")

        for region_idx, text_region in enumerate(page_ctx.layout_regions):
            candidate_to_process: Optional[MainCandidate] = None

            # Step 1: Determine the candidate text to post-process
            if text_region.consensus_candidate:
                # If a consensus step already ran, use its output
                # Create a new MainCandidate based on it to avoid modifying original consensus
                candidate_to_process = MainCandidate(
                    text=text_region.consensus_candidate.text,
                    confidence=text_region.consensus_candidate.confidence,
                    source_engines=list(text_region.consensus_candidate.source_engines)
                )
                self.logger.debug(f"Region {text_region.region_id}: Using existing consensus candidate for post-processing.")
            elif text_region.raw_ocr_results:
                # MVP: If no consensus, take the first raw OCR result as the base
                # A more sophisticated approach would pick the "best" based on confidence or other metrics.
                best_raw_result = text_region.raw_ocr_results[0] # Assuming at least one result
                candidate_to_process = MainCandidate(
                    text=best_raw_result.text,
                    confidence=best_raw_result.confidence,
                    source_engines=[best_raw_result.engine_id or "unknown_engine"]
                )
                self.logger.debug(f"Region {text_region.region_id}: No consensus, using first raw OCR result for post-processing.")
            else:
                self.logger.warning(f"Region {text_region.region_id}: No raw OCR results or consensus candidate. Skipping post-processing for this region.")
                text_region.postprocessed_candidate = None # Ensure it's None
                continue

            # Step 2: Apply dictionary-based correction
            if self.correction_dict:
                original_text = candidate_to_process.text
                candidate_to_process.text = self._apply_dictionary_correction(original_text)
                if original_text != candidate_to_process.text:
                    self.logger.debug(f"Region {text_region.region_id}: Dictionary correction applied. Original: '{original_text}', Corrected: '{candidate_to_process.text}'")

            # Step 3: S2S Model Correction (Placeholder for MVP)
            if current_config.get("s2s_model_enabled", False):
                self.logger.info(f"Region {text_region.region_id}: S2S model correction placeholder - no changes applied.")
                # In future: candidate_to_process.text = self.s2s_model.correct(candidate_to_process.text)

            # Store the post-processed candidate
            text_region.postprocessed_candidate = candidate_to_process
            self.logger.debug(f"Region {text_region.region_id}: Post-processing complete. Final text: '{candidate_to_process.text}'")

        self.logger.info(f"Post-processing completed for page {page_ctx.page_number}.")
