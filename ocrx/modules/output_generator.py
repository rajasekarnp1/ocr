import logging
import os
from typing import Any, Dict, Optional, List
from pathlib import Path
import html # For escaping text in hOCR

from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.data_objects import DocumentContext, PageContext, TextRegion, MainCandidate, RecognitionResult
from ocrx.core.exceptions import OCRXConfigurationError, OCRXProcessingError

class OutputGenerator(OCRXModuleBase):
    """
    Generates output files (e.g., TXT, hOCR) from the processed DocumentContext.
    """

    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)

    def _validate_config(self) -> None:
        super()._validate_config()
        self.config.setdefault("output_dir", "output") # Default output directory
        self.config.setdefault("formats", ["txt", "hocr"]) # Default formats to generate

        if not isinstance(self.config["output_dir"], str):
            raise OCRXConfigurationError(f"{self.module_id}: 'output_dir' must be a string path.")
        if not isinstance(self.config["formats"], list) or \
           not all(isinstance(f, str) and f in ["txt", "hocr", "alto", "pagexml"] for f in self.config["formats"]): # Added more common formats
            raise OCRXConfigurationError(f"{self.module_id}: 'formats' must be a list of supported strings (e.g., 'txt', 'hocr').")

        self.logger.info(f"{self.module_id} validated config: Output dir '{self.config['output_dir']}', Formats {self.config['formats']}")

    def _get_text_for_region(self, text_region: TextRegion) -> str:
        """Helper to get the best available text from a TextRegion."""
        if text_region.postprocessed_candidate:
            return text_region.postprocessed_candidate.text
        elif text_region.consensus_candidate:
            return text_region.consensus_candidate.text
        elif text_region.raw_ocr_results:
            # Fallback: naive concatenation of raw results if multiple segments from one engine
            # Or just take the first one if assuming one result per region from engine for MVP
            return " ".join(res.text for res in text_region.raw_ocr_results if res.text).strip()
        return ""

    def _generate_txt(self, doc_context: DocumentContext, page_ctx: PageContext, page_output_path: Path) -> None:
        """Generates a plain text file for a single page."""
        full_text = []
        # Assume regions are already sorted in reading order by LayoutAnalyzer
        for region in page_ctx.layout_regions:
            region_text = self._get_text_for_region(region)
            if region_text:
                full_text.append(region_text)

        # Join regions with double newlines (paragraphs) or single (lines)
        # For MVP, simple join with single newline between regions.
        content = "\n".join(full_text)

        txt_file_path = page_output_path.with_suffix(".txt")
        try:
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Generated TXT output for page {page_ctx.page_number} at: {txt_file_path}")
        except IOError as e:
            err_msg = f"IOError writing TXT for page {page_ctx.page_number} to {txt_file_path}: {e}"
            self.logger.error(err_msg, exc_info=True)
            page_ctx.errors.append(err_msg) # Add error to page context


    def _generate_hocr(self, doc_context: DocumentContext, page_ctx: PageContext, page_output_path: Path) -> None:
        """Generates an hOCR file for a single page."""
        h, w = (0,0)
        if page_ctx.original_image is not None: # Preprocessed might be altered in size/channels
            h, w = page_ctx.original_image.shape[:2]
        elif page_ctx.preprocessed_image is not None:
            h, w = page_ctx.preprocessed_image.shape[:2]
        else: # Fallback if no image dimensions available (less ideal)
             # Try to get from largest bounding box if regions exist
            if page_ctx.layout_regions:
                max_x = 0
                max_y = 0
                for r in page_ctx.layout_regions:
                    if r.bounding_box:
                        max_x = max(max_x, r.bounding_box[2])
                        max_y = max(max_y, r.bounding_box[3])
                w, h = max_x, max_y
            if w == 0 or h == 0: # Still zero
                w,h = 2000, 3000 # Arbitrary default
                self.logger.warning(f"Page {page_ctx.page_number}: Could not determine image dimensions for hOCR. Using defaults ({w}x{h}).")


        hocr_content = [
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">",
            "<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">",
            "<head>",
            f"<title>OCR Output for {html.escape(doc_context.document_id)}, Page {page_ctx.page_number}</title>",
            "<meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\" />",
            "<meta name='ocr-system' content='OCRX MVP' />", # Replace with actual engine if known
            f"<meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_line ocrx_word'/>", # Hypothetical word capability
            "</head>",
            "<body>",
            f"<div class='ocr_page' id='page_{page_ctx.page_number}' title='image \"{html.escape(doc_context.document_id)}_page_{page_ctx.page_number}\"; bbox 0 0 {w} {h}'>",
        ]

        for region in page_ctx.layout_regions:
            x1, y1, x2, y2 = region.bounding_box
            region_text = self._get_text_for_region(region)

            # For MVP, treat each TextRegion as a 'ocr_carea' (component area) or 'ocr_par' (paragraph)
            # If raw_ocr_results has multiple entries, they could be ocr_line within this area
            hocr_content.append(f"<div class='ocr_carea' id='{region.region_id}_area' title='bbox {x1} {y1} {x2} {y2}'>")

            # If we have per-line results within the region (e.g. from PaddleOCR segments)
            if region.postprocessed_candidate and hasattr(region.postprocessed_candidate, 'segments') and region.postprocessed_candidate.segments: # type: ignore
                # This assumes MainCandidate might have segments in future
                pass # TODO: Iterate segments for ocr_line
            elif region.raw_ocr_results and len(region.raw_ocr_results) > 1 : # Multiple raw results might be lines
                 for idx, line_res in enumerate(region.raw_ocr_results):
                    lx1,ly1,lx2,ly2 = line_res.char_boxes[0] if line_res.char_boxes else (x1,y1,x2,y2) # Use region box if no line box
                    line_id = f"{region.region_id}_line_{idx}"
                    hocr_content.append(f"<span class='ocr_line' id='{line_id}' title='bbox {lx1} {ly1} {lx2} {ly2}'>{html.escape(line_res.text)}</span>")
            else: # Single block of text for the region
                 hocr_content.append(f"<span class='ocr_line' id='{region.region_id}_line' title='bbox {x1} {y1} {x2} {y2}'>{html.escape(region_text)}</span>")

            hocr_content.append("</div>") # Close ocr_carea

        hocr_content.extend(["</div>", "</body>", "</html>"])

        hocr_file_path = page_output_path.with_suffix(".hocr")
        try:
            with open(hocr_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(hocr_content))
            self.logger.info(f"Generated hOCR output for page {page_ctx.page_number} at: {hocr_file_path}")
        except IOError as e:
            err_msg = f"IOError writing hOCR for page {page_ctx.page_number} to {hocr_file_path}: {e}"
            self.logger.error(err_msg, exc_info=True)
            page_ctx.errors.append(err_msg)


    def process(self, doc_context: DocumentContext, config_override: Optional[Dict] = None) -> None:
        """
        Generates output files for the processed document based on configured formats.

        Args:
            doc_context: The DocumentContext object containing all processed data.
            config_override: Optional runtime configuration (not used by this module for now).
        """
        if not self.is_enabled():
            self.logger.info(f"Module {self.module_id} is disabled. Skipping output generation.")
            return

        current_config = {**self.config, **(config_override or {})}
        output_dir = Path(current_config["output_dir"])
        formats_to_generate = current_config["formats"]

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OCRXProcessingError(f"Cannot create output directory {output_dir}: {e}") from e

        self.logger.info(f"Starting output generation for document '{doc_context.document_id}' to '{output_dir}'. Formats: {formats_to_generate}")

        for page_ctx in doc_context.pages:
            # Base path for this page's outputs, e.g. <output_dir>/<doc_id>_page_<page_num>
            # Suffix will be added by specific generator methods.
            sanitized_doc_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in doc_context.document_id)
            page_output_base_path = output_dir / f"{sanitized_doc_id}_page_{page_ctx.page_number}"

            if "txt" in formats_to_generate:
                self._generate_txt(doc_context, page_ctx, page_output_base_path)
            if "hocr" in formats_to_generate:
                self._generate_hocr(doc_context, page_ctx, page_output_base_path)
            # Add calls to other format generators here (e.g., ALTO, PageXML)

        self.logger.info(f"Output generation completed for document '{doc_context.document_id}'.")
