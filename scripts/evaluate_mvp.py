"""
MVP E2E Evaluation Script for OCR-X.

This script processes a directory of input documents (images/PDFs),
generates plain text output using OCRWorkflowOrchestrator,
compares the output against ground truth text files, and calculates
Character Error Rate (CER) and Word Error Rate (WER).
"""
import argparse
import logging
from pathlib import Path
import jiwer
import time

# Ensure ocrx modules are importable, assuming script is run from project root
# or ocrx package is in PYTHONPATH.
from ocrx.ocr_workflow_orchestrator import OCRWorkflowOrchestrator
from ocrx.core.data_objects import DocumentContext # For type hinting

# Setup basic logging for the script itself
script_logger = logging.getLogger("evaluate_mvp")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_gt_path(ocr_output_path: Path, gt_dir: Path) -> Optional[Path]:
    """
    Determines the corresponding ground truth file path for a given OCR output file.
    Assumes OCR output is <doc_id>_page_<page_num>.txt and GT is <doc_id>_page_<page_num>.gt.txt or <doc_id>.gt.txt.
    Adjust logic as needed based on actual GT naming conventions.
    """
    base_name = ocr_output_path.stem # e.g., "mydoc_page_0"

    # Try GT convention: <doc_id>_page_<page_num>.gt.txt
    gt_path_specific = gt_dir / f"{base_name}.gt.txt"
    if gt_path_specific.exists():
        return gt_path_specific

    # Try GT convention: <doc_id>.gt.txt (if GT is per document, not per page)
    # This requires knowing if the OCR output is from a multi-page doc's first page or single-page doc.
    # For MVP, assume one GT file per input image, or one GT file per page of a PDF.
    # If OCR output is "mydoc_page_0.txt", try "mydoc.gt.txt".
    if "_page_0" in base_name:
        doc_id_part = base_name.rsplit("_page_0", 1)[0]
        gt_path_doc_level = gt_dir / f"{doc_id_part}.gt.txt"
        if gt_path_doc_level.exists():
            return gt_path_doc_level

    # Fallback: if OCR output is "mydoc.txt" (no page part), try "mydoc.gt.txt"
    if "_page_" not in base_name:
        gt_path_direct = gt_dir / f"{base_name}.gt.txt"
        if gt_path_direct.exists():
            return gt_path_direct

    script_logger.warning(f"No corresponding GT file found for OCR output: {ocr_output_path.name} in {gt_dir}")
    return None


def main(args):
    input_path = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    config_file = args.config_file

    if not input_path.is_dir():
        script_logger.error(f"Input directory not found: {input_path}")
        return
    if not gt_dir.is_dir():
        script_logger.error(f"Ground truth directory not found: {gt_dir}")
        return
    if not Path(config_file).is_file():
        script_logger.error(f"Configuration file not found: {config_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    script_logger.info(f"Initializing OCRWorkflowOrchestrator with config: {config_file}")
    # Pass the output_dir from script args to the orchestrator's OutputGenerator module config
    # This demonstrates overriding a specific module's parameter at runtime.
    runtime_overrides = {
        "modules": {
            "output_generator": {
                "output_dir": str(output_dir)
            }
        }
    }
    orchestrator = OCRWorkflowOrchestrator(config_path=config_file)

    all_gt_texts: List[str] = []
    all_ocr_texts: List[str] = []

    # Determine files to process
    # Simple approach: iterate all files in input_dir. Could be refined (e.g. by extension).
    files_to_process = [f for f in input_path.iterdir() if f.is_file()]
    script_logger.info(f"Found {len(files_to_process)} files to process in {input_path}")

    for i, file_path in enumerate(files_to_process):
        script_logger.info(f"Processing file {i+1}/{len(files_to_process)}: {file_path.name}...")
        try:
            doc_start_time = time.time()
            # The orchestrator's process_document now takes 'source' and 'runtime_config_override'
            # The 'runtime_config_override' here is for the orchestrator itself, not specific modules directly.
            # We've updated the global config for the orchestrator instance with the output_dir.
            # If other overrides are needed per document, they can be passed here.
            doc_context: DocumentContext = orchestrator.process_document(
                source=str(file_path),
                runtime_config_override=runtime_overrides # Pass output_dir override
            )
            doc_process_time = time.time() - doc_start_time
            script_logger.info(f"Finished processing {file_path.name} in {doc_process_time:.2f}s. Status: {doc_context.overall_status}")

            if doc_context.overall_status == "failed" or doc_context.document_errors:
                script_logger.error(f"Failed to process {file_path.name}. Errors: {doc_context.document_errors}")
                # For evaluation, we might still try to find partial outputs if any page succeeded.
                # Or skip this document for CER/WER if it's a full failure.

            # Assuming OutputGenerator creates one .txt file per page in the output_dir
            # The format is <doc_id>_page_<page_num>.txt
            for page_idx, page_ctx in enumerate(doc_context.pages):
                # Construct expected OCR output text file path
                sanitized_doc_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in doc_context.document_id)
                ocr_text_file_name = f"{sanitized_doc_id}_page_{page_idx}.txt"
                ocr_text_file_path = output_dir / ocr_text_file_name

                if ocr_text_file_path.exists():
                    ocr_text = ocr_text_file_path.read_text(encoding="utf-8").strip()

                    # Find corresponding GT file
                    # For MVP, assume GT file is named same as input image + .gt.txt, or per page.
                    # E.g., if input is "doc1.png", GT is "doc1.gt.txt".
                    # If input is "mydoc.pdf" (processed as "mydoc_page_0.txt"), GT could be "mydoc_page_0.gt.txt" or "mydoc.gt.txt".
                    # This logic needs to be robust based on dataset naming.

                    # For this script, let's assume GT matches the OCR output base name + .gt.txt
                    # E.g., for "mydoc_page_0.txt", GT is "mydoc_page_0.gt.txt"
                    gt_file_path = get_gt_path(ocr_text_file_path, gt_dir)

                    if gt_file_path and gt_file_path.exists():
                        gt_text = gt_file_path.read_text(encoding="utf-8").strip()
                        all_gt_texts.append(gt_text)
                        all_ocr_texts.append(ocr_text)
                        script_logger.info(f"Found GT for {ocr_text_file_name}. Added for evaluation.")
                    else:
                        script_logger.warning(f"Ground truth file not found for {ocr_text_file_name} (expected at {gt_file_path}). Skipping this output.")
                else:
                    script_logger.warning(f"OCR output text file not found: {ocr_text_file_path}. Page status: {page_ctx.errors}")

        except Exception as e:
            script_logger.error(f"Critical error processing file {file_path.name}: {e}", exc_info=True)

    if not all_gt_texts or not all_ocr_texts:
        script_logger.error("No OCR outputs or ground truths were successfully paired. Cannot calculate error rates.")
        return

    script_logger.info(f"\n--- Evaluation Summary ---")
    script_logger.info(f"Processed {len(all_ocr_texts)} text pairs for evaluation.")

    # Calculate overall CER and WER
    # jiwer.compute_measures requires lists of strings (ground_truth_list, hypothesis_list)
    try:
        measures = jiwer.compute_measures(all_gt_texts, all_ocr_texts)
        cer = measures['cer']
        wer = measures['wer']
        script_logger.info(f"Overall CER: {cer:.4f}")
        script_logger.info(f"Overall WER: {wer:.4f}")

        # Detailed output (hits, subs, dels, ins)
        script_logger.info(f"WER Details: Hits={measures['hits']}, Substitutions={measures['substitutions']}, Deletions={measures['deletions']}, Insertions={measures['insertions']}")

    except Exception as e:
        script_logger.error(f"Error calculating CER/WER with jiwer: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR-X MVP E2E Evaluation Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images/PDFs.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth .txt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save OCR output .txt files.")
    parser.add_argument("--config_file", type=str, default="configs/mvp_config.yaml", help="Path to the OCR-X YAML configuration file.")

    args = parser.parse_args()
    main(args)
