import pytest
import os
import numpy as np
from PIL import Image
import fitz # PyMuPDF
from pathlib import Path
import yaml

from ocrx.ocr_workflow_orchestrator import OCRWorkflowOrchestrator
from ocrx.core.data_objects import DocumentContext

# --- Helper Functions from unit tests (or could be moved to a common test utils) ---
def create_dummy_image_file(filepath: Path, img_format: str = "PNG", size: tuple = (100, 80), color: str = "blue") -> None:
    try:
        img = Image.new('RGB', size, color=color)
        img.save(filepath, format=img_format)
    except Exception as e:
        pytest.skip(f"Pillow could not save {img_format} file, skipping test: {e}")

def create_dummy_pdf_file(filepath: Path, num_pages: int = 1, size: tuple = (200, 300)) -> None:
    try:
        pdf_doc = fitz.open()
        for i in range(num_pages):
            page = pdf_doc.new_page(width=size[0], height=size[1])
            page.insert_text((50, 72), f"Page {i+1} content.", fontsize=11)
        pdf_doc.save(str(filepath))
        pdf_doc.close()
    except Exception as e:
        pytest.skip(f"PyMuPDF could not create PDF file, skipping test: {e}")

# --- Fixtures ---
@pytest.fixture
def tmp_fixtures_dir(tmp_path: Path) -> Path:
    fixtures_dir = tmp_path / "integration_test_fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    return fixtures_dir

@pytest.fixture
def integration_test_config(tmp_path: Path) -> str:
    """Creates a YAML config file for integration tests."""
    config_dict = {
        "app_settings": {
            "default_ocr_engine": "dummy_engine_placeholder" # Not used in this test
        },
        "modules": {
            "image_loader": {
                "default_dpi": 150 # Lower DPI for faster PDF processing in tests
            },
            "preprocessor": {
                "enabled": True,
                "osd_skew_correction": {"enabled": True, "min_skew_angle_to_correct": 0.1},
                "blur": {"enabled": False}, # Keep blur off to make checks easier
                "binarization": {"enabled": True, "method": "otsu"}
            }
            # No OCR engine configs needed as we are not testing recognition here
        },
        "ocr_engines": {}, # Empty, not testing OCR part
        "logging": { # Basic logging for tests
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"simple": {'format': '%(levelname)s - %(name)s: %(message)s'}},
            "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple", "level": "DEBUG"}},
            "root": {"handlers": ["console"], "level": "WARNING"},
            "loggers": {
                "ocrx": {"handlers": ["console"], "level": "INFO", "propagate": False}
            }
        }
    }
    config_file_path = tmp_path / "integration_config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
    return str(config_file_path)

@pytest.fixture
def orchestrator_for_integration(integration_test_config: str) -> OCRWorkflowOrchestrator:
    return OCRWorkflowOrchestrator(config_path=integration_test_config)

# --- Integration Test Cases ---

def test_ingestion_preprocessing_flow_png(orchestrator_for_integration: OCRWorkflowOrchestrator, tmp_fixtures_dir: Path):
    """Test processing a PNG image through image loading and preprocessing."""
    img_filename = "test_integration.png"
    img_path = tmp_fixtures_dir / img_filename
    create_dummy_image_file(img_path, "PNG", size=(120, 90), color="green")

    doc_context = orchestrator_for_integration.process_document(source=str(img_path))

    assert doc_context is not None
    assert doc_context.document_id == img_filename
    assert not doc_context.document_errors, f"Document errors: {doc_context.document_errors}"
    assert doc_context.overall_status == "completed" or doc_context.overall_status == "completed_with_errors" # if page errors exist

    assert len(doc_context.pages) == 1
    page_ctx = doc_context.pages[0]

    assert page_ctx.original_image is not None
    assert isinstance(page_ctx.original_image, np.ndarray)
    assert page_ctx.original_image.shape == (90, 120, 3) # H, W, C

    assert page_ctx.preprocessed_image is not None
    assert isinstance(page_ctx.preprocessed_image, np.ndarray)
    assert page_ctx.preprocessed_image.shape == (90, 120, 3)

    # Check if preprocessing (binarization) was applied (image should be different)
    # And preprocessed image should be binary (in BGR format, so each channel binary)
    if orchestrator_for_integration.preprocessor and orchestrator_for_integration.preprocessor.is_enabled() and \
       orchestrator_for_integration.preprocessor.config.get("binarization",{}).get("enabled"):
        assert not np.array_equal(page_ctx.original_image, page_ctx.preprocessed_image)
        unique_vals_ch0 = np.unique(page_ctx.preprocessed_image[:,:,0])
        assert len(unique_vals_ch0) <= 2 # Should be 0 and 255 mostly

    assert "preprocessing" in page_ctx.processing_times
    assert page_ctx.processing_times["preprocessing"] > 0

def test_ingestion_preprocessing_flow_pdf(orchestrator_for_integration: OCRWorkflowOrchestrator, tmp_fixtures_dir: Path):
    """Test processing a PDF document through image loading and preprocessing."""
    pdf_filename = "test_integration.pdf"
    pdf_path = tmp_fixtures_dir / pdf_filename
    create_dummy_pdf_file(pdf_path, num_pages=2, size=(100,120)) # W,H for PDF page

    doc_context = orchestrator_for_integration.process_document(source=str(pdf_path))

    assert doc_context is not None
    assert doc_context.document_id == pdf_filename
    assert not doc_context.document_errors, f"Document errors: {doc_context.document_errors}"
    assert doc_context.overall_status == "completed" or doc_context.overall_status == "completed_with_errors"

    assert len(doc_context.pages) == 2

    for i, page_ctx in enumerate(doc_context.pages):
        assert page_ctx.page_number == i
        assert page_ctx.original_image is not None
        assert isinstance(page_ctx.original_image, np.ndarray)

        # Expected H, W for 150 DPI: (120*150/72, 100*150/72) -> (250, 208)
        dpi = orchestrator_for_integration.image_loader.config.get("default_dpi", 150)
        expected_h = int(120 * dpi / 72)
        expected_w = int(100 * dpi / 72)
        assert abs(page_ctx.original_image.shape[0] - expected_h) < 2
        assert abs(page_ctx.original_image.shape[1] - expected_w) < 2

        assert page_ctx.preprocessed_image is not None
        assert isinstance(page_ctx.preprocessed_image, np.ndarray)
        assert page_ctx.preprocessed_image.shape == page_ctx.original_image.shape

        if orchestrator_for_integration.preprocessor and orchestrator_for_integration.preprocessor.is_enabled() and \
           orchestrator_for_integration.preprocessor.config.get("binarization",{}).get("enabled"):
            assert not np.array_equal(page_ctx.original_image, page_ctx.preprocessed_image)

        assert "preprocessing" in page_ctx.processing_times
        assert page_ctx.processing_times["preprocessing"] > 0

def test_ingestion_error_propagates_to_document_context(orchestrator_for_integration: OCRWorkflowOrchestrator, tmp_fixtures_dir: Path):
    """Test that an error during image loading is correctly reported in DocumentContext."""
    invalid_source = str(tmp_fixtures_dir / "non_existent_file.png")

    doc_context = orchestrator_for_integration.process_document(source=invalid_source)

    assert doc_context is not None
    assert doc_context.overall_status == "image_loading_failed" # Updated by ImageLoader
    assert len(doc_context.document_errors) > 0
    assert "ImageLoader: File not found" in doc_context.document_errors[0] or \
           "ImageLoader: Unsupported source type" in doc_context.document_errors[0] # if path doesn't exist before process call

    # Ensure no pages were partially processed or added if loading failed critically
    assert len(doc_context.pages) == 0 or \
           all(p.original_image is None for p in doc_context.pages) # Or pages list is empty / page has errors

# More tests could be added:
# - Test with byte inputs for images and PDFs.
# - Test specific preprocessing outcomes (e.g., if skew correction changed image orientation).
# - Test with runtime_config_override to change preprocessing steps.
