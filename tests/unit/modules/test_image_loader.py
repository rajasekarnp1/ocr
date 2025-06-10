import pytest
import os
import numpy as np
from PIL import Image
import fitz # PyMuPDF
from pathlib import Path

from ocrx.core.data_objects import DocumentContext
from ocrx.modules.image_loader import ImageLoader
from ocrx.core.exceptions import OCRXInputError, OCRXProcessingError

# --- Helper Functions to Create Dummy Files ---

def create_dummy_image_file(filepath: Path, img_format: str = "PNG", size: tuple = (100, 80)) -> None:
    """Creates a dummy image file."""
    try:
        img = Image.new('RGB', size, color='red')
        img.save(filepath, format=img_format)
    except Exception as e:
        pytest.skip(f"Pillow could not save {img_format} file, skipping test: {e}")

def create_dummy_pdf_file(filepath: Path, num_pages: int = 1, size: tuple = (200, 300)) -> None:
    """Creates a dummy PDF file with specified number of blank pages."""
    try:
        pdf_doc = fitz.open() # New empty PDF
        for _ in range(num_pages):
            page = pdf_doc.new_page(width=size[0], height=size[1])
            # Optionally add some text or simple graphics if needed for more complex tests
            # page.insert_text((50, 72), "Hello, PDF page!", fontsize=11)
        pdf_doc.save(str(filepath))
        pdf_doc.close()
    except Exception as e:
        pytest.skip(f"PyMuPDF could not create PDF file, skipping test: {e}")

# --- Fixtures ---

@pytest.fixture
def image_loader_instance() -> ImageLoader:
    """Returns an instance of ImageLoader with default config."""
    return ImageLoader(module_id="test_loader", config={"default_dpi": 150})

@pytest.fixture
def tmp_fixtures_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for fixture files."""
    fixtures_dir = tmp_path / "test_fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    return fixtures_dir

# --- Test Cases ---

def test_load_png_image_file(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_ctx = DocumentContext(document_id="test_png", source_path_or_id=str(tmp_fixtures_dir / "test.png"))
    img_path = tmp_fixtures_dir / "test.png"
    create_dummy_image_file(img_path, "PNG", size=(60,40))

    result_ctx = image_loader_instance.process(doc_ctx, str(img_path))

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 1
    page = result_ctx.pages[0]
    assert page.original_image is not None
    assert isinstance(page.original_image, np.ndarray)
    assert page.original_image.shape == (40, 60, 3) # H, W, C (BGR)
    assert page.page_number == 0

def test_load_jpeg_image_file(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_ctx = DocumentContext(document_id="test_jpg", source_path_or_id=str(tmp_fixtures_dir / "test.jpg"))
    img_path = tmp_fixtures_dir / "test.jpg"
    create_dummy_image_file(img_path, "JPEG") # Default size

    result_ctx = image_loader_instance.process(doc_ctx, str(img_path))

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 1
    assert result_ctx.pages[0].original_image is not None
    assert result_ctx.pages[0].original_image.shape[:2] == (80, 100)

def test_load_tiff_image_file(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_ctx = DocumentContext(document_id="test_tiff", source_path_or_id=str(tmp_fixtures_dir / "test.tiff"))
    img_path = tmp_fixtures_dir / "test.tiff"
    create_dummy_image_file(img_path, "TIFF")

    result_ctx = image_loader_instance.process(doc_ctx, str(img_path))

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 1
    assert result_ctx.pages[0].original_image is not None

def test_load_bmp_image_file(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_ctx = DocumentContext(document_id="test_bmp", source_path_or_id=str(tmp_fixtures_dir / "test.bmp"))
    img_path = tmp_fixtures_dir / "test.bmp"
    create_dummy_image_file(img_path, "BMP")

    result_ctx = image_loader_instance.process(doc_ctx, str(img_path))

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 1
    assert result_ctx.pages[0].original_image is not None

def test_load_pdf_single_page(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_id = "test_single.pdf"
    pdf_path = tmp_fixtures_dir / doc_id
    create_dummy_pdf_file(pdf_path, num_pages=1, size=(100,150)) # W, H for PDF page
    doc_ctx = DocumentContext(document_id=doc_id, source_path_or_id=str(pdf_path))

    result_ctx = image_loader_instance.process(doc_ctx, str(pdf_path))

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 1
    page = result_ctx.pages[0]
    assert page.original_image is not None
    assert isinstance(page.original_image, np.ndarray)
    # PyMuPDF renders at DPI. Expected H, W for 150 DPI: (150/72 * 150, 150/72 * 100) -> (312, 208)
    # Default DPI for ImageLoader is 300, but fixture sets to 150 for this test.
    # Page size (100,150) at 150 DPI: Height = 150 * 150/72 = 312.5 -> 312 or 313
    # Width = 100 * 150/72 = 208.33 -> 208
    # Shape is H, W, C. So (312 or 313, 208, 3)
    assert abs(page.original_image.shape[0] - (150 * 150 / 72)) < 2 # Allow for rounding
    assert abs(page.original_image.shape[1] - (100 * 150 / 72)) < 2
    assert page.page_number == 0


def test_load_pdf_multi_page(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_id = "test_multi.pdf"
    pdf_path = tmp_fixtures_dir / doc_id
    create_dummy_pdf_file(pdf_path, num_pages=3)
    doc_ctx = DocumentContext(document_id=doc_id, source_path_or_id=str(pdf_path))

    result_ctx = image_loader_instance.process(doc_ctx, str(pdf_path))

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 3
    for i, page in enumerate(result_ctx.pages):
        assert page.original_image is not None
        assert page.page_number == i

def test_load_image_from_bytes(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    img_path = tmp_fixtures_dir / "bytes_test.png"
    create_dummy_image_file(img_path, "PNG", size=(50,50))

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    doc_ctx = DocumentContext(document_id="test_img_bytes", source_path_or_id="bytes_input")
    result_ctx = image_loader_instance.process(doc_ctx, img_bytes)

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 1
    assert result_ctx.pages[0].original_image is not None
    assert result_ctx.pages[0].original_image.shape == (50, 50, 3)

def test_load_pdf_from_bytes(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    pdf_path = tmp_fixtures_dir / "bytes_test.pdf"
    create_dummy_pdf_file(pdf_path, num_pages=2)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    doc_ctx = DocumentContext(document_id="test_pdf_bytes", source_path_or_id="bytes_input_pdf")
    result_ctx = image_loader_instance.process(doc_ctx, pdf_bytes)

    assert not result_ctx.document_errors
    assert len(result_ctx.pages) == 2
    assert result_ctx.pages[0].original_image is not None
    assert result_ctx.pages[1].original_image is not None


def test_load_file_not_found(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_id = "non_existent.png"
    non_existent_path = str(tmp_fixtures_dir / doc_id)
    doc_ctx = DocumentContext(document_id=doc_id, source_path_or_id=non_existent_path)

    result_ctx = image_loader_instance.process(doc_ctx, non_existent_path)

    assert len(result_ctx.document_errors) == 1
    assert "File not found" in result_ctx.document_errors[0]
    assert result_ctx.overall_status == "loading_failed"

def test_load_corrupted_image_file(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_id = "corrupted.png"
    corrupted_file_path = tmp_fixtures_dir / doc_id
    with open(corrupted_file_path, "w") as f:
        f.write("This is not an image")
    doc_ctx = DocumentContext(document_id=doc_id, source_path_or_id=str(corrupted_file_path))

    result_ctx = image_loader_instance.process(doc_ctx, str(corrupted_file_path))

    assert len(result_ctx.document_errors) >= 1 # PyMuPDF might also try and fail
    assert any("Cannot identify image file" in err for err in result_ctx.document_errors) or \
           any("Unsupported file extension" in err for err in result_ctx.document_errors) # If suffix is .txt for example
    assert result_ctx.overall_status == "loading_failed"


def test_load_invalid_pdf_file(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_id = "invalid.pdf"
    invalid_pdf_path = tmp_fixtures_dir / doc_id
    with open(invalid_pdf_path, "w") as f:
        f.write("This is not a PDF")
    doc_ctx = DocumentContext(document_id=doc_id, source_path_or_id=str(invalid_pdf_path))

    result_ctx = image_loader_instance.process(doc_ctx, str(invalid_pdf_path))

    assert len(result_ctx.document_errors) == 1
    assert "PyMuPDF (Fitz) error" in result_ctx.document_errors[0].lower() or \
           "unsupported file extension" in result_ctx.document_errors[0].lower() # if not .pdf
    assert result_ctx.overall_status == "loading_failed"


def test_unsupported_source_type(image_loader_instance: ImageLoader):
    doc_ctx = DocumentContext(document_id="test_unsupported_type", source_path_or_id="unsupported")

    result_ctx = image_loader_instance.process(doc_ctx, 12345) # type: ignore

    assert len(result_ctx.document_errors) == 1
    assert "Unsupported source type" in result_ctx.document_errors[0]
    assert result_ctx.overall_status == "loading_failed"

def test_unsupported_file_extension(image_loader_instance: ImageLoader, tmp_fixtures_dir: Path):
    doc_id = "test.txt"
    txt_path = tmp_fixtures_dir / doc_id
    txt_path.write_text("This is a text file.")
    doc_ctx = DocumentContext(document_id=doc_id, source_path_or_id=str(txt_path))

    result_ctx = image_loader_instance.process(doc_ctx, str(txt_path))
    assert len(result_ctx.document_errors) == 1
    assert "Unsupported file extension: .txt" in result_ctx.document_errors[0]
    assert result_ctx.overall_status == "loading_failed"

def test_default_dpi_validation_in_imageloader():
    with pytest.raises(OCRXConfigurationError, match="'default_dpi' must be a positive integer"):
        ImageLoader(module_id="bad_dpi_loader", config={"default_dpi": "not_an_int"})
    with pytest.raises(OCRXConfigurationError, match="'default_dpi' must be a positive integer"):
        ImageLoader(module_id="bad_dpi_loader_2", config={"default_dpi": 0})

    # Should not raise
    loader = ImageLoader(module_id="good_dpi_loader", config={"default_dpi": 200})
    assert loader.default_dpi == 200
    loader_default = ImageLoader(module_id="default_dpi_loader", config={}) # Uses class default 300
    assert loader_default.default_dpi == 300
