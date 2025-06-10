import pytest
from pathlib import Path
import xml.etree.ElementTree as ET # For hOCR validation
import html

from ocrx.modules.output_generator import OutputGenerator
from ocrx.core.data_objects import DocumentContext, PageContext, TextRegion, MainCandidate, RecognitionResult
from ocrx.core.exceptions import OCRXConfigurationError, OCRXProcessingError

# --- Fixtures ---
@pytest.fixture
def default_output_generator(tmp_path: Path) -> OutputGenerator:
    # Configure output to a temporary directory for tests
    return OutputGenerator(module_id="test_output_gen", config={"output_dir": str(tmp_path)})

@pytest.fixture
def sample_doc_context_for_output() -> DocumentContext:
    doc_ctx = DocumentContext(document_id="doc_test_123", source_path_or_id="/path/to/test_doc.pdf")

    # Page 1
    page1 = PageContext(page_number=0, original_image=np.zeros((100,100,3), dtype=np.uint8)) # Dummy image for dims

    # Region 1 on Page 1
    r1p1_text = "Hello World."
    r1p1_cand = MainCandidate(text=r1p1_text, confidence=0.9)
    r1p1 = TextRegion(region_id="p0_r1", bounding_box=(10, 10, 100, 30), postprocessed_candidate=r1p1_cand)

    # Region 2 on Page 1 (using consensus candidate as fallback)
    r2p1_text = "This is a test."
    r2p1_cons = MainCandidate(text=r2p1_text, confidence=0.8)
    r2p1 = TextRegion(region_id="p0_r2", bounding_box=(10, 40, 150, 60), consensus_candidate=r2p1_cons)

    page1.layout_regions = [r1p1, r2p1]
    doc_ctx.pages.append(page1)

    # Page 2 (empty, to test handling of empty pages)
    page2 = PageContext(page_number=1, original_image=np.zeros((50,50,3), dtype=np.uint8))
    # Region 3 on Page 2 (using raw results as fallback)
    r3p2_text_seg1 = "Raw segment 1"
    r3p2_text_seg2 = "Raw segment 2"
    r3p2_raw1 = RecognitionResult(text=r3p2_text_seg1, confidence=0.7, engine_id="e1", char_boxes=[(5,5,50,15)])
    r3p2_raw2 = RecognitionResult(text=r3p2_text_seg2, confidence=0.7, engine_id="e1", char_boxes=[(5,15,50,25)])
    r3p2 = TextRegion(region_id="p1_r1", bounding_box=(5,5,50,25), raw_ocr_results=[r3p2_raw1, r3p2_raw2])
    page2.layout_regions = [r3p2]
    doc_ctx.pages.append(page2)

    return doc_ctx

# --- Test Cases ---

def test_output_generator_init_default(tmp_path: Path):
    gen = OutputGenerator("default_gen", {"output_dir": str(tmp_path)})
    assert gen.is_enabled()
    assert gen.config["output_dir"] == str(tmp_path)
    assert gen.config["formats"] == ["txt", "hocr"] # Default formats

def test_output_generator_validate_config_errors():
    with pytest.raises(OCRXConfigurationError, match="'output_dir' must be a string path"):
        OutputGenerator("err_gen1", {"output_dir": 123})
    with pytest.raises(OCRXConfigurationError, match="'formats' must be a list of supported strings"):
        OutputGenerator("err_gen2", {"formats": "not_a_list"})
    with pytest.raises(OCRXConfigurationError, match="'formats' must be a list of supported strings"):
        OutputGenerator("err_gen3", {"formats": ["txt", "unsupported_format"]})


def test_txt_output_generation(default_output_generator: OutputGenerator, sample_doc_context_for_output: DocumentContext, tmp_path: Path):
    default_output_generator.config["formats"] = ["txt"] # Only generate TXT
    default_output_generator.process(sample_doc_context_for_output)

    # Page 1 assertions
    expected_txt_path_p0 = tmp_path / f"{sample_doc_context_for_output.document_id}_page_0.txt"
    assert expected_txt_path_p0.exists()
    content_p0 = expected_txt_path_p0.read_text(encoding="utf-8")
    assert "Hello World." in content_p0
    assert "This is a test." in content_p0
    # Check if regions are separated by newline (current simple behavior)
    assert content_p0 == "Hello World.\nThis is a test."

    # Page 2 assertions
    expected_txt_path_p1 = tmp_path / f"{sample_doc_context_for_output.document_id}_page_1.txt"
    assert expected_txt_path_p1.exists()
    content_p1 = expected_txt_path_p1.read_text(encoding="utf-8")
    assert "Raw segment 1\nRaw segment 2" in content_p1 # Based on current _get_text_for_region fallback for multiple raw


def test_hocr_output_generation(default_output_generator: OutputGenerator, sample_doc_context_for_output: DocumentContext, tmp_path: Path):
    default_output_generator.config["formats"] = ["hocr"] # Only generate hOCR
    default_output_generator.process(sample_doc_context_for_output)

    # Page 1 assertions
    expected_hocr_path_p0 = tmp_path / f"{sample_doc_context_for_output.document_id}_page_0.hocr"
    assert expected_hocr_path_p0.exists()

    try:
        tree = ET.parse(str(expected_hocr_path_p0))
        root = tree.getroot()
    except ET.ParseError as e:
        pytest.fail(f"hOCR file {expected_hocr_path_p0} is not valid XML: {e}")

    # Basic hOCR structure checks for Page 1
    assert root.tag == "{http://www.w3.org/1999/xhtml}html"
    body = root.find("{http://www.w3.org/1999/xhtml}body")
    assert body is not None
    page_div = body.find("{http://www.w3.org/1999/xhtml}div[@class='ocr_page']")
    assert page_div is not None
    assert page_div.get("id") == "page_0"

    page_dims = page_div.get("title", "").split("bbox ")[1].split(" ") if page_div.get("title") else []
    assert len(page_dims) == 4 and page_dims[2] == "100" and page_dims[3] == "100" # From dummy image H,W

    careas = page_div.findall("{http://www.w3.org/1999/xhtml}div[@class='ocr_carea']")
    assert len(careas) == 2 # Two regions on page 1

    # Check first carea (region1)
    carea1_lines = careas[0].findall("{http://www.w3.org/1999/xhtml}span[@class='ocr_line']")
    assert len(carea1_lines) == 1
    assert careas[0].get("title") == "bbox 10 10 100 30"
    assert careas[0].get("id") == "p0_r1_area"
    assert html.unescape(carea1_lines[0].text) == "Hello World." # type: ignore

    # Page 2 assertions
    expected_hocr_path_p1 = tmp_path / f"{sample_doc_context_for_output.document_id}_page_1.hocr"
    assert expected_hocr_path_p1.exists()
    tree_p1 = ET.parse(str(expected_hocr_path_p1))
    root_p1 = tree_p1.getroot()
    page_div_p1 = root_p1.find(".//{http://www.w3.org/1999/xhtml}div[@class='ocr_page']")
    assert page_div_p1 is not None
    careas_p1 = page_div_p1.findall("{http://www.w3.org/1999/xhtml}div[@class='ocr_carea']")
    assert len(careas_p1) == 1

    # Region on Page 2 had multiple raw results, check if they became multiple lines
    lines_p1_r1 = careas_p1[0].findall("{http://www.w3.org/1999/xhtml}span[@class='ocr_line']")
    assert len(lines_p1_r1) == 2 # Expecting two lines from the two raw results
    assert html.unescape(lines_p1_r1[0].text) == "Raw segment 1" # type: ignore
    assert html.unescape(lines_p1_r1[1].text) == "Raw segment 2" # type: ignore
    assert lines_p1_r1[0].get("title") == "bbox 5 5 50 15" # From raw_ocr_results[0].char_boxes[0]


def test_process_disabled_generator(default_output_generator: OutputGenerator, sample_doc_context_for_output: DocumentContext, tmp_path: Path):
    default_output_generator.config["enabled"] = False
    default_output_generator.process(sample_doc_context_for_output)

    # No files should be created
    expected_txt_path_p0 = tmp_path / f"{sample_doc_context_for_output.document_id}_page_0.txt"
    assert not expected_txt_path_p0.exists()

def test_output_dir_creation_failure(sample_doc_context_for_output: DocumentContext, tmp_path: Path):
    # Create a file where the output directory should be, to cause mkdir to fail
    invalid_output_dir = tmp_path / "output_conflict_file"
    invalid_output_dir.write_text("I am a file, not a directory.")

    generator = OutputGenerator("fail_gen", {"output_dir": str(invalid_output_dir)})

    with pytest.raises(OCRXProcessingError, match="Cannot create output directory"):
        generator.process(sample_doc_context_for_output)

def test_get_text_for_region_fallbacks():
    """Test the _get_text_for_region helper's fallback logic."""
    gen = OutputGenerator("test", {"output_dir": "test_output"}) # Not actually writing files

    # 1. Has postprocessed_candidate
    r1 = TextRegion("r1", (0,0,1,1), postprocessed_candidate=MainCandidate("Postprocessed", 0.9))
    assert gen._get_text_for_region(r1) == "Postprocessed"

    # 2. No postprocessed, has consensus
    r2 = TextRegion("r2", (0,0,1,1), consensus_candidate=MainCandidate("Consensus", 0.8))
    assert gen._get_text_for_region(r2) == "Consensus"

    # 3. No postprocessed or consensus, has raw_ocr_results
    raw1 = RecognitionResult("Raw1", 0.7)
    raw2 = RecognitionResult("Raw2", 0.6)
    r3 = TextRegion("r3", (0,0,1,1), raw_ocr_results=[raw1, raw2])
    assert gen._get_text_for_region(r3) == "Raw1 Raw2" # Concatenates

    # 4. Empty raw_ocr_results
    r4 = TextRegion("r4", (0,0,1,1), raw_ocr_results=[RecognitionResult("",0.0)])
    assert gen._get_text_for_region(r4) == ""

    # 5. No text candidates at all
    r5 = TextRegion("r5", (0,0,1,1))
    assert gen._get_text_for_region(r5) == ""
