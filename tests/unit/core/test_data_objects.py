import pytest
import numpy as np
from ocrx.core.data_objects import (
    TextRegion,
    RecognitionResult,
    MainCandidate,
    PageContext,
    DocumentContext
)

def test_text_region_instantiation_defaults():
    """Test TextRegion instantiation with minimal required fields and check defaults."""
    region_id = "region_1"
    bounding_box = (10, 20, 100, 50) # x_min, y_min, x_max, y_max

    region = TextRegion(region_id=region_id, bounding_box=bounding_box)

    assert region.region_id == region_id
    assert region.bounding_box == bounding_box
    assert region.image_crop is None
    assert region.region_type == "text_line"
    assert region.sequence_id == 0
    assert region.raw_ocr_results == []
    assert region.consensus_candidate is None
    assert region.postprocessed_candidate is None

def test_text_region_instantiation_custom_values():
    """Test TextRegion instantiation with all fields provided."""
    dummy_image_crop = np.array([[[0,0,0]]], dtype=np.uint8)
    rec_result = RecognitionResult(text="test", confidence=0.9)
    main_cand = MainCandidate(text="final", confidence=0.95)

    region = TextRegion(
        region_id="region_custom",
        image_crop=dummy_image_crop,
        bounding_box=(0, 0, 10, 10),
        region_type="paragraph",
        sequence_id=1,
        raw_ocr_results=[rec_result],
        consensus_candidate=main_cand,
        postprocessed_candidate=main_cand
    )

    assert np.array_equal(region.image_crop, dummy_image_crop)
    assert region.region_type == "paragraph"
    assert region.sequence_id == 1
    assert len(region.raw_ocr_results) == 1
    assert region.raw_ocr_results[0].text == "test"
    assert region.consensus_candidate is not None
    assert region.consensus_candidate.text == "final"

def test_recognition_result_instantiation_defaults():
    """Test RecognitionResult instantiation with minimal required fields and check defaults."""
    text = "Recognized text"
    confidence = 0.88

    result = RecognitionResult(text=text, confidence=confidence)

    assert result.text == text
    assert result.confidence == confidence
    assert result.char_confidences is None
    assert result.char_boxes is None
    assert result.word_confidences is None
    assert result.word_boxes is None
    assert result.engine_id is None

def test_recognition_result_instantiation_custom_values():
    """Test RecognitionResult instantiation with all fields provided."""
    result = RecognitionResult(
        text="More text",
        confidence=0.75,
        char_confidences=[0.9, 0.8, 0.7, 0.95],
        char_boxes=[(0,0,5,10), (5,0,10,10)],
        word_confidences=[0.8, 0.7],
        word_boxes=[(0,0,8,10), (9,0,15,10)],
        engine_id="engine_dummy_v2"
    )
    assert len(result.char_confidences) == 4
    assert len(result.char_boxes) == 2
    assert result.engine_id == "engine_dummy_v2"


def test_main_candidate_instantiation_defaults():
    """Test MainCandidate instantiation with minimal required fields and check defaults."""
    text = "Main candidate text"
    confidence = 0.92

    candidate = MainCandidate(text=text, confidence=confidence)

    assert candidate.text == text
    assert candidate.confidence == confidence
    assert candidate.source_engines == []

def test_main_candidate_instantiation_custom_values():
    """Test MainCandidate instantiation with all fields provided."""
    candidate = MainCandidate(
        text="Final choice",
        confidence=0.99,
        source_engines=["engine_A", "engine_B"]
    )
    assert len(candidate.source_engines) == 2
    assert "engine_A" in candidate.source_engines


def test_page_context_instantiation_defaults():
    """Test PageContext instantiation with minimal required fields and check defaults."""
    page_number = 1

    page_ctx = PageContext(page_number=page_number)

    assert page_ctx.page_number == page_number
    assert page_ctx.original_image is None
    assert page_ctx.preprocessed_image is None
    assert page_ctx.layout_regions == []
    assert page_ctx.errors == []
    assert page_ctx.processing_times == {}

def test_page_context_instantiation_custom_values():
    """Test PageContext instantiation with all fields provided."""
    dummy_image = np.array([[[255,255,255]]], dtype=np.uint8)
    text_reg = TextRegion(region_id="r1", bounding_box=(0,0,1,1))

    page_ctx = PageContext(
        page_number=5,
        original_image=dummy_image,
        preprocessed_image=dummy_image,
        layout_regions=[text_reg],
        errors=["Warning: Low resolution"],
        processing_times={"loading": 0.1, "preprocessing": 0.2}
    )
    assert np.array_equal(page_ctx.original_image, dummy_image)
    assert len(page_ctx.layout_regions) == 1
    assert "Warning: Low resolution" in page_ctx.errors
    assert page_ctx.processing_times["loading"] == 0.1


def test_document_context_instantiation_defaults():
    """Test DocumentContext instantiation with minimal required fields and check defaults."""
    doc_id = "doc001"
    source_path = "/path/to/doc001.tiff"

    doc_ctx = DocumentContext(document_id=doc_id, source_path_or_id=source_path)

    assert doc_ctx.document_id == doc_id
    assert doc_ctx.source_path_or_id == source_path
    assert doc_ctx.global_config == {}
    assert doc_ctx.pages == []
    assert doc_ctx.overall_status == "pending"
    assert doc_ctx.document_errors == []
    assert doc_ctx.total_processing_time is None

def test_document_context_instantiation_custom_values():
    """Test DocumentContext instantiation with all fields provided."""
    page1 = PageContext(page_number=0)
    doc_ctx = DocumentContext(
        document_id="doc002",
        source_path_or_id="id_002",
        global_config={"mode": "fast"},
        pages=[page1],
        overall_status="processing",
        document_errors=["Failed to load metadata"],
        total_processing_time=10.5
    )
    assert doc_ctx.global_config["mode"] == "fast"
    assert len(doc_ctx.pages) == 1
    assert doc_ctx.overall_status == "processing"
    assert "Failed to load metadata" in doc_ctx.document_errors
    assert doc_ctx.total_processing_time == 10.5

def test_numpy_array_in_data_objects():
    """Test that numpy arrays are correctly handled."""
    image_data = np.random.rand(100, 100, 3).astype(np.uint8)
    page_ctx = PageContext(page_number=1, original_image=image_data)
    assert np.array_equal(page_ctx.original_image, image_data)

    region = TextRegion(region_id="r1", bounding_box=(0,0,1,1), image_crop=image_data)
    assert np.array_equal(region.image_crop, image_data)

def test_data_objects_are_mutable():
    """Test that fields like lists and dicts are mutable and independent between instances."""
    doc1 = DocumentContext(document_id="d1", source_path_or_id="s1")
    doc2 = DocumentContext(document_id="d2", source_path_or_id="s2")

    doc1.document_errors.append("Error in doc1")
    doc1.global_config["key"] = "value_doc1"

    assert "Error in doc1" in doc1.document_errors
    assert "Error in doc1" not in doc2.document_errors
    assert doc2.document_errors == []

    assert doc1.global_config["key"] == "value_doc1"
    assert "key" not in doc2.global_config
    assert doc2.global_config == {}

    page1 = PageContext(page_number=0)
    page2 = PageContext(page_number=0) # Same page number, different instance
    page1.errors.append("Page1 error")

    assert "Page1 error" in page1.errors
    assert page2.errors == []
