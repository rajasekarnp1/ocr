import pytest
import json
import csv
from pathlib import Path
import logging

from ocrx.modules.postprocessor import AdvancedPostprocessor
from ocrx.core.data_objects import PageContext, TextRegion, MainCandidate, RecognitionResult
from ocrx.core.exceptions import OCRXConfigurationError

# --- Fixtures ---
@pytest.fixture
def default_postprocessor() -> AdvancedPostprocessor:
    return AdvancedPostprocessor(module_id="test_postproc", config={})

@pytest.fixture
def sample_page_ctx_for_postprocessing() -> PageContext:
    page_ctx = PageContext(page_number=0)

    # Region 1: Has a consensus candidate
    region1_consensus = MainCandidate(text="This is teh original text.", confidence=0.9)
    region1 = TextRegion(region_id="r1", bounding_box=(0,0,10,10), consensus_candidate=region1_consensus)

    # Region 2: Only has raw OCR results
    raw_res1 = RecognitionResult(text="An apliccation for testing.", confidence=0.8, engine_id="engineA")
    raw_res2 = RecognitionResult(text="An aplication for testing.", confidence=0.7, engine_id="engineB") # Slightly worse
    region2 = TextRegion(region_id="r2", bounding_box=(10,10,20,20), raw_ocr_results=[raw_res1, raw_res2])

    # Region 3: Has nothing to postprocess
    region3 = TextRegion(region_id="r3", bounding_box=(20,20,30,30))

    page_ctx.layout_regions = [region1, region2, region3]
    return page_ctx

@pytest.fixture
def correction_dict_file_json(tmp_path: Path) -> Path:
    dict_content = {"teh": "the", "aplication": "application", "tsting": "testing"}
    file_path = tmp_path / "corrections.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dict_content, f)
    return file_path

@pytest.fixture
def correction_dict_file_csv(tmp_path: Path) -> Path:
    # misspelled,corrected
    dict_content = [
        ("teh", "the"),
        ("aplication", "application"),
        ("wrld", "world")
    ]
    file_path = tmp_path / "corrections.csv"
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dict_content)
    return file_path

# --- Test Cases ---

def test_postprocessor_init_default(default_postprocessor: AdvancedPostprocessor):
    assert default_postprocessor.is_enabled()
    assert default_postprocessor.config["dictionary_path"] is None
    assert default_postprocessor.config["s2s_model_enabled"] is False
    assert not default_postprocessor.correction_dict # Empty

def test_postprocessor_validate_config_errors():
    with pytest.raises(OCRXConfigurationError, match="'dictionary_path' must be a string path"):
        AdvancedPostprocessor(module_id="err_pp", config={"dictionary_path": 123})

def test_initialize_resources_no_dict_path(default_postprocessor: AdvancedPostprocessor, caplog):
    with caplog.at_level(logging.INFO):
        default_postprocessor._initialize_resources() # Should run without error
    assert not default_postprocessor.correction_dict
    # No specific log for not having a dict, that's fine.

def test_initialize_resources_dict_json_valid(correction_dict_file_json: Path):
    config = {"dictionary_path": str(correction_dict_file_json)}
    postprocessor = AdvancedPostprocessor("pp_json", config) # _initialize_resources called by __init__
    assert len(postprocessor.correction_dict) == 3
    assert postprocessor.correction_dict["teh"] == "the"

def test_initialize_resources_dict_csv_valid(correction_dict_file_csv: Path):
    config = {"dictionary_path": str(correction_dict_file_csv)}
    postprocessor = AdvancedPostprocessor("pp_csv", config)
    assert len(postprocessor.correction_dict) == 3
    assert postprocessor.correction_dict["wrld"] == "world"

def test_initialize_resources_dict_not_found(caplog):
    config = {"dictionary_path": "non_existent_dict.json"}
    with caplog.at_level(logging.WARNING):
        postprocessor = AdvancedPostprocessor("pp_notfound", config)
    assert not postprocessor.correction_dict
    assert any("Correction dictionary not found" in record.message for record in caplog.records)

def test_initialize_resources_s2s_model_placeholder(caplog):
    config = {"s2s_model_enabled": True}
    with caplog.at_level(logging.INFO):
        AdvancedPostprocessor("pp_s2s", config)
    assert any("S2S model correction is enabled in config, but it's a placeholder" in record.message for record in caplog.records)

def test_apply_dictionary_correction(correction_dict_file_json: Path):
    postprocessor = AdvancedPostprocessor("pp_apply", {"dictionary_path": str(correction_dict_file_json)})

    assert postprocessor._apply_dictionary_correction("This is teh test.") == "This is the test."
    assert postprocessor._apply_dictionary_correction("Fix this apliccation.") == "Fix this application."
    assert postprocessor._apply_dictionary_correction("No corrections here.") == "No corrections here."
    assert postprocessor._apply_dictionary_correction("teh teh aplication") == "the the application"

    # Test with no dict loaded
    postprocessor_no_dict = AdvancedPostprocessor("pp_no_dict_apply", {})
    assert postprocessor_no_dict._apply_dictionary_correction("Some teh text") == "Some teh text"


def test_process_page_corrections_and_candidate_creation(sample_page_ctx_for_postprocessing: PageContext, correction_dict_file_json: Path):
    config = {"dictionary_path": str(correction_dict_file_json)}
    postprocessor = AdvancedPostprocessor("pp_process", config)

    postprocessor.process(sample_page_ctx_for_postprocessing)

    # Region 1: Had consensus, should be postprocessed
    region1 = sample_page_ctx_for_postprocessing.layout_regions[0]
    assert region1.postprocessed_candidate is not None
    assert region1.postprocessed_candidate.text == "This is the original text."
    assert region1.postprocessed_candidate.confidence == region1.consensus_candidate.confidence # Confidence preserved
    assert region1.postprocessed_candidate.source_engines == region1.consensus_candidate.source_engines

    # Region 2: Had raw results, first one used and postprocessed
    region2 = sample_page_ctx_for_postprocessing.layout_regions[1]
    assert region2.postprocessed_candidate is not None
    assert region2.postprocessed_candidate.text == "An application for testing."
    assert region2.postprocessed_candidate.confidence == region2.raw_ocr_results[0].confidence
    assert region2.postprocessed_candidate.source_engines == [region2.raw_ocr_results[0].engine_id]

    # Region 3: Had no input, should have no postprocessed candidate
    region3 = sample_page_ctx_for_postprocessing.layout_regions[2]
    assert region3.postprocessed_candidate is None


def test_process_s2s_placeholder_logging(sample_page_ctx_for_postprocessing: PageContext, caplog):
    config = {"s2s_model_enabled": True} # No dictionary
    postprocessor = AdvancedPostprocessor("pp_process_s2s", config)

    with caplog.at_level(logging.INFO):
        postprocessor.process(sample_page_ctx_for_postprocessing)

    # Check S2S placeholder log for regions that had text
    # Region 1 and Region 2 should trigger this log.
    s2s_log_count = sum(1 for record in caplog.records if "S2S model correction placeholder" in record.message)
    assert s2s_log_count >= 2 # One for region1, one for region2

    # Text should pass through unchanged as S2S is a placeholder
    assert sample_page_ctx_for_postprocessing.layout_regions[0].postprocessed_candidate.text == "This is teh original text."


def test_process_disabled_module(default_postprocessor: AdvancedPostprocessor, sample_page_ctx_for_postprocessing: PageContext):
    default_postprocessor.config["enabled"] = False
    # Create copies to avoid modifying fixture for other tests if PageContext becomes mutable in more ways
    original_text_r1 = sample_page_ctx_for_postprocessing.layout_regions[0].consensus_candidate.text # type: ignore

    default_postprocessor.process(sample_page_ctx_for_postprocessing)

    assert sample_page_ctx_for_postprocessing.layout_regions[0].postprocessed_candidate is None
    assert sample_page_ctx_for_postprocessing.layout_regions[0].consensus_candidate.text == original_text_r1 # type: ignore

def test_process_no_regions_on_page(default_postprocessor: AdvancedPostprocessor):
    page_ctx_empty = PageContext(page_number=1, layout_regions=[])
    default_postprocessor.process(page_ctx_empty) # Should run without error
    assert not page_ctx_empty.errors # No errors should be added for this case.
