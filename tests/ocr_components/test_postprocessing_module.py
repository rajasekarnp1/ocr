import pytest
import logging
from typing import Dict, Any, List

# Add project root to sys.path to allow direct imports from ocr_components
# This is often handled by test runners or conftest.py, but included here for explicitness if running file directly
import sys
import os
# Calculate the path to the project root (assuming tests/ocr_components/ is two levels down from root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocr_components.postprocessing_module import OCRPostprocessor


@pytest.fixture
def mock_logger():
    """Fixture to create a mock logger."""
    return logging.getLogger("mock_test_logger")

@pytest.fixture
def postprocessor(mock_logger):
    """Fixture to create a default OCRPostprocessor instance with no rules."""
    return OCRPostprocessor(config=None, logger=mock_logger)


# --- Initialization Tests ---

def test_postprocessor_init_no_config(mock_logger):
    """Test OCRPostprocessor initialization with no config."""
    pp = OCRPostprocessor(config=None, logger=mock_logger)
    assert pp.custom_rules == []
    assert pp.config == {}

def test_postprocessor_init_empty_config(mock_logger):
    """Test OCRPostprocessor initialization with an empty config."""
    pp = OCRPostprocessor(config={}, logger=mock_logger)
    assert pp.custom_rules == []
    assert pp.config == {}

def test_postprocessor_init_with_valid_rules(mock_logger):
    """Test OCRPostprocessor initialization with valid custom rules."""
    config = {
        "postprocessing": {
            "custom_text_replacements": [
                ["hullo", "hello"],
                ["wørld", "world"]
            ]
        }
    }
    pp = OCRPostprocessor(config=config, logger=mock_logger)
    assert pp.custom_rules == [("hullo", "hello"), ("wørld", "world")]

def test_postprocessor_init_with_malformed_rules_not_list(mock_logger, caplog):
    """Test init with custom_text_replacements not being a list."""
    config = {"postprocessing": {"custom_text_replacements": "not-a-list"}}
    pp = OCRPostprocessor(config=config, logger=mock_logger)
    assert pp.custom_rules == []
    assert any("should be a list" in record.message for record in caplog.records if record.levelno == logging.WARNING)

def test_postprocessor_init_with_malformed_rules_item_not_list(mock_logger, caplog):
    """Test init with an item in custom_text_replacements not being a list."""
    config = {"postprocessing": {"custom_text_replacements": ["not-a-list-pair"]}}
    pp = OCRPostprocessor(config=config, logger=mock_logger)
    assert pp.custom_rules == []
    assert any("Skipping invalid custom replacement rule" in record.message for record in caplog.records if record.levelno == logging.WARNING)

def test_postprocessor_init_with_malformed_rules_item_wrong_length(mock_logger, caplog):
    """Test init with a rule list not having exactly two elements."""
    config = {"postprocessing": {"custom_text_replacements": [["find", "replace", "extra"]]}}
    pp = OCRPostprocessor(config=config, logger=mock_logger)
    assert pp.custom_rules == []
    assert any("Skipping invalid custom replacement rule" in record.message for record in caplog.records if record.levelno == logging.WARNING)

def test_postprocessor_init_with_malformed_rules_item_not_strings(mock_logger, caplog):
    """Test init with a rule list containing non-string elements."""
    config = {"postprocessing": {"custom_text_replacements": [["find", 123]]}}
    pp = OCRPostprocessor(config=config, logger=mock_logger)
    assert pp.custom_rules == []
    assert any("Skipping invalid custom replacement rule" in record.message for record in caplog.records if record.levelno == logging.WARNING)


# --- _clean_whitespace Tests ---

@pytest.mark.parametrize("input_text, expected_text", [
    ("  Hello World  ", "Hello World"),
    ("\tHello\tWorld\t", "Hello World"),
    ("Hello \t World", "Hello World"),
    ("Hello   World", "Hello World"),
    ("Hello\nWorld", "Hello\nWorld"),
    ("Hello\n\nWorld", "Hello\nWorld"), # Default is single newline max
    ("Hello\n\n\nWorld", "Hello\nWorld"),
    ("Hello\r\nWorld", "Hello\nWorld"),
    ("Hello\rWorld", "Hello\nWorld"),
    ("  \nHello\nWorld\n  ", "Hello\nWorld"),
    ("", ""),
    ("NoIssuesHere", "NoIssuesHere"),
    ("   ", ""), # Only whitespace, should become empty
    ("\n\n   \n\n", ""), # Only whitespace and newlines
    ("Line1\n  Line2  \nLine3", "Line1\n Line2 \nLine3"), # Spaces within lines preserved unless multiple
])
def test_clean_whitespace_various_cases(postprocessor, input_text, expected_text):
    assert postprocessor._clean_whitespace(input_text) == expected_text

def test_clean_whitespace_non_string_input(postprocessor, caplog):
    """Test _clean_whitespace with non-string input."""
    assert postprocessor._clean_whitespace(123) == 123
    assert any("expected string input" in record.message for record in caplog.records if record.levelno == logging.WARNING)


# --- _apply_custom_rules Tests ---

def test_apply_custom_rules_no_rules(postprocessor):
    """Test _apply_custom_rules when no rules are defined."""
    postprocessor.custom_rules = [] # Ensure no rules
    text = "Some text here."
    assert postprocessor._apply_custom_rules(text) == text

def test_apply_custom_rules_with_rules(mock_logger):
    """Test _apply_custom_rules with defined rules."""
    config = {
        "postprocessing": {
            "custom_text_replacements": [
                ["OCR err0r", "OCR error"],
                ["123", "onetwothree"],
                ["ex ample", "example"], # Test rule with space
                ["sensitivE", "sensitive"] # Case-sensitive test
            ]
        }
    }
    pp_with_rules = OCRPostprocessor(config=config, logger=mock_logger)

    assert pp_with_rules._apply_custom_rules("This is an OCR err0r.") == "This is an OCR error."
    assert pp_with_rules._apply_custom_rules("Test 123 now.") == "Test onetwothree now."
    assert pp_with_rules._apply_custom_rules("An ex ample text.") == "An example text."
    assert pp_with_rules._apply_custom_rules("This is sensitivE.") == "This is sensitive."
    assert pp_with_rules._apply_custom_rules("This is SENSITIVE.") == "This is SENSITIVE." # No change for different case
    assert pp_with_rules._apply_custom_rules("No matching rules here.") == "No matching rules here."
    assert pp_with_rules._apply_custom_rules("") == "" # Empty string

def test_apply_custom_rules_non_string_input(postprocessor, caplog):
    """Test _apply_custom_rules with non-string input."""
    # Ensure some rules exist to try to apply them
    postprocessor.custom_rules = [("a", "b")]
    assert postprocessor._apply_custom_rules(None) is None
    assert any("expected string input" in record.message for record in caplog.records if record.levelno == logging.WARNING)
    caplog.clear()
    assert postprocessor._apply_custom_rules([1,2]) == [1,2]
    assert any("expected string input" in record.message for record in caplog.records if record.levelno == logging.WARNING)


# --- process_output Tests ---

def test_process_output_only_whitespace_cleaning(postprocessor):
    """Test process_output with only whitespace cleaning (no custom rules)."""
    postprocessor.custom_rules = [] # Explicitly set no rules
    ocr_result = {"text": "  extra \t\n spaces  \n\n here  ", "segments": [], "confidence": 0.9}
    expected_text = "extra spaces\nhere" # After strip, multi-space, multi-newline

    processed_result = postprocessor.process_output(ocr_result.copy()) # Pass a copy
    assert processed_result['text'] == expected_text
    assert processed_result['segments'] == ocr_result['segments'] # Segments should be untouched

def test_process_output_with_custom_rules(mock_logger):
    """Test process_output with both whitespace cleaning and custom rules."""
    config = {
        "postprocessing": {
            "custom_text_replacements": [
                ["err0r", "error"],
                ["m0re", "more"]
            ]
        }
    }
    pp = OCRPostprocessor(config=config, logger=mock_logger)
    ocr_result = {"text": "  Test err0r and   m0re \n spaces  ", "segments": []}
    # After whitespace: "Test err0r and m0re\nspaces"
    # After rules: "Test error and more\nspaces"
    expected_text = "Test error and more\nspaces"

    processed_result = pp.process_output(ocr_result.copy())
    assert processed_result['text'] == expected_text

def test_process_output_no_text_field(postprocessor, caplog):
    """Test process_output when 'text' field is missing."""
    ocr_result_no_text = {"segments": [], "confidence": 0.7}
    processed_result = postprocessor.process_output(ocr_result_no_text.copy())
    assert processed_result == ocr_result_no_text # Should return original dict
    assert any("'text' field in ocr_result is not a string or is missing" in record.message for record in caplog.records if record.levelno == logging.WARNING)

def test_process_output_text_field_not_string(postprocessor, caplog):
    """Test process_output when 'text' field is not a string."""
    ocr_result_non_string_text = {"text": 12345, "segments": []}
    processed_result = postprocessor.process_output(ocr_result_non_string_text.copy())
    assert processed_result == ocr_result_non_string_text # Should return original dict
    assert any("'text' field in ocr_result is not a string or is missing" in record.message for record in caplog.records if record.levelno == logging.WARNING)

def test_process_output_empty_text(postprocessor):
    """Test process_output with an empty 'text' field."""
    ocr_result_empty_text = {"text": "", "segments": []}
    processed_result = postprocessor.process_output(ocr_result_empty_text.copy())
    assert processed_result['text'] == ""

def test_process_output_input_not_dict(postprocessor, caplog):
    """Test process_output when the input itself is not a dictionary."""
    # Pyright/Pylance will warn about type, which is intended for this test
    processed_result = postprocessor.process_output("not a dictionary") # type: ignore
    assert processed_result == "not a dictionary"
    assert any("process_output expected a dictionary" in record.message for record in caplog.records if record.levelno == logging.ERROR)

def test_process_output_original_dict_unmodified(postprocessor):
    """Test that the original ocr_result dictionary is not modified if a copy is made."""
    ocr_result = {"text": "  test  "}
    original_text = ocr_result["text"] # Keep a reference

    postprocessor.process_output(ocr_result) # OCRPostprocessor makes a copy internally

    assert ocr_result["text"] == original_text # Verify original is unchanged
```
