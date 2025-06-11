import logging
import re
from typing import Optional, Dict, Any, List, Tuple

module_logger = logging.getLogger(__name__)

class OCRPostprocessor:
    """
    Handles post-processing of OCR results, including text cleaning and custom rule application.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the OCRPostprocessor.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary.
                Can contain a 'postprocessing' key, which in turn can have:
                - 'custom_text_replacements': A list of [find_str, replace_str] pairs.
                  e.g., {"postprocessing": {"custom_text_replacements": [["hte", "the"], ["l ", "1 "]]}}
            logger (Optional[logging.Logger]): Logger instance.
        """
        self.config = config if config is not None else {}
        self.logger = logger if logger is not None else module_logger

        # Load custom replacement rules from the 'postprocessing' sub-dictionary in config
        postprocessing_config = self.config.get('postprocessing', {})
        self.custom_rules: List[Tuple[str, str]] = []
        raw_rules = postprocessing_config.get('custom_text_replacements', [])

        if isinstance(raw_rules, list):
            for rule in raw_rules:
                if isinstance(rule, list) and len(rule) == 2 and all(isinstance(item, str) for item in rule):
                    self.custom_rules.append((rule[0], rule[1]))
                else:
                    self.logger.warning(f"Skipping invalid custom replacement rule: {rule}. Rule must be a list of two strings.")
        else:
            self.logger.warning(f"'custom_text_replacements' should be a list of [find, replace] pairs. Found: {type(raw_rules)}")

        self.logger.info(f"OCRPostprocessor initialized with {len(self.custom_rules)} custom replacement rules.")
        if self.custom_rules:
             self.logger.debug(f"Custom rules loaded: {self.custom_rules}")


    def _clean_whitespace(self, text: str) -> str:
        """
        Cleans whitespace in the given text.
        - Removes leading/trailing whitespace.
        - Replaces multiple consecutive spaces/tabs with a single space.
        - Normalizes newline characters (CRLF, CR to LF).
        - Replaces multiple blank lines with a single blank line.

        Args:
            text (str): The input text.

        Returns:
            str: The text with cleaned whitespace.
        """
        if not isinstance(text, str):
            self.logger.warning(f"_clean_whitespace expected string input, got {type(text)}. Returning as is.")
            return text

        self.logger.debug("Applying whitespace cleaning.")

        # 1. Normalize newline characters first
        processed_text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 2. Remove leading/trailing whitespace (including newlines if they are at ends)
        processed_text = processed_text.strip()

        # 3. Replace multiple spaces/tabs with a single space
        processed_text = re.sub(r'[ \t]+', ' ', processed_text)

        # 4. Replace multiple consecutive newlines with a single newline
        # This effectively removes completely blank lines if they result from just newlines.
        # If the goal is to ensure at most one blank line (two newlines), use r'\n{3,}' -> '\n\n'
        processed_text = re.sub(r'\n{2,}', '\n', processed_text) # Results in no purely blank lines
        # To allow one blank line (max two consecutive newlines):
        # processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)


        return processed_text

    def _apply_custom_rules(self, text: str) -> str:
        """
        Applies custom find-and-replace rules to the text.
        Rules are applied in the order they are defined.

        Args:
            text (str): The input text.

        Returns:
            str: The text after applying custom rules.
        """
        if not isinstance(text, str):
            self.logger.warning(f"_apply_custom_rules expected string input, got {type(text)}. Returning as is.")
            return text

        if not self.custom_rules:
            return text

        self.logger.debug(f"Applying {len(self.custom_rules)} custom replacement rules.")
        processed_text = text
        for find_str, replace_str in self.custom_rules:
            processed_text = processed_text.replace(find_str, replace_str)

        return processed_text

    def process_output(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to post-process an OCR result dictionary.
        Currently processes the main 'text' field.

        Args:
            ocr_result (Dict[str, Any]): The standardized OCR result dictionary.
                                         Expected to have a 'text' field.

        Returns:
            Dict[str, Any]: The modified OCR result dictionary with processed text.
        """
        if not isinstance(ocr_result, dict):
            self.logger.error(f"process_output expected a dictionary, got {type(ocr_result)}. Returning as is.")
            return ocr_result

        original_text = ocr_result.get('text')
        if not isinstance(original_text, str):
            self.logger.warning(f"'text' field in ocr_result is not a string or is missing. Skipping post-processing. Found: {type(original_text)}")
            return ocr_result

        self.logger.info("Starting OCR output post-processing.")

        # Step 1: Clean whitespace
        processed_text = self._clean_whitespace(original_text)
        if processed_text != original_text:
            self.logger.debug("Whitespace cleaning applied.")
        else:
            self.logger.debug("No changes from whitespace cleaning.")

        # Step 2: Apply custom rules
        text_after_rules = self._apply_custom_rules(processed_text)
        if text_after_rules != processed_text:
            self.logger.debug("Custom rules applied.")
        else:
            self.logger.debug("No changes from custom rules application.")

        processed_text = text_after_rules

        # Update the 'text' field in the result
        modified_ocr_result = ocr_result.copy() # Avoid modifying original dict directly if passed by ref elsewhere
        modified_ocr_result['text'] = processed_text

        self.logger.info("OCR output post-processing completed.")

        # For MVP, segment-level processing is skipped.
        # If implemented, it would iterate ocr_result.get('segments', [])
        # and apply similar cleaning to each segment['text'].

        return modified_ocr_result


if __name__ == '__main__':
    # Basic logging setup for the __main__ block
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger("OCRPostprocessorTest")
    main_logger.info("Running OCRPostprocessor self-test...")

    # --- Test Case 1: Whitespace Cleaning Only ---
    sample_ocr_result_1 = {
        "text": "  Hello \t World!  \n\n\nThis is a   test. \r\nNew line.\rAnother.  ",
        "segments": [], "confidence": 0.9, "engine_name": "test_engine"
    }
    main_logger.info(f"\nOriginal Text 1:\n'''{sample_ocr_result_1['text']}'''")

    postprocessor_no_rules = OCRPostprocessor()
    processed_result_1 = postprocessor_no_rules.process_output(sample_ocr_result_1)
    main_logger.info(f"Processed Text 1 (Whitespace only):\n'''{processed_result_1['text']}'''")

    expected_text_1 = "Hello World!\nThis is a test.\nNew line.\nAnother."
    assert processed_result_1['text'] == expected_text_1, f"Test Case 1 Failed. Expected:\n'''{expected_text_1}'''\nGot:\n'''{processed_result_1['text']}'''"
    main_logger.info("Test Case 1 Passed.")

    # --- Test Case 2: With Custom Rules ---
    config_with_rules = {
        "postprocessing": {
            "custom_text_replacements": [
                ["hte", "the"],    # Common OCR error
                ["wrold", "world"],  # Misspelling
                ["  ", " "],       # Redundant space (should be handled by whitespace, but good test)
                ["!.", "!"],       # Punctuation correction
                ["( ", "("],       # Space after parenthesis
                [" )", ")"]        # Space before parenthesis
            ]
        }
    }
    sample_ocr_result_2 = {
        "text": "  Ths is hte wrold ( example !. ).  \n\nExtra   spaces.  ",
        "segments": [], "confidence": 0.85, "engine_name": "test_engine"
    }
    main_logger.info(f"\nOriginal Text 2:\n'''{sample_ocr_result_2['text']}'''")

    postprocessor_with_rules = OCRPostprocessor(config=config_with_rules, logger=main_logger) # Pass main_logger
    processed_result_2 = postprocessor_with_rules.process_output(sample_ocr_result_2)
    main_logger.info(f"Processed Text 2 (Whitespace & Custom Rules):\n'''{processed_result_2['text']}'''")

    # Expected after whitespace: "Ths is hte wrold ( example !. ).\nExtra spaces."
    # Expected after rules: "Ths is the world (example!).\nExtra spaces."
    # Note: "Ths" is not in rules.
    expected_text_2 = "Ths is the world (example!).\nExtra spaces."
    assert processed_result_2['text'] == expected_text_2, f"Test Case 2 Failed. Expected:\n'''{expected_text_2}'''\nGot:\n'''{processed_result_2['text']}'''"
    main_logger.info("Test Case 2 Passed.")

    # --- Test Case 3: No Text Field or Invalid Input ---
    main_logger.info("\nTesting with invalid inputs...")
    invalid_input_1 = {"message": "Not an OCR result"}
    processed_invalid_1 = postprocessor_no_rules.process_output(invalid_input_1)
    assert processed_invalid_1 == invalid_input_1, "Test Case 3a (Missing 'text') Failed."
    main_logger.info("Test Case 3a (Missing 'text' field) Passed.")

    invalid_input_2 = "Just a string, not a dict"
    # Pyright/Pylance will complain about type here, which is good.
    processed_invalid_2 = postprocessor_no_rules.process_output(invalid_input_2) # type: ignore
    assert processed_invalid_2 == invalid_input_2, "Test Case 3b (Non-dict input) Failed."
    main_logger.info("Test Case 3b (Non-dict input) Passed.")

    invalid_input_3 = {"text": 12345} # Text is not a string
    processed_invalid_3 = postprocessor_no_rules.process_output(invalid_input_3)
    assert processed_invalid_3 == invalid_input_3, "Test Case 3c (Non-string 'text') Failed."
    main_logger.info("Test Case 3c (Non-string 'text' field) Passed.")


    # --- Test Case 4: Empty Text ---
    sample_ocr_result_4 = {"text": "   \n\t\n   "}
    main_logger.info(f"\nOriginal Text 4:\n'''{sample_ocr_result_4['text']}'''")
    processed_result_4 = postprocessor_no_rules.process_output(sample_ocr_result_4)
    main_logger.info(f"Processed Text 4 (Empty after cleaning):\n'''{processed_result_4['text']}'''")
    expected_text_4 = "" # strip() removes all if only whitespace
    assert processed_result_4['text'] == expected_text_4, f"Test Case 4 Failed. Expected: '''{expected_text_4}''', Got: '''{processed_result_4['text']}'''"
    main_logger.info("Test Case 4 (Empty text) Passed.")

    main_logger.info("\nOCRPostprocessor self-test finished.")
```
