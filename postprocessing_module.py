import logging
import os
import re

# Assuming custom_exceptions.py is in the same directory or PYTHONPATH
try:
    from custom_exceptions import OCRFileNotFoundError, OCRPipelineError
except ImportError:
    # Basic fallback if custom_exceptions is not found
    OCRFileNotFoundError = FileNotFoundError
    OCRPipelineError = RuntimeError # General pipeline error for text processing issues


DEFAULT_WHITELIST_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'()[]-"

class TextCleaner:
    def __init__(self, whitelist_chars: str | None = None):
        self.logger = logging.getLogger(__name__)
        if whitelist_chars is None:
            self.whitelist_chars = DEFAULT_WHITELIST_CHARS
            self.logger.info(f"TextCleaner initialized with default whitelist: '{DEFAULT_WHITELIST_CHARS}'")
        else:
            self.whitelist_chars = whitelist_chars
            self.logger.info(f"TextCleaner initialized with custom whitelist: '{whitelist_chars}'")

        # Create a set for efficient lookup
        self._whitelist_set = set(self.whitelist_chars)

    def clean_text(self, text_data: dict) -> dict:
        """
        Cleans the text string within text_data using the whitelist.
        text_data is expected to be a dictionary, e.g., {"text": "RawText...", "confidence": 0.9}.
        Returns a new dictionary with cleaned_text, original_text, and other metadata.
        """
        if not isinstance(text_data, dict):
            self.logger.error(f"Input text_data must be a dictionary, got {type(text_data)}.")
            # Return a dict indicating error or raise TypeError
            return {
                "cleaned_text": "",
                "original_text": f"Error: Input was not a dict, was {type(text_data)}",
                "confidence": 0.0,
                "error": "Invalid input type"
            }

        original_text = text_data.get('text', '')
        confidence = text_data.get('confidence', 0.0) # Preserve confidence
        other_metadata = {k: v for k, v in text_data.items() if k not in ['text', 'confidence']}


        self.logger.debug(f"Original text for cleaning: '{original_text[:100]}...'")

        cleaned_text_chars = [char for char in original_text if char in self._whitelist_set]
        cleaned_text = "".join(cleaned_text_chars)

        self.logger.info(f"Text cleaning complete. Original length: {len(original_text)}, Cleaned length: {len(cleaned_text)}")
        self.logger.debug(f"Cleaned text: '{cleaned_text[:100]}...'")

        result = {
            "cleaned_text": cleaned_text,
            "original_text": original_text,
            "confidence": confidence
        }
        result.update(other_metadata) # Add back any other metadata
        return result

# Keep PostprocessingModulePlaceholder for now if other parts of the project might use it,
# or if its __main__ block has useful examples. For this task, TextCleaner is primary.
class PostprocessingModulePlaceholder:
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        self.settings = settings # Settings might include whitelist for a more complex module
        self.logger.info(f"PostprocessingModulePlaceholder initialized with settings: {self.settings}")
        # Example: could internally use TextCleaner if settings indicate
        # default_wl = "abcdefghijklmnopqrstuvwxyz "
        # self.cleaner = TextCleaner(whitelist_chars=settings.get('whitelist', default_wl))


    def run_all(self, ocr_data):
        """
        Runs all postprocessing steps on the OCR data.
        ocr_data is expected to be a dictionary, possibly with a 'text' key.
        """
        if not isinstance(ocr_data, dict):
            self.logger.warning(f"Expected dict for ocr_data, got {type(ocr_data)}. Using empty string for text.")
            text_to_process = ""
            confidence = 0.0
        else:
            text_to_process = ocr_data.get('text', '')
            confidence = ocr_data.get('confidence', 0.0)

        self.logger.info(f"Postprocessing placeholder (run_all) on text: '{text_to_process[:50]}...'")

        # Example: Could use TextCleaner internally
        # cleaned_data = self.cleaner.clean_text({"text": text_to_process, "confidence": confidence})
        # final_text = f"PlaceholderProcessed_{cleaned_data['cleaned_text']}"
        final_text = f"FinalText_for_{text_to_process}" # Original placeholder behavior

        self.logger.debug(f"Postprocessing placeholder result: '{final_text[:50]}...'")
        # To align with TextCleaner's output format for orchestrator:
        return {
            "cleaned_text": final_text, # This is not really "cleaned" by whitelist in placeholder
            "original_text": text_to_process,
            "confidence": confidence,
            "placeholder_info": "Processed by PostprocessingModulePlaceholder"
        }


if __name__ == '__main__':
    # Basic logging setup for standalone execution
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Basic logging setup for standalone execution
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Test TextCleaner ---
    logging.info("\n--- Testing TextCleaner ---")

    # Test with default whitelist
    cleaner_default = TextCleaner()
    test_data_1 = {"text": "Hello World! This is a test. 123 numbers included & some symbols $^%.", "confidence": 0.92, "source": "doc1.png"}
    cleaned_result_1 = cleaner_default.clean_text(test_data_1)
    logging.info(f"Test 1 (Default Whitelist) - Original: '{test_data_1['text']}'")
    logging.info(f"Test 1 (Default Whitelist) - Cleaned: '{cleaned_result_1.get('cleaned_text')}', Confidence: {cleaned_result_1.get('confidence')}, Source: {cleaned_result_1.get('source')}")
    expected_cleaned_1 = "Hello World! This is a test. 123 numbers included  some symbols ." # Based on DEFAULT_WHITELIST_CHARS
    assert cleaned_result_1.get('cleaned_text') == expected_cleaned_1
    assert cleaned_result_1.get('confidence') == 0.92
    assert cleaned_result_1.get('source') == "doc1.png"


    # Test with a custom whitelist (e.g., only lowercase letters and space)
    custom_whitelist = "abcdefghijklmnopqrstuvwxyz "
    cleaner_custom = TextCleaner(whitelist_chars=custom_whitelist)
    test_data_2 = {"text": "Hello World! This is a test. 123 numbers included.", "confidence": 0.88}
    cleaned_result_2 = cleaner_custom.clean_text(test_data_2)
    logging.info(f"Test 2 (Custom Whitelist: '{custom_whitelist}') - Original: '{test_data_2['text']}'")
    logging.info(f"Test 2 (Custom Whitelist) - Cleaned: '{cleaned_result_2.get('cleaned_text')}', Confidence: {cleaned_result_2.get('confidence')}")
    expected_cleaned_2 = "ello orld his is a test  numbers included" # Based on custom_whitelist
    assert cleaned_result_2.get('cleaned_text') == expected_cleaned_2
    assert cleaned_result_2.get('confidence') == 0.88

    # Test with empty text
    test_data_3 = {"text": "", "confidence": 0.99}
    cleaned_result_3 = cleaner_default.clean_text(test_data_3)
    logging.info(f"Test 3 (Empty Text) - Original: '{test_data_3['text']}'")
    logging.info(f"Test 3 (Empty Text) - Cleaned: '{cleaned_result_3.get('cleaned_text')}', Confidence: {cleaned_result_3.get('confidence')}")
    assert cleaned_result_3.get('cleaned_text') == ""

    # Test with text containing only non-whitelisted characters
    test_data_4 = {"text": "$%^&*@#", "confidence": 0.75}
    cleaned_result_4 = cleaner_default.clean_text(test_data_4) # Uses default whitelist
    logging.info(f"Test 4 (Non-whitelisted Text) - Original: '{test_data_4['text']}'")
    logging.info(f"Test 4 (Non-whitelisted Text) - Cleaned: '{cleaned_result_4.get('cleaned_text')}', Confidence: {cleaned_result_4.get('confidence')}")
    assert cleaned_result_4.get('cleaned_text') == "" # Should be empty if default whitelist doesn't include these

    # Test with invalid input (not a dict)
    test_data_5 = "This is just a string"
    cleaned_result_5 = cleaner_default.clean_text(test_data_5)
    logging.info(f"Test 5 (Invalid Input Type) - Original: '{test_data_5}'")
    logging.info(f"Test 5 (Invalid Input Type) - Result: {cleaned_result_5}")
    assert "error" in cleaned_result_5
    assert cleaned_result_5.get('cleaned_text') == ""

    # --- Example of PostprocessingModulePlaceholder (kept for reference) ---
    # logging.info("\n--- Testing PostprocessingModulePlaceholder ---")
    # postproc_settings = {"nlp_model_path": "dummy_nlp_model.onnx", "language": "en"}
    # # ... (rest of placeholder test code can be here if needed)


import re

class SpellCorrector:
    def __init__(self, dictionary_path: str | None = None):
        self.logger = logging.getLogger(__name__)
        self.dictionary_path = dictionary_path if dictionary_path else "default_dict.txt"
        self.dictionary = set()
        self._load_dictionary()

    def _load_dictionary(self):
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.dictionary.add(line.strip().lower())
            self.logger.info(f"SpellCorrector initialized. Loaded {len(self.dictionary)} words from '{self.dictionary_path}'.")
        except FileNotFoundError:
            self.logger.error(f"Dictionary file '{self.dictionary_path}' not found. SpellCorrector will operate with an empty dictionary.")
        except Exception as e:
            self.logger.error(f"Error loading dictionary '{self.dictionary_path}': {e}", exc_info=True)


    def correct_text(self, text_data_dict: dict) -> dict:
        """
        Corrects spelling in the 'cleaned_text' field of the input dictionary.
        Adds '[?]' to words not found in the dictionary (case-insensitive check).
        """
        if not isinstance(text_data_dict, dict):
            self.logger.error(f"Input must be a dictionary, got {type(text_data_dict)}.")
            return {**text_data_dict, "spell_checked_text": "Error: Input was not a dict", "spell_correction_error": "Invalid input type"}

        cleaned_text = text_data_dict.get('cleaned_text', '')
        if not cleaned_text: # Handle empty or None cleaned_text
            self.logger.info("No cleaned text provided for spell checking.")
            return {**text_data_dict, "spell_checked_text": ""}

        self.logger.debug(f"Original text for spell checking: '{cleaned_text[:100]}...'")

        # Split text into words, keeping punctuation attached to words for now, or separating them.
        # A simple regex split that keeps delimiters might be: re.findall(r"[\w']+|[.,!?;]", cleaned_text)
        # For this task, let's use a simpler split by space and then handle punctuation.
        # The TextCleaner's default whitelist includes basic punctuation.

        words = cleaned_text.split(' ') # Simple space-based split
        corrected_words = []

        for word_token in words:
            if not word_token: # Handles multiple spaces
                corrected_words.append(word_token)
                continue

            # Separate leading/trailing punctuation from the word for dictionary lookup
            # This is a basic approach. A more robust solution would use regex or tokenization.
            word_to_check = word_token
            prefix_punct = ""
            suffix_punct = ""

            # Example: Extract trailing punctuation (simplified)
            # This regex will find common trailing punctuation.
            match = re.match(r"^(.*?)(\W*)$", word_token)
            if match:
                word_to_check = match.group(1)
                suffix_punct = match.group(2)

            # If word_to_check itself is empty (e.g. token was just punctuation), keep original token
            if not word_to_check:
                 corrected_words.append(word_token)
                 continue

            if word_to_check.lower() in self.dictionary:
                corrected_words.append(word_token) # Keep original casing and punctuation
            else:
                # Check if it's a number (which TextCleaner allows) - numbers shouldn't be marked as misspelled
                # A simple check:
                if word_to_check.isnumeric():
                    corrected_words.append(word_token)
                else:
                    corrected_words.append(word_to_check + "[?]" + suffix_punct)

        spell_checked_text = " ".join(corrected_words)

        self.logger.info(f"Spell checking complete. Corrected text: '{spell_checked_text[:100]}...'")

        # Create a new dictionary to avoid modifying the input dict directly, and add the new key
        result_dict = text_data_dict.copy()
        result_dict["spell_checked_text"] = spell_checked_text
        return result_dict

if __name__ == '__main__':
    # Basic logging setup for standalone execution (repeated from above, ensure it's fine)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Test TextCleaner (from previous task) ---
    logging.info("\n--- Testing TextCleaner ---")
    cleaner_default = TextCleaner()
    test_data_tc = {"text": "Hello World! This is a test. 123 numbers included & some symbols $^%.", "confidence": 0.92}
    cleaned_result_tc = cleaner_default.clean_text(test_data_tc)
    logging.info(f"TextCleaner - Original: '{test_data_tc['text']}'")
    logging.info(f"TextCleaner - Cleaned: '{cleaned_result_tc.get('cleaned_text')}'")

    # --- Test SpellCorrector ---
    logging.info("\n--- Testing SpellCorrector ---")
    # Ensure default_dict.txt exists in /app/ for this test to work with default path
    # If not, SpellCorrector will log an error and use an empty dictionary.
    # Create a dummy default_dict.txt for testing if it doesn't exist
    default_dict_path_for_test = "default_dict.txt"
    if not os.path.exists(default_dict_path_for_test):
        logging.warning(f"'{default_dict_path_for_test}' not found for test. Creating a dummy one.")
        with open(default_dict_path_for_test, "w") as f:
            f.write("hello\nworld\nis\na\ntest\nnumbers\npython\nexample\ntext\n")

    spell_checker_default = SpellCorrector() # Uses default_dict.txt

    # Use the output from TextCleaner as input for SpellCorrector
    input_for_spell_checker = cleaned_result_tc
    # Expected from TextCleaner: "Hello World! This is a test. 123 numbers included  some symbols ."

    spell_checked_result_1 = spell_checker_default.correct_text(input_for_spell_checker)
    logging.info(f"SpellChecker Test 1 - Input (cleaned): '{spell_checked_result_1.get('cleaned_text')}'")
    logging.info(f"SpellChecker Test 1 - Output (spell-checked): '{spell_checked_result_1.get('spell_checked_text')}'")
    # Expected: "Hello World! This is a test. 123 numbers included[?]  some[?] symbols[?]." (if 'included', 'some', 'symbols' not in dummy dict)
    # Based on dummy dict created: "Hello World! This is a test. 123 numbers included[?]  some[?] symbols[?]."
    # Actually, with the current dummy dict: "Hello World! This is a test. 123 numbers included[?]  some[?] symbols[?]."
    # Let's refine the dummy dict for better testing:
    # Dummy dict: hello, world, is, a, test, numbers, python, example, text, included, some, symbols
    if os.path.exists("temp_test_dict.txt"): os.remove("temp_test_dict.txt") # clean if exists
    with open("temp_test_dict.txt", "w") as f:
        f.write("hello\nworld\nis\na\ntest\nnumbers\npython\nexample\ntext\nincluded\nsome\nsymbols\n")
    spell_checker_temp_dict = SpellCorrector(dictionary_path="temp_test_dict.txt")
    spell_checked_result_1_temp = spell_checker_temp_dict.correct_text(input_for_spell_checker)
    logging.info(f"SpellChecker Test 1 (temp dict) - Output (spell-checked): '{spell_checked_result_1_temp.get('spell_checked_text')}'")
    # Expected with temp_test_dict.txt: "Hello World! This is a test. 123 numbers included  some symbols ." (no [?])
    assert spell_checked_result_1_temp.get('spell_checked_text') == "Hello World! This is a test. 123 numbers included  some symbols ."


    test_data_sc_2 = {"cleaned_text": "This is an exampl of mispelled wrds.", "confidence": 0.80}
    spell_checked_result_2 = spell_checker_temp_dict.correct_text(test_data_sc_2)
    logging.info(f"SpellChecker Test 2 - Input: '{test_data_sc_2['cleaned_text']}'")
    logging.info(f"SpellChecker Test 2 - Output: '{spell_checked_result_2.get('spell_checked_text')}'")
    # Expected: "This is an exampl[?] of[?] mispelled[?] wrds[?]."
    assert spell_checked_result_2.get('spell_checked_text') == "This is an exampl[?] of[?] mispelled[?] wrds[?]."

    # Test with numbers and punctuation
    test_data_sc_3 = {"cleaned_text": "Test 123 numbers! Go.", "confidence": 0.99}
    spell_checked_result_3 = spell_checker_temp_dict.correct_text(test_data_sc_3)
    logging.info(f"SpellChecker Test 3 - Input: '{test_data_sc_3['cleaned_text']}'")
    logging.info(f"SpellChecker Test 3 - Output: '{spell_checked_result_3.get('spell_checked_text')}'")
    # Expected: "Test 123 numbers! Go[?]." (if "go" not in dict)
    assert spell_checked_result_3.get('spell_checked_text') == "Test 123 numbers! Go[?]."

    if os.path.exists("temp_test_dict.txt"): os.remove("temp_test_dict.txt")

    # Test with PostprocessingModulePlaceholder for structural check
    # logging.info("\n--- Testing PostprocessingModulePlaceholder (structure check) ---")
    # placeholder_settings = {}
    # placeholder_module = PostprocessingModulePlaceholder(placeholder_settings)
    # placeholder_input = {"text": "This is some text for placeholder.", "confidence": 0.9}
    # placeholder_output = placeholder_module.run_all(placeholder_input)
    # logging.info(f"Placeholder output: {placeholder_output}")
    # assert "cleaned_text" in placeholder_output # Placeholder now returns a dict
    # assert "original_text" in placeholder_output
    # assert "confidence" in placeholder_output
