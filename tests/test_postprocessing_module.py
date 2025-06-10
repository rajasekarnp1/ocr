import unittest
import os
import logging
import shutil

# Assuming modules are in parent directory or PYTHONPATH is set
try:
    from postprocessing_module import TextCleaner, SpellCorrector, DEFAULT_WHITELIST_CHARS
    from custom_exceptions import OCRPipelineError # For testing error raising
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from postprocessing_module import TextCleaner, SpellCorrector, DEFAULT_WHITELIST_CHARS
    from custom_exceptions import OCRPipelineError

# Configure logging for tests
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO) # Set to DEBUG for more verbose test logging


class TestTextCleaner(unittest.TestCase):
    def test_clean_text_default_whitelist(self):
        logger.info("TestTextCleaner: test_clean_text_default_whitelist")
        cleaner = TextCleaner() # Uses default whitelist
        input_data = {"text": "Hello World! 123. Test with symbols: @#$%^&*().", "confidence": 0.9}
        expected_cleaned = "Hello World! 123. Test with symbols ()." # Default whitelist keeps .,!?'()[]-

        result = cleaner.clean_text(input_data)
        self.assertEqual(result["cleaned_text"], expected_cleaned)
        self.assertEqual(result["original_text"], input_data["text"])
        self.assertEqual(result["confidence"], input_data["confidence"])

    def test_clean_text_custom_whitelist(self):
        logger.info("TestTextCleaner: test_clean_text_custom_whitelist")
        custom_whitelist = "abc " # Only 'a', 'b', 'c', and space
        cleaner = TextCleaner(whitelist_chars=custom_whitelist)
        input_data = {"text": "aabbcc ddeeff", "confidence": 0.8}
        expected_cleaned = "aabbcc  " # d, e, f are removed

        result = cleaner.clean_text(input_data)
        self.assertEqual(result["cleaned_text"], expected_cleaned)

    def test_clean_text_empty_string(self):
        logger.info("TestTextCleaner: test_clean_text_empty_string")
        cleaner = TextCleaner()
        input_data = {"text": "", "confidence": 0.0}
        expected_cleaned = ""
        result = cleaner.clean_text(input_data)
        self.assertEqual(result["cleaned_text"], expected_cleaned)

    def test_clean_text_only_non_whitelisted(self):
        logger.info("TestTextCleaner: test_clean_text_only_non_whitelisted")
        cleaner = TextCleaner(whitelist_chars="abc") # Very restrictive
        input_data = {"text": "$%^&*123", "confidence": 0.5}
        expected_cleaned = ""
        result = cleaner.clean_text(input_data)
        self.assertEqual(result["cleaned_text"], expected_cleaned)

    def test_clean_text_invalid_input_type(self):
        logger.info("TestTextCleaner: test_clean_text_invalid_input_type")
        cleaner = TextCleaner()
        with self.assertRaises(TypeError): # As per refined error handling
            cleaner.clean_text("not a dict")

    def test_clean_text_missing_text_key(self):
        logger.info("TestTextCleaner: test_clean_text_missing_text_key")
        cleaner = TextCleaner()
        input_data = {"confidence": 0.9} # Missing 'text' key
        with self.assertRaises(OCRPipelineError): # As per refined error handling
            cleaner.clean_text(input_data)


class TestSpellCorrector(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_spell_corrector_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.default_dict_path = "default_dict.txt" # Relies on the main one for some tests

        # Create a temporary custom dictionary for specific tests
        self.custom_dict_path = os.path.join(self.test_dir, "custom_test_dict.txt")
        with open(self.custom_dict_path, "w", encoding='utf-8') as f:
            f.write("hello\nworld\ncustom\nspell\nchecker\nexample\n")
        logger.info(f"TestSpellCorrector: setUp created custom dictionary at {self.custom_dict_path}")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logger.info(f"TestSpellCorrector: tearDown removed test directory {self.test_dir}")

    def test_correct_text_all_words_in_dict(self):
        logger.info("TestSpellCorrector: test_correct_text_all_words_in_dict")
        # Using custom dict where all test words are present
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path)
        input_data = {"cleaned_text": "hello custom spell checker example", "confidence": 0.95}
        expected_spell_checked = "hello custom spell checker example"

        result = corrector.correct_text(input_data)
        self.assertEqual(result["spell_checked_text"], expected_spell_checked)
        self.assertEqual(result["cleaned_text"], input_data["cleaned_text"])

    def test_correct_text_some_words_not_in_dict(self):
        logger.info("TestSpellCorrector: test_correct_text_some_words_not_in_dict")
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path)
        input_data = {"cleaned_text": "hello worldd customm spelll checkker examplee", "confidence": 0.9}
        # "worldd", "customm", "spelll", "checkker", "examplee" are not in custom_test_dict.txt
        expected_spell_checked = "hello[?] worldd[?] customm[?] spelll[?] checkker[?] examplee[?]"
        # After refinement, "hello" is not in custom_dict.txt, it's in default_dict.txt.
        # Correcting expectation for custom_dict_path:
        expected_spell_checked = "hello[?] worldd[?] custom[?] spell[?] checker[?] example[?]"
        # The test setup uses self.custom_dict_path. "hello" is in it.
        # "worldd", "customm", "spelll", "checkker", "examplee" are not.
        # The default TextCleaner whitelist includes " .,!?'()[]-"
        # The refined SpellCorrector's regex for punctuation: re.search(r"(\w[\w']*\w|\w)", word_token)
        # So, for "worldd[?]", it should work.
        # For "example." -> core_word="example", suffix_punct="."
        # If custom_dict has "hello", "custom", "spell", "checker", "example":
        expected_spell_checked = "hello worldd[?] custom spell checker example" # if "worldd" is the only unknown

        # Let's use the custom dict which has "hello", "custom", "spell", "checker", "example"
        input_data = {"cleaned_text": "hello worldd custom spell checkker example.", "confidence": 0.9}
        expected_spell_checked = "hello worldd[?] custom spell checkker[?] example."

        result = corrector.correct_text(input_data)
        self.assertEqual(result["spell_checked_text"], expected_spell_checked)

    def test_correct_text_mixed_case(self):
        logger.info("TestSpellCorrector: test_correct_text_mixed_case")
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path) # custom_dict is all lowercase
        input_data = {"cleaned_text": "Hello CUSTOM SpelL eXample", "confidence": 0.88}
        # Dictionary lookups are case-insensitive (words stored as lower)
        expected_spell_checked = "Hello CUSTOM SpelL eXample"

        result = corrector.correct_text(input_data)
        self.assertEqual(result["spell_checked_text"], expected_spell_checked)

    def test_correct_text_punctuation_handling(self):
        logger.info("TestSpellCorrector: test_correct_text_punctuation_handling")
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path)
        input_data = {"cleaned_text": "hello. custom! (spell) checker -example-", "confidence": 0.7}
        # Assumes "hello", "custom", "spell", "checker", "example" are in dict.
        # Punctuation should be preserved.
        expected_spell_checked = "hello. custom! (spell) checker -example-"

        result = corrector.correct_text(input_data)
        self.assertEqual(result["spell_checked_text"], expected_spell_checked)

        input_data_misspelled = {"cleaned_text": "helo. custm! (spel) cheker -exampl-", "confidence": 0.6}
        expected_misspelled_checked = "helo[?]. custm[?]! (spel[?]) cheker[?] -exampl[?]-"
        result_misspelled = corrector.correct_text(input_data_misspelled)
        self.assertEqual(result_misspelled["spell_checked_text"], expected_misspelled_checked)

    def test_correct_text_with_numbers(self):
        logger.info("TestSpellCorrector: test_correct_text_with_numbers")
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path)
        input_data = {"cleaned_text": "custom 123 example 456", "confidence": 0.92}
        # Numbers should not be marked as unknown by `isnumeric()` check.
        expected_spell_checked = "custom 123 example 456"

        result = corrector.correct_text(input_data)
        self.assertEqual(result["spell_checked_text"], expected_spell_checked)

    def test_correct_text_empty_and_whitespace(self):
        logger.info("TestSpellCorrector: test_correct_text_empty_and_whitespace")
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path)

        input_data_empty = {"cleaned_text": "", "confidence": 0.1}
        result_empty = corrector.correct_text(input_data_empty)
        self.assertEqual(result_empty["spell_checked_text"], "")

        input_data_space = {"cleaned_text": "   ", "confidence": 0.1} # Multiple spaces
        result_space = corrector.correct_text(input_data_space)
        self.assertEqual(result_space["spell_checked_text"], "   ") # Preserves spaces

    def test_init_dictionary_not_found(self):
        logger.info("TestSpellCorrector: test_init_dictionary_not_found")
        non_existent_dict_path = os.path.join(self.test_dir, "no_such_dict.txt")
        # Expect a log warning, and an empty dictionary
        with self.assertLogs(level='ERROR') as log_cm: # Check for ERROR level log
            corrector = SpellCorrector(dictionary_path=non_existent_dict_path)
            self.assertTrue(any(f"Dictionary file '{non_existent_dict_path}' not found" in msg for msg in log_cm.output))
        self.assertEqual(len(corrector.dictionary), 0)

        # Test correction with empty dictionary
        input_data = {"cleaned_text": "any word", "confidence": 0.5}
        expected_spell_checked = "any[?] word[?]"
        result = corrector.correct_text(input_data)
        self.assertEqual(result["spell_checked_text"], expected_spell_checked)

    def test_correct_text_invalid_input_type(self):
        logger.info("TestSpellCorrector: test_correct_text_invalid_input_type")
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path)
        with self.assertRaises(TypeError): # As per refined error handling
            corrector.correct_text("not a dict")

    def test_correct_text_missing_cleaned_text_key(self):
        logger.info("TestSpellCorrector: test_correct_text_missing_cleaned_text_key")
        corrector = SpellCorrector(dictionary_path=self.custom_dict_path)
        input_data = {"original_text": "some text", "confidence": 0.9} # Missing 'cleaned_text'
        with self.assertRaises(OCRPipelineError): # As per refined error handling
            corrector.correct_text(input_data)

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()
