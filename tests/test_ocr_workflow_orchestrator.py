import unittest
from unittest.mock import patch, MagicMock
import os
import logging
import shutil
import yaml
import numpy as np # For dummy image data in orchestrator tests

# Assuming modules are in parent directory or PYTHONPATH is set
try:
    from ocr_workflow_orchestrator import OCRWorkflowOrchestrator
    from custom_exceptions import OCRFileNotFoundError, OCRConfigurationError, OCRPipelineError, OCRModelError, OCRImageProcessingError
    from config_loader import DEFAULT_LOGGING_CONFIG # For checking fallback logging
    # For generating dummy ONNX models for tests
    from generate_dummy_geometric_model import generate_model as generate_geometric_onnx
    from generate_dummy_recognition_model import generate_model as generate_recognition_onnx
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ocr_workflow_orchestrator import OCRWorkflowOrchestrator
    from custom_exceptions import OCRFileNotFoundError, OCRConfigurationError, OCRPipelineError, OCRModelError, OCRImageProcessingError
    from config_loader import DEFAULT_LOGGING_CONFIG
    from generate_dummy_geometric_model import generate_model as generate_geometric_onnx
    from generate_dummy_recognition_model import generate_model as generate_recognition_onnx

# Configure logging for tests
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TestOCRWorkflowOrchestrator(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_orchestrator_temp_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.config_filename = "dummy_orchestrator_config.yaml"
        self.config_path = os.path.join(self.test_dir, self.config_filename)

        self.dummy_image_filename = "dummy_test_image.png"
        self.dummy_image_path = os.path.join(self.test_dir, self.dummy_image_filename)

        self.geom_model_path = os.path.join(self.models_dir, "dummy_geom_test.onnx")
        self.ocr_model_path = os.path.join(self.models_dir, "dummy_ocr_test.onnx")
        self.dict_path = os.path.join(self.test_dir, "default_dict.txt") # Changed from nlp_model_path

        # Generate dummy ONNX models
        try:
            generate_geometric_onnx(self.geom_model_path)
            generate_recognition_onnx(self.ocr_model_path)
        except Exception as e:
            logger.error(f"Failed to generate dummy ONNX models for orchestrator tests: {e}", exc_info=True)
            raise

        # Create dummy dictionary file
        with open(self.dict_path, "w") as f:
            f.write("hello\nworld\ntext\nocr\nimage\n")

        self.dummy_config_data = {
            "logging": DEFAULT_LOGGING_CONFIG, # Use a known default for logging part of test
            "preprocessing_settings": {"model_path": self.geom_model_path},
            "recognition_settings": {
                "svtr_recognizer": self.ocr_model_path, # Correct key for ONNXRecognizer
                "use_directml": False # Easier for testing
            },
            "postprocessing_settings": {
                "whitelist_chars": None, # Use default
                "dictionary_path": self.dict_path
            },
            "deskewer_settings": {} # Use defaults
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.dummy_config_data, f)

        with open(self.dummy_image_path, "w") as f: # Create a dummy file, content doesn't matter as load_image is mocked or returns np array
            f.write("dummy image content")

        # Reset root logger handlers to avoid duplicate logs if tests run multiple times
        # This is important because load_config applies logging.config.dictConfig
        # logging.shutdown()
        # for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        # logging.basicConfig(level=logging.DEBUG) # Re-init for test output visibility

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        # Clean up config_dev.yaml if created by orchestrator's default config logic
        if os.path.exists("config_dev.yaml"):
            os.remove("config_dev.yaml")

    def test_successful_initialization(self):
        logger.info("TestOCRWorkflowOrchestrator: test_successful_initialization")
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        self.assertIsNotNone(orchestrator.preprocessor, "GeometricCorrector should be initialized.")
        self.assertIsNotNone(orchestrator.recognizer, "ONNXRecognizer should be initialized.")
        self.assertIsNotNone(orchestrator.binarizer, "ImageBinarizer should be initialized.")
        self.assertIsNotNone(orchestrator.deskewer, "ImageDeskewer should be initialized.")
        self.assertIsNotNone(orchestrator.text_cleaner, "TextCleaner should be initialized.")
        self.assertIsNotNone(orchestrator.spell_corrector, "SpellCorrector should be initialized.")
        self.assertEqual(orchestrator.config["preprocessing_settings"]["model_path"], self.geom_model_path)

    def test_initialization_config_not_found(self):
        logger.info("TestOCRWorkflowOrchestrator: test_initialization_config_not_found")
        non_existent_config_path = os.path.join(self.test_dir, "this_config_does_not_exist.yaml")
        with self.assertRaises(OCRFileNotFoundError): # Orchestrator re-raises this from load_config
            OCRWorkflowOrchestrator(config_path=non_existent_config_path)

    def test_initialization_model_file_not_found_in_config(self):
        logger.info("TestOCRWorkflowOrchestrator: test_initialization_model_file_not_found_in_config")
        faulty_config_data = self.dummy_config_data.copy()
        # Point to a model that won't exist
        faulty_config_data["recognition_settings"]["svtr_recognizer"] = os.path.join(self.models_dir, "actually_missing_ocr.onnx")

        faulty_config_path = os.path.join(self.test_dir, "faulty_model_path_config.yaml")
        with open(faulty_config_path, 'w') as f:
            yaml.dump(faulty_config_data, f)

        # Orchestrator's __init__ catches errors from module initializations (like OCRFileNotFoundError from ONNXRecognizer)
        # and wraps them in OCRPipelineError.
        with self.assertRaises(OCRPipelineError) as context:
            OCRWorkflowOrchestrator(config_path=faulty_config_path)
        self.assertIsInstance(context.exception.__cause__, OCRFileNotFoundError,
                              "Cause should be OCRFileNotFoundError from the sub-module.")

    @patch('ocr_workflow_orchestrator.OCRWorkflowOrchestrator.load_image') # Mock load_image
    def test_process_document_successful_flow(self, mock_load_image):
        logger.info("TestOCRWorkflowOrchestrator: test_process_document_successful_flow")
        # Setup mock for load_image to return a consistent dummy NumPy array
        # This should be a uint8 grayscale-like image for binarization
        dummy_np_image = np.random.randint(0, 256, size=(100, 150), dtype=np.uint8)
        mock_load_image.return_value = dummy_np_image

        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        result_dict = orchestrator.process_document(self.dummy_image_path)

        self.assertIsInstance(result_dict, dict)
        self.assertNotIn("error", result_dict, f"Processing unexpectedly returned an error dict: {result_dict}")
        self.assertNotIn("Error:", result_dict.get("spell_checked_text", ""), "Processing result string indicates an error.")

        self.assertIn("spell_checked_text", result_dict)
        self.assertIn("cleaned_text", result_dict)
        self.assertIn("original_text", result_dict)
        self.assertIn("confidence", result_dict)
        # Example check based on dummy models (add 1, then multiply by 2)
        # Binarized (0, 255) -> deskewed (0, 255) -> normalized (0.0, 1.0)
        # GeomCorrect (adds 1.0) -> (1.0, 2.0)
        # Recognition (mult by 2.0) -> (2.0, 4.0) -> sum for dummy text
        # This part is complex to assert precisely without knowing exact binarization/deskew output.
        # A simpler check is that some text is produced.
        self.assertTrue(len(result_dict["spell_checked_text"]) > 0, "Spell checked text is empty.")


    def test_process_document_load_image_raises_ocrfilenotfound(self):
        logger.info("TestOCRWorkflowOrchestrator: test_process_document_load_image_raises_ocrfilenotfound")
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        non_existent_image = os.path.join(self.test_dir, "no_such_image.png")

        # Orchestrator's process_document catches OCRFileNotFoundError from load_image
        # and re-raises it as OCRPipelineError.
        with self.assertRaises(OCRPipelineError) as context:
            orchestrator.process_document(non_existent_image)
        self.assertIsInstance(context.exception.__cause__, OCRFileNotFoundError)

    @patch('ocr_workflow_orchestrator.ImageBinarizer.binarize')
    def test_process_document_binarizer_error(self, mock_binarize):
        logger.info("TestOCRWorkflowOrchestrator: test_process_document_binarizer_error")
        mock_binarize.side_effect = OCRImageProcessingError("Test Binarization Error")
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        with self.assertRaises(OCRPipelineError) as context:
            orchestrator.process_document(self.dummy_image_path)
        self.assertIsInstance(context.exception.__cause__, OCRImageProcessingError)
        self.assertTrue("Test Binarization Error" in str(context.exception.__cause__))

    @patch('ocr_workflow_orchestrator.ONNXRecognizer.predict')
    def test_process_document_recognizer_error(self, mock_predict):
        logger.info("TestOCRWorkflowOrchestrator: test_process_document_recognizer_error")
        mock_predict.side_effect = OCRModelError("Test Recognition Error")
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        with self.assertRaises(OCRPipelineError) as context:
            orchestrator.process_document(self.dummy_image_path)
        self.assertIsInstance(context.exception.__cause__, OCRModelError)
        self.assertTrue("Test Recognition Error" in str(context.exception.__cause__))


    def test_get_results_formatting(self):
        logger.info("TestOCRWorkflowOrchestrator: test_get_results_formatting")
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)

        processed_data_ok = {
            "spell_checked_text": "Hello Wørld[?]",
            "cleaned_text": "Hello Wørld",
            "original_text": "Hello Wørld! 123",
            "confidence": 0.88,
            "custom_meta": "some_value"
        }
        formatted_ok = orchestrator.get_results(processed_data_ok)
        self.assertIn("Spell-Checked Text: 'Hello Wørld[?]'", formatted_ok)
        self.assertIn("Cleaned Text      : 'Hello Wørld'", formatted_ok)
        self.assertIn("Original OCR Text : 'Hello Wørld! 123'", formatted_ok)
        self.assertIn("Confidence        : 0.88", formatted_ok)
        self.assertIn("'custom_meta': 'some_value'", formatted_ok)

        # Test with error string (older pattern, now orchestrator raises exceptions)
        # However, get_results might still be called with an error string if process_document had early exit
        # For example, if load_image directly returned an error string (which it doesn't anymore).
        # The refined orchestrator now raises exceptions, so this path in get_results is less likely
        # to be hit with an "Error:" string from the pipeline itself, but good to keep for robustness.
        formatted_err_str = orchestrator.get_results("Error: Something bad happened early.")
        self.assertEqual(formatted_err_str, "No valid data to format. Processing status: Error: Something bad happened early.")

        formatted_none = orchestrator.get_results(None)
        self.assertEqual(formatted_none, "No data to format. Processing result was None.")

        # Test with a dictionary that indicates an internal error from a sub-module
        # (e.g., if spell_corrector.correct_text returned such a dict - it now raises exceptions)
        # This test case may need adjustment based on how errors are propagated as dicts vs exceptions.
        # The current get_results is mostly for formatting successful dicts or error strings.
        # If process_document raises an exception, get_results won't be called with its return value.


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers(): # Ensure basic logging for standalone run
         logging.basicConfig(level=logging.DEBUG)
    unittest.main()
