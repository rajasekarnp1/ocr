import unittest
import os
import logging
import shutil
import yaml
from ocr_workflow_orchestrator import OCRWorkflowOrchestrator
# Assuming other modules (config_loader, preprocessing_module, etc.) are in /app
# and tests are run from /app using 'python -m unittest discover tests'
# or sys.path is adjusted.

class TestOCRWorkflowOrchestrator(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_orchestrator_temp_dir"
        os.makedirs(self.test_dir, exist_ok=True)

        self.config_filename = "dummy_orchestrator_config.yaml"
        self.config_path = os.path.join(self.test_dir, self.config_filename)

        self.dummy_image_filename = "dummy_test_image.png"
        self.dummy_image_path = os.path.join(self.test_dir, self.dummy_image_filename)

        # Model paths for dummy config - these will be created as dummy files
        self.geom_model_path = os.path.join(self.test_dir, "models", "dummy_geom_test.onnx")
        self.ocr_model_path = os.path.join(self.test_dir, "models", "dummy_ocr_test.onnx")
        self.nlp_model_path = os.path.join(self.test_dir, "models", "dummy_nlp_test.onnx") # From postprocessing settings

        # Create dummy config file
        self.dummy_config_data = {
            "logging": {
                "version": 1,
                "root": {"level": "INFO", "handlers": ["console"]},
                "handlers": {"console": {"class": "logging.StreamHandler", "level": "INFO"}},
                "loggers": {"OCRWorkflowOrchestrator": {"level": "DEBUG"}} 
            },
            "preprocessing_settings": {"model_path": self.geom_model_path},
            "recognition_settings": {"model_path": self.ocr_model_path, "use_directml": False},
            "postprocessing_settings": {"nlp_model_path": self.nlp_model_path}
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.dummy_config_data, f)

        # Create dummy model files and their directories
        os.makedirs(os.path.dirname(self.geom_model_path), exist_ok=True)
        with open(self.geom_model_path, "w") as f: f.write("dummy geom model data")
        
        os.makedirs(os.path.dirname(self.ocr_model_path), exist_ok=True)
        with open(self.ocr_model_path, "w") as f: f.write("dummy ocr model data")

        os.makedirs(os.path.dirname(self.nlp_model_path), exist_ok=True)
        with open(self.nlp_model_path, "w") as f: f.write("dummy nlp model data")
        
        # Create dummy image file
        with open(self.dummy_image_path, "w") as f:
            f.write("dummy image content for testing orchestrator")

        self.addCleanup(self.cleanup_test_dir)

        # Basic logging setup for tests if not already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Ensure that the default config_dev.yaml, if created by a previous test run's module, is removed
        # as orchestrator might try to load it if its own config path is problematic
        if os.path.exists("config_dev.yaml"):
            os.remove("config_dev.yaml")


    def cleanup_test_dir(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists("config_dev.yaml"): # Clean up if orchestrator created it due to error
            os.remove("config_dev.yaml")


    def test_successful_initialization(self):
        try:
            orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
            self.assertIsNotNone(orchestrator.preprocessor, "Preprocessor should be initialized.")
            self.assertIsNotNone(orchestrator.recognizer, "Recognizer should be initialized.")
            self.assertIsNotNone(orchestrator.postprocessor, "Postprocessor should be initialized.")
            self.assertEqual(orchestrator.config["preprocessing_settings"]["model_path"], self.geom_model_path)
            logging.info("OCRWorkflowOrchestrator initialized successfully for test.")
        except Exception as e:
            self.fail(f"OCRWorkflowOrchestrator initialization failed unexpectedly: {e}")

    def test_initialization_config_not_found(self):
        non_existent_config_path = os.path.join(self.test_dir, "non_existent_config.yaml")
        # The orchestrator's __init__ currently re-raises FileNotFoundError from load_config
        with self.assertRaises(FileNotFoundError):
            OCRWorkflowOrchestrator(config_path=non_existent_config_path)
        logging.info("OCRWorkflowOrchestrator correctly raised error for non-existent config.")

    def test_initialization_model_file_not_found(self):
        # Test case: config file is present, but a model file path it contains is missing.
        faulty_config_path = os.path.join(self.test_dir, "faulty_model_path_config.yaml")
        faulty_config_data = self.dummy_config_data.copy()
        faulty_config_data["recognition_settings"]["model_path"] = os.path.join(self.test_dir, "models", "actually_missing_ocr.onnx")
        
        with open(faulty_config_path, 'w') as f:
            yaml.dump(faulty_config_data, f)
            
        # GeometricCorrector/ONNXRecognizer init raises FileNotFoundError if model file is missing,
        # which then should be caught and re-raised as RuntimeError by Orchestrator's init.
        with self.assertRaises(RuntimeError) as context:
            OCRWorkflowOrchestrator(config_path=faulty_config_path)
        
        self.assertTrue("Could not initialize" in str(context.exception) or "model file not found" in str(context.exception).lower())
        logging.info("OCRWorkflowOrchestrator correctly raised RuntimeError for missing model file specified in config.")


    def test_process_document_successful_placeholder_flow(self):
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        result = orchestrator.process_document(self.dummy_image_path)

        self.assertIsNotNone(result, "Result should not be None for successful processing.")
        self.assertFalse(result.startswith("Error:"), f"Processing unexpectedly returned an error: {result}")
        
        # Check if the result string indicates it went through the placeholder modules
        # Expected: "FinalText_for_RawText_from_dummy_ocr_test.onnx for image of shape MockImageData_for_dummy_test_image.png" (approx)
        # from Postprocessing(Recognition(Preprocessing(Load)))
        # Actual result based on current placeholders:
        # Load: "MockImageData_for_dummy_test_image.png"
        # Preproc: "MockImageData_for_dummy_test_image.png" * 0.95 (if input is numeric, but it's string)
        # Recognizer: ("Raw OCR from dummy_ocr_test.onnx for image of shape MockImageData_for_dummy_test_image.png", 0.92)
        # Postproc: "FinalText_for_Raw OCR from dummy_ocr_test.onnx for image of shape MockImageData_for_dummy_test_image.png"
        
        # Let's adjust expectations for GeometricCorrector's placeholder with string input
        # GeometricCorrector's placeholder is `lambda image_data_np: image_data_np * 0.95`
        # If `image_data` is "MockImageData_for_dummy_test_image.png", this will actually fail with TypeError.
        # The `correct` method in GeometricCorrector returns the original image_data_np on exception.
        # So, `preprocessed_image` will be "MockImageData_for_dummy_test_image.png".
        
        expected_recognition_text_part = f"Raw OCR from {os.path.basename(self.ocr_model_path)} for image of shape MockImageData_for_{self.dummy_image_filename}"
        expected_final_text = f"FinalText_for_{expected_recognition_text_part}"
        
        self.assertEqual(result, expected_final_text, "process_document did not return the expected placeholder result string.")
        logging.info(f"OCRWorkflowOrchestrator process_document successful placeholder flow returned: {result}")


    def test_process_document_image_not_found(self):
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        non_existent_image_path = os.path.join(self.test_dir, "non_existent_test_image.png")
        result = orchestrator.process_document(non_existent_image_path)

        self.assertTrue(result.startswith("Error: File not found"), f"Result '{result}' does not indicate FileNotFoundError correctly.")
        logging.info("OCRWorkflowOrchestrator process_document handled non-existent image file correctly.")

    def test_get_results_formatting(self):
        orchestrator = OCRWorkflowOrchestrator(config_path=self.config_path)
        
        # Test with valid data
        processed_data_ok = "Some processed text"
        formatted_ok = orchestrator.get_results(processed_data_ok)
        self.assertEqual(formatted_ok, f"Formatted Results: {processed_data_ok}")

        # Test with None data
        formatted_none = orchestrator.get_results(None)
        self.assertEqual(formatted_none, "No valid data to format. Processing status: None")
        
        # Test with error string data
        processed_data_error = "Error: Something went wrong"
        formatted_error = orchestrator.get_results(processed_data_error)
        self.assertEqual(formatted_error, f"No valid data to format. Processing status: {processed_data_error}")


if __name__ == '__main__':
    # Add /app to sys.path to ensure modules can be found if run directly
    import sys
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
         sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    unittest.main()
