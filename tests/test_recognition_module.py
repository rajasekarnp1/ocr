import unittest
import os
import logging
import shutil
from recognition_module import ONNXRecognizer

# Attempt to import Mojo to check availability. This will determine if Mojo-dependent tests can run.
try:
    from mojo.mojo.python import Python
    mojo_available = True
    # Try to import the specific module to see if it's found by the environment
    # This is a preliminary check; the actual import is in the predict method.
    # Python.import_module("mojo_recognizer_utils") # This might raise if file not found by Mojo
except (ImportError, ModuleNotFoundError):
    mojo_available = False
    logging.warning("Mojo SDK not found or 'mojo_recognizer_utils.mojo' not discoverable. Mojo-dependent tests will be skipped.")
except Exception as e: # Catch any other error during Mojo import for robustness
    mojo_available = False
    logging.error(f"An unexpected error occurred while trying to import Mojo or mojo_recognizer_utils: {e}. Assuming Mojo is unavailable.")


class TestONNXRecognizer(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_recognition_temp_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.dummy_model_filename = "dummy_ocr_model_test.onnx"
        self.dummy_model_path = os.path.join(self.test_dir, self.dummy_model_filename)

        with open(self.dummy_model_path, "w") as f:
            f.write("dummy onnx model data for testing recognizer")

        self.addCleanup(self.cleanup_test_dir)

        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create dummy mojo_recognizer_utils.mojo if it doesn't exist, so Python.import_module doesn't fail on file not found
        # This is only for the Python side of import_module to find *a* file. Actual Mojo execution still needs SDK.
        self.dummy_mojo_file_path = "mojo_recognizer_utils.mojo"
        if not os.path.exists(self.dummy_mojo_file_path) and not mojo_available: # Only create if real one is not there due to no SDK
            logging.info(f"Creating a dummy '{self.dummy_mojo_file_path}' for Python import testing as Mojo SDK seems unavailable.")
            with open(self.dummy_mojo_file_path, "w") as f:
                f.write("# Dummy Mojo file for Python import testing\nfn main():\n  print(\"Dummy Mojo main\")\n")


    def cleanup_test_dir(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        # Remove the dummy mojo file if it was created by setUp
        # Be cautious not to remove a real one if mojo_available was true but file was missing.
        # This logic is tricky; for now, only remove if mojo_available is False.
        if not mojo_available and os.path.exists(self.dummy_mojo_file_path) and "Dummy Mojo main" in open(self.dummy_mojo_file_path).read():
             # os.remove(self.dummy_mojo_file_path) # Keep it for now to avoid deleting actual file if logic is flawed
             pass


    def test_successful_initialization(self):
        try:
            recognizer = ONNXRecognizer(model_path=self.dummy_model_path)
            self.assertIsNotNone(recognizer.session, "Session should be initialized.")
            self.assertEqual(recognizer.model_path, self.dummy_model_path)
            logging.info("ONNXRecognizer initialized successfully for test.")
        except Exception as e:
            self.fail(f"ONNXRecognizer initialization failed unexpectedly: {e}")

    def test_initialization_model_not_found(self):
        non_existent_model_path = os.path.join(self.test_dir, "non_existent_ocr_model.onnx")
        with self.assertRaises(FileNotFoundError):
            ONNXRecognizer(model_path=non_existent_model_path)
        logging.info("ONNXRecognizer correctly raised FileNotFoundError for non-existent model.")

    def test_predict_method_placeholder_logic_no_mojo(self):
        recognizer = ONNXRecognizer(model_path=self.dummy_model_path)

        # Temporarily "disable" Mojo for this test by ensuring Python.import_module fails
        # This tests the Python-only path of the predict method.
        original_python_import_module = None
        if mojo_available: # Only if mojo was found initially
            try:
                # This is a bit of a hack. If mojo.mojo.python.Python is a real module,
                # we try to mock its import_module method.
                from mojo.mojo.python import Python
                original_python_import_module = Python.import_module
                def mock_import_module_that_fails(module_name):
                    logging.info(f"Mocked Python.import_module for '{module_name}' - raising ImportError to simulate Mojo unavailability.")
                    raise ImportError("Simulated Mojo import failure for testing Python-only path")
                Python.import_module = mock_import_module_that_fails
            except Exception as e:
                logging.warning(f"Could not mock Python.import_module for testing no-Mojo path: {e}. Test might not fully isolate Python logic.")


        mock_image_data = "SamplePreprocessedImageData"
        text, confidence = recognizer.predict(mock_image_data)

        # Restore original import_module if it was mocked
        if original_python_import_module and mojo_available:
            from mojo.mojo.python import Python
            Python.import_module = original_python_import_module

        # Check placeholder ONNX output
        expected_text_part = f"Raw OCR from {os.path.basename(self.dummy_model_path)}"
        self.assertTrue(text.startswith(expected_text_part), f"Predict method text output '{text}' does not match expected placeholder output.")
        self.assertEqual(confidence, 0.92, "Predict method confidence does not match placeholder output.")
        logging.info("ONNXRecognizer predict method (Python-only path) returned expected placeholder output.")

    def test_predict_method_with_none_input(self):
        recognizer = ONNXRecognizer(model_path=self.dummy_model_path)
        text, confidence = recognizer.predict(None)
        self.assertIsNone(text, "Text should be None for None input.")
        self.assertEqual(confidence, 0.0, "Confidence should be 0.0 for None input.")

    @unittest.skipUnless(mojo_available, "Mojo SDK and/or mojo_recognizer_utils.mojo not available or not correctly configured.")
    def test_predict_method_with_mojo_call_expected_success(self):
        """
        Tests the predict method including the call to the Mojo function.
        This test requires the Mojo SDK to be installed and mojo_recognizer_utils.mojo
        to be in the correct path (e.g., same directory or in MODULE_PATH/PYTHONPATH).
        If Mojo is properly set up, this test should pass and show logs from Mojo.
        The dummy mojo_recognizer_utils.mojo created in setUp won't be used if a real one is found by Mojo.
        """
        # Ensure the real mojo_recognizer_utils.mojo (created in previous subtask) is present
        # This test relies on that file being present in /app/
        real_mojo_file_path = "mojo_recognizer_utils.mojo"
        if not os.path.exists(real_mojo_file_path):
            self.skipTest(f"Real '{real_mojo_file_path}' not found. Cannot test Mojo integration.")

        recognizer = ONNXRecognizer(model_path=self.dummy_model_path)
        mock_image_data = "SamplePreprocessedImageDataForMojoTest"

        # The predict method internally calls the Mojo function.
        # We expect it to log information about the Mojo call and its results.
        # We also expect the ONNX placeholder output.
        text, confidence = recognizer.predict(mock_image_data)

        expected_text_part = f"Raw OCR from {os.path.basename(self.dummy_model_path)}"
        self.assertTrue(text.startswith(expected_text_part))
        self.assertEqual(confidence, 0.92)

        # To truly verify Mojo's effect, you'd check logs or if Mojo modified something.
        # The current Mojo function `example_mojo_tensor_operation` returns a list,
        # and the Python code logs it. We can't directly assert its output here without
        # capturing logs or modifying the predict method to return it.
        # For now, a successful run without error and with expected ONNX output is a pass.
        logging.info("ONNXRecognizer predict method with Mojo call attempt completed. Check logs for Mojo output.")
        # Example: If the dummy_tensor_data was [10, 20, 30, 40, 50], Mojo output should be [25, 45, 65, 85, 105]
        # This test assumes that if Mojo runs, it will print to stdout/stderr, which will appear in test logs.


    def test_predict_method_mojo_call_graceful_failure_if_mojo_file_is_missing_but_sdk_present(self):
        """
        This test simulates a scenario where Mojo SDK is present, but the specific .mojo file is not.
        The Python.import_module call should ideally raise an error that is caught.
        """
        if not mojo_available:
            self.skipTest("Mojo SDK itself is not available, cannot test this scenario.")

        # Ensure the target .mojo file does NOT exist for this test
        original_mojo_file_path = "mojo_recognizer_utils.mojo" # Assuming it's in /app
        temp_renamed_path = "mojo_recognizer_utils.mojo.renamed_for_test"

        mojo_file_actually_exists = os.path.exists(original_mojo_file_path)

        if mojo_file_actually_exists:
            try:
                os.rename(original_mojo_file_path, temp_renamed_path)
                logging.info(f"Temporarily renamed '{original_mojo_file_path}' to '{temp_renamed_path}' for test.")
            except OSError as e:
                self.fail(f"Failed to rename mojo file for testing: {e}")
        else:
            # If the file doesn't exist in the first place, this test's premise is met.
            logging.info(f"'{original_mojo_file_path}' does not exist, proceeding with test.")


        recognizer = ONNXRecognizer(model_path=self.dummy_model_path)
        mock_image_data = "SampleDataForMojoFileMissingTest"

        # Call predict. The try-except block around Python.import_module in predict should catch the error.
        # The method should still return the ONNX placeholder output.
        text, confidence = recognizer.predict(mock_image_data)

        expected_text_part = f"Raw OCR from {os.path.basename(self.dummy_model_path)}"
        self.assertTrue(text.startswith(expected_text_part), "Predict should return ONNX output even if Mojo file import fails.")
        self.assertEqual(confidence, 0.92)
        logging.info("ONNXRecognizer predict method handled missing .mojo file gracefully (as expected).")

        # Restore the .mojo file if it was renamed
        if mojo_file_actually_exists:
            try:
                os.rename(temp_renamed_path, original_mojo_file_path)
                logging.info(f"Restored '{original_mojo_file_path}'.")
            except OSError as e:
                # This is problematic for subsequent tests if it fails.
                logging.error(f"CRITICAL: Failed to restore '{original_mojo_file_path}'. Subsequent tests might be affected. Error: {e}")
                # self.fail(f"Failed to restore mojo file after testing: {e}") # Optional: make test fail loudly


if __name__ == '__main__':
    # This allows running the test file directly.
    # It's important that Python can find 'recognition_module.py' from this 'tests' directory.
    # This often means the parent directory ('/app') should be in PYTHONPATH.
    # For `python -m unittest discover tests` from /app, this is handled.
    # If running `python tests/test_recognition_module.py` directly, ensure PYTHONPATH is set.

    # Add /app to sys.path to ensure modules can be found if run directly
    import sys
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
         sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    unittest.main()
