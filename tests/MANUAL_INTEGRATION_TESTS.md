# Manual Integration Tests for OCR Application CLI (`main.py`)

This document outlines manual integration test cases for the command-line interface (`main.py`) of the OCR application. These tests verify the end-to-end functionality of the OCR pipeline when invoked via the CLI.

## Prerequisites for All Tests:
*   The Python environment should have all necessary dependencies installed (e.g., `opencv-python`, `numpy`, `onnxruntime`, `PyYAML`, `customtkinter`, `Pillow`).
*   The application modules (`ocr_workflow_orchestrator.py`, `preprocessing_module.py`, etc.) must be correctly implemented and importable.
*   A default `config_dev.yaml` should be generatable by the application if not present (e.g., by running `python ocr_workflow_orchestrator.py` once, or `python main.py path/to/dummy_image.png` which should trigger its creation if missing).
*   Dummy ONNX models (`dummy_geometric_model.onnx`, `dummy_recognition_model.onnx`) should be present in the `models/` directory, and `default_dict.txt` should exist, as per the default configuration. The `ocr_workflow_orchestrator.py` or `config_loader.py`'s default creation logic should handle these.
*   A dummy image file (e.g., `dummy_image_for_orchestrator.png` created by `ocr_workflow_orchestrator.py`, or a simple user-created PNG/JPG) should be available for testing. For these tests, let's assume `dummy_image_for_orchestrator.png` exists in `/app/`.

---

### Test Area: Basic CLI Operations and Successful OCR Processing

**Test Case ID:** CLI_TC_001
*   **Description:** Verify successful OCR processing of a valid image using the default configuration, with image path as a positional argument.
*   **Prerequisites:**
    *   `dummy_image_for_orchestrator.png` exists in `/app/`.
    *   Default `config_dev.yaml` is present or can be auto-generated correctly.
*   **Command:** `python main.py dummy_image_for_orchestrator.png`
*   **Expected Outcome:**
    *   The script executes without unhandled Python exceptions.
    *   Logs from various modules (INFO level by default) are printed to stdout/stderr.
    *   The final output to stdout includes "--- OCR Results ---", followed by formatted results including "Spell-Checked Text:", "Cleaned Text:", "Original OCR Text:", and "Confidence:".
    *   The text content will be based on dummy model operations (e.g., "DummyTextOutputfromshape...").
    *   Exit code should be 0.

**Test Case ID:** CLI_TC_002
*   **Description:** Verify successful OCR processing of a valid image using the default configuration, with image path via `--image` option.
*   **Prerequisites:** Same as CLI_TC_001.
*   **Command:** `python main.py --image dummy_image_for_orchestrator.png`
*   **Expected Outcome:** Same as CLI_TC_001.

**Test Case ID:** CLI_TC_003
*   **Description:** Verify successful OCR processing using a specified valid configuration file.
*   **Prerequisites:**
    *   `dummy_image_for_orchestrator.png` exists in `/app/`.
    *   A valid configuration file `config_dev.yaml` exists (or a copy, e.g., `custom_config_test.yaml` with similar valid content). For this test, using the auto-generated `config_dev.yaml` is sufficient.
*   **Command:** `python main.py --image dummy_image_for_orchestrator.png --config config_dev.yaml`
*   **Expected Outcome:** Same as CLI_TC_001.

**Test Case ID:** CLI_TC_004
*   **Description:** Verify usage of the `--verbose` flag to enable DEBUG level logging.
*   **Prerequisites:** Same as CLI_TC_001.
*   **Command:** `python main.py --image dummy_image_for_orchestrator.png --verbose`
*   **Expected Outcome:**
    *   Same OCR results as CLI_TC_001.
    *   Log output should be more verbose, including DEBUG level messages from various modules (e.g., detailed shapes, parameters, Mojo fallback messages if applicable).
    *   The initial log from `OCRMain` should state "Verbose DEBUG logging enabled...".

---

### Test Area: Error Handling - File Issues

**Test Case ID:** CLI_TC_005
*   **Description:** Test error handling when the input image file is not found.
*   **Prerequisites:** None (ensure the specified image does *not* exist).
*   **Command:** `python main.py --image non_existent_image.png`
*   **Expected Outcome:**
    *   A user-friendly error message is printed to stderr (e.g., "Error: Input image file not found at 'non_existent_image.png'. Please check the path.").
    *   Relevant error logs appear from `OCRMain`.
    *   The script exits with a non-zero exit code (e.g., 1).
    *   No Python traceback should be shown to the user on stderr for this handled error.

**Test Case ID:** CLI_TC_006
*   **Description:** Test error handling when the specified config file is not found.
*   **Prerequisites:** `dummy_image_for_orchestrator.png` exists.
*   **Command:** `python main.py --image dummy_image_for_orchestrator.png --config non_existent_config.yaml`
*   **Expected Outcome:**
    *   A user-friendly error message is printed to stderr (e.g., "Error: A required file was not found. Details: Configuration file 'non_existent_config.yaml' not found...").
    *   Relevant error logs appear (e.g., from `OCRMain` and `config_loader`).
    *   The script exits with a non-zero exit code.

---

### Test Area: Error Handling - Configuration Issues

**Test Case ID:** CLI_TC_007
*   **Description:** Test error handling when the specified config file is malformed or invalid.
*   **Prerequisites:**
    *   `dummy_image_for_orchestrator.png` exists.
    *   Create a malformed config file, e.g., `malformed_config.yaml` with invalid YAML syntax or incorrect data types for expected values.
        Example `malformed_config.yaml`:
        ```yaml
        app_settings:
          version: 1.0
        logging:
          version: "should_be_int" # Invalid type
          handlers:
            console: console_is_not_a_dict
        ```
*   **Command:** `python main.py --image dummy_image_for_orchestrator.png --config malformed_config.yaml`
*   **Expected Outcome:**
    *   A user-friendly error message is printed to stderr (e.g., "Error: There was a problem with the application configuration. Details: Failed to parse configuration file...").
    *   Detailed error logs appear, including information about the parsing failure or configuration validation error.
    *   The script exits with a non-zero exit code.

**Test Case ID:** CLI_TC_008
*   **Description:** Test error handling if a model file path specified in a valid config file is missing.
*   **Prerequisites:**
    *   `dummy_image_for_orchestrator.png` exists.
    *   Create a config file `config_missing_model.yaml` that is structurally valid YAML but points to a non-existent ONNX model file (e.g., `models/missing_model.onnx` in `preprocessing_settings` or `recognition_settings`).
        Example `config_missing_model.yaml` (copy `config_dev.yaml` and change a model path):
        ```yaml
        # ... (other settings copied from a valid config_dev.yaml)
        app_settings:
          # ...
          preprocessing_settings:
            model_path: "models/actually_missing_geom_model.onnx" # This file should not exist
          recognition_settings:
            svtr_recognizer: "models/dummy_recognition_model.onnx" # Assume this one is OK
          # ...
        ```
*   **Command:** `python main.py --image dummy_image_for_orchestrator.png --config config_missing_model.yaml`
*   **Expected Outcome:**
    *   A user-friendly error message is printed to stderr (e.g., "Error: An error occurred during the OCR processing pipeline. Details: Failed to initialize OCR processing modules... Caused by: OCRFileNotFoundError: Geometric correction model file not found...").
    *   Detailed error logs appear, indicating failure to initialize the orchestrator due to a module (e.g., `GeometricCorrector`) failing to load its model.
    *   The script exits with a non-zero exit code.

---

### Test Area: CLI Argument Parsing

**Test Case ID:** CLI_TC_009
*   **Description:** Test providing image path both positionally and with `--image` option (should ideally be handled by argparse or custom logic).
*   **Prerequisites:** `dummy_image_for_orchestrator.png` exists.
*   **Command:** `python main.py dummy_image_for_orchestrator.png --image dummy_image_for_orchestrator.png`
*   **Expected Outcome:**
    *   The script should run successfully (as in CLI_TC_001) if paths are identical. If paths are different, `main.py` currently has logic to error out: "Please provide the image path either positionally or with --image, not both differently." This error should be shown if different paths are given.
    *   If paths are identical, runs successfully.
    *   If paths are different (e.g., `python main.py image1.png --image image2.png`), it should show the argparse error.

**Test Case ID:** CLI_TC_010
*   **Description:** Test CLI without providing any image path.
*   **Prerequisites:** None.
*   **Command:** `python main.py`
*   **Expected Outcome:**
    *   Argparse should show a usage error message indicating that the image path is required.
    *   The script should exit with a non-zero exit code (typically 2 for argparse errors).

**Test Case ID:** CLI_TC_011
*   **Description:** Test CLI help message.
*   **Prerequisites:** None.
*   **Command:** `python main.py --help`
*   **Expected Outcome:**
    *   The help message defined in `main.py`'s `ArgumentParser` is displayed, showing usage, description, argument details, and the epilog.
    *   The script exits with code 0.

---

This set of manual integration tests covers the main success paths and various error conditions for the CLI application.
