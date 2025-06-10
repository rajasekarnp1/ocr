# Manual GUI Test Plan for OCR Application (`gui_app.py`)

This document outlines a manual test plan for the Graphical User Interface (GUI) of the OCR application, implemented in `gui_app.py`.

## Prerequisites for All Tests:
*   The Python environment has all necessary dependencies installed: `customtkinter`, `Pillow`, and all OCR pipeline modules (`opencv-python`, `numpy`, `onnxruntime`, `PyYAML`).
*   **Crucially, a working `tkinter` installation is required for `customtkinter` to run.** If `tkinter` is missing, the GUI will not launch.
*   The application modules (`ocr_workflow_orchestrator.py`, `config_loader.py`, etc.) are correctly implemented and located in the `/app` directory or accessible via PYTHONPATH.
*   A default `config_dev.yaml` should be present or auto-generatable by the application (as handled by `config_loader.py` and `ocr_workflow_orchestrator.py`). This includes paths to dummy ONNX models and `default_dict.txt`.
*   Dummy ONNX models (`dummy_geometric_model.onnx`, `dummy_recognition_model.onnx`) and `default_dict.txt` should exist in their configured locations (typically `/app/models/` and `/app/` respectively).
*   Have a few sample image files (e.g., PNG, JPG) available for testing the "Browse Image" functionality. A dummy image like `dummy_image_for_orchestrator.png` (if created by running `ocr_workflow_orchestrator.py`'s main block) can also be used.

## Test Environment:
*   Operating System: (Specify OS, e.g., Windows 10, Ubuntu 22.04, macOS Monterey)
*   Python Version: (Specify Python version, e.g., 3.10.x)
*   Key Library Versions: `customtkinter` (e.g., 5.2.2), `Pillow` (e.g., 10.x.x)

---

### Test Area: Application Launch and Initial State

**Test Case ID:** GUI_TC_001
*   **Description:** Verify the application launches correctly and GUI elements are in their initial state.
*   **Test Steps:**
    1.  Navigate to the `/app` directory in the terminal.
    2.  Run the command: `python gui_app.py`.
*   **Expected Result:**
    *   The GUI window titled "OCR Application - Integrated Pipeline" appears.
    *   Window dimensions are approximately 800x600.
    *   The "Browse Image..." button is visible and enabled.
    *   The "Run OCR" button is visible but **disabled**.
    *   The image display area shows the text "No image selected."
    *   The OCR output text area shows placeholder text like "OCR results will appear here." and is read-only.
    *   The status bar at the bottom shows "Ready." (or "Critical Error: OCR system failed to initialize." if orchestrator setup failed, in which case buttons should be disabled).

**Test Case ID:** GUI_TC_002
*   **Description:** Verify behavior if the OCR orchestrator fails to initialize during application startup.
*   **Prerequisites:** Modify `config_dev.yaml` to point to a non-existent model file for `GeometricCorrector` or `ONNXRecognizer` to force an initialization error in `OCRWorkflowOrchestrator`.
*   **Test Steps:**
    1.  Modify `config_dev.yaml` as described in prerequisites.
    2.  Run `python gui_app.py`.
*   **Expected Result:**
    *   An error dialog from `CTkMessagebox` appears, titled "Initialization Error" or "Critical Error", indicating failure to initialize the OCR system (e.g., "Failed to initialize OCRWorkflowOrchestrator: Failed to initialize OCR processing modules. Caused by: OCRFileNotFoundError: Geometric correction model file not found...").
    *   After closing the dialog, the main GUI window might appear (or the app might exit depending on error severity).
    *   If the window appears, the status bar should show "Critical Error: OCR system failed to initialize. Check logs."
    *   Both "Browse Image..." and "Run OCR" buttons should be **disabled**.

---

### Test Area: Image Selection and Display

**Test Case ID:** GUI_TC_003
*   **Description:** Verify successful image selection, display, and enabling of the "Run OCR" button.
*   **Prerequisites:** Application launched successfully (GUI_TC_001 passed). Have a valid image file (e.g., `sample.png`) ready.
*   **Test Steps:**
    1.  Click the "Browse Image..." button.
    2.  A file dialog opens. Select a valid image file (e.g., PNG, JPG).
    3.  Click "Open" in the file dialog.
*   **Expected Result:**
    *   The selected image is displayed in the image display area. The placeholder text "No image selected." is replaced.
    *   The status bar updates to show the name of the selected image (e.g., "Selected: sample.png").
    *   The "Run OCR" button becomes **enabled**.
    *   Logs in the console should indicate the image selection and display.

**Test Case ID:** GUI_TC_004
*   **Description:** Verify behavior when the user cancels the file dialog.
*   **Prerequisites:** Application launched successfully.
*   **Test Steps:**
    1.  Click the "Browse Image..." button.
    2.  A file dialog opens. Click "Cancel" (or close the dialog).
*   **Expected Result:**
    *   No image is displayed (or the previous image/placeholder text remains).
    *   The status bar text updates to "No image selected." or remains unchanged if no image was previously selected.
    *   The "Run OCR" button remains **disabled** (or becomes disabled if an image was previously loaded).

**Test Case ID:** GUI_TC_005
*   **Description:** Verify behavior when an invalid/corrupted image file is selected.
*   **Prerequisites:** Application launched successfully. Have an invalid or corrupted image file ready.
*   **Test Steps:**
    1.  Click the "Browse Image..." button.
    2.  Select the invalid/corrupted image file.
    3.  Click "Open".
*   **Expected Result:**
    *   The image display area shows an error message (e.g., "Error loading image...").
    *   The status bar updates to show an error message (e.g., "Error loading image: filename.ext").
    *   The "Run OCR" button remains **disabled** or becomes disabled.
    *   Console logs should show an error related to image loading/display.

---

### Test Area: OCR Execution and Results Display

**Test Case ID:** GUI_TC_006
*   **Description:** Verify successful OCR execution and display of results for a selected image.
*   **Prerequisites:** A valid image has been selected and displayed (GUI_TC_003 passed).
*   **Test Steps:**
    1.  Click the "Run OCR" button.
*   **Expected Result:**
    *   The "Run OCR" and "Browse Image..." buttons become **disabled**.
    *   The status bar updates to "Processing OCR...".
    *   The OCR output text area is cleared and shows "Processing, please wait...".
    *   After a short delay (simulating processing or actual processing if fast), the OCR output text area is updated with the (dummy) OCR results (e.g., "Formatted Results: Spell-Checked Text: ...").
    *   The status bar updates to "OCR complete." (or similar success message).
    *   The "Run OCR" and "Browse Image..." buttons become **re-enabled**.
    *   The GUI should remain responsive during the (brief) processing period.

**Test Case ID:** GUI_TC_007
*   **Description:** Verify behavior if "Run OCR" is clicked without an image selected.
*   **Prerequisites:** Application launched successfully, no image selected, or previous image selection failed.
*   **Test Steps:**
    1.  Ensure "Run OCR" button is somehow active (e.g., if a previous image was loaded then cleared, or if the button state logic has a bug). Or, directly attempt to call the action if testing programmatically. (Manual test: button should be disabled).
    2.  If enabled, click "Run OCR".
*   **Expected Result:**
    *   A `CTkMessagebox` with title "Error" should appear stating "Please select an image first...".
    *   The status bar should show an error message (e.g., "Error: No image selected.").
    *   No OCR processing should start. "Run OCR" button should remain disabled or return to its previous state.

---

### Test Area: Error Handling During OCR Processing

**Test Case ID:** GUI_TC_008
*   **Description:** Verify GUI behavior if the OCR pipeline (orchestrator) encounters an error during processing.
*   **Prerequisites:**
    *   A valid image is selected.
    *   Simulate an error condition in the OCR pipeline. This might involve:
        *   Modifying `config_dev.yaml` to point to a non-existent model that is only loaded during `process_document` (if applicable, though most are loaded at init).
        *   Temporarily modifying a sub-module (e.g., `ONNXRecognizer.predict`) to artificially raise an `OCRPipelineError` or other exception.
*   **Test Steps:**
    1.  Set up the error condition as per prerequisites.
    2.  Select a valid image.
    3.  Click "Run OCR".
*   **Expected Result:**
    *   "Run OCR" and "Browse Image..." buttons are initially disabled.
    *   Status bar shows "Processing OCR...".
    *   After processing attempt, the OCR output text area displays an error message (e.g., "OCR Error: OCRPipelineError: Details of the error...").
    *   The status bar updates to "OCR failed with error." or similar.
    *   "Run OCR" and "Browse Image..." buttons are re-enabled.
    *   Console logs should show details of the exception from the OCR pipeline.

---

### Test Area: GUI Responsiveness

**Test Case ID:** GUI_TC_009
*   **Description:** Verify the GUI remains responsive while the (threaded) OCR process is running.
*   **Prerequisites:** A valid image is selected. The OCR process should have a noticeable (even if simulated) delay. (The current `self.after(2000, ...)` simulates this for placeholder, actual pipeline might be fast/slow).
*   **Test Steps:**
    1.  Select a valid image.
    2.  Click "Run OCR".
    3.  While the status bar shows "Processing OCR...", attempt to:
        *   Move the GUI window.
        *   Minimize and restore the window.
        *   (If other interactive elements were present, try interacting with them).
*   **Expected Result:**
    *   The GUI window should remain responsive (movable, minimizable/restorable) during the OCR processing.
    *   Buttons ("Browse Image...", "Run OCR") should correctly remain disabled during this period.

---

This manual test plan covers the primary functionalities and error conditions of the GUI application.
