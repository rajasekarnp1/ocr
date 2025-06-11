import sys
import os
import logging
import yaml # For dummy config in __main__

# Add project root to sys.path for imports if running directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QComboBox, QTextEdit, QFileDialog,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSlot

# Assuming orchestrator and config loader are accessible
from ocr_workflow_orchestrator import OCRWorkflowOrchestrator
from config_loader import load_config # For __main__

# --- Configuration for __main__ dummy setup ---
DUMMY_UI_CONFIG_FILE = "temp_ui_config.yaml"
DUMMY_UI_MODEL_DIR = "ui_dummy_models"
DUMMY_UI_DET_MODEL = os.path.join(DUMMY_UI_MODEL_DIR, "dummy_det.onnx")
DUMMY_UI_REC_MODEL = os.path.join(DUMMY_UI_MODEL_DIR, "dummy_rec.onnx")
DUMMY_UI_CHARS_FILE = os.path.join(DUMMY_UI_MODEL_DIR, "dummy_chars.txt")


class MainWindow(QMainWindow):
    """
    Main window for the OCR-X Client UI.
    """
    def __init__(self, orchestrator: OCRWorkflowOrchestrator, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__) # Get logger from orchestrator or new one

        self.setWindowTitle("OCR-X Client MVP")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        self._selected_file_path: Optional[str] = None
        self._init_ui()
        self.logger.info("MainWindow initialized.")

    def _init_ui(self):
        """Initialize UI elements."""
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # File Selection
        file_selection_layout = QHBoxLayout()
        self.select_file_button = QPushButton("Select Image/PDF File")
        self.select_file_button.clicked.connect(self._select_file)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        file_selection_layout.addWidget(self.select_file_button)
        file_selection_layout.addWidget(self.file_path_edit)
        self.main_layout.addLayout(file_selection_layout)

        # Engine Selection
        engine_selection_layout = QHBoxLayout()
        engine_label = QLabel("Select OCR Engine:")
        self.engine_combo = QComboBox()

        available_engines = self.orchestrator.engine_manager.get_available_engines()
        if available_engines:
            for engine_name in available_engines:
                # Get the user-friendly name from the engine instance if possible
                engine_instance = self.orchestrator.engine_manager.get_engine(engine_name)
                display_name = engine_instance.get_engine_name() if engine_instance else engine_name
                self.engine_combo.addItem(display_name, engine_name) # Store internal name as data
            self.engine_combo.currentIndexChanged.connect(self._update_api_key_field_visibility)
        else:
            self.engine_combo.addItem("No engines available")
            self.engine_combo.setEnabled(False)
            self.logger.warning("No OCR engines available to populate engine selection ComboBox.")

        engine_selection_layout.addWidget(engine_label)
        engine_selection_layout.addWidget(self.engine_combo)
        self.main_layout.addLayout(engine_selection_layout)

        # API Key Input (Conditional for Google Cloud Vision)
        self.api_key_layout = QHBoxLayout()
        self.api_key_label = QLabel("Google API Key Path:")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Path to service_account_key.json (if Google Engine selected)")
        self.api_key_layout.addWidget(self.api_key_label)
        self.api_key_layout.addWidget(self.api_key_edit)
        self.main_layout.addLayout(self.api_key_layout)

        # Initially hide API key field; visibility updated by engine selection
        self._update_api_key_field_visibility()


        # OCR Trigger Button
        self.run_ocr_button = QPushButton("Run OCR")
        self.run_ocr_button.clicked.connect(self._run_ocr)
        self.main_layout.addWidget(self.run_ocr_button)

        # Results Display
        results_label = QLabel("OCR Results:")
        self.results_text_edit = QTextEdit()
        self.results_text_edit.setReadOnly(True)
        self.main_layout.addWidget(results_label)
        self.main_layout.addWidget(self.results_text_edit)

        # Status Bar
        self.statusBar().showMessage("Ready. Select a file and engine.")

    @pyqtSlot()
    def _select_file(self):
        """Open file dialog to select an image or PDF file."""
        # For PDF, additional handling/library (e.g., pdf2image) would be needed in orchestrator/preprocessor
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", file_filter)
        if path:
            self._selected_file_path = path
            self.file_path_edit.setText(path)
            self.statusBar().showMessage(f"File selected: {os.path.basename(path)}")
            self.logger.info(f"User selected file: {path}")

    @pyqtSlot()
    def _update_api_key_field_visibility(self):
        """Show or hide the API key input field based on selected engine."""
        current_engine_internal_name = self.engine_combo.currentData()
        # Check if the selected engine is Google Cloud Vision (based on its internal name or display name)
        # This relies on the 'name' field in the config or the engine's get_engine_name()
        is_google_engine = False
        if current_engine_internal_name: # currentData() holds the key (internal name)
             engine_instance = self.orchestrator.engine_manager.get_engine(current_engine_internal_name)
             if engine_instance and "google" in engine_instance.get_engine_name().lower():
                 is_google_engine = True

        self.api_key_label.setVisible(is_google_engine)
        self.api_key_edit.setVisible(is_google_engine)
        if is_google_engine:
            self.logger.debug("Google engine selected, showing API key field.")
            # For MVP, this field is informational. The actual key path is from engine's initial config.
            # A real app might use this to update config or re-init the engine.
            # Populate with configured path if available:
            g_engine_cfg = self.orchestrator.config.get('ocr_engines', {}).get(current_engine_internal_name, {}).get('config', {})
            self.api_key_edit.setText(g_engine_cfg.get('api_key_path', ''))
        else:
            self.logger.debug("Non-Google engine selected, hiding API key field.")


    @pyqtSlot()
    def _run_ocr(self):
        """Execute the OCR process."""
        if not self._selected_file_path:
            QMessageBox.warning(self, "No File", "Please select an image file first.")
            self.statusBar().showMessage("Error: No file selected.")
            return

        if not os.path.exists(self._selected_file_path):
            QMessageBox.critical(self, "File Error", f"File not found: {self._selected_file_path}")
            self.statusBar().showMessage(f"Error: File not found - {os.path.basename(self._selected_file_path)}")
            return

        selected_engine_internal_name = self.engine_combo.currentData() # Get internal name
        if not selected_engine_internal_name:
            QMessageBox.warning(self, "No Engine", "No OCR engine selected or available.")
            self.statusBar().showMessage("Error: No engine selected.")
            return

        self.logger.info(f"Running OCR on '{self._selected_file_path}' with engine key '{selected_engine_internal_name}'.")
        self.results_text_edit.clear()
        self.statusBar().showMessage(f"Processing with {self.engine_combo.currentText()}...")
        QApplication.processEvents()  # Allow UI to update

        # --- Threading Note ---
        # In a real application, the following orchestrator call MUST be in a separate QThread
        # to prevent the UI from freezing during potentially long OCR operations.
        # For this MVP, it's synchronous. Example:
        # self.ocr_worker = OCRWorker(self.orchestrator, self._selected_file_path, selected_engine_internal_name)
        # self.ocr_worker.finished.connect(self._on_ocr_finished)
        # self.ocr_worker.start()

        try:
            # The API key path from the UI is currently informational for MVP.
            # The GoogleCloudOCREngine instance uses the api_key_path from its initial configuration.
            # A more advanced implementation might allow dynamic updates.
            # For now, we just log if the user provided something in the UI field for Google.
            if self.api_key_label.isVisible() and self.api_key_edit.text():
                self.logger.info(f"User provided API key path in UI: {self.api_key_edit.text()} (informational for MVP).")

            result = self.orchestrator.process_document(
                self._selected_file_path,
                requested_engine_name=selected_engine_internal_name
            )

            if result and "text" in result:
                self.results_text_edit.setText(result.get("text", ""))
                confidence = result.get("confidence", "N/A")
                engine_used = result.get("engine_name", selected_engine_internal_name)
                self.statusBar().showMessage(f"OCR Complete with {engine_used}. Overall Confidence: {confidence if isinstance(confidence, float) else 'N/A'}")
                self.logger.info(f"OCR successful. Engine: {engine_used}, Confidence: {confidence}")
            elif result and "error" in result:
                error_msg = result.get("error", "Unknown error during OCR.")
                self.results_text_edit.setText(f"OCR Failed: {error_msg}")
                self.statusBar().showMessage(f"OCR Failed. See details in text area.")
                self.logger.error(f"OCR failed: {error_msg}")
                QMessageBox.critical(self, "OCR Error", f"OCR process failed: {error_msg}")
            else: # Should not happen if orchestrator always returns dict with text or error
                self.results_text_edit.setText("OCR Failed: No result or unexpected format from orchestrator.")
                self.statusBar().showMessage("OCR Failed: Unexpected error.")
                self.logger.error("OCR failed: No result or unexpected format from orchestrator.")
                QMessageBox.warning(self, "OCR Error", "OCR process failed to return a valid result.")

        except FileNotFoundError as fnf_error: # Should be caught by pre-check, but as safety
            self.logger.error(f"File not found during OCR processing: {fnf_error}", exc_info=True)
            QMessageBox.critical(self, "File Error", str(fnf_error))
            self.statusBar().showMessage(f"Error: {fnf_error}")
        except RuntimeError as rt_error: # E.g., engine not available or processing pipeline error
             self.logger.error(f"Runtime error during OCR processing: {rt_error}", exc_info=True)
             QMessageBox.critical(self, "Processing Error", str(rt_error))
             self.statusBar().showMessage(f"Error: {rt_error}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during OCR: {e}", exc_info=True)
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {str(e)}")
            self.statusBar().showMessage("Error: An unexpected error occurred.")


def setup_dummy_config_for_ui_test():
    """Creates a dummy config file and necessary dummy model files for UI testing."""
    # Create dummy model files for LocalOCREngine (so it can be listed as "available" if init checks paths)
    os.makedirs(DUMMY_UI_MODEL_DIR, exist_ok=True)
    with open(DUMMY_UI_DET_MODEL, 'w') as f: f.write("dummy_onnx_det")
    with open(DUMMY_UI_REC_MODEL, 'w') as f: f.write("dummy_onnx_rec")
    with open(DUMMY_UI_CHARS_FILE, 'w') as f: f.write("a\nb\nc")

    dummy_config_content = f"""
app_settings:
  default_ocr_engine: "local_engine_for_ui_test"

ocr_engines:
  local_engine_for_ui_test:
    enabled: true
    module: "ocr_components.local_ocr_engine"
    class: "LocalOCREngine"
    name: "Local Dummy Engine (UI Test)"
    config:
      use_gpu_directml: false
      detection_model_path: "{DUMMY_UI_DET_MODEL}"
      recognition_model_path: "{DUMMY_UI_REC_MODEL}"
      character_dict_path: "{DUMMY_UI_CHARS_FILE}"
      # Add other minimal required config for LocalOCREngine if its __init__ is strict
      det_input_size: [100, 100]
      rec_image_shape: [1, 32, 100]


  google_engine_for_ui_test:
    enabled: true
    module: "ocr_components.google_ocr_engine"
    class: "GoogleCloudOCREngine"
    name: "Google Cloud Vision (UI Test)"
    config:
      api_key_path: "PLACEHOLDER_PATH_FOR_GOOGLE_KEY.json" # User might change this in UI or env
      default_language_hints: ["en"]

logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    default: {{format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}}
  handlers:
    console: {{class: logging.StreamHandler, formatter: default, level: DEBUG, stream: ext://sys.stdout}}
  root: {{level: DEBUG, handlers: [console]}}
"""
    with open(DUMMY_UI_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(dummy_config_content)
    logging.info(f"Created dummy UI config: {DUMMY_UI_CONFIG_FILE}")

def cleanup_dummy_config_for_ui_test():
    if os.path.exists(DUMMY_UI_CONFIG_FILE):
        os.remove(DUMMY_UI_CONFIG_FILE)
    if os.path.exists(DUMMY_UI_DET_MODEL): os.remove(DUMMY_UI_DET_MODEL)
    if os.path.exists(DUMMY_UI_REC_MODEL): os.remove(DUMMY_UI_REC_MODEL)
    if os.path.exists(DUMMY_UI_CHARS_FILE): os.remove(DUMMY_UI_CHARS_FILE)
    if os.path.exists(DUMMY_UI_MODEL_DIR):
        try:
            os.rmdir(DUMMY_UI_MODEL_DIR) # Only if empty
        except OSError:
            pass # Not empty, leave it
    logging.info("Cleaned up dummy UI config and model files.")


if __name__ == '__main__':
    # Ensure logging is configured at least basically if config fails to load
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    app = QApplication(sys.argv)

    orchestrator_instance = None
    try:
        setup_dummy_config_for_ui_test()
        # Config loader will set up more detailed logging from the YAML
        loaded_config = load_config(DUMMY_UI_CONFIG_FILE) # Load_config also calls logging.config.dictConfig
        orchestrator_instance = OCRWorkflowOrchestrator(config_path=DUMMY_UI_CONFIG_FILE)
    except Exception as e:
        logging.critical(f"Failed to setup/load config or initialize orchestrator for UI test: {e}", exc_info=True)
        # Show a message box if GUI context is available, otherwise just exit
        try:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.setText(f"Fatal Error during Application Setup:\n{e}\n\nApplication will exit.")
            error_box.setWindowTitle("Application Setup Error")
            error_box.exec()
        except RuntimeError: # If QApplication not fully up
             pass
        cleanup_dummy_config_for_ui_test()
        sys.exit(1) # Exit if orchestrator can't be created

    if orchestrator_instance:
        main_window = MainWindow(orchestrator=orchestrator_instance)
        main_window.show()
        exit_code = app.exec()
        cleanup_dummy_config_for_ui_test()
        sys.exit(exit_code)
    else: # Should have been caught by try-except already
        logging.error("Orchestrator instance is None after setup. Exiting.")
        cleanup_dummy_config_for_ui_test()
        sys.exit(1)
```
