import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import logging
import os
import threading
from customtkinter import CTkMessagebox # Import CTkMessagebox

# Assuming ocr_workflow_orchestrator.py and custom_exceptions.py are in the same directory or PYTHONPATH
from ocr_workflow_orchestrator import OCRWorkflowOrchestrator
from custom_exceptions import OCRPipelineError, OCRFileNotFoundError, OCRConfigurationError

# Configure basic logging for the GUI application
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("OCR_GUI_App")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OCR Application - Integrated Pipeline")
        self.geometry("850x650") # Slightly wider for new buttons
        logger.info("Application window initialized.")

        self.current_image_path = None
        self.current_config_path = "config_dev.yaml" # Default config path

        # Initialize OCR Orchestrator
        try:
            logger.info(f"Initializing OCRWorkflowOrchestrator with config: {self.current_config_path}...")
            self.orchestrator = OCRWorkflowOrchestrator(config_path=self.current_config_path)
            logger.info("OCRWorkflowOrchestrator initialized successfully.")
        except (OCRFileNotFoundError, OCRConfigurationError, OCRPipelineError) as e:
            logger.critical(f"Failed to initialize OCRWorkflowOrchestrator with '{self.current_config_path}': {e}", exc_info=True)
            CTkMessagebox(title="Initialization Error",
                          message=f"Failed to initialize OCR system with '{self.current_config_path}': {e}\nApplication might not function correctly. Please load a valid configuration.",
                          icon="cancel")
            self.orchestrator = None
        except Exception as e_init:
            logger.critical(f"Unexpected critical error during OCRWorkflowOrchestrator initialization: {e_init}", exc_info=True)
            CTkMessagebox(title="Critical Startup Error",
                          message=f"An unexpected error occurred during startup: {e_init}\nApplication will exit.",
                          icon="cancel")
            self.destroy()
            return

        # --- Configure grid layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # --- Top Controls Frame ---
        self.top_controls_frame = ctk.CTkFrame(self, height=50)
        self.top_controls_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        # Configure columns for more buttons
        self.top_controls_frame.grid_columnconfigure(0, weight=0) # Browse
        self.top_controls_frame.grid_columnconfigure(1, weight=0) # Load Config
        self.top_controls_frame.grid_columnconfigure(2, weight=0) # Run OCR
        self.top_controls_frame.grid_columnconfigure(3, weight=0) # Clear All
        self.top_controls_frame.grid_columnconfigure(4, weight=1) # Spacer

        self.browse_button = ctk.CTkButton(self.top_controls_frame, text="Browse Image...", command=self.browse_image_command)
        self.browse_button.grid(row=0, column=0, padx=(10,5), pady=10)

        self.load_config_button = ctk.CTkButton(self.top_controls_frame, text="Load Config...", command=self.load_configuration_command)
        self.load_config_button.grid(row=0, column=1, padx=5, pady=10)

        self.run_ocr_button = ctk.CTkButton(self.top_controls_frame, text="Run OCR", command=self.run_ocr_command, state="disabled")
        self.run_ocr_button.grid(row=0, column=2, padx=5, pady=10)

        self.clear_all_button = ctk.CTkButton(self.top_controls_frame, text="Clear All", command=self.clear_all_command)
        self.clear_all_button.grid(row=0, column=3, padx=(5,10), pady=10)


        # --- Main Content Frame ---
        self.main_content_frame = ctk.CTkFrame(self)
        self.main_content_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.main_content_frame.grid_columnconfigure(0, weight=1) # Image display area
        self.main_content_frame.grid_columnconfigure(1, weight=1) # OCR output text area
        self.main_content_frame.grid_rowconfigure(0, weight=1)

        # Image Display Area
        self.image_display_label = ctk.CTkLabel(self.main_content_frame, text="No image selected.", height=400) # Min height
        self.image_display_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # OCR Output Text Area
        self.ocr_output_textbox = ctk.CTkTextbox(self.main_content_frame, wrap="word", state="disabled") # Read-only initially
        self.ocr_output_textbox.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.ocr_output_textbox.insert("0.0", "OCR results will appear here.")
        # self.ocr_output_textbox.configure(state="disabled") # Make it read-only after inserting text

        # --- Status Bar ---
        self.status_bar_label = ctk.CTkLabel(self, text="Ready.", height=25, anchor="w")
        self.status_bar_label.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")

        logger.info("GUI layout created.")
        if self.orchestrator is None: # If orchestrator failed to init
            self.browse_button.configure(state="disabled")
            self.run_ocr_button.configure(state="disabled")
            self.status_bar_label.configure(text="Critical Error: OCR system failed to initialize. Check logs.")


    def browse_image_command(self):
        logger.info("Browse Image button clicked.")
        file_types = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=file_types)

        if file_path:
            self.current_image_path = file_path
            filename = os.path.basename(file_path)
            logger.info(f"Image selected: {file_path}")
            self.status_bar_label.configure(text=f"Selected: {filename}")

            try:
                pil_image = Image.open(file_path)

                # Determine display size for the image preview
                # Use a fixed portion of the main content frame width/height or aspect ratio logic
                # For simplicity, using a fixed size for the CTkImage, let the label scale it if possible.
                # A more dynamic approach would calculate this based on available space.
                label_width = self.image_display_label.winfo_width()
                label_height = self.image_display_label.winfo_height()

                # Get actual label dimensions for accurate fitting
                # Ensure window is drawn to get correct winfo_width/height
                self.image_display_label.update_idletasks() # Process pending geometry changes
                label_width = self.image_display_label.winfo_width()
                label_height = self.image_display_label.winfo_height()

                # Fallback if initial dimensions are still too small (e.g. window not fully visible)
                if label_width < 50: label_width = 350 # Default reasonable width
                if label_height < 50: label_height = 350 # Default reasonable height

                img_w, img_h = pil_image.size

                # Calculate scaling factor to fit image within label, maintaining aspect ratio
                # Image should not be upscaled if smaller than the label.
                scale_w = label_width / img_w
                scale_h = label_height / img_h
                scale = min(scale_w, scale_h) # Scale to fit both dimensions

                # If image is smaller than label area, display at original size (don't upscale)
                # by ensuring scale is not > 1.0 (unless we want to allow small images to fill space)
                # For this requirement: "not be upscaled if it's smaller"
                if scale > 1.0:
                    scale = 1.0

                display_w = int(img_w * scale)
                display_h = int(img_h * scale)

                if display_w <= 0: display_w = min(img_w, 200) # Min fallback, use original if tiny
                if display_h <= 0: display_h = min(img_h, 200) # Min fallback

                logger.debug(f"Label size: {label_width}x{label_height}. Image original: {img_w}x{img_h}. Calculated scale: {scale:.2f}. Final display: {display_w}x{display_h}")

                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(display_w, display_h))
                self.image_display_label.configure(image=ctk_image, text="")
                logger.info(f"Image '{filename}' loaded. Display size: ({display_w}x{display_h})")

                self.run_ocr_button.configure(state="normal" if self.orchestrator else "disabled")
            except FileNotFoundError: # Should not happen if filedialog returns valid path
                logger.error(f"Selected image file disappeared after dialog: {file_path}", exc_info=True)
                self.status_bar_label.configure(text=f"Error: Image file '{filename}' not found.")
                self.image_display_label.configure(image=None, text=f"Error: Image file not found:\n{filename}")
                self.current_image_path = None
                self.run_ocr_button.configure(state="disabled")
            except Exception as e:
                logger.error(f"Failed to load or display image '{file_path}': {e}", exc_info=True)
                self.status_bar_label.configure(text=f"Error loading image: {filename}")
                self.image_display_label.configure(image=None, text=f"Error loading image:\n{filename}\nSee logs.")
                self.current_image_path = None
                self.run_ocr_button.configure(state="disabled")
        else:
            logger.info("No image selected from dialog.")
            self.status_bar_label.configure(text="No image selected.")


    def run_ocr_command(self):
        if not self.current_image_path:
            logger.warning("Run OCR: No image selected.")
            self.status_bar_label.configure(text="Error: No image selected.")
            ctk.CTkMessagebox(title="Error", message="Please select an image first using the 'Browse Image...' button.", icon="cancel")
            return
        if not self.orchestrator:
            logger.error("Run OCR: Orchestrator not initialized.")
            self.status_bar_label.configure(text="Error: OCR system not initialized.")
            ctk.CTkMessagebox(title="Error", message="OCR system is not initialized. Please restart the application.", icon="cancel")
            return

        logger.info(f"Run OCR clicked for: {self.current_image_path}")
        self.status_bar_label.configure(text="Processing OCR...")
        self.run_ocr_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")

        self.ocr_output_textbox.configure(state="normal")
        self.ocr_output_textbox.delete("0.0", "end")
        self.ocr_output_textbox.insert("0.0", "Processing, please wait...\n")
        self.ocr_output_textbox.configure(state="disabled")

        # Run OCR in a separate thread
        ocr_thread = threading.Thread(target=self._actual_ocr_task, args=(self.current_image_path,), daemon=True)
        ocr_thread.start()

    def _actual_ocr_task(self, image_path: str):
        logger.info(f"OCR Thread: Starting processing for '{image_path}'")
        result_to_display = ""
        status_message = ""
        try:
            # This is the actual call to the OCR pipeline
            processed_data_dict = self.orchestrator.process_document(image_path)

            # Check if orchestrator returned an error string (older error handling pattern)
            if isinstance(processed_data_dict, str) and processed_data_dict.startswith("Error:"):
                result_to_display = processed_data_dict
                status_message = "OCR failed."
                logger.error(f"OCR Thread: Orchestrator returned an error string: {result_to_display}")
            else: # Assume dictionary output
                result_to_display = self.orchestrator.get_results(processed_data_dict)
                status_message = "OCR complete."
            logger.info("OCR Thread: Processing successful.")

        except (OCRFileNotFoundError, OCRConfigurationError, OCRPipelineError) as e_ocr:
            logger.error(f"OCR Thread: OCR pipeline error: {e_ocr}", exc_info=True) # Log with traceback for these
            result_to_display = f"OCR Error:\n{type(e_ocr).__name__}: {e_ocr}"
            status_message = "OCR failed with error."
        except Exception as e_thread: # Catch any other unexpected exceptions from the thread
            logger.error(f"OCR Thread: Unexpected exception: {e_thread}", exc_info=True)
            result_to_display = f"An unexpected error occurred in OCR processing:\n{type(e_thread).__name__}: {e_thread}"
            status_message = "OCR failed unexpectedly."

        # Schedule GUI update on the main thread
        self.after(0, lambda: self._update_gui_post_ocr(result_to_display, status_message))

    def _update_gui_post_ocr(self, result_text: str, status_message: str):
        logger.info(f"Updating GUI post OCR. Status: {status_message}")

        self.ocr_output_textbox.configure(state="normal")
        self.ocr_output_textbox.delete("0.0", "end")
        self.ocr_output_textbox.insert("0.0", result_text)
        self.ocr_output_textbox.configure(state="disabled")

        self.status_bar_label.configure(text=status_message)
        self.run_ocr_button.configure(state="normal" if self.current_image_path else "disabled") # Re-enable if image still valid
        self.browse_button.configure(state="normal")
        logger.info("GUI updated with OCR results/status.")

    def load_configuration_command(self):
        logger.info("Load Configuration button clicked.")
        file_types = [("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        new_config_path = filedialog.askopenfilename(title="Select Configuration File", filetypes=file_types)

        if new_config_path:
            logger.info(f"New configuration file selected: {new_config_path}")
            try:
                # Attempt to re-initialize orchestrator with the new config
                logger.info(f"Attempting to re-initialize OCRWorkflowOrchestrator with: {new_config_path}")
                new_orchestrator = OCRWorkflowOrchestrator(config_path=new_config_path)
                self.orchestrator = new_orchestrator # Replace old orchestrator
                self.current_config_path = new_config_path

                self.status_bar_label.configure(text=f"Configuration loaded: {os.path.basename(new_config_path)}")
                logger.info(f"Successfully re-initialized orchestrator with '{new_config_path}'.")

                # Clear current state as new config might mean different models/settings
                self.clear_all_command(called_by_load_config=True)

            except (OCRFileNotFoundError, OCRConfigurationError, OCRPipelineError) as e:
                logger.error(f"Failed to load or apply new configuration from '{new_config_path}': {e}", exc_info=True)
                CTkMessagebox(title="Configuration Error",
                              message=f"Failed to load or apply configuration from '{os.path.basename(new_config_path)}':\n\n{type(e).__name__}: {e}",
                              icon="cancel")
                self.status_bar_label.configure(text=f"Error loading config: {os.path.basename(new_config_path)}")
                # Orchestrator might be None or still the old one. If new one failed, set to None.
                # If self.orchestrator was already None, it remains None.
                # If new_orchestrator assignment failed, self.orchestrator is still the old one.
                # For simplicity, if load fails, disable OCR until a good config or image is loaded.
                # Consider if self.orchestrator should be set to None here if new_orchestrator failed.
                # If new_orchestrator = OCRWorkflowOrchestrator throws error, self.orchestrator isn't updated.
                # This means it would continue with the old config if new one is bad.
                # Forcing user to fix or reload:
                # self.orchestrator = None
                # self.run_ocr_button.configure(state="disabled")
                # self.browse_button.configure(state="disabled" if self.orchestrator is None else "normal")

            except Exception as e_unexpected:
                logger.critical(f"Unexpected error during configuration loading from '{new_config_path}': {e_unexpected}", exc_info=True)
                CTkMessagebox(title="Critical Configuration Error",
                              message=f"An unexpected error occurred while loading configuration:\n{e_unexpected}",
                              icon="cancel")
                self.status_bar_label.configure(text="Critical error loading new configuration.")
        else:
            logger.info("No new configuration file selected.")

    def clear_all_command(self, called_by_load_config=False):
        logger.info("Clear All button clicked or called internally.")
        self.current_image_path = None
        self.image_display_label.configure(image=None, text="No image selected.")

        self.ocr_output_textbox.configure(state="normal")
        self.ocr_output_textbox.delete("0.0", "end")
        self.ocr_output_textbox.insert("0.0", "OCR results will appear here.")
        self.ocr_output_textbox.configure(state="disabled")

        if not called_by_load_config: # Avoid overwriting "Config loaded" message
            self.status_bar_label.configure(text="Interface cleared. Ready.")

        self.run_ocr_button.configure(state="disabled")
        logger.info("GUI interface cleared.")


if __name__ == "__main__":
    logger.info("Application starting...")
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = App()
    # Check if app.destroy() was called in __init__ due to critical error
    if app.winfo_exists():
        app.mainloop()
        logger.info("Application closed.")
    else:
        logger.critical("Application failed to start due to error during initialization.")
