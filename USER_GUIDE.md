# OCR Application - User Guide

## 1. Introduction

Welcome to the OCR Application! This program allows you to extract text from images using a multi-stage processing pipeline. It provides both a Command-Line Interface (CLI) for batch processing or scripting, and a Graphical User Interface (GUI) for interactive use.

## 2. Dependencies

To run this application, you'll need the following:

*   **Python:** Version 3.x (developed with 3.10).
*   **Python Packages:**
    *   `Pillow` (for image manipulation)
    *   `numpy` (for numerical operations, especially on image data)
    *   `PyYAML` (for reading YAML configuration files)
    *   `opencv-python` (for image preprocessing like binarization and deskewing)
    *   `onnxruntime` (for running ONNX-based text recognition and geometric correction models)
    *   `customtkinter` (for the graphical user interface)
    You can typically install these using pip:
    ```bash
    pip install Pillow numpy PyYAML opencv-python onnxruntime customtkinter
    ```
*   **Special Note on Mojo SDK:**
    *   Some performance-critical functions (like image normalization) are written in Mojo for potential speed benefits.
    *   To use these Mojo-accelerated functions, the **Mojo SDK must be installed and correctly configured** in your environment (including `MODULE_PATH` pointing to where `.mojo` files are located).
    *   If the Mojo SDK is not detected, the application will automatically use Python/NumPy fallback implementations for these functions. These fallbacks are fully functional but may be slower. You will see a warning in the console logs if Mojo SDK is not found.
*   **Special Note on Tkinter (for GUI):**
    *   The GUI application (`gui_app.py`) is built using `customtkinter`, which relies on Tkinter.
    *   Most Python installations include Tkinter by default. However, in some minimal environments (like certain Docker containers or specific Linux distributions), you might need to install it separately. For example, on Debian-based Linux systems (like Ubuntu), this can often be installed with:
        ```bash
        sudo apt-get update
        sudo apt-get install python3-tk
        ```

## 3. Using the Command-Line Interface (`main.py`)

The CLI is ideal for processing single images quickly or for integrating the OCR tool into scripts.

*   **How to Run:**
    Open your terminal or command prompt, navigate to the `/app/` directory of the application, and run:
    ```bash
    python main.py [options] <image_path>
    ```

*   **Command-Line Arguments:**
    *   `image_path`: (Positional argument) The path to the input image file you want to process.
    *   `--image <path>`: (Optional) An alternative way to specify the input image file path.
    *   `--config <path>`: (Optional) Path to a custom YAML configuration file. If not provided, the application defaults to using `config_dev.yaml` located in the application's root directory. If `config_dev.yaml` is not found, one will be created with default settings.
    *   `--verbose`: (Optional) Enables detailed DEBUG level logging to the console. This provides more insight into the OCR pipeline steps.
    *   `-h, --help`: Shows a help message detailing all available arguments and exits.

*   **Example Usages:**
    *   Process an image using default settings:
        ```bash
        python main.py path/to/your/image.png
        ```
    *   Process an image using the `--image` flag:
        ```bash
        python main.py --image path/to/your/image.jpg
        ```
    *   Process an image with a custom configuration file and verbose logging:
        ```bash
        python main.py --image path/to/image.tiff --config path/to/my_custom_config.yaml --verbose
        ```

*   **Expected Output Format:**
    The CLI will print the OCR results to the standard output. The output includes:
    *   Spell-Checked Text: The OCR text after spell correction (unknown words marked with `[?]`).
    *   Cleaned Text: Text after applying character whitelisting.
    *   Original OCR Text: Raw text output from the recognition model (truncated for brevity).
    *   Confidence: The confidence score from the recognition model (currently a dummy value).
    *   Additional metadata may also be included.
    Console logs will also show information about the processing stages, and more detail if `--verbose` is used.

## 4. Using the Graphical User Interface (`gui_app.py`)

The GUI provides an interactive way to load an image, run OCR, and view results.

*   **How to Run:**
    Open your terminal or command prompt, navigate to the `/app/` directory, and run:
    ```bash
    python gui_app.py
    ```
    (Requires Tkinter to be installed, see "Dependencies" section).

*   **GUI Components & Workflow:**
    1.  **Window Launch:** The application window titled "OCR Application - Integrated Pipeline" will appear.
    2.  **"Browse Image..." Button:**
        *   Click this button to open a file dialog.
        *   Select an image file (e.g., PNG, JPG, TIFF, BMP).
        *   The selected image will be displayed in the left-hand image display area.
        *   The status bar at the bottom will show the name of the selected file.
        *   The "Run OCR" button will become enabled.
    3.  **Image Display Area:**
        *   Located on the left side of the main window.
        *   Shows a preview of the currently loaded image. If the image is larger than the display area, it will be scaled down to fit while maintaining its aspect ratio. Small images are not upscaled.
    4.  **"Load Configuration..." Button:**
        *   Click this button to open a file dialog, filtering for YAML configuration files (`*.yaml`, `*.yml`).
        *   If you select a valid configuration file, the OCR system will re-initialize using these settings.
        *   The status bar will confirm "Configuration loaded: [filename]".
        *   The image display and OCR output areas will be cleared. You will need to select an image again.
        *   If the selected configuration is invalid or causes an error, an error message box will appear, and the application will attempt to continue with the previous (or default) configuration if possible.
    5.  **"Run OCR" Button:**
        *   This button is enabled only after an image has been successfully loaded (and if the OCR system initialized correctly).
        *   Click this button to start the OCR process on the displayed image.
        *   While processing, this button (and "Browse Image...", "Load Configuration...") will be temporarily disabled.
        *   The status bar will indicate "Processing OCR...".
        *   The OCR output text area will show "Processing, please wait...".
    6.  **OCR Output Text Area:**
        *   Located on the right side of the main window.
        *   Displays the final OCR results (Spell-Checked Text, Cleaned Text, Original Text, Confidence) after processing is complete.
        *   The text area is scrollable for longer results and is read-only.
    7.  **"Clear All" Button:**
        *   Click this button to reset the GUI to its initial state.
        *   It clears the image display area ("No image selected.").
        *   Clears the OCR output text area ("OCR results will appear here.").
        *   Resets the status bar to "Interface cleared. Ready."
        *   Disables the "Run OCR" button.
    8.  **Status Bar:**
        *   Located at the bottom of the window.
        *   Provides feedback on the application's current state, such as "Ready.", "Selected image: ...", "Processing OCR...", "OCR complete.", or error messages.

*   **Basic Workflow:**
    1.  Launch `gui_app.py`.
    2.  (Optional) Click "Load Configuration..." to load custom settings.
    3.  Click "Browse Image..." to select an image. It will be displayed.
    4.  Click "Run OCR".
    5.  View results in the text area and status in the status bar.
    6.  Click "Clear All" to process another image or load a new configuration.

## 5. Configuration (`config_dev.yaml`)

The application uses a YAML file (defaulting to `config_dev.yaml` in the application's root directory) for configuration. While the application can run with defaults, advanced users might want to tweak this file. Key configurable aspects include:

*   **Model Paths (`model_paths`):** Paths to the ONNX models for geometric correction and text recognition.
*   **Performance Settings (`performance`):** Options like `use_directml` for ONNX Runtime.
*   **Postprocessing Settings (`postprocessing_settings`):**
    *   `whitelist_chars`: A string of characters to keep during the text cleaning phase. If `null` or not specified, a default whitelist is used.
    *   `dictionary_path`: Path to the dictionary file (e.g., `default_dict.txt`) used by the spell checker.
*   **Deskewer Settings (`deskewer_settings`):** Parameters for the image deskewing algorithm, such as `angle_threshold_degrees`.
*   **Logging (`logging`):** Granular control over logging levels and handlers for different modules.

If `config_dev.yaml` is not found, the application will attempt to create a default one with predefined paths and settings (including creating placeholder dummy models and a dictionary if they are also missing).
