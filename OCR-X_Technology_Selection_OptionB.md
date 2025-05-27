# OCR-X Project: Technology Selection (Option B - On-Premise Powerhouse)

This document details the specific technology selections for each component and sub-component of the OCR-X project, based on the chosen architecture: Option B - On-Premise Powerhouse.

## 1. Core OCR Pipeline Components

### 1.1. Input Handling & Preprocessing Module

*   **Sub-component: Image Acquisition**
    *   **Primary Technology:** Pillow (PIL Fork)
    *   **Exact Version:** Pillow 10.1.0
    *   **Alternative Options:** OpenCV (can also read images, but Pillow is more focused on image file I/O and format handling).
    *   **Justification:** Lightweight, extensive image format support, widely used and well-documented for image I/O in Python.
    *   **Key Dependencies:** Python 3.8+

    *   **Primary Technology (for PDF):** PyMuPDF (fitz)
    *   **Exact Version:** PyMuPDF 1.23.7
    *   **Alternative Options:** `pdf2image` (simpler, but often relies on `pdftoppler` external dependency).
    *   **Justification:** Efficient and feature-rich PDF manipulation library, including robust image extraction from PDFs. Good performance.
    *   **Key Dependencies:** Python 3.8+

    *   **Primary Technology (for Clipboard):** PyQt6 (or chosen UI framework's clipboard API)
    *   **Exact Version:** PyQt6 6.6.1
    *   **Alternative Options:** `pyperclip` (cross-platform, simpler but less integrated with a specific UI).
    *   **Justification:** Integration with the chosen UI framework (PyQt6) provides seamless access to clipboard data.
    *   **Key Dependencies:** Python 3.8+, Qt 6 libraries.

*   **Sub-component: Format Conversion & Initial Validation**
    *   **Primary Technology:** OpenCV-Python
    *   **Exact Version:** OpenCV-Python 4.8.1.78
    *   **Alternative Options:** Pillow (for some conversions, but OpenCV is more comprehensive for image manipulation).
    *   **Justification:** Standard for image manipulation and processing; provides `cv::Mat` equivalent via NumPy arrays, which is the internal standard.
    *   **Key Dependencies:** Python 3.8+, NumPy 1.26.2.

*   **Sub-component: Adaptive Binarization**
    *   **Primary Technology (Deep Learning Model Inference):** ONNX Runtime with DirectML
    *   **Exact Version:** ONNX Runtime 1.16.3 (DirectML execution provider)
    *   **Alternative Options:** TensorFlow Lite (if models are converted to TFLite format, potentially for more edge-focused scenarios).
    *   **Justification:** Official Microsoft support for DirectML acceleration on Windows, framework-agnostic model deployment (U-Net model trained in TF/PyTorch then converted).
    *   **Key Dependencies:** ONNX 1.15.0 (for model format), DirectML-compatible GPU drivers.

    *   **Primary Technology (Model Training Framework - for U-Net):** TensorFlow/Keras
    *   **Exact Version:** TensorFlow 2.15.0
    *   **Alternative Options:** PyTorch 2.1.2 (equally capable for U-Net training).
    *   **Justification:** Robust framework, good documentation for U-Net. TensorFlow has a mature DirectML plugin for training acceleration if needed, and good ONNX export.
    *   **Key Dependencies (Training):** Python 3.9+, CUDA 12.x (for NVIDIA GPU training if not using TF-DirectML plugin), cuDNN 8.x.

*   **Sub-component: Geometric Correction**
    *   **Primary Technology (Deep Learning Model Inference):** ONNX Runtime with DirectML
    *   **Exact Version:** ONNX Runtime 1.16.3
    *   **Alternative Options:** None primary for DL model inference if DirectML is the target.
    *   **Justification:** Consistent with binarization module for DirectML acceleration of learned geometric correction models (e.g., DeepXY simulated).
    *   **Key Dependencies:** ONNX 1.15.0, DirectML-compatible GPU drivers.

    *   **Primary Technology (Model Training Framework - for DeepXY-like model):** PyTorch
    *   **Exact Version:** PyTorch 2.1.2
    *   **Alternative Options:** TensorFlow 2.15.0.
    *   **Justification:** PyTorch is widely used in research for geometric correction models and offers flexible model definition. Good ONNX export.
    *   **Key Dependencies (Training):** Python 3.9+, CUDA 12.x (for NVIDIA GPU training), cuDNN 8.x.

    *   **Primary Technology (Classical Transformations):** OpenCV-Python
    *   **Exact Version:** OpenCV-Python 4.8.1.78
    *   **Justification:** Provides necessary functions for applying affine/perspective transformations derived from models or classical algorithms.

*   **Sub-component: Noise Reduction & Enhancement**
    *   **Primary Technology:** OpenCV-Python
    *   **Exact Version:** OpenCV-Python 4.8.1.78
    *   **Alternative Options:** scikit-image 0.22.0 (provides some advanced/alternative filters).
    *   **Justification:** Comprehensive set of standard denoising filters (Gaussian, median, non-local means) and enhancement techniques (CLAHE).
    *   **Key Dependencies:** NumPy 1.26.2.

    *   **Primary Technology (Wavelet Transforms):** PyWavelets
    *   **Exact Version:** PyWavelets 1.5.0
    *   **Justification:** Standard library for wavelet transforms in Python, useful for advanced denoising or feature extraction.

### 1.2. Recognition Module (Ensemble Engine)

*   **Sub-component: PaddleOCR Engine Integration**
    *   **Primary Technology:** PaddleOCR Python library (for PP-OCRv4 models)
    *   **Exact Version:** PaddleOCR 2.7.3, PaddlePaddle 2.6.0
    *   **Alternative Options:** Direct ONNX inference of PaddleOCR models if pre-converted (reduces direct PaddlePaddle dependency at runtime).
    *   **Justification:** Official library for SOTA PP-OCR models. Provides tools for model download and basic usage. Models will be converted to ONNX for final deployment.
    *   **Key Dependencies:** PaddlePaddle (for initial model handling/conversion).

*   **Sub-component: SVTR Engine Integration**
    *   **Primary Technology (Model Source):** PyTorch (for SVTR-Large or similar model implementation)
    *   **Exact Version:** PyTorch 2.1.2
    *   **Alternative Options:** TensorFlow (if a suitable SVTR implementation exists and is preferred).
    *   **Justification:** Many SOTA text recognition models, including SVTR variants, have reference implementations in PyTorch. Models will be converted to ONNX.
    *   **Key Dependencies (Model Handling/Conversion):** Python 3.9+.

*   **Sub-component: ONNX Conversion & DirectML Optimization**
    *   **Primary Technology:** ONNX
    *   **Exact Version:** ONNX 1.15.0
    *   **Justification:** Standard format for model interoperability, target for DirectML deployment.

    *   **Primary Technology:** ONNX Runtime
    *   **Exact Version:** ONNX Runtime 1.16.3 (with DirectML execution provider)
    *   **Justification:** High-performance inference engine for ONNX models with DirectML support.

    *   **Primary Technology (Conversion tools):** `tf2onnx` 1.15.1, `paddle2onnx` 1.1.0, `pytorch-onnx` (usually part of PyTorch core via `torch.onnx.export`).
    *   **Justification:** Official or widely used tools for converting models from their native frameworks to ONNX.

*   **Sub-component: Ensemble/Voting Logic**
    *   **Primary Technology:** Python
    *   **Exact Version:** Python 3.9.13 (or later 3.9.x / 3.10.x / 3.11.x)
    *   **Justification:** Custom logic implemented in Python for flexibility and ease of integration with other Python-based components.
    *   **Key Dependencies:** NumPy 1.26.2 (for numerical operations on confidence scores, etc.).

### 1.3. Post-Processing Module

*   **Sub-component: NLP-based Error Correction**
    *   **Primary Technology (Model):** Hugging Face Transformers (for ByT5 model)
    *   **Exact Version:** Transformers 4.35.2
    *   **Alternative Options:** SentencePiece 0.1.99 (for ByT5 tokenizer if not bundled).
    *   **Justification:** Provides easy access to pre-trained SOTA NLP models like ByT5 and tools for fine-tuning/inference.
    *   **Key Dependencies:** PyTorch 2.1.2 or TensorFlow 2.15.0 (as backend for Transformers), ONNX Runtime 1.16.3 (for inference of converted ByT5).

*   **Sub-component: Simulated Quantum Error Correction**
    *   **Primary Technology:** Qiskit
    *   **Exact Version:** Qiskit 0.45.1 (includes Terra, Aer)
    *   **Alternative Options:** Pennylane (another quantum computing library, but Qiskit is more established for QUBO-like problem simulations via its optimization module).
    *   **Justification:** Leading quantum computing framework, good for simulating QUBO problems on classical hardware. The optimization module can be used for VQE/QAOA simulations.
    *   **Key Dependencies:** Python 3.9+, NumPy 1.26.2.

*   **Sub-component: Formatting & Output Generation**
    *   **Primary Technology (Plain Text, JSON):** Python built-in capabilities
    *   **Exact Version:** Python 3.9.13+
    *   **Justification:** Standard library is sufficient for these tasks.

    *   **Primary Technology (Searchable PDF):** PyMuPDF (fitz)
    *   **Exact Version:** PyMuPDF 1.23.7
    *   **Alternative Options:** `reportlab` (more general PDF generation, but PyMuPDF is very efficient for text overlay on existing PDF images).
    *   **Justification:** Excellent for creating searchable PDFs by adding invisible text layers. Already a dependency for input handling.

## 2. Application & Utility Components

### 2.1. Windows Client Application

*   **Sub-component: User Interface (UI)**
    *   **Primary Technology:** PyQt6
    *   **Exact Version:** PyQt6 6.6.1
    *   **Alternative Options:**
        *   `WinUI 3` (via Python bindings like `CsWinRT` or C# interop): For the most native Windows 11 look and feel, but more complex to integrate with Python backend.
        *   `.NET MAUI` (C#): If a shift to C# for the frontend was considered for stronger Windows alignment, but Option B implies a Python-centric approach for the backend.
    *   **Justification:** Mature, feature-rich, cross-platform (though focus is Windows), good Python bindings, active community. Allows for complex and responsive UIs.
    *   **Key Dependencies:** Qt 6 libraries.

*   **Sub-component: OCR Workflow Orchestrator**
    *   **Primary Technology:** Python (with `asyncio` or `QThreads`)
    *   **Exact Version:** Python 3.9.13+
    *   **Justification:** Python for overall application logic. `asyncio` (for modern async programming) or `QThreads` (for traditional multi-threading with PyQt) to keep UI responsive during OCR tasks.
    *   **Key Dependencies:** PyQt6 (if using `QThreads`).

*   **Sub-component: Configuration Manager**
    *   **Primary Technology:** Python `configparser` or `json`
    *   **Exact Version:** Python 3.9.13+ (part of standard library)
    *   **Alternative Options:** `PyYAML` (for YAML config files, more human-readable for complex configs).
    *   **Justification:** Standard Python libraries are sufficient and easy to use for managing application settings.

### 2.2. Synthetic Data Generation Pipeline

*   **Sub-component: Configuration & Scripting**
    *   **Primary Technology:** TextRecognitionDataGenerator (TRDG)
    *   **Exact Version:** TRDG 1.7.0 (or latest compatible)
    *   **Alternative Options:** Custom scripts using Pillow/OpenCV for simpler synthetic data, or tools like SynthDoG for more advanced generation (if its complexity is warranted).
    *   **Justification:** Specifically designed for generating text recognition training data with many configurable options.
    *   **Key Dependencies:** Python 3.8+, Pillow, NumPy, OpenCV-Python.

*   **Sub-component: Data Storage & Management**
    *   **Primary Technology:** File System (Python `os`, `shutil` modules)
    *   **Exact Version:** Python 3.9.13+
    *   **Alternative Options:** SQLite (for managing metadata if dataset becomes very large and complex).
    *   **Justification:** Simple and effective for managing image files and ground truth text files in structured directories.

### 2.3. Model Management & Retraining Framework

*   **Sub-component: Model Repository**
    *   **Primary Technology:** Git Large File Storage (LFS)
    *   **Exact Version:** Git LFS 3.4.1 (or latest)
    *   **Alternative Options:** DVC (Data Version Control - more feature-rich for data/model versioning but adds complexity).
    *   **Justification:** Integrates with Git for versioning large model files effectively. Simpler than DVC for a project of this scale.
    *   **Key Dependencies:** Git 2.43.0.

*   **Sub-component: Training/Fine-tuning Scripts**
    *   **Primary Technology:** PaddlePaddle, PyTorch, TensorFlow (as per model origins)
    *   **Exact Version:** PaddlePaddle 2.6.0, PyTorch 2.1.2, TensorFlow 2.15.0
    *   **Justification:** Use the native framework of each model for fine-tuning to ensure compatibility and access to all training features.
    *   **Key Dependencies:** Respective ML framework dependencies (CUDA, cuDNN, etc., for GPU training). `scikit-learn` 1.3.2 (for evaluation metrics).

---

This technology selection aims to balance performance, feature requirements, Windows integration (DirectML), and developer experience for the On-Premise Powerhouse architecture. Versions are specified based on stable releases in late 2023/early 2024 and may be updated as the project progresses.
