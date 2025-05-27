# OCR-X Project: Testing Strategy Pyramid (Option B - Flexible Hybrid Powerhouse)

This document defines the testing strategy for the OCR-X project, specifically for Option B (Flexible Hybrid Powerhouse). This strategy considers the hybrid nature of the system, which includes local open-source OCR engines, integration with commercial cloud OCR APIs, and a mechanism for switching between them.

## I. Overview of Testing Philosophy

The testing philosophy for OCR-X emphasizes a comprehensive, multi-layered approach to ensure the reliability, accuracy, and performance of the application. Given the hybrid nature of Option B, testing must rigorously validate:
1.  The accuracy and performance of the local open-source OCR engine ensemble (PaddleOCR + SVTR with ONNX/DirectML).
2.  The correct integration and data handling of commercial cloud OCR APIs (Google Document AI, Azure AI Vision).
3.  The robustness and correctness of the OCR engine abstraction layer and the engine selection/switching mechanism.
4.  The overall user experience and functionality of the Windows client application.

Where applicable, particularly for backend modules and business logic, principles of **Test-Driven Development (TDD)** or **Behavior-Driven Development (BDD)** will be encouraged. For instance, defining expected outputs for specific image inputs before implementing the processing logic for a new filter, or defining user stories for UI interactions and writing tests that reflect these behaviors. This ensures that development is guided by testability and clear requirements.

Continuous testing throughout the development lifecycle is paramount, integrated into the CI/CD pipeline to catch regressions and issues early.

## II. Testing Levels (The Pyramid)

The testing strategy follows the standard testing pyramid model, with a broad base of unit tests, a narrower layer of integration tests, and a highly focused set of end-to-end tests at the top. Performance tests are considered a specialized category applied across different levels.

### A. Unit Tests (Foundation - Largest number of tests)

*   **Scope:** Focus on isolating and testing the smallest, individual, and independently testable parts of the codebase, such as functions, methods within classes, and individual modules. External dependencies (like network calls or complex runtimes) are typically mocked or stubbed.
*   **Components to Test:**
    *   **Preprocessing Sub-components:**
        *   Each image processing filter (e.g., binarization algorithms, noise reduction filters): Verify output for given input images (e.g., specific pixel values, image statistics).
        *   Geometric correction algorithms: Test with sample images having known distortions; mock model outputs for U-Net/DeepXY simulations to test the surrounding logic (data conversion, parameter passing).
        *   Image acquisition utilities: Test loading from different file formats (mock file system interactions), clipboard handling logic (mock clipboard API).
    *   **Recognition Module:**
        *   Wrappers for local ONNX models (PaddleOCR, SVTR):
            *   Test model loading logic (mock `onnxruntime.InferenceSession` to check if correct path and providers are passed).
            *   Test input preparation functions (e.g., image resizing, normalization, transposing for model input).
            *   Test output parsing functions (e.g., decoding raw model output tensors into text, confidence, and bounding boxes).
            *   For pure unit tests, mock the `InferenceSession.run()` method to return predefined outputs. For slightly broader unit tests, use very small, dummy ONNX models that run quickly on CPU.
        *   Commercial API Client Wrappers (Google, Azure):
            *   Test request formatting logic (ensure correct JSON/protobuf structures are built for API calls).
            *   Test response parsing logic (ensure successful and error responses from APIs are correctly interpreted and mapped to the internal common OCR data format).
            *   Test error handling for various API error codes (e.g., authentication errors, rate limits, invalid input).
            *   **Crucially, mock the actual SDK calls** (e.g., using Python's `unittest.mock.patch`) to avoid network dependencies, costs, and non-deterministic behavior in unit tests.
        *   OCR Engine Abstraction Layer (`OCREngine` interface and concrete implementations):
            *   Test correct instantiation of different engine types (`LocalOCREngine`, `GoogleCloudOCREngine`, `AzureAIVisionOCREngine`).
            *   Test that each engine implementation adheres to the defined interface.
        *   Engine Selection/Switching Logic:
            *   Test the logic that chooses the correct engine (local vs. specific cloud provider) based on user configuration or other criteria.
    *   **Post-Processing Sub-components:**
        *   NLP error correction logic (e.g., ByT5 ONNX wrapper): Test input preparation for the ByT5 model, output parsing, and application of corrections to sample text. Mock the ONNX session for pure unit tests.
        *   Simulated Quantum Error Correction logic: Test QUBO problem formulation for specific ambiguity patterns with sample data. Test the interface to the Qiskit simulation.
        *   Output Formatters: Test conversion of internal OCR data structures to plain text, structured JSON, and (conceptually) searchable PDF elements.
    *   **Configuration Manager:**
        *   Test loading configurations from different file structures (e.g., YAML, JSON).
        *   Test saving configurations.
        *   Test validation of configuration values (e.g., correct data types, valid choices for settings).
        *   Test secure loading/saving of API keys (mocking the actual Windows Credential Manager or OS keychain interactions).
    *   **Utility Functions:** Any helper functions (e.g., for image manipulation, file handling, data structure conversion) should have dedicated unit tests.
*   **Tools:** `pytest` (Python), `unittest.mock` (for mocking).
*   **Goal:** Verify the logical correctness of individual software units. Ensure each component behaves as designed in isolation. Aim for >80% code coverage for critical modules.

### B. Integration Tests (Middle Layer)

*   **Scope:** Focus on testing the interaction and interfaces between different components or modules. Ensure that data flows correctly and components work together as specified.
*   **Key Integrations to Test:**
    *   **Preprocessing Pipeline:**
        *   Test the full chain of preprocessing steps: Image Acquisition -> Format Conversion -> Adaptive Binarization -> Geometric Correction -> Noise Reduction. Verify that the output of one step is correctly passed to and processed by the next. Use actual small test images.
    *   **Full OCR Workflow (Local Engine):**
        *   Test the flow: Preprocessing Module -> OCR Engine Abstraction Layer (selecting Local Engine) -> Local OCR Ensemble (PaddleOCR + SVTR with actual small ONNX models running on DirectML/CPU) -> Post-Processing Module.
        *   Use a small set of test images and verify the final output format and basic accuracy.
    *   **Full OCR Workflow (Cloud API - Mocked/Staged):**
        *   Test the flow: Preprocessing Module -> OCR Engine Abstraction Layer (selecting a Cloud Engine) -> Commercial API Client Wrapper -> Post-Processing Module.
        *   **Mocking Strategy:**
            *   Use `unittest.mock.patch` or similar to mock the actual cloud SDK calls (e.g., `google.cloud.vision.ImageAnnotatorClient.document_text_detection` or `azure.ai.vision.imageanalysis.ImageAnalysisClient.analyze`).
            *   The mock should return realistic API success and error responses (predefined JSON/protobuf-like structures).
            *   If available and feasible, a dedicated staging/test API endpoint from the cloud provider could be used with caution for a limited set_of tests, but this often has cost/quota implications.
    *   **Engine Switching Mechanism:**
        *   Test that the OCR Workflow Orchestrator, based on current configuration (from Configuration Manager), correctly routes processing requests to the selected engine (Local, Google, or Azure) via the OCR Engine Abstraction Layer.
        *   Verify that outputs from different engines are transformed into a consistent internal format by the abstraction layer before being sent to post-processing.
    *   **Windows Client & Orchestrator:**
        *   Test that UI actions (e.g., button clicks to select a file and start OCR) correctly trigger the corresponding methods in the OCR Workflow Orchestrator.
        *   Test that results and error messages from the Orchestrator are correctly displayed back in the UI (mocking the Orchestrator's processing methods to return predefined results/errors).
    *   **Configuration Manager & Engine Loading/Behavior:**
        *   Test that changes made to engine selection or API key configurations via the UI are correctly saved by the Configuration Manager.
        *   Test that on application startup or when settings are changed, the OCR Engine Abstraction Layer correctly initializes/re-initializes the selected engine with the appropriate configuration (e.g., API keys for cloud, model paths for local).
*   **Tools:** `pytest` with fixtures (for managing test data and component setup), `unittest.mock`, potentially `Docker` (if any backend components are containerized for development/testing, though less likely for Option B's client focus).
*   **Goal:** Ensure that integrated components communicate and interoperate correctly, and that data flows through the system as designed.

### C. End-to-End (E2E) Tests (System Level - Fewer tests, broader scope)

*   **Scope:** Test the entire application from the user's perspective, simulating real user scenarios. This involves interacting with the Windows Client UI and verifying outputs against expected results or ground truth.
*   **Scenarios:**
    *   **Local Engine Scenarios:**
        *   OCR a clean, high-quality scanned document (JPG/PNG/single-page PDF) using the local open-source engine ensemble. Verify accuracy against ground truth.
        *   OCR a moderately noisy document using the local engine. Verify output.
        *   OCR a document with a common layout (e.g., single column text) using the local engine.
        *   (Future) OCR multi-page PDFs using the local engine.
    *   **Cloud API Scenarios (requires valid, potentially sandboxed, API keys for test environment):**
        *   OCR a clean document using Google Document AI. Verify output and basic accuracy.
        *   OCR a clean document using Azure AI Vision. Verify output and basic accuracy.
        *   Test handling of API authentication errors (e.g., invalid key provided in UI) – ensure user-friendly error messages.
    *   **UI and Workflow Scenarios:**
        *   Full user flow: Launch app -> Select file -> Choose "Local Engine" -> Start OCR -> View plain text result -> Save result.
        *   Full user flow: Launch app -> Configure Google API Key -> Select file -> Choose "Google Cloud OCR" -> Start OCR -> View plain text result.
        *   Verify that the UI correctly displays progress and status messages during OCR.
        *   Verify that UI error dialogs are shown for critical errors (e.g., file not found, engine failure).
    *   **Output Verification:**
        *   Compare OCR text output against ground truth text for a curated E2E test set. Calculate CER/WER.
        *   Visually inspect searchable PDF outputs for correct text overlay and searchability.
    *   **Engine Switching (Key for Hybrid):**
        *   Perform OCR on the same document using the local engine, then switch to a cloud engine (e.g., Google) via UI settings, and re-OCR. Verify the switch occurs and output reflects the chosen engine.
        *   If dynamic failover is implemented (e.g., cloud API fails, falls back to local), test this scenario.
*   **Tools:**
    *   UI Automation: `PyAutoGUI` (for controlling mouse/keyboard on Windows), `Windows Application Driver` (WinAppDriver with Appium), or framework-specific tools if PyQt/WinUI3 offer testability interfaces (e.g., `pytest-qt` for some interactions).
    *   Ground truth comparison scripts (custom Python scripts using `ocreval`, `jiwer`).
    *   A dedicated, version-controlled E2E test dataset.
*   **Goal:** Validate that the entire system functions as expected from a user's point of view, meets key functional requirements, and achieves acceptable quality on representative documents.

### D. Performance Tests (Specialized)

*   **Scope:** Evaluate the application's speed, resource consumption, and stability under various conditions, particularly for the local engine's DirectML and CPU performance.
*   **Tests:**
    *   **Latency Tests:**
        *   Measure the time taken from initiating OCR to displaying results for single documents of varying complexity (simple text, dense text, images with noise).
        *   Test with local engine on CPU.
        *   Test with local engine on various DirectML-compatible GPUs (NVIDIA, AMD, Intel integrated).
        *   Test with each configured cloud API (to measure typical API + network latency).
    *   **Throughput Tests:**
        *   Measure pages per minute (PPM) for batch processing of a standardized set of documents (e.g., 50-100 pages).
        *   Test for local engine (CPU and DirectML).
        *   Test for cloud APIs (consider API rate limits and implement appropriate client-side throttling/batching).
    *   **Resource Monitoring:**
        *   Monitor CPU usage (overall and per-core if possible) during local OCR tasks.
        *   Monitor GPU usage (specifically DirectML activity via tools like Task Manager's GPU view or NVIDIA/AMD utilities) during local ONNX model inference.
        *   Monitor RAM (application process working set) and VRAM (dedicated GPU memory usage) during OCR tasks.
    *   **Stress Tests:**
        *   Test the system with very large single image files or PDFs with many pages (local engine).
        *   Test prolonged operation (e.g., OCRing a large batch of documents for an extended period) to check for memory leaks or performance degradation over time.
*   **Tools:**
    *   Custom Python scripts using `time.perf_counter()` for timing critical code sections.
    *   Windows Performance Monitor (`perfmon`) for system-level resource tracking.
    *   Task Manager (GPU utilization tab).
    *   GPU-specific monitoring tools: `NVIDIA System Monitor` (nvsmi), AMD equivalent.
    *   Python memory profiling tools: `memory-profiler`.
*   **Goal:** Ensure the system meets the performance targets (PPM, latency) defined in FR5 of the requirements specification, operates efficiently on target Windows hardware (especially with DirectML), and remains stable under sustained load.

## III. Test Data Management

A robust test data management strategy is crucial for effective testing.

*   **Unit Test Data:**
    *   Small, focused data snippets (e.g., short text strings for NLP correction, small NumPy arrays representing image segments, mock JSON responses for API clients).
    *   Stored directly within the test code or as small files in a dedicated test assets directory (e.g., `tests/unit/assets/`).
*   **Integration Test Data:**
    *   A diverse set of small, representative images (e.g., 5-10 clean images, 5-10 noisy images, a few different layouts).
    *   Simple single-page and few-page PDF files containing these images.
    *   Stored in a version-controlled test data directory (e.g., `tests/integration/test_data/`).
*   **End-to-End (E2E) Test Set:**
    *   A curated set of 20-50 documents representing various real-world scenarios (different qualities, fonts, layouts, document types like letters, articles, simple forms).
    *   Accurate ground truth text files for each document in this set.
    *   Stored in a dedicated, version-controlled E2E benchmark directory (e.g., `data/e2e_benchmark_set/`). This set will be expanded over time.
    *   Consider using subsets of publicly available datasets (e.g., ICDAR, SROIE) if licensing permits and they align with target use cases.
*   **Specialized Datasets:**
    *   **Noisy/Distorted Images:** A specific set of images with common augmentations (blur, noise, low contrast, specific geometric warps) to test preprocessing and engine robustness.
    *   **Language-Specific Datasets:** As new languages are supported, dedicated test sets with ground truth for each language will be created/sourced.
*   **Synthetic Data Usage:**
    *   The TRDG pipeline will be used to generate large volumes of synthetic text images with diverse augmentations.
    *   This data will primarily be used for:
        *   Training/fine-tuning local OCR models.
        *   Augmenting robustness test sets for E2E and performance testing (e.g., testing how the system handles thousands of slightly varied images).
*   **Versioning:** All test data (especially ground truth for E2E and benchmark sets) will be versioned using Git LFS alongside the source code to ensure test reproducibility and track changes to evaluation criteria.

This testing strategy pyramid, combined with robust test data management, aims to build a high-quality, reliable, and performant OCR-X application.The `OCR-X_Testing_Strategy_Pyramid.md` file has been created.

**Summary of the Document:**

The document outlines a comprehensive testing strategy for the OCR-X project (Option B: Flexible Hybrid Powerhouse), emphasizing its hybrid nature.

1.  **Overview of Testing Philosophy:**
    *   Stresses comprehensive testing at all levels for local engines, cloud API integrations, and the switching mechanism.
    *   Suggests TDD/BDD for certain components.

2.  **Testing Levels (The Pyramid):**
    *   **A. Unit Tests (Foundation):**
        *   Covers individual modules/classes/functions of Preprocessing, Recognition (local ONNX wrappers, cloud API wrappers, abstraction layer, switching logic), Post-Processing, Configuration Manager, and utilities.
        *   Emphasizes mocking for external dependencies (ONNX runtime, cloud SDKs).
        *   Tool: `pytest`. Goal: >80% code coverage.
    *   **B. Integration Tests (Middle Layer):**
        *   Focuses on interactions: Preprocessing pipeline, full local OCR workflow (with actual small ONNX models), full cloud OCR workflow (mocked/staged APIs), engine switching, UI-Orchestrator interaction, and Config Manager-Engine loading.
        *   Tools: `pytest` with fixtures, `unittest.mock`, potentially Docker.
    *   **C. End-to-End (E2E) Tests (System Level):**
        *   User-perspective testing of the entire application (Windows Client).
        *   Scenarios include OCRing various documents with local and cloud engines, UI flow for engine selection/API key config, output verification, error handling, and dynamic engine switching.
        *   Tools: UI automation frameworks (`PyAutoGUI`, etc.), ground truth comparison scripts.
    *   **D. Performance Tests (Specialized):**
        *   Evaluates speed, resource usage, and stability.
        *   Tests include latency, throughput (PPM), resource monitoring (CPU, GPU/DirectML, RAM/VRAM), and stress tests for both local and cloud operations.
        *   Tools: Custom scripts, Windows Performance Monitor, GPU-specific monitors, Python memory profilers.

3.  **Test Data Management:**
    *   Defines strategies for unit, integration, and E2E test data (including curated sets with ground truth like ICDAR subsets).
    *   Includes specialized datasets for noise/distortion and different languages.
    *   Highlights the use of synthetic data (TRDG) for augmenting test sets.
    *   Specifies versioning of test data with Git LFS.

The document provides a structured approach to ensure the quality and reliability of the OCR-X application.
