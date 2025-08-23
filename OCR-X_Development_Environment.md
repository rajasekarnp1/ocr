# OCR-X Project: Development Environment Setup (Option B - Flexible Hybrid Powerhouse)

This document outlines the development environment setup for the OCR-X project, specifically tailored for Option B: Flexible Hybrid Powerhouse. This setup supports development for an architecture that integrates robust local OCR capabilities (utilizing DirectML for acceleration) with the ability to connect to commercial cloud OCR APIs.

## 1. Local Development Environment Setup (for Windows)

This setup is crucial for developers working directly on the Windows client and DirectML integrations.

*   **Core Requirements:**
    *   **Python:**
        *   **Version:** Python 3.9.13. This specific patch version is recommended for stability with the selected libraries.
        *   **Management:**
            *   **pyenv-win:** Recommended for managing multiple Python versions on Windows (`pyenv-win` GitHub repository).
            *   **Anaconda/Miniconda:** A viable alternative, especially if developers prefer Conda environments.
        *   **Virtual Environments:** Mandatory. Use `venv` (built-in) or `conda create -n ocrx_env python=3.9.13`.
    *   **Git:**
        *   **Version Control:** Latest stable version of Git for Windows.
        *   **Git LFS:** Required for managing large model files, as specified in `OCR-X_Technology_Selection_OptionB.md`.
    *   **IDE Recommendations:**
        *   **Visual Studio Code (VS Code):**
            *   **Primary Choice:** Excellent Python support, debugging, and terminal integration.
            *   **Essential Extensions:**
                *   `Python` (Microsoft)
                *   `Pylance` (Microsoft - for enhanced IntelliSense)
                *   `GitLens` (for advanced Git integration)
                *   `Docker` (Microsoft - if using Docker for testing)
                *   ` Ruff` (charliermarsh - for linting and formatting, can replace Flake8/Black)
        *   **PyCharm Professional:** A strong alternative with robust Python-specific features, though VS Code is often preferred for its versatility.

*   **Key Python Libraries Installation:**
    *   A `requirements.txt` file will be maintained at the root of the project.
    *   **Installation Method:** After setting up the virtual environment, run:
        ```bash
        pip install -r requirements.txt
        ```
    *   **Key Libraries (from `OCR-X_Technology_Selection_OptionB.md` - versions as specified there):**
        *   `opencv-python==4.8.1.78`
        *   `onnxruntime-directml==1.16.3` (or `onnxruntime-gpu` if needing CUDA for other tasks, but DirectML is primary for Option B deployment)
        *   `tensorflow==2.15.0` (if using TensorFlow for model training/conversion, consider `tensorflow-directml-plugin` for training acceleration on Windows)
        *   `torch==2.1.2` (if using PyTorch for model training/conversion, check for `torch-directml` package or PyTorch's native DirectML support status for training)
        *   `paddlepaddle==2.6.0` (for model conversion from PaddleOCR)
        *   `paddleocr==2.7.3`
        *   `transformers==4.35.2`
        *   `qiskit==0.45.1`
        *   `PyQt6==6.6.1` (UI Framework)
        *   `Pillow==10.1.0`
        *   `PyMuPDF==1.23.7`
        *   `numpy==1.26.2`
        *   `scikit-image==0.22.0`
        *   `PyWavelets==1.5.0`
        *   `tf2onnx==1.15.1`
        *   `paddle2onnx==1.1.0`
        *   `trdg==1.7.0`
        *   `scikit-learn==1.3.2` (for evaluation)
        *   Testing: `pytest`, `coverage`
        *   Testing: `pytest`, `coverage`
        *   Linting/Formatting: `ruff` (preferred), `flake8`, `black`
        *   Cloud SDKs (as per `OCR-X_Technology_Selection_OptionB.md`):
            *   `google-cloud-documentai==2.20.0` (or latest stable)
            *   `google-auth==2.23.4` (or latest stable, often a dependency)
            *   `azure-ai-vision-imageanalysis==1.0.0b1` (or latest stable for Read API)
            *   `azure-ai-formrecognizer==3.3.2` (or latest stable, if using Document Intelligence features)
            *   `azure-identity==1.15.0` (or latest stable)
        *   Secure Credential Management:
            *   `keyring` (latest stable)

    *   **Cloud SDK Setup/Authentication (Local Development):**
        *   Developers will need to configure authentication for Google Cloud and Azure APIs for their local testing and development against live cloud services (where necessary). This typically involves:
            *   **Google Cloud:** Setting up Application Default Credentials (ADC) by running `gcloud auth application-default login` through the Google Cloud CLI, OR by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a downloaded service account JSON keyfile. For development, service accounts should have minimal, role-specific permissions.
            *   **Azure:** Logging in via Azure CLI (`az login`) for user-based authentication during development is often simplest. Alternatively, configure environment variables for a service principal (e.g., `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`) or use certificate-based authentication if required by project policy.
            *   **Important:** Actual API keys or sensitive credential files should NOT be committed to the repository. Use environment variables, local user-specific configuration files not checked into Git, or tools like `python-dotenv` to manage them locally during development. Secure credential handling for deployed application is detailed in `OCR-X_Security_Implementation.md`.
            *   Refer to official Google Cloud and Azure documentation for detailed, up-to-date authentication instructions.

*   **DirectML Setup (for Local OCR Engine Components):**
    *   **Operating System:** Windows 10 version 1709 (Fall Creators Update) or later, or Windows 11.
    *   **GPU Drivers:** Ensure GPU drivers (NVIDIA, AMD, Intel) are up-to-date and support DirectML. Refer to GPU manufacturer's websites.
    *   **ONNX Runtime:** `onnxruntime-directml` package installation via pip is the primary method for local model inference. No separate SDK is typically needed.
    *   **TensorFlow with DirectML Plugin:** For training TensorFlow models using DirectML (if not solely relying on ONNX for inference of local models).
        *   Follow Microsoft's official guide: `https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin`
    *   **PyTorch with DirectML:** PyTorch has been building native DirectML support. For local model training/conversion:
        *   Follow PyTorch's official installation instructions for the version that includes DirectML support: `https://pytorch.org/get-started/locally/` (select appropriate options for Windows, DirectML).
    *   **Verification:** Test simple local ONNX models with `onnxruntime-directml` to ensure DirectML is being utilized (e.g., check GPU usage in Task Manager, library-specific logging).

*   **WSL2 (Windows Subsystem for Linux 2 - Optional but Recommended):**
    *   **Use Cases:**
        *   Running Linux-specific data processing or model training tools if they are not easily available or performant on Windows directly.
        *   Easier management of some Python environments if developers are more comfortable with Linux toolchains.
        *   Running Docker containers if Docker Desktop for Windows has performance issues for specific tasks (though Docker Desktop now integrates well with WSL2).
    *   **Setup:** Follow Microsoft's official WSL2 installation guide: `https://learn.microsoft.com/en-us/windows/wsl/install`
    *   **Integration:** VS Code offers excellent WSL2 integration, allowing development "inside" WSL2 from the Windows GUI.

## 2. Docker/Container Configuration

For consistent testing, component isolation during development, and potentially for packaging some backend parts.

*   **Base Image:**
    *   **For General Testing/CPU-bound tasks:** `python:3.9-slim-bullseye` or `python:3.9-windowsservercore-ltsc2022` (for Windows containers, though Linux containers are generally more common and versatile).
    *   **For GPU-accelerated tasks (model conversion/training not using DirectML, or specific component tests needing CUDA):**
        *   NVIDIA CUDA base image: `nvidia/cuda:12.1.0-devel-ubuntu22.04` (example, version should align with chosen ML framework requirements). This would be for tasks *before* DirectML-specific inference. DirectML itself is a Windows host technology and typically not used *inside* a Linux Docker container in this context.
*   **`Dockerfile` Example (Conceptual for a Linux-based test environment):**

    ```dockerfile
    # Base image with Python 3.9
    FROM python:3.9.13-slim-bullseye

    # Set working directory
    WORKDIR /app

    # Install system dependencies (example: OpenCV might need some)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        # Add other necessary system libraries for your Python packages
        && rm -rf /var/lib/apt/lists/*

    # Copy requirements file
    COPY requirements.txt .

    # Install Python dependencies
    # Consider using a multi-stage build for smaller final images if needed
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY . .

    # Environment variables (example)
    ENV PYTHONPATH=/app
    ENV OCRX_MODEL_PATH=/app/models 
    # (Models would need to be ADDed or mounted in a real scenario)

    # Default command (e.g., for running tests)
    # CMD ["pytest", "tests/"] 
    # Or an entrypoint script for more complex startup
    ENTRYPOINT ["/app/docker-entrypoint.sh"] 
    # Example: docker-entrypoint.sh could run tests or a specific component
    # If using mock API servers, this Dockerfile would also include their setup and dependencies.
    ```

*   **Use Cases:**
    *   **Consistent Testing:** Running unit and integration tests in a standardized Linux environment, ensuring that tests pass consistently regardless of the developer's local Windows setup nuances (for non-DirectML specific parts).
    *   **Component Isolation:** Developing and testing specific backend modules (e.g., the core PaddleOCR/SVTR model wrappers before ONNX conversion, or the NLP correction logic) in isolation.
    *   **Build Environment:** Using a Docker container as a clean build environment for creating ONNX models from PyTorch/TensorFlow/PaddlePaddle, ensuring all conversion tool dependencies are met.
    *   **Synthetic Data Generation:** The TRDG pipeline can be run within a container to ensure consistent data generation across different systems.
    *   **Testing Cloud API Client Logic without Live Calls:** A container can be configured to run mock API servers (e.g., using Python libraries like `aiohttp`, `FastAPI`, or `Flask` to simulate Google/Azure responses). This allows for isolated testing of the `CloudOCRClient` components' request formatting, response parsing, and error handling logic without incurring costs or relying on live external services, which is particularly useful for automated testing in CI. The Dockerfile might include dependencies and commands to run these mock servers.

## 3. CI/CD Pipeline Definition (e.g., using GitHub Actions)

Automated pipeline for building, testing, and potentially deploying OCR-X.

*   **Trigger Events:**
    *   On `push` to `main` branch.
    *   On `pull_request` targeting `main` branch.
    *   Optionally, on `push` to feature branches for early validation.

*   **Key Stages/Jobs:**
    1.  **Checkout Code:** `actions/checkout@v4`
    2.  **Set up Python Environment:** `actions/setup-python@v4` with Python 3.9.
    3.  **Install Dependencies:** `pip install -r requirements.txt`. Cache dependencies for faster builds.
    4.  **Lint & Format Check:**
        *   Run `flake8` (or `ruff check`).
        *   Run `black --check` (or `ruff format --check`).
    5.  **Unit Tests:** Execute `pytest tests/unit` (or your chosen test runner and path). Generate coverage reports.
    6.  **Integration Tests:** Execute `pytest tests/integration`. These tests will verify interactions between major components. This stage might require more setup (e.g., dummy model files, test data).
    7.  **OCR Accuracy Benchmarking (on a small, standardized dataset):**
        *   **Action:** Run a dedicated script that uses OCR-X's core pipeline to process a small, version-controlled benchmark dataset. This should be executed for:
            *   The local OCR engine ensemble.
            *   The Google Cloud OCR engine integration (if CI environment is configured with credentials/mocks).
            *   The Azure AI Vision engine integration (if CI environment is configured with credentials/mocks).
        *   **Metrics:** Use `ocreval` (for CER) and `jiwer` (for WER) to compare output against ground truth.
        *   **Thresholding:** Optionally, configure the job to fail if CER/WER exceeds predefined thresholds for any of the benchmarked engine paths.
        *   **Reporting:** Output CER/WER for each tested engine path to the job summary or as artifacts.
    8.  **Testing Cloud API Integrations:**
        *   Cloud API integration testing in CI will primarily rely on mocked API responses (as part of integration tests) to ensure client logic (request formatting, response parsing, error handling for known conditions) works correctly without incurring costs or external dependencies.
        *   Occasional, carefully controlled integration tests against sandboxed or development-tier cloud provider accounts (using dedicated, restricted CI-specific API keys stored as GitHub secrets) might be run on a less frequent schedule (e.g., nightly builds on the `main` branch, or manually triggered) to catch real integration issues or undocumented API changes.
    9.  **Build Windows Package (Manual Trigger or on Tag/Release):**
        *   Use tools like `PyInstaller` (for Python-based applications) or MSIX packaging tools if building a .NET or C++ component.
        *   Upload the built package as a workflow artifact.
    9.  **Create Release (Manual Trigger or on Tag):** Draft a new GitHub release and attach the packaged application.

*   **`workflow.yml` Example (Conceptual Snippet for GitHub Actions):**

    ```yaml
    name: OCR-X CI/CD Windows

    on:
      push:
        branches: [ main ]
      pull_request:
        branches: [ main ]

    jobs:
      build-and-test-windows:
        runs-on: windows-latest # Crucial for DirectML related testing if possible in CI
        strategy:
          matrix:
            python-version: ['3.9']

        steps:
          - uses: actions/checkout@v4
            with:
              lfs: true # Checkout LFS files (models, benchmark data)

          - name: Set up Python ${{ matrix.python-version }}
            uses: actions/setup-python@v4
            with:
              python-version: ${{ matrix.python-version }}

          - name: Install Dependencies
            run: |
              python -m pip install --upgrade pip
              pip install -r requirements.txt
            # Caching dependencies can be added here for pip

          - name: Lint with Ruff and Bandit
            run: |
              pip install ruff bandit # Ensure ruff and bandit are available
              ruff check .
              ruff format --check .
              bandit -r ocrx_project_root_dir_or_src_dir -ll # Adjust ocrx_project_root_dir_or_src_dir and level (e.g. -ll for medium)

          - name: Run Unit Tests with Pytest
            run: |
              pip install pytest pytest-cov # Ensure pytest is available
              pytest tests/unit --cov=./ --cov-report=xml

          # Placeholder for DirectML availability check / specific setup if needed in CI
          # This is complex in GitHub-hosted runners; might need self-hosted runners with GPUs.
          # - name: Check DirectML
          #   run: |
          #     # Script to test onnxruntime-directml basic functionality
          #     python -c "import onnxruntime as rt; print(rt.get_available_providers())" 
          #     # This will list 'DmlExecutionProvider' if available

          - name: Run Integration Tests (CPU fallback if no DirectML in CI, includes mocked cloud clients)
            run: |
              pytest tests/integration

          - name: Run OCR Accuracy Benchmark (CPU fallback for local models, mocked for cloud)
            env:
                # Ensure local tests run with CPU provider if DirectML is not reliably available in CI
                ONNXRUNTIME_PREFERRED_PROVIDERS: 'CPUExecutionProvider' 
                # Variables to enable mock mode for cloud clients during benchmark script run
                OCR_GOOGLE_MOCK_MODE: "true"
                OCR_AZURE_MOCK_MODE: "true"
            run: |
              python scripts/run_benchmark.py --dataset data/benchmark_small --output results/benchmark_ci --engines local,google,azure
              python scripts/calculate_accuracy.py --pred results/benchmark_ci --gt data/benchmark_small/groundtruth.json --cer-threshold 3.0 --wer-threshold 7.0
            # Note: Actual benchmark script and accuracy calculation would need to be developed.

          - name: Upload Coverage Report
            uses: actions/upload-artifact@v3
            with:
              name: coverage-report-${{ matrix.python-version }}
              path: coverage.xml

          - name: Upload Benchmark Results
            if: always() # Upload even if accuracy thresholds fail
            uses: actions/upload-artifact@v3
            with:
              name: benchmark-results-${{ matrix.python-version }}
              path: results/benchmark_ci/
    ```
    *Note: Testing DirectML capabilities directly in standard GitHub-hosted runners is challenging as they typically lack GPUs or the necessary Windows environment. For full DirectML testing, self-hosted Windows runners with appropriate hardware would be required. The example above shows a CPU fallback for benchmarks.*

## 4. Code Quality Gates

*   **Linters:**
    *   **Ruff:** Recommended (`ruff check .`). It's extremely fast and can replace Flake8, Black, isort, and many other tools.
    *   *Alternative:* `Flake8` (with plugins: `flake8-bugbear`, `flake8-comprehensions`, `flake8-docstrings`, `pep8-naming`).
*   **Formatters:**
    *   **Ruff Formatter:** Recommended (`ruff format .`).
    *   *Alternative:* `Black` (enforced via CI and ideally pre-commit hooks).
*   **Pre-commit Hooks:**
    *   Use `pre-commit` framework to run linters/formatters automatically before each commit.
    *   Example `.pre-commit-config.yaml`:
        ```yaml
        repos:
        -   repo: https://github.com/astral-sh/ruff-pre-commit
            rev: v0.1.9 # Use latest ruff version (or specific version used in project)
            hooks:
            -   id: ruff
                args: [--fix, --exit-non-zero-on-fix]
            -   id: ruff-format
        -   repo: https://github.com/PyCQA/bandit
            rev: 1.7.6 # Use latest bandit version
            hooks:
            -   id: bandit
                args: ["-r", "ocrx_project_root_dir_or_src_dir", "-ll"] # Adjust path and level
        -   repo: https://github.com/pre-commit/pre-commit-hooks
            rev: v4.5.0
            hooks:
            -   id: check-yaml
            -   id: end-of-file-fixer
            -   id: trailing-whitespace
        ```
*   **Static Analysis (Integrated into CI):**
    *   **Ruff:** Already covers many static analysis checks (pyflakes, pycodestyle, etc.).
    *   **Bandit:** For security-specific static analysis, integrated into CI and pre-commit.
    *   **SonarLint (IDE):** Recommended for on-the-fly analysis in the IDE (VS Code, PyCharm).
    *   **Pylint (Optional):** Can be added for more comprehensive (and often more verbose) static analysis if desired, but Ruff + Bandit cover many common needs.
*   **Code Review Checklists:**
    *   Mandatory peer reviews for all Pull Requests to `main`.
    *   Checklist items:
        *   Code readability and clarity.
        *   Adherence to project coding style and conventions.
        *   Correctness and completeness of implementation against requirements.
        *   Adequacy of unit tests (coverage, edge cases).
        *   Performance considerations (especially for critical path code).
        *   Security implications (if any).
        *   Documentation (docstrings, comments for complex logic).
        *   No leftover debugging code or commented-out large blocks.
*   **Test Coverage:**
    *   **Tool:** `pytest-cov` (plugin for Pytest, using `coverage.py`).
    *   **Target:** Aim for >80% unit test line coverage for new code.
    *   **CI Integration:** Fail CI job if coverage drops below a certain threshold or if the coverage report for a PR shows a decrease.

By establishing this comprehensive development environment and CI/CD pipeline, the OCR-X project aims to maintain high code quality, ensure consistent testing, and facilitate efficient development of the On-Premise Powerhouse application.I have generated the content for `OCR-X_Development_Environment.md`. It includes:

1.  **Local Development Environment Setup (for Windows):**
    *   Core requirements: Python version (3.9.13 via pyenv-win or Conda), Git, IDE (VS Code with extensions).
    *   Key Python libraries installation via `pip install -r requirements.txt`.
    *   DirectML setup instructions/links for ONNX Runtime, TensorFlow, PyTorch.
    *   WSL2 recommendation.

2.  **Docker/Container Configuration:**
    *   Base image recommendations (`python:3.9-slim-bullseye`, NVIDIA CUDA image for specific tasks).
    *   A conceptual `Dockerfile` example for a Linux-based test environment, including system dependencies, Python library installation, and environment setup.
    *   Use cases: consistent testing, component isolation, build environment for ONNX, synthetic data generation.

3.  **CI/CD Pipeline Definition (GitHub Actions):**
    *   Trigger events (push to `main`, pull request to `main`).
    *   Key stages/jobs: Checkout, Python setup, dependency installation, Lint & Format (Ruff), Unit Tests (pytest), Integration Tests, OCR Accuracy Benchmarking (with CER/WER calculation and thresholding), optional packaging.
    *   A conceptual `workflow.yml` snippet for GitHub Actions, including LFS checkout and a note on DirectML testing challenges in CI.

4.  **Code Quality Gates:**
    *   Linters (Ruff or Flake8).
    *   Formatters (Ruff Formatter or Black).
    *   Pre-commit hooks setup.
    *   Static Analysis recommendations (SonarLint, Pylint).
    *   Code review checklist points.
    *   Test coverage targets (>80% with `pytest-cov`).

This structure and content should fulfill the requirements of the subtask. The file is ready to be created with this content.
