# OCR-X Project: Environment Strategy (Option B - Flexible Hybrid Powerhouse)

This document outlines the environment strategy for the OCR-X project (Option B: Flexible Hybrid Powerhouse), detailing the setup for Development, Staging, and Production environments.

## I. Overall Environment Philosophy

*   **Goal:** Ensure consistency, reliability, and predictability from development through to production deployment on user machines. This minimizes "works on my machine" issues and streamlines the testing and release process.
*   **Principle of Least Privilege:** Access to resources (especially cloud API keys and production build systems) will be granted on a need-to-know basis.
*   **Automated Configuration:** Where feasible, automate the setup and configuration of environments to reduce manual errors and ensure consistency. For the client-side production environment, this is primarily managed by the installer.
*   **Isolation:** Keep Development, Staging, and Production environments (especially regarding API keys and configurations) strictly separate.

## II. Development Environment (Dev)

*   **Purpose:** Individual developer workspaces for coding, local testing (unit and component integration), and debugging new features and bug fixes.
*   **Setup (as per `OCR-X_Development_Environment.md`):**
    *   **Hardware:** Local Windows machines (Windows 10/11, various hardware to ensure broad compatibility testing).
    *   **Software Stack:**
        *   **Python:** Specific version (e.g., Python 3.9.13) managed via `pyenv-win` (recommended) or Anaconda/Miniconda.
        *   **Virtual Environments:** Mandatory (e.g., using `venv` or `conda create`).
        *   **Git:** Latest stable version of Git for Windows, with Git LFS installed and configured for model files.
        *   **IDE:** Visual Studio Code (recommended with Python, Pylance, Ruff, GitLens extensions) or other preferred Python IDEs (e.g., PyCharm).
        *   **Python Libraries:** All project dependencies installed from `requirements.txt` (or `pyproject.toml` if using Poetry/PDM) within the activated virtual environment.
        *   **DirectML Stack:** Up-to-date GPU drivers supporting DirectML. Relevant SDKs or plugins for TensorFlow/PyTorch if used for model conversion/testing (though primary inference is ONNX Runtime). `onnxruntime-directml` is key.
    *   **Local Models:** Developers will have access to development/test versions of local ONNX models (for preprocessing, recognition, postprocessing) typically pulled via Git LFS. These might be smaller, faster-to-load versions for quicker iteration.
    *   **Cloud API Access (Dev/Test):**
        *   **API Keys:**
            *   Option 1 (Preferred for Isolation): Developers use personal/sandbox development-tier API keys for Google Cloud Vision / Azure AI Vision to test integration. This ensures individual quotas and activities are sandboxed.
            *   Option 2 (Shared Dev Key): A shared, centrally managed development/test API key with a limited quota can be provided by a team lead for specific, coordinated testing. This requires careful management to avoid quota exhaustion.
        *   **Security:** Strict guidelines against committing API keys to version control. Keys should be managed via user-specific environment variables, secure local configuration files (e.g., `.env` files loaded by Python, not committed), or IDE-specific secure storage.
    *   **WSL2:** Optional, but recommended for specific Linux-based tools (e.g., certain data processing scripts, advanced Git operations) or for cross-platform compatibility checks if future Linux support is envisioned.
*   **Testing:** Developers are responsible for running unit tests and relevant integration tests for their changes locally before committing. Local debugging of features and bug fixes is standard practice.

## III. Staging Environment (Pre-Production/QA)

*   **Purpose:** To test the fully integrated application in an environment that closely mirrors a typical end-user's production setup. Used for User Acceptance Testing (UAT), final Quality Assurance (QA), and performance benchmarking before a release.
*   **Setup:**
    *   **Hardware:** Dedicated Windows machines (physical or Virtual Machines) with hardware specifications representative of target user environments. This should include:
        *   Systems meeting minimum specified CPU, RAM, and GPU (DirectML-capable, including integrated GPUs).
        *   Systems meeting recommended specifications for optimal performance.
    *   **Operating System:** Target Windows versions (e.g., Windows 10 Pro/Enterprise 22H2, Windows 11 Pro/Enterprise 23H2). Kept patched and updated similar to typical user environments.
    *   **Application Installation:** Full OCR-X application installed using the official MSIX package (or chosen distribution method for that release candidate). This tests the installer and the installed application.
    *   **Local Models:** All local OCR models (ONNX) deployed exactly as they would be packaged for production (e.g., included in the MSIX or downloaded by the installer). These should be the release candidate versions of the models.
    *   **Cloud API Access (Staging):**
        *   **API Keys:** Dedicated, separate API keys for the Staging/QA environment for Google Cloud Vision / Azure AI Vision. These keys must not be the same as any development or production keys.
        *   **Quotas:** API quotas for these staging keys should be sufficient for thorough testing of all cloud-dependent features, including some level of load/stress testing if applicable.
        *   **Configuration:** The application's configuration in the staging environment will point to these staging-specific API keys. Cloud service endpoints are typically the same production endpoints, as most providers do not offer distinct staging endpoints for standard APIs (authentication/project ID segregates usage).
    *   **Network Configuration:** Ensure staging machines have reliable internet access to reach commercial cloud OCR APIs without overly restrictive firewalls that might not be present on typical user machines (or document necessary firewall exceptions).
*   **Testing:**
    *   Full End-to-End (E2E) test suite execution, including UI automation tests.
    *   Performance benchmarks as outlined in `OCR-X_Quality_Gates.md`.
    *   User Acceptance Testing (UAT) by stakeholders (e.g., product owner, beta testers if applicable).
    *   Exploratory testing by the QA team.
    *   Verification of all requirements outlined in `OCR-X_Requirements_Specification.md`.
*   **Data:** Use a diverse and comprehensive set of test documents, including challenging cases, edge cases, and potentially anonymized samples representative of real user data (if available and ethically permissible for internal QA).

## IV. Production Environment (Prod)

*   **Purpose:** The live environment where end-users install and run the OCR-X application on their own Windows machines.
*   **Setup (User's Machine for Option B - Flexible Hybrid Powerhouse):**
    *   **Installation:** OCR-X is a desktop application installed by the end-user on their Windows machine using the provided MSIX package.
    *   **Local Components:**
        *   All necessary runtimes (e.g., Python embedded within the package, required C++ redistributables) are packaged with or installed as dependencies by the MSIX installer.
        *   Up-to-date DirectML-capable GPU drivers are a user prerequisite (application should check and guide if possible).
        *   Local ONNX models are included in the installer and deployed to the user's machine.
    *   **Cloud API Access (User-Provided/Managed):**
        *   Users who wish to utilize commercial cloud OCR engines (Google Cloud Vision, Azure AI Vision) are responsible for obtaining and providing their own API keys.
        *   The application must provide a secure mechanism for users to enter, store, and manage these API keys. Options include:
            *   Leveraging Windows Credential Manager (preferred for security, via Python bindings like `keyring`).
            *   Storing in an encrypted local configuration file (requires robust encryption and key management).
            *   Clear UI guidance and documentation on how users can obtain API keys from cloud providers and configure them within OCR-X.
        *   The application will use these user-provided keys for all cloud API interactions.
    *   **Configuration:** Application settings (e.g., default engine selection, paths for local models if configurable post-install, user-specific API keys) are stored locally on the user's machine (e.g., in `%APPDATA%\OCR-X` or similar, using secure storage for sensitive items).
*   **Deployment Method:** Primarily via MSIX package for controlled installation, updates, and uninstallation. This could be distributed via a website, Microsoft Store, or enterprise deployment mechanisms.
*   **Monitoring (Client-Side):**
    *   **Local Application Logging:** The application will generate local logs (errors, warnings, key operational info) that users can access to help troubleshoot issues or include in bug reports. Log files should have size limits and rotation.
    *   **Optional Telemetry (Opt-In by User - Future Phase):**
        *   If implemented in later phases for product improvement, users could opt-in to send anonymized performance metrics (e.g., processing times per engine, error rates for specific non-sensitive categories) and crash reports to a central system.
        *   This would require a backend service for telemetry ingestion and analysis, which is currently out of scope for the core "on-premise" aspect of Option B but is a common enhancement for desktop applications.
        *   Privacy implications would need to be clearly communicated, and data anonymized.

## V. Environment Consistency & Configuration Management

*   **Dependency Management:**
    *   Use of `requirements.txt` (generated from a `setup.py` or `pyproject.toml` if using Poetry/PDM) ensures consistent Python library versions across Dev and Staging (for build/test). The MSIX package will bundle these dependencies for Production.
*   **Version Control:**
    *   All source code, build scripts, installer scripts, CI/CD pipeline configurations, and configuration file templates will be stored in Git.
    *   ONNX models and large test data assets will be versioned using Git LFS.
*   **Staging Environment Setup:**
    *   For managing multiple Staging VMs or physical machines, consider using scripting (PowerShell DSC, simple PowerShell scripts) or lightweight configuration management tools to ensure consistent setup (OS features, baseline software, directory structures).
    *   If Staging involves many machines, a more robust solution like Ansible (with Ansible generating PowerShell scripts for Windows targets) could be explored, but might be overkill initially.
*   **Production Environment Setup (User's Machine):**
    *   The MSIX installer is the primary mechanism for ensuring a consistent installation of OCR-X application files, local models, and bundled runtimes on the user's machine.
    *   The application itself should handle the creation and management of user-specific configuration files in appropriate local directories upon first run or when settings are changed.

By maintaining this structured environment strategy, the OCR-X project aims to facilitate efficient development, thorough testing, and reliable deployment to end-users.
