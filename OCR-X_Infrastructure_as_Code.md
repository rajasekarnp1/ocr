# OCR-X Project: Infrastructure as Code (IaC) Strategy (Option B - Flexible Hybrid Powerhouse)

This document outlines the Infrastructure as Code (IaC) strategy for the OCR-X project (Option B: Flexible Hybrid Powerhouse). It details IaC practices relevant to a primarily desktop application with needs for consistent staging environments and automated packaging.

## I. IaC Philosophy for OCR-X (Option B)

*   **Focus:**
    *   **Automation of Environment Setup:** Primarily for Staging environments to ensure consistency and repeatability for testing.
    *   **Automation of Application Packaging:** Streamlining the build and packaging process for releases.
    *   **Configuration Management:** Managing default application configurations and facilitating consistent developer setups.
*   **Scope Context:**
    *   Given Option B is a Windows desktop application, the IaC scope differs from cloud-native applications. Extensive cloud resource provisioning (e.g., Terraform for complex AWS/Azure services) is less central.
    *   However, IaC principles (automation, versioning, repeatability) are still highly applicable to the development, testing, and packaging lifecycle.
*   **Goals:**
    *   **Consistency:** Ensure Development and Staging environments are as similar as possible to each other and to the eventual user environment (within practical limits).
    *   **Repeatability:** Enable reliable recreation of environments and application packages.
    *   **Efficiency:** Reduce manual effort and errors in environment setup and application deployment.
    *   **Version Control:** Manage all infrastructure and configuration scripts under version control.

## II. Staging Environment Automation

*   **Purpose:** To ensure staging Windows VMs (if used for QA and UAT) are configured consistently, providing a reliable pre-production testing ground.
*   **Target Environment:** Windows 10/11 VMs or physical machines.
*   **Tools & Technologies (Examples):**

    *   **PowerShell Desired State Configuration (DSC):**
        *   **Use Case:** Define and enforce the configuration of Windows machines used for staging.
        *   **Scope:**
            *   Installation of required Windows features (e.g., .NET Framework versions, Hyper-V for nested virtualization if needed for specific tests).
            *   Installation of specific Python runtime versions.
            *   Ensuring DirectML-capable GPU drivers are present (or scripting a check/basic update attempt).
            *   Deployment of common testing tools, test datasets, and any shared test harnesses.
            *   Setting up environment variables or registry keys needed for the application or test tools.
        *   **Example Snippet (Conceptual PowerShell DSC):**
            ```powershell
            Configuration StagingOCRNode {
                Import-DscResource -ModuleName 'PSDesiredStateConfiguration'
                Import-DscResource -ModuleName 'ComputerManagementDsc' # Example for managing features/registry

                Node "localhost" { # Can target remote nodes as well
                    WindowsFeature PythonRuntimeSupport { # Conceptual - actual feature may vary
                        Ensure = "Present"
                        Name   = "NetFx4-AdvSrvs" # Example, real feature for Python might be different or handled by script
                    }

                    Script PythonInstallation {
                        GetScript = { @{ Result = (Test-Path "C:\Python39") } }
                        SetScript = {
                            # Script to download and silently install Python 3.9.13
                            # Example:
                            # Start-Process "python-3.9.13-amd64.exe" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
                        }
                        TestScript = { Test-Path "C:\Python39\python.exe" }
                    }

                    Script DirectMLDriverCheck {
                        GetScript = { @{ Result = "Driver status placeholder" } } # More complex check needed
                        SetScript = {
                            # Placeholder for script that attempts to check/update GPU drivers
                            # This is highly hardware-dependent and complex to fully automate via DSC alone.
                            # Might involve vendor-specific updaters or manual intervention for staging.
                            Write-Warning "Manual DirectML driver verification/update might be needed."
                        }
                        TestScript = {
                            # Placeholder for a script that attempts to verify DirectML functionality
                            # e.g., running a small ONNXRuntime-DirectML test.
                            $true # Assume passes for placeholder
                        }
                    }
                    # Add configurations for other tools, environment variables etc.
                }
            }

            # To apply:
            # StagingOCRNode
            # Start-DscConfiguration -Path ./StagingOCRNode -Wait -Verbose -Force
            ```

    *   **Vagrant with Hyper-V or VirtualBox Provider (for local or CI-managed staging VMs):**
        *   **Use Case:** Define and provision reproducible Windows virtual machines for staging or automated E2E testing.
        *   **`Vagrantfile`:**
            *   Defines VM specifications (OS base image, CPU, memory).
            *   Links to provisioning scripts (PowerShell DSC, simple PowerShell, or shell scripts).
        *   **Example Snippet (Conceptual `Vagrantfile`):**
            ```ruby
            Vagrant.configure("2") do |config|
              config.vm.box = "gusztavvargadr/windows-10" # Example public Windows box
              config.vm.communicator = "winrm"
              config.vm.guest = :windows
              config.vm.network "private_network", ip: "192.168.50.10" # Example network

              config.vm.provider "hyperv" do |h|
                h.memory = 8192
                h.cpus = 4
                h.vm_integration_services = { # Ensure guest services are enabled
                    guest_service_interface: true,
                    heartbeat: true,
                    key_value_pair_exchange: true,
                    shutdown: true,
                    time_synchronization: true,
                    vss: true
                }
              end

              # Provision with PowerShell
              config.vm.provision "shell", path: "scripts/provision_staging_vm.ps1", args: "-PythonVersion 3.9.13"

              # Or provision with DSC
              # config.vm.provision "dsc", configuration_file: "DscConfigurations/StagingOCRNode.ps1",
              #   configuration_name: "StagingOCRNode", module_path: "DscModules"
            end
            ```
        *   **`scripts/provision_staging_vm.ps1` (Conceptual):**
            ```powershell
            param(
                [string]$PythonVersion = "3.9.13"
            )
            Write-Host "Provisioning Staging VM..."
            # Chocolatey for software package management
            # Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
            # choco install python --version=$PythonVersion -y --params "/InstallDir:C:\Python\Python$PythonVersion"
            # choco install git -y
            # choco install vscode -y
            # Add steps for DirectML driver checks, test tool installations, etc.
            Write-Host "Python $PythonVersion environment setup (conceptual)."
            Write-Host "DirectML components check (conceptual)."
            ```

    *   **Ansible (for managing multiple Windows VMs in a more complex or persistent staging setup):**
        *   **Use Case:** If the staging environment consists of multiple, persistent Windows nodes that require consistent configuration management over time.
        *   **Playbooks:** Define roles and tasks to install software, manage services, configure Windows features, and deploy application builds.
        *   **WinRM:** Ansible uses WinRM (Windows Remote Management) to connect to and manage Windows nodes.

*   **Scope of Automation:** Installation of OS patches (via Windows Update configuration), Python environments (specific versions), DirectML components (driver checks, runtime installations), browsers for UI testing (if applicable), shared testing tools (e.g., specific image viewers, dataset management scripts), and ensuring consistent directory structures or environment variables.

## III. Application Packaging and Deployment Automation

*   **Purpose:** To automate the creation of the MSIX package for OCR-X, ensuring a repeatable and reliable build process.
*   **MSIX Packaging Automation:**
    *   **Tools:** `MakeAppx.exe` (from Windows SDK), `signtool.exe` (for signing), PowerShell.
    *   **Scripts:** PowerShell or Python scripts to:
        1.  Collect all application artifacts (Python code, embedded Python runtime if used, ONNX models, images, icons, `config.yaml` template).
        2.  Generate or update the `AppxManifest.xml` file with correct version numbers, capabilities, and file mappings.
        3.  Invoke `MakeAppx.exe` to bundle the application into an `.msix` or `.msixbundle` package.
        4.  Sign the package using `signtool.exe` and a code signing certificate.
    *   **CI/CD Integration (e.g., GitHub Actions):**
        *   Trigger packaging on every push to a release branch or on creating a Git tag.
        *   Securely manage code signing certificates (e.g., using GitHub Secrets or Azure Key Vault).
        *   Upload the built and signed MSIX package as a build artifact.
    *   **Example Snippet (Conceptual PowerShell for MSIX - simplified):**
        ```powershell
        param(
            [string]$Version = "1.0.0.0",
            [string]$SourceDir = "./dist/app_files", # Directory with all application files
            [string]$ManifestPath = "./packaging/AppxManifest.xml", # Template manifest
            [string]$PackageName = "OCR-X_OptionB",
            [string]$PublisherName = "CN=MyCompany", # From your code signing certificate
            [string]$OutputPackagePath = "./release/OCR-X_OptionB_$Version.msix",
            [string]$CertificatePath = $env:SIGNING_CERT_PATH, # Store path in environment variable
            [string]$CertificatePassword = $env:SIGNING_CERT_PASSWORD # Store password in environment variable (securely)
        )

        Write-Host "Starting MSIX Packaging for version $Version..."

        # 1. Update Manifest (example: version number)
        # (Get-Content $ManifestPath) -replace '<Identity Name=".*?" Publisher=".*?" Version=".*?" />', "<Identity Name=`"$PackageName`" Publisher=`"$PublisherName`" Version=`"$Version`" />" | Set-Content $ManifestPath

        # 2. Create Package using MakeAppx.exe
        # $makeAppxPath = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\makeappx.exe" # Adjust path
        # & $makeAppxPath pack /d "$SourceDir" /p "$OutputPackagePath" /o /l # /o for overwrite, /l for localization if needed

        # 3. Sign Package using SignTool.exe
        # $signToolPath = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe" # Adjust path
        # & $signToolPath sign /fd SHA256 /a /f "$CertificatePath" /p "$CertificatePassword" "$OutputPackagePath"

        Write-Host "MSIX Packaging script placeholder for $OutputPackagePath completed."
        # Actual script would involve more robust error handling and path management.
        # Tools like msbuild with a .wapproj (Windows Application Packaging Project) can also automate this.
        ```

*   **ONNX Model Management:**
    *   **Versioning:** Models are versioned in a repository (e.g., Git LFS, dedicated model registry like DVC, MLflow Model Registry, or Artifactory/Nexus).
    *   **Packaging Integration:** Scripts in the CI/CD pipeline (or packaging scripts) will:
        1.  Fetch the specified/latest approved versions of ONNX models from their storage location.
        2.  Place them into the correct directory structure within the application bundle before `MakeAppx.exe` is run.

## IV. Development Environment Consistency (Optional IaC)

*   **Purpose:** To help developers quickly set up a consistent and correct local development environment.
*   **Devcontainer (VS Code):**
    *   **`devcontainer.json`:** Defines a Docker container configuration for development.
        *   Specifies a base image (e.g., a Windows image with Python and DirectML tools, or a Linux image if cross-compilation or WSL2-based dev is preferred for certain tasks).
        *   Lists VS Code extensions to install automatically within the container.
        *   Can define post-create commands to install Python dependencies, configure Git LFS, etc.
    *   **Use Case:** Ideal for new team members or for ensuring that all developers are using the exact same set of core tools and Python package versions, minimizing "works on my machine" issues related to environment discrepancies.
*   **Scripts for Local Setup:**
    *   **PowerShell or Python scripts:** Provide scripts that developers can run to:
        *   Check for and guide the installation of correct Python versions (e.g., using `pyenv-win`).
        *   Verify/guide installation of GPU drivers and DirectML components.
        *   Install all required Python packages from `requirements.txt` into a virtual environment.
        *   Set up pre-commit hooks.
        *   Clone and set up Git LFS for model repositories.

## V. Configuration Management for the Application

*   **Default Configuration:**
    *   A template `config.yaml` (or similar format) containing default settings for the application (e.g., default OCR engine, logging levels, paths to bundled models) will be included in the source code and packaged with the installer.
    *   On first run, the application copies this template to a user-specific directory (e.g., `%APPDATA%\OCR-X\config.yaml`).
*   **User Configuration Updates:**
    *   Primarily managed by the user through the application's UI (as per `OCR-X_Component_Breakdown_OptionB.md`).
    *   The application ensures that changes are saved correctly to the user's configuration file.
    *   IaC's role here is limited to the initial template and the application's logic for managing it, rather than actively configuring user environments post-installation.
*   **Secure Configuration (API Keys):**
    *   The default configuration will **not** contain placeholders for API keys.
    *   The application UI will guide users to input their API keys, which are then stored securely using Windows Credential Manager (via Python's `keyring` library) or similar OS-level secure storage, not in plain text configuration files.

## VI. Version Control for IaC Scripts

*   All IaC scripts and configuration files (PowerShell DSC configurations, `Vagrantfile`s, `Dockerfile`s for devcontainers, `devcontainer.json`, application packaging scripts, default `config.yaml` templates, CI/CD workflow files like `.github/workflows/main.yml`) **must be stored in the project's Git repository.**
*   This ensures that infrastructure and environment definitions are versioned alongside the application code, enabling traceability, rollback capabilities, and collaborative development of the environment setup.

By applying these IaC principles, the OCR-X project aims to achieve more reliable and efficient development, testing, and packaging processes, particularly for its Option B Windows desktop application.
