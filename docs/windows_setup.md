# Windows Setup Guide

This guide walks through setting up the Snowflake2025 project on a Windows 10/11 laptop, including creating a Python virtual environment with PyTorch and checking out the `oct10` branch.

## 1. Install Prerequisites

1. **Windows updates**: Ensure Windows Update has been run recently and reboot if needed.
2. **Git**: Download and install the latest Git for Windows from [https://git-scm.com/download/win](https://git-scm.com/download/win). During setup you can accept the default options.
3. **Python 3.10+**:
   - Download from the [official Python website](https://www.python.org/downloads/windows/).
   - During installation, enable **"Add python.exe to PATH"**.
4. **Microsoft C++ Build Tools (required for some Python packages)**:
   - Download the Visual Studio Build Tools installer from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
   - In the installer select the **"Desktop development with C++"** workload and complete the installation.
5. **(Optional) NVIDIA CUDA Toolkit**: Only required if you have an NVIDIA GPU and want GPU-accelerated PyTorch. Install CUDA 12.1 or the version recommended by [PyTorch](https://pytorch.org/get-started/locally/).

> **Tip**: If you only need CPU execution you can skip CUDA. PyTorch wheels for CPU are provided below.

## 2. Clone the Repository

Open **Windows Terminal** or **PowerShell**, then run:

```powershell
# Pick a directory where you want to keep the project
cd ~\source

# Clone the repo
git clone https://github.com/niallc/Snowflake2025.git
cd Snowflake2025

# Checkout the desired branch (after the initial clone you can switch between branches)
git checkout oct10
```

If you prefer to start on `main` first, run `git checkout main`, then switch to `oct10` later with `git checkout oct10`.

## 3. Create and Activate a Virtual Environment

```powershell
# Create the virtual environment in the project folder
python -m venv .venv

# Activate the virtual environment for the current shell session
.\.venv\Scripts\Activate.ps1
```

If you see an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once in PowerShell and retry the activation command.

## 4. Install PyTorch and Project Dependencies

PyTorch provides platform-specific wheels. Choose **one** of the following commands while the virtual environment is active:

- **CPU only** (works on all Windows machines):
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```
- **CUDA 12.1 (NVIDIA GPU)**:
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

After PyTorch is installed, install the rest of the dependencies from the repo:

```powershell
pip install -r requirements.txt
```

> **Note**: `requirements.txt` lists PyTorch packages, but installing from the official index first ensures you get the correct Windows wheels.

## 5. Configure Environment Variables

Most project commands expect `PYTHONPATH` to include the repository root. You can set it for the current session with:

```powershell
setx PYTHONPATH "$PWD"
```

Close and reopen your terminal (or run `RefreshEnv`) so the new `PYTHONPATH` value is available. When using a new terminal session, remember to reactivate the virtual environment with `.\.venv\Scripts\Activate.ps1`.

## 6. Verify the Installation

With the virtual environment active and `PYTHONPATH` set, run the quick setup validation scripts:

```powershell
python scripts\agent_setup.py
python scripts\validate_environment.py
```

If you want to confirm PyTorch can see your hardware:

```powershell
python scripts\check_device.py
```

## 7. Create Required Project Directories

The project expects several directories to exist the first time you run it:

```powershell
python scripts\setup_directories.py
```

This command is safe to run multiple times.

## 8. Next Steps

- **Run the web app**: `PYTHONPATH=. python -m hex_ai.web.app --port 5001`
- **Start training**: Follow the command examples in `README.md` under "Training"
- **Update to latest branch**: Pull new changes with `git pull` and switch branches with `git checkout <branch>`

## Troubleshooting

- **"cl.exe not found" errors** during `pip install`: Ensure the C++ Build Tools are installed and that you opened a new terminal after installation.
- **SSL or proxy issues**: If your network uses a proxy, configure `pip` with `pip config set global.proxy http://user:pass@proxy:port`.
- **Virtual environment activation fails**: Run PowerShell as Administrator and execute `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`.
- **CUDA not detected**: Confirm your GPU supports CUDA 12.1, the CUDA toolkit is installed, and you installed the matching PyTorch wheel.

If problems persist, collect the full error message and open an issue or contact the maintainer.
