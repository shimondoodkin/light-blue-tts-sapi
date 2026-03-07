# LightBlue SAPI - Hebrew Neural TTS for Windows

A native Windows SAPI 5 text-to-speech engine for Hebrew, powered by the [LightBlue](https://github.com/notmax123/LightBlue) neural TTS model.

Registers as a standard Windows voice — works with any SAPI 5 application (Narrator, screen readers, Balabolka, etc.).

## Features

- **SAPI 5 integration** — appears as a standard Windows voice
- **Male and female voices** with multiple quality/speed presets (4, 8, 32, 64 diffusion steps)
- **CPU (DirectML)** — works out of the box, no GPU required
- **GPU (CUDA/TensorRT)** — optional, for faster synthesis on NVIDIA GPUs
- **Hebrew diacritization** — automatic nikud via [phonikud-rs](https://github.com/shimondoodkin/phonikud-rs)
- **CLI tool** — `lightblue-tts.exe` for command-line speech synthesis

## Installation

### CPU-Only (Simplest)

1. Download and run **LightBlue-TTS-SAPI-Setup.exe** (app installer)
2. Download and run **LightBlue-TTS-SAPI-Models.exe** (model files)
3. Done! The voice appears in Windows speech settings.

### GPU (TensorRT / CUDA)

For faster synthesis on NVIDIA GPUs:

1. Install the latest **CUDA 12.x** from [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Install the latest **cuDNN 9.x** from [cuDNN Archive](https://developer.nvidia.com/cudnn-archive)
3. Download and run **LightBlue-TTS-SAPI-Setup.exe** (install the app first)
4. Download and run **LightBlue-TTS-SAPI-Models.exe** (installs into the app folder)
5. Download and run **LightBlue-TTS-SAPI-ORT-GPU.exe** (overwrites CPU onnxruntime.dll with GPU version — must be installed after the app setup)
6. Select one of the "GPU" voice variants in Windows speech settings.

## Download Models

You can either download a pre-built models ZIP from the [Releases](https://github.com/shimondoodkin/light-blue-tts-sapi/releases) page, or prepare them from scratch:

### Prepare models from scratch

```bash
# 1. Download raw models from HuggingFace
python scripts/download_models.py

# 2. Apply backbone surgery (GPU optimization)
pip install onnx numpy
python scripts/backbone_surgery.py

# 3. Simplify all models with onnxsim
pip install onnxsim
python scripts/simplify_models_in_place.py
```

> **Note:** `onnxsim` may not have prebuilt wheels for your Python version on Windows.
> If `pip install onnxsim` fails, use the included builder script:
> ```bash
> python scripts/build_onnxsim.py
> ```
> It checks prerequisites (Git, CMake, MSVC), installs what's missing, then clones and builds onnxsim from source.

## Building from Source

### Prerequisites

- Windows 10/11 x64
- [Rust](https://rustup.rs/) (stable, MSVC toolchain)
- [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++" workload
- [Python 3.x](https://www.python.org/downloads/) (for model scripts)
- [NSIS](https://nsis.sourceforge.io/) (for building installers)

```bash
# Install Rust (if not installed)
winget install Rustlang.Rustup

# Install NSIS (if not installed)
choco install nsis -y
```

### Build

```bash
# CPU/DirectML build (default — build.rs auto-downloads WinML ORT)
cargo build --release
```

Build outputs in `target/release/`:
- `lightblue_sapi.dll` — SAPI 5 COM engine
- `lightblue-tts.exe` — CLI tool
- `onnxruntime.dll` — ONNX Runtime (auto-downloaded by build.rs)

### Download Models

```bash
python scripts/download_models.py
```

### Register the SAPI Voice

```bash
regsvr32 target\release\lightblue_sapi.dll
```

### Deploy to Program Files (development)

PowerShell scripts for deploying built files to the install directory without running the full NSIS installer:

```powershell
# Deploy DLL + ORT + register COM (requires admin)
powershell -ExecutionPolicy Bypass -File installer\install.ps1

# Deploy models only (requires admin)
powershell -ExecutionPolicy Bypass -File installer\deploy-models.ps1

# Uninstall (unregister COM + remove install directory)
powershell -ExecutionPolicy Bypass -File installer\uninstall.ps1
```

### Build NSIS Installers

```bash
mkdir target\installer

# App installer
makensis installer\setup.nsi

# Models installer (requires models/ to be populated)
makensis installer\setup-models.nsi
```

### Build ORT GPU Installer

The ORT GPU installer repackages the official [ONNX Runtime GPU release](https://github.com/microsoft/onnxruntime/releases) DLLs into an NSIS installer that extracts them to the app folder.

```bash
# Download ONNX Runtime GPU (example version)
curl -L -o ort-gpu.zip https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip
unzip ort-gpu.zip

# Copy DLLs to installer directory
cp onnxruntime-win-x64-gpu-1.23.2/lib/onnxruntime.dll installer/
cp onnxruntime-win-x64-gpu-1.23.2/lib/onnxruntime_providers_shared.dll installer/
cp onnxruntime-win-x64-gpu-1.23.2/lib/onnxruntime_providers_cuda.dll installer/
cp onnxruntime-win-x64-gpu-1.23.2/lib/onnxruntime_providers_tensorrt.dll installer/

# Build installer
makensis installer\setup-ort-gpu.nsi
```

## Model Surgery (GPU Optimization)

The `backbone_keys.onnx` model contains CPU-only ops that cause CPU<->GPU memory transfers under CUDA EP. The surgery script replaces these with GPU-compatible equivalents:

```bash
pip install onnx numpy
python scripts/backbone_surgery.py models/backbone_keys_orig.onnx models/backbone_keys.onnx
```

All models are also processed with [onnxsim](https://github.com/daquexian/onnx-simplifier) (`python scripts/simplify_models_in_place.py`) to optimize the graph.

See [docs/backbone-onnx-surgery.md](docs/backbone-onnx-surgery.md) for details.

## Credits

- **LightBlue TTS** model by [notmax123](https://github.com/notmax123/LightBlue) (HuggingFace: [notmax123/LightBlue](https://huggingface.co/notmax123/LightBlue))
- **Phonikud** Hebrew diacritization by [thewh1teagle](https://github.com/thewh1teagle/phonikud) (ONNX: [thewh1teagle/phonikud-onnx](https://huggingface.co/thewh1teagle/phonikud-onnx))
- **ONNX Runtime** by Microsoft
- SAPI 5 engine implementation by Shimon Doodkin

## License

MIT License. See [LICENSE](LICENSE).
