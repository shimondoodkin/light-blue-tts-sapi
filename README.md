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

1. Download and run **LightBlue-TTS-Setup.exe** (app installer)
2. Download and run **LightBlue-TTS-Models.exe** (model files)
3. Done! The voice appears in Windows speech settings.

### GPU (TensorRT / CUDA)

For faster synthesis on NVIDIA GPUs:

1. Install **CUDA 12** and **cuDNN 9**:
   ```
   winget install Nvidia.CUDA
   winget install Nvidia.cuDNN
   ```
   Or download manually from [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN Archive](https://developer.nvidia.com/cudnn-archive).

2. Download and run **LightBlue-TTS-Setup.exe** (app installer)
3. Download and run **LightBlue-TTS-Models.exe** (model files)
4. Download and run **LightBlue-TTS-ORT-GPU.exe** (replaces CPU runtime with GPU version)
5. Select one of the "GPU" voice variants in Windows speech settings.

## Building from Source

### Prerequisites

- [Rust](https://rustup.rs/) (stable, MSVC toolchain)
- Windows 10/11 x64

### Build

```bash
# CPU/DirectML build (default — build.rs auto-downloads WinML ORT)
cargo build --release

# CUDA/TensorRT build (downloads GPU ORT automatically)
cargo build --release --features cuda
```

Build outputs in `target/release/`:
- `lightblue_sapi.dll` — SAPI 5 COM engine
- `lightblue-tts.exe` — CLI tool
- `lightblue-download.exe` — model downloader
- `onnxruntime.dll` — ONNX Runtime (auto-downloaded)

### Download Models

```bash
# From original HuggingFace repos
python scripts/download_models.py --original

# Or use the Rust downloader
cargo run --release --bin lightblue-download -- --dir .
```

### Register the SAPI Voice

```bash
regsvr32 target\release\lightblue_sapi.dll
```

### Build NSIS Installers

Install [NSIS](https://nsis.sourceforge.io/), then:

```bash
mkdir target\installer
makensis installer\setup.nsi
makensis installer\setup-models.nsi
```

## Model Surgery (GPU Optimization)

The `backbone_keys.onnx` model contains CPU-only ops that cause CPU<->GPU memory transfers under CUDA EP. The surgery script replaces these with GPU-compatible equivalents:

```bash
pip install onnx numpy
python scripts/backbone_surgery.py models/onnx/backbone_keys_orig.onnx models/onnx/backbone_keys.onnx
```

All models are also processed with [onnxsim](https://github.com/daquexian/onnx-simplifier) (`python -m onnxsim model.onnx model.onnx`) to optimize the graph.

See [docs/backbone-onnx-surgery.md](docs/backbone-onnx-surgery.md) for details.

## Credits

- **LightBlue TTS** model by [notmax123](https://github.com/notmax123/LightBlue) (HuggingFace: [notmax123/LightBlue](https://huggingface.co/notmax123/LightBlue))
- **Phonikud** Hebrew diacritization by [thewh1teagle](https://github.com/thewh1teagle/phonikud) (ONNX: [thewh1teagle/phonikud-onnx](https://huggingface.co/thewh1teagle/phonikud-onnx))
- **ONNX Runtime** by Microsoft
- SAPI 5 engine implementation by Shimon Doodkin

## License

MIT License. See [LICENSE](LICENSE).
