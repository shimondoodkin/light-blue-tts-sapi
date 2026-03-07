//! LightBlue TTS model downloader with progress bars.
//!
//! Downloads ONNX models, phonikud model, tokenizer, style configs,
//! and ONNX Runtime from HuggingFace / GitHub.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const ORT_VERSION: &str = "1.23.0-dev-20250730-1206-a89b038cd2";
const ORT_PACKAGE: &str = "microsoft.ai.machinelearning";

const ORT_CUDA_VERSION: &str = "1.23.2";
/// DLLs to extract from the CUDA ORT tgz (relative paths inside the archive lib/ folder)
const ORT_CUDA_DLLS: &[&str] = &[
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
];


struct DownloadFile {
    url: &'static str,
    /// Path relative to install dir
    rel_path: &'static str,
}

const FILES: &[DownloadFile] = &[
    // TTS ONNX models
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/backbone_keys.onnx", rel_path: "models/backbone_keys.onnx" },
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/text_encoder.onnx", rel_path: "models/text_encoder.onnx" },
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/reference_encoder.onnx", rel_path: "models/reference_encoder.onnx" },
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/vocoder.onnx", rel_path: "models/vocoder.onnx" },
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/length_pred.onnx", rel_path: "models/length_pred.onnx" },
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/length_pred_style.onnx", rel_path: "models/length_pred_style.onnx" },
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/stats.npz", rel_path: "models/stats.npz" },
    DownloadFile { url: "https://huggingface.co/notmax123/LightBlue/resolve/main/uncond.npz", rel_path: "models/uncond.npz" },
    // Voice styles
    DownloadFile { url: "https://raw.githubusercontent.com/maxmelichov/Light-BlueTTS/main/voices/male1.json", rel_path: "models/voices/male1.json" },
    DownloadFile { url: "https://raw.githubusercontent.com/maxmelichov/Light-BlueTTS/main/voices/female1.json", rel_path: "models/voices/female1.json" },
    // Phonikud
    DownloadFile { url: "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.onnx", rel_path: "models/phonikud.onnx" },
    DownloadFile { url: "https://huggingface.co/dicta-il/dictabert-large-char-menaked/raw/main/tokenizer.json", rel_path: "models/tokenizer.json" },
];

fn default_install_dir() -> PathBuf {
    // Try to find install dir: same directory as this executable, or Program Files
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    if let Some(dir) = &exe_dir {
        // If we're in the install directory (has lightblue_sapi.dll), use it
        if dir.join("lightblue_sapi.dll").exists() {
            return dir.clone();
        }
    }

    PathBuf::from(r"C:\Program Files\LightBlue TTS")
}

fn download_file(client: &reqwest::blocking::Client, url: &str, dest: &Path, pb: &ProgressBar) -> Result<(), String> {
    let resp = client
        .get(url)
        .send()
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let total = resp.content_length().unwrap_or(0);
    if total > 0 {
        pb.set_length(total);
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir failed: {e}"))?;
    }

    let mut file = fs::File::create(dest).map_err(|e| format!("create file failed: {e}"))?;
    let bytes = resp.bytes().map_err(|e| format!("read failed: {e}"))?;
    file.write_all(&bytes).map_err(|e| format!("write failed: {e}"))?;
    pb.set_position(bytes.len() as u64);

    Ok(())
}

fn download_ort(client: &reqwest::blocking::Client, install_dir: &Path, pb: &ProgressBar) -> Result<(), String> {
    let ort_dll = install_dir.join("onnxruntime.dll");
    if ort_dll.exists() {
        pb.finish_with_message("onnxruntime.dll (exists, skipped)");
        return Ok(());
    }

    // WinML nightly build from Azure DevOps ORT-Nightly feed (GPU via DirectML)
    let url = format!(
        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/flat2/{pkg}/{ver}/{pkg}.{ver}.nupkg",
        pkg = ORT_PACKAGE,
        ver = ORT_VERSION,
    );
    pb.set_message(format!("onnxruntime WinML v{ORT_VERSION} (downloading...)"));

    let resp = client
        .get(&url)
        .send()
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let total = resp.content_length().unwrap_or(0);
    if total > 0 {
        pb.set_length(total);
    }

    let bytes = resp.bytes().map_err(|e| format!("read failed: {e}"))?;
    pb.set_position(bytes.len() as u64);

    // Extract onnxruntime.dll from WinML nupkg (runtimes/win-x64/_native/)
    let cursor = std::io::Cursor::new(&bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| format!("zip open failed: {e}"))?;

    let mut found_ort = false;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| format!("zip entry failed: {e}"))?;
        let name = entry.name().to_string();
        if name.ends_with("onnxruntime.dll") {
            let mut out = fs::File::create(&ort_dll).map_err(|e| format!("create failed: {e}"))?;
            std::io::copy(&mut entry, &mut out).map_err(|e| format!("extract failed: {e}"))?;
            found_ort = true;
        }
    }

    if found_ort {
        pb.finish_with_message("onnxruntime.dll (WinML, OK)");
        Ok(())
    } else {
        Err("onnxruntime.dll not found in nupkg".to_string())
    }
}

fn download_ort_cuda(client: &reqwest::blocking::Client, install_dir: &Path, pb: &ProgressBar) -> Result<(), String> {
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ver}/onnxruntime-win-x64-gpu-{ver}.tgz",
        ver = ORT_CUDA_VERSION,
    );
    pb.set_message(format!("onnxruntime-cuda v{ORT_CUDA_VERSION} (downloading...)"));

    let resp = client
        .get(&url)
        .send()
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {} from {url}", resp.status()));
    }

    let total = resp.content_length().unwrap_or(0);
    if total > 0 {
        pb.set_length(total);
    }

    let bytes = resp.bytes().map_err(|e| format!("read failed: {e}"))?;
    pb.set_position(bytes.len() as u64);

    // Extract DLLs from tgz
    let cursor = std::io::Cursor::new(&bytes);
    let gz = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(gz);

    let mut extracted = 0usize;
    for entry in archive.entries().map_err(|e| format!("tar read failed: {e}"))? {
        let mut entry = entry.map_err(|e| format!("tar entry failed: {e}"))?;
        let path = entry.path().map_err(|e| format!("tar path failed: {e}"))?.to_path_buf();
        let fname = path.file_name().and_then(|f| f.to_str()).unwrap_or("");

        if ORT_CUDA_DLLS.contains(&fname) {
            let dest = install_dir.join(fname);
            let mut out = fs::File::create(&dest).map_err(|e| format!("create {fname} failed: {e}"))?;
            std::io::copy(&mut entry, &mut out).map_err(|e| format!("extract {fname} failed: {e}"))?;
            extracted += 1;
        }
    }

    if extracted == ORT_CUDA_DLLS.len() {
        pb.finish_with_message(format!("onnxruntime-cuda v{ORT_CUDA_VERSION} ({extracted} DLLs) OK"));
        Ok(())
    } else {
        Err(format!("only extracted {extracted}/{} DLLs from tgz", ORT_CUDA_DLLS.len()))
    }
}

/// Recursively search for a DLL file under a directory, returning the parent dir path.
fn find_dll_recursive(dir: &Path, dll_name: &str) -> Option<String> {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|f| f.to_str()) == Some(dll_name) {
                return Some(path.parent().unwrap().to_string_lossy().to_string());
            }
            if path.is_dir() {
                if let Some(found) = find_dll_recursive(&path, dll_name) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Check if CUDA 12 toolkit and cuDNN 9 are installed, print instructions if not.
/// Returns true if both are found.
fn check_cuda_prerequisites() -> bool {
    let mut ok = true;

    // Check for CUDA 12 — look for nvcc in common install paths
    let cuda_found = (|| {
        // Check CUDA_PATH env var
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let p = PathBuf::from(&cuda_path);
            if p.join("bin").join("nvcc.exe").exists() {
                // Check version — CUDA 12+ folder is typically named v12.x
                let name = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
                if name.starts_with("v12") {
                    return Some(cuda_path);
                }
            }
        }
        // Check common paths
        let base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if base.exists() {
            if let Ok(entries) = fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("v12")
                        && entry.path().join("bin").join("nvcc.exe").exists()
                    {
                        return Some(entry.path().to_string_lossy().to_string());
                    }
                }
            }
        }
        None
    })();

    match cuda_found {
        Some(path) => println!("[OK] CUDA Toolkit found: {path}"),
        None => {
            ok = false;
            println!("[!!] CUDA 12 Toolkit NOT found.");
            println!();
            println!("     ONNX Runtime CUDA requires CUDA Toolkit 12.x.");
            println!("     Download and install from:");
            println!("     https://developer.nvidia.com/cuda-12-6-3-download-archive");
            println!();
            println!("     After installing, restart your terminal so PATH is updated.");
            println!();
        }
    }

    // Check for cuDNN 9 — look for cudnn64_9.dll on PATH, CUDA dirs, or new NVIDIA cuDNN dir
    let cudnn_found = (|| {
        // Check PATH
        if let Ok(path_var) = env::var("PATH") {
            for dir in path_var.split(';') {
                let p = PathBuf::from(dir).join("cudnn64_9.dll");
                if p.exists() {
                    return Some(dir.to_string());
                }
            }
        }
        // Check classic CUDA bin dirs
        let base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if base.exists() {
            if let Ok(entries) = fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let bin = entry.path().join("bin").join("cudnn64_9.dll");
                    if bin.exists() {
                        return Some(entry.path().join("bin").to_string_lossy().to_string());
                    }
                }
            }
        }
        // Check new cuDNN 9 install location (C:\Program Files\NVIDIA\CUDNN\v9.x\...)
        let cudnn_base = PathBuf::from(r"C:\Program Files\NVIDIA\CUDNN");
        if cudnn_base.exists() {
            if let Ok(versions) = fs::read_dir(&cudnn_base) {
                for ver in versions.flatten() {
                    // Search recursively for cudnn64_9.dll under each version dir
                    let ver_path = ver.path();
                    if let Some(found) = find_dll_recursive(&ver_path, "cudnn64_9.dll") {
                        return Some(found);
                    }
                }
            }
        }
        None
    })();

    match cudnn_found {
        Some(path) => println!("[OK] cuDNN 9 found: {path}"),
        None => {
            ok = false;
            println!("[!!] cuDNN 9 NOT found (cudnn64_9.dll).");
            println!();
            println!("     ONNX Runtime CUDA requires cuDNN 9.x for CUDA 12.");
            println!("     Download the installer from (requires free NVIDIA account):");
            println!("     https://developer.nvidia.com/cudnn-downloads");
            println!();
            println!("     Select: cuDNN 9.x.x for CUDA 12.x, Windows, exe installer.");
            println!("     Run the installer and restart your terminal.");
            println!();
        }
    }

    ok
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut install_dir = default_install_dir();
    let mut force = false;
    let mut cuda = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" | "-d" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --dir requires a path argument");
                    process::exit(1);
                }
                install_dir = PathBuf::from(&args[i]);
            }
            "--force" | "-f" => {
                force = true;
            }
            "--cuda" => {
                cuda = true;
            }
            "--help" | "-h" => {
                println!("LightBlue TTS Model Downloader");
                println!();
                println!("Usage: lightblue-download [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -d, --dir <PATH>   Install directory (default: auto-detect or Program Files)");
                println!("  -f, --force        Re-download all files even if they exist");
                println!("  --cuda             Download CUDA-enabled ONNX Runtime + check CUDA/cuDNN");
                println!("  -h, --help         Show this help");
                process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    println!("LightBlue TTS Model Downloader");
    println!("Install directory: {}", install_dir.display());
    println!();

    // Create directories
    fs::create_dir_all(install_dir.join("models")).ok();
    fs::create_dir_all(install_dir.join("models/voices")).ok();
    fs::create_dir_all(install_dir.join("dictionaries")).ok();

    let client = reqwest::blocking::Client::builder()
        .user_agent("LightBlue-TTS-Downloader/1.0")
        .build()
        .expect("failed to create HTTP client");

    let multi = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("{msg:40} [{bar:30.cyan/dim}] {bytes}/{total_bytes}")
        .expect("invalid template")
        .progress_chars("=> ");

    let done_style = ProgressStyle::default_bar()
        .template("{msg}")
        .expect("invalid template");

    let mut errors = Vec::new();

    // Download ONNX Runtime
    if cuda {
        // Check CUDA prerequisites first
        println!("Checking CUDA prerequisites...");
        println!();
        let prereqs_ok = check_cuda_prerequisites();
        println!();

        if !prereqs_ok {
            println!("Please install the missing prerequisites above, then re-run:");
            println!("  lightblue-download --cuda");
            println!();
            println!("Continuing with ORT CUDA download anyway...");
            println!();
        }

        // CUDA variant — downloads from GitHub releases, overwrites default DLLs
        let pb = multi.add(ProgressBar::new(0));
        pb.set_style(style.clone());
        match download_ort_cuda(&client, &install_dir, &pb) {
            Ok(()) => pb.set_style(done_style.clone()),
            Err(e) => {
                pb.finish_with_message(format!("onnxruntime-cuda FAILED: {e}"));
                pb.set_style(done_style.clone());
                errors.push(format!("onnxruntime-cuda: {e}"));
            }
        }
    } else {
        let pb = multi.add(ProgressBar::new(0));
        pb.set_style(style.clone());
        pb.set_message("onnxruntime.dll");
        match download_ort(&client, &install_dir, &pb) {
            Ok(()) => pb.set_style(done_style.clone()),
            Err(e) => {
                pb.finish_with_message(format!("onnxruntime.dll FAILED: {e}"));
                pb.set_style(done_style.clone());
                errors.push(format!("onnxruntime.dll: {e}"));
            }
        }
    }

    // Download model files
    for file in FILES {
        let dest = install_dir.join(file.rel_path);
        let name = Path::new(file.rel_path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let pb = multi.add(ProgressBar::new(0));
        pb.set_style(style.clone());
        pb.set_message(name.clone());

        if !force && dest.exists() {
            pb.finish_with_message(format!("{name} (exists, skipped)"));
            pb.set_style(done_style.clone());
            continue;
        }

        let url = file.url;
        println!("Downloading {name}");
        println!("  from: {url}");
        println!("  to:   {}", dest.display());

        match download_file(&client, url, &dest, &pb) {
            Ok(()) => {
                let size = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                let size_mb = size as f64 / 1_048_576.0;
                pb.finish_with_message(format!("{name} ({size_mb:.1} MB) OK"));
                pb.set_style(done_style.clone());
            }
            Err(e) => {
                pb.finish_with_message(format!("{name} FAILED: {e}"));
                pb.set_style(done_style.clone());
                errors.push(format!("{name}: {e}"));
            }
        }
    }

    println!();
    if errors.is_empty() {
        println!("All files downloaded successfully!");
    } else {
        println!("Download completed with {} error(s):", errors.len());
        for e in &errors {
            println!("  - {e}");
        }
        process::exit(1);
    }
}
