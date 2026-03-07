use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// WinML nightly build — onnxruntime.dll with DirectML support (default).
const ORT_WINML_VERSION: &str = "1.23.0-dev-20250730-1206-a89b038cd2";
const ORT_WINML_PACKAGE: &str = "microsoft.ai.machinelearning";

/// CUDA-enabled ORT from official GitHub releases.
const ORT_CUDA_VERSION: &str = "1.23.2";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let use_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();

    let (ort_lib_dir, variant) = if use_cuda {
        let folder = format!("onnxruntime-win-x64-gpu-{}", ORT_CUDA_VERSION);
        let ort_dir = manifest_dir.join(&folder);
        let lib_dir = ort_dir.join("lib");

        if !lib_dir.exists() {
            println!("cargo:warning=Downloading ORT CUDA v{}...", ORT_CUDA_VERSION);
            download_ort_cuda(&manifest_dir, &folder);
        }

        (lib_dir, "cuda")
    } else {
        let folder = format!("ort-winml-{}", ORT_WINML_VERSION);
        let ort_dir = manifest_dir.join(&folder);
        let lib_dir = ort_dir.join("runtimes").join("win-x64").join("_native");

        if !lib_dir.exists() {
            println!("cargo:warning=Downloading ORT WinML nightly {}...", ORT_WINML_VERSION);
            download_ort_winml(&manifest_dir, &folder);
        }

        (lib_dir, "winml")
    };

    copy_dlls_to_target(&ort_lib_dir, use_cuda);

    println!(
        "cargo:rustc-env=ORT_DYLIB_PATH={}",
        ort_lib_dir.join("onnxruntime.dll").display()
    );
    println!("cargo:warning=Using ORT variant: {} ({})", variant, ort_lib_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
}

// ---------------------------------------------------------------------------
// WinML (DirectML) download
// ---------------------------------------------------------------------------

fn download_ort_winml(manifest_dir: &Path, folder: &str) {
    let nupkg_url = format!(
        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/flat2/{pkg}/{ver}/{pkg}.{ver}.nupkg",
        pkg = ORT_WINML_PACKAGE,
        ver = ORT_WINML_VERSION,
    );
    let zip_path = manifest_dir.join(format!("{}.zip", folder));
    let ort_dir = manifest_dir.join(folder);

    curl_download(&nupkg_url, &zip_path);
    extract_zip(&zip_path, &ort_dir);
    let _ = fs::remove_file(&zip_path);

    println!("cargo:warning=ORT WinML extracted to {}", ort_dir.display());
}

// ---------------------------------------------------------------------------
// CUDA download
// ---------------------------------------------------------------------------

fn download_ort_cuda(manifest_dir: &Path, folder: &str) {
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ver}/onnxruntime-win-x64-gpu-{ver}.zip",
        ver = ORT_CUDA_VERSION,
    );
    let zip_path = manifest_dir.join(format!("{}.zip", folder));

    curl_download(&url, &zip_path);
    extract_zip(&zip_path, manifest_dir);
    let _ = fs::remove_file(&zip_path);

    let ort_dir = manifest_dir.join(folder);
    println!("cargo:warning=ORT CUDA extracted to {}", ort_dir.display());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn curl_download(url: &str, dest: &Path) {
    let status = Command::new("curl")
        .args(["-L", "-o", dest.to_str().unwrap(), url])
        .status()
        .expect("Failed to run curl");
    assert!(status.success(), "curl download failed for {}", url);
}

fn extract_zip(zip_path: &Path, dest_dir: &Path) {
    let _ = fs::create_dir_all(dest_dir);
    let status = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                zip_path.display(),
                dest_dir.display()
            ),
        ])
        .status()
        .expect("Failed to run PowerShell Expand-Archive");
    assert!(status.success(), "Expand-Archive failed");
}

fn copy_dlls_to_target(ort_lib_dir: &Path, use_cuda: bool) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("Could not determine target directory");

    let mut dlls: Vec<&str> = vec!["onnxruntime.dll"];
    if use_cuda {
        dlls.push("onnxruntime_providers_shared.dll");
        dlls.push("onnxruntime_providers_cuda.dll");
        dlls.push("onnxruntime_providers_tensorrt.dll");
    }

    for dll_name in &dlls {
        let src = ort_lib_dir.join(dll_name);
        if src.exists() {
            let dst = target_dir.join(dll_name);
            match fs::copy(&src, &dst) {
                Ok(_) => println!("cargo:warning=Copied {} to {}", dll_name, dst.display()),
                Err(e) => println!("cargo:warning=Failed to copy {}: {}", dll_name, e),
            }
        } else {
            println!("cargo:warning={} not found in {}", dll_name, ort_lib_dir.display());
        }
    }
}
