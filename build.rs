use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// CUDA-enabled ORT from official GitHub releases.
const ORT_CUDA_VERSION: &str = "1.23.2";

/// Intel OpenVINO ORT NuGet package.
const ORT_OPENVINO_PACKAGE: &str = "intel.ml.onnxruntime.openvino";

/// NuGet package for DirectML-enabled ORT.
const NUGET_ORT_DIRECTML_PACKAGE: &str = "microsoft.ml.onnxruntime.directml";

/// NuGet package for DirectML redistributable DLL.
const NUGET_DIRECTML_PACKAGE: &str = "microsoft.ai.directml";

/// NuGet v3 flat-container base URL for version discovery.
const NUGET_FLAT_BASE: &str = "https://api.nuget.org/v3-flatcontainer";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let use_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let use_openvino = env::var("CARGO_FEATURE_OPENVINO").is_ok();

    let (ort_lib_dir, variant) = if use_cuda {
        let folder = format!("onnxruntime-win-x64-gpu-{}", ORT_CUDA_VERSION);
        let ort_dir = manifest_dir.join(&folder);
        let lib_dir = ort_dir.join("lib");

        if !lib_dir.exists() {
            println!("cargo:warning=Downloading ORT CUDA v{}...", ORT_CUDA_VERSION);
            download_ort_cuda(&manifest_dir, &folder);
        }

        (lib_dir, "cuda")
    } else if use_openvino {
        let (lib_dir, version) = download_or_reuse_openvino(&manifest_dir);
        println!("cargo:warning=Using ORT OpenVINO v{}", version);
        (lib_dir, "openvino")
    } else {
        let (lib_dir, version) = download_or_reuse_directml(&manifest_dir);
        println!("cargo:warning=Using ORT DirectML v{}", version);
        (lib_dir, "directml")
    };

    copy_dlls_to_target(&ort_lib_dir, use_cuda, use_openvino);

    println!(
        "cargo:rustc-env=ORT_DYLIB_PATH={}",
        ort_lib_dir.join("onnxruntime.dll").display()
    );
    println!("cargo:warning=Using ORT variant: {} ({})", variant, ort_lib_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENVINO");
}

// ---------------------------------------------------------------------------
// DirectML download (default)
//   - onnxruntime.dll (with DirectML EP) from Microsoft.ML.OnnxRuntime.DirectML NuGet
//   - DirectML.dll from Microsoft.AI.DirectML NuGet
// ---------------------------------------------------------------------------

/// Download or reuse ORT DirectML from NuGet packages.
///
/// Sources:
///   - `Microsoft.ML.OnnxRuntime.DirectML` NuGet → onnxruntime.dll with DirectML EP compiled in
///   - `Microsoft.AI.DirectML` NuGet → DirectML.dll redistributable
///
/// The previous approach downloaded the CPU-only onnxruntime.dll from GitHub releases
/// and placed DirectML.dll next to it, but that build doesn't have the DirectML
/// execution provider — DirectML would never actually activate.
fn download_or_reuse_directml(manifest_dir: &Path) -> (PathBuf, String) {
    let version = nuget_latest_version(NUGET_ORT_DIRECTML_PACKAGE);
    let folder = format!("ort-directml-{}", version);
    let lib_dir = manifest_dir.join(&folder).join("lib");

    if !lib_dir.exists() {
        println!("cargo:warning=Downloading ORT DirectML v{} from NuGet...", version);
        let _ = fs::create_dir_all(&lib_dir);

        // 1. Download ORT DirectML NuGet package
        let ort_nupkg_url = format!(
            "https://globalcdn.nuget.org/packages/{pkg}.{ver}.nupkg?packageVersion={ver}",
            pkg = NUGET_ORT_DIRECTML_PACKAGE,
            ver = version,
        );
        let ort_dir = manifest_dir.join(format!("_tmp_ort_directml_{}", version));
        let ort_zip = manifest_dir.join(format!("_tmp_ort_directml_{}.zip", version));
        curl_download(&ort_nupkg_url, &ort_zip);
        extract_zip(&ort_zip, &ort_dir);
        let _ = fs::remove_file(&ort_zip);

        // Copy onnxruntime.dll and onnxruntime_providers_shared.dll from NuGet
        let native_dir = ort_dir
            .join("runtimes")
            .join("win-x64")
            .join("native");
        for dll in ["onnxruntime.dll", "onnxruntime_providers_shared.dll"] {
            let src = native_dir.join(dll);
            if src.exists() {
                fs::copy(&src, lib_dir.join(dll)).expect(&format!("Failed to copy {}", dll));
                println!("cargo:warning=Copied {} from NuGet ORT DirectML", dll);
            }
        }
        // Also copy the .lib if present (for linking)
        let lib_file = native_dir.join("onnxruntime.lib");
        if lib_file.exists() {
            let _ = fs::copy(&lib_file, lib_dir.join("onnxruntime.lib"));
        }
        let _ = fs::remove_dir_all(&ort_dir);

        // 2. Download DirectML.dll from Microsoft.AI.DirectML NuGet
        let dml_version = nuget_latest_version(NUGET_DIRECTML_PACKAGE);
        let dml_nupkg_url = format!(
            "https://globalcdn.nuget.org/packages/{pkg}.{ver}.nupkg?packageVersion={ver}",
            pkg = NUGET_DIRECTML_PACKAGE,
            ver = dml_version,
        );
        let dml_dir = manifest_dir.join(format!("_tmp_directml_{}", dml_version));
        let dml_zip = manifest_dir.join(format!("_tmp_directml_{}.zip", dml_version));
        curl_download(&dml_nupkg_url, &dml_zip);
        extract_zip(&dml_zip, &dml_dir);
        let _ = fs::remove_file(&dml_zip);

        // DirectML.dll is at bin/x64-win/DirectML.dll in the NuGet package
        let directml_src = dml_dir.join("bin").join("x64-win").join("DirectML.dll");
        if directml_src.exists() {
            fs::copy(&directml_src, lib_dir.join("DirectML.dll"))
                .expect("Failed to copy DirectML.dll");
            println!("cargo:warning=Copied DirectML.dll v{} from NuGet", dml_version);
        } else {
            // Fallback: search recursively
            if let Some(found) = find_file_recursive(&dml_dir, "DirectML.dll") {
                fs::copy(&found, lib_dir.join("DirectML.dll"))
                    .expect("Failed to copy DirectML.dll");
                println!("cargo:warning=Copied DirectML.dll from NuGet (fallback path)");
            } else {
                println!("cargo:warning=DirectML.dll not found in NuGet package!");
            }
        }
        let _ = fs::remove_dir_all(&dml_dir);

        println!("cargo:warning=ORT DirectML assembled in {}", lib_dir.display());
    }

    (lib_dir, version)
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
// OpenVINO download
// ---------------------------------------------------------------------------

/// Download or reuse the OpenVINO ORT NuGet package.
/// Returns (lib_dir, version_string).
fn download_or_reuse_openvino(manifest_dir: &Path) -> (PathBuf, String) {
    let version = nuget_latest_version(ORT_OPENVINO_PACKAGE);
    let folder = format!("ort-openvino-{}", version);
    let ort_dir = manifest_dir.join(&folder);
    let lib_dir = ort_dir.join("runtimes").join("win-x64").join("native");

    if !lib_dir.exists() {
        println!("cargo:warning=Downloading ORT OpenVINO v{}...", version);
        let nupkg_url = format!(
            "https://globalcdn.nuget.org/packages/{pkg}.{ver}.nupkg?packageVersion={ver}",
            pkg = ORT_OPENVINO_PACKAGE,
            ver = version,
        );
        let zip_path = manifest_dir.join(format!("{}.zip", folder));

        curl_download(&nupkg_url, &zip_path);
        extract_zip(&zip_path, &ort_dir);
        let _ = fs::remove_file(&zip_path);

        println!("cargo:warning=ORT OpenVINO extracted to {}", ort_dir.display());
    }

    (lib_dir, version)
}

// ---------------------------------------------------------------------------
// Version discovery helpers
// ---------------------------------------------------------------------------

/// Query the NuGet v3 flat-container index to find the latest version of a package.
fn nuget_latest_version(package: &str) -> String {
    let index_url = format!("{}/{}/index.json", NUGET_FLAT_BASE, package);
    let json_str = curl_fetch_string(&index_url);

    // Parse: {"versions":["1.20.0","1.21.0",...]}
    extract_json_string_array_last(&json_str, "versions")
        .expect("No versions found in NuGet index")
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

fn curl_fetch_string(url: &str) -> String {
    let output = Command::new("curl")
        .args(["-s", "-L", url])
        .output()
        .expect("Failed to run curl");
    assert!(output.status.success(), "Failed to fetch {}", url);
    String::from_utf8_lossy(&output.stdout).to_string()
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

/// Recursively find a file by name under a directory.
fn find_file_recursive(dir: &Path, name: &str) -> Option<PathBuf> {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|f| f.to_str()) == Some(name) {
                return Some(path);
            }
            if path.is_dir() {
                if let Some(found) = find_file_recursive(&path, name) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Extract the last string from a JSON string array field.
/// e.g. for `"versions":["a","b","c"]` with key "versions", returns Some("c").
fn extract_json_string_array_last(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    let start = json.find(&search)?;
    let bracket_end = json[start..].find(']')? + start;
    let section = &json[start..=bracket_end];

    let mut last = None;
    let mut i = 0;
    let bytes = section.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'"' {
            i += 1;
            let s_start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            let s = &section[s_start..i];
            if s != key && !s.is_empty() {
                last = Some(s.to_string());
            }
        }
        i += 1;
    }
    last
}

/// DLLs to ship for DirectML (default CPU build).
const DIRECTML_DLLS: &[&str] = &[
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "DirectML.dll",
];

/// DLLs to ship for OpenVINO (everything except debug DLLs, tbbmalloc, tbbbind,
/// and frontends we don't use — tensorflow, paddle, pytorch).
const OPENVINO_DLLS: &[&str] = &[
    "onnxruntime.dll",
    "onnxruntime_providers_openvino.dll",
    "onnxruntime_providers_shared.dll",
    "openvino.dll",
    "openvino_c.dll",
    "openvino_intel_cpu_plugin.dll",
    "openvino_intel_gpu_plugin.dll",
    "openvino_intel_npu_plugin.dll",
    "openvino_onnx_frontend.dll",
    "openvino_ir_frontend.dll",
    "openvino_auto_plugin.dll",
    "tbb12.dll",
];

fn copy_dlls_to_target(ort_lib_dir: &Path, use_cuda: bool, use_openvino: bool) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("Could not determine target directory");

    let dlls: Vec<&str> = if use_openvino {
        OPENVINO_DLLS.to_vec()
    } else if use_cuda {
        vec![
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_cuda.dll",
            "onnxruntime_providers_tensorrt.dll",
        ]
    } else {
        DIRECTML_DLLS.to_vec()
    };

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
