use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// CUDA-enabled ORT from official GitHub releases.
const ORT_CUDA_VERSION: &str = "1.23.2";

/// Intel OpenVINO ORT NuGet package.
const ORT_OPENVINO_PACKAGE: &str = "intel.ml.onnxruntime.openvino";

/// PyPI package name for DirectML DLL extraction.
const PYPI_DIRECTML_PACKAGE: &str = "onnxruntime-directml";

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
//   - onnxruntime.dll + onnxruntime_providers_shared.dll from GitHub releases
//   - DirectML.dll from PyPI onnxruntime-directml wheel
// ---------------------------------------------------------------------------

/// Get the latest ORT release version from GitHub API.
fn github_latest_ort_version() -> String {
    let json_str = curl_fetch_string("https://api.github.com/repos/microsoft/onnxruntime/releases/latest");

    // Extract "tag_name":"vX.Y.Z"
    let marker = "\"tag_name\":";
    let pos = json_str.find(marker).expect("No tag_name in GitHub API response");
    let after = &json_str[pos + marker.len()..];
    let after = after.trim_start();
    assert!(after.starts_with('"'), "Expected quoted tag_name");
    let start = 1;
    let end = after[start..].find('"').expect("Unterminated tag_name") + start;
    let tag = &after[start..end];
    // Strip leading 'v' if present
    tag.strip_prefix('v').unwrap_or(tag).to_string()
}

/// Download or reuse ORT CPU from GitHub + DirectML.dll from PyPI.
/// Returns (lib_dir, version_string).
fn download_or_reuse_directml(manifest_dir: &Path) -> (PathBuf, String) {
    let version = github_latest_ort_version();
    let folder = format!("ort-directml-{}", version);
    let lib_dir = manifest_dir.join(&folder).join("lib");

    if !lib_dir.exists() {
        println!("cargo:warning=Downloading ORT v{} from GitHub + DirectML from PyPI...", version);
        let _ = fs::create_dir_all(&lib_dir);

        // 1. Download ORT CPU from GitHub releases
        let cpu_folder = format!("onnxruntime-win-x64-{}", version);
        let cpu_dir = manifest_dir.join(&cpu_folder);
        let cpu_lib_dir = cpu_dir.join("lib");
        if !cpu_lib_dir.exists() {
            let url = format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{ver}/onnxruntime-win-x64-{ver}.zip",
                ver = version,
            );
            let zip_path = manifest_dir.join(format!("{}.zip", cpu_folder));
            curl_download(&url, &zip_path);
            extract_zip(&zip_path, manifest_dir);
            let _ = fs::remove_file(&zip_path);
        }

        // Copy onnxruntime.dll and onnxruntime_providers_shared.dll
        for dll in ["onnxruntime.dll", "onnxruntime_providers_shared.dll"] {
            let src = cpu_lib_dir.join(dll);
            if src.exists() {
                fs::copy(&src, lib_dir.join(dll)).expect(&format!("Failed to copy {}", dll));
            }
        }

        // 2. Download DirectML.dll from PyPI wheel
        let (_, wheel_url) = pypi_latest_wheel(PYPI_DIRECTML_PACKAGE);
        let wheel_dir = manifest_dir.join("_tmp_directml_wheel");
        let wheel_zip = manifest_dir.join("_tmp_directml.zip");
        curl_download(&wheel_url, &wheel_zip);
        extract_zip(&wheel_zip, &wheel_dir);
        let _ = fs::remove_file(&wheel_zip);

        // Find DirectML.dll in the extracted wheel (may be in different paths)
        if let Some(directml_path) = find_file_recursive(&wheel_dir, "DirectML.dll") {
            fs::copy(&directml_path, lib_dir.join("DirectML.dll"))
                .expect("Failed to copy DirectML.dll");
            println!("cargo:warning=Copied DirectML.dll from PyPI wheel");
        } else {
            println!("cargo:warning=DirectML.dll not found in PyPI wheel!");
        }
        let _ = fs::remove_dir_all(&wheel_dir);

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

/// Query the PyPI JSON API to find the latest version and a win_amd64 wheel URL.
/// Returns (version, wheel_download_url).
fn pypi_latest_wheel(package: &str) -> (String, String) {
    let api_url = format!("https://pypi.org/pypi/{}/json", package);
    let json_str = curl_fetch_string(&api_url);

    // Extract version from "version":"X.Y.Z"
    let version = {
        let marker = "\"version\":";
        let pos = json_str.find(marker).expect("No version field in PyPI JSON");
        let after = &json_str[pos + marker.len()..];
        let after = after.trim_start();
        assert!(after.starts_with('"'), "Expected quoted version string");
        let start = 1;
        let end = after[start..].find('"').expect("Unterminated version string") + start;
        after[start..end].to_string()
    };

    // Find a cp313+ win_amd64 wheel URL in the "urls" array.
    // We need cp313+ because older wheels (cp311/cp312) use a .data/purelib/ layout
    // that separates onnxruntime.dll into a pybind module, while cp313+ wheels have
    // a flat onnxruntime/capi/ layout with all native DLLs together.
    let wheel_url = {
        let mut url = None;
        let needles = ["cp313-cp313-win_amd64.whl", "cp314-cp314-win_amd64.whl", "win_amd64.whl"];
        'outer: for needle in &needles {
            let mut search_from = 0;
            while let Some(pos) = json_str[search_from..].find(needle) {
                let abs_pos = search_from + pos;
                let region = &json_str[..abs_pos];
                if let Some(url_key_pos) = region.rfind("\"url\"") {
                    let after_key = &json_str[url_key_pos + 5..];
                    let after_key = after_key.trim_start().trim_start_matches(':').trim_start();
                    if after_key.starts_with('"') {
                        let start = 1;
                        let end = after_key[start..].find('"').unwrap() + start;
                        url = Some(after_key[start..end].to_string());
                        break 'outer;
                    }
                }
                search_from = abs_pos + needle.len();
            }
        }
        url.expect("No win_amd64 wheel found in PyPI JSON")
    };

    (version, wheel_url)
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
