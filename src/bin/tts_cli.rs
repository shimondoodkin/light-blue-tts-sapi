//! CLI tool for LightBlue Hebrew TTS.
//!
//! Usage:
//!   lightblue-tts <output.wav> <text>                       — synthesize speech
//!   lightblue-tts register --name "Name" --style file.json  — register custom voice
//!   lightblue-tts unregister --name "Name"                  — unregister custom voice
//!   lightblue-tts list                                      — list registered voices

use std::path::PathBuf;
use std::sync::Arc;

use lightblue_sapi::bridge::LightBlueSynthesizer;
use lightblue_sapi::tts::TTSConfig;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "register" => cmd_register(&args),
        "unregister" => cmd_unregister(&args),
        "list" => cmd_list(),
        _ => cmd_synthesize(&args),
    }
}

fn print_usage(prog: &str) {
    eprintln!("LightBlue Hebrew TTS — CLI tool\n");
    eprintln!("Usage:");
    eprintln!("  {prog} <output.wav> [--style file.json] <hebrew text>   Synthesize speech");
    eprintln!("  {prog} register [options]                  Register a custom SAPI voice");
    eprintln!("    --name  \"Display Name\"                   Voice name (required)");
    eprintln!("    --style path/to/style.json               Style JSON file (required)");
    eprintln!("    --gender Male|Female                     Voice gender (default: Male)");
    eprintln!("  {prog} unregister --name \"Display Name\"    Remove a custom SAPI voice");
    eprintln!("  {prog} list                                List registered LightBlue voices");
}

// ---------------------------------------------------------------------------
// Synthesize command
// ---------------------------------------------------------------------------

fn cmd_synthesize(args: &[String]) {
    if args.len() < 3 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let output_path = &args[1];

    // Parse optional --style before collecting remaining text
    let mut style_override: Option<String> = None;
    let mut text_args: Vec<&str> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--style" {
            i += 1;
            style_override = args.get(i).cloned();
        } else {
            text_args.push(&args[i]);
        }
        i += 1;
    }
    let text = text_args.join(" ");
    if text.is_empty() {
        eprintln!("Error: no text provided");
        std::process::exit(1);
    }

    let models_dir = find_models_dir();

    let phonikud_model = models_dir.join("phonikud.onnx");
    let phonikud_tokenizer = models_dir.join("tokenizer.json");
    let style_json = models_dir.join("style.json");

    for (name, path) in [
        ("models dir", models_dir.as_path()),
        ("phonikud.onnx", phonikud_model.as_path()),
        ("tokenizer.json", phonikud_tokenizer.as_path()),
    ] {
        if !path.exists() {
            eprintln!("Error: {} not found at {}", name, path.display());
            std::process::exit(1);
        }
    }

    let style_json_path = if style_json.exists() {
        Some(style_json.to_string_lossy().into_owned())
    } else {
        None
    };

    let mut tts_config = TTSConfig::new(&models_dir);
    tts_config.default_style_json = style_json_path.as_ref().map(PathBuf::from);

    eprintln!("Loading models from: {}", models_dir.display());
    let t_load = std::time::Instant::now();

    let synth = LightBlueSynthesizer::new(
        tts_config,
        &phonikud_model.to_string_lossy(),
        &phonikud_tokenizer.to_string_lossy(),
        style_json_path,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error initializing synthesizer: {e}");
        std::process::exit(1);
    });
    eprintln!("Models loaded in {:.2?}", t_load.elapsed());

    let synth: Arc<dyn lightblue_sapi::sapi::TtsSynthesizer> = Arc::new(synth);

    eprintln!("Synthesizing: \"{}\"", text);
    let t_synth = std::time::Instant::now();

    let output = synth.synthesize(&text, style_override.as_deref(), None, false).unwrap_or_else(|e| {
        eprintln!("Synthesis error: {e}");
        std::process::exit(1);
    });
    let samples = output.samples;
    let sample_rate = output.sample_rate;

    eprintln!("Synthesis took {:.2?}", t_synth.elapsed());
    eprintln!(
        "Generated {} samples ({:.2}s) at {} Hz",
        samples.len(),
        samples.len() as f64 / sample_rate as f64,
        sample_rate
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec).unwrap_or_else(|e| {
        eprintln!("Error creating WAV file: {e}");
        std::process::exit(1);
    });
    for &s in &samples {
        let val = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(val).unwrap();
    }
    writer.finalize().unwrap();

    eprintln!("Written to: {}", output_path);
}

// ---------------------------------------------------------------------------
// Register command
// ---------------------------------------------------------------------------

fn cmd_register(args: &[String]) {
    let mut name: Option<String> = None;
    let mut style: Option<String> = None;
    let mut gender = "Male".to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                i += 1;
                name = args.get(i).cloned();
            }
            "--style" => {
                i += 1;
                style = args.get(i).cloned();
            }
            "--gender" => {
                i += 1;
                gender = args.get(i).cloned().unwrap_or_else(|| "Male".into());
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let name = name.unwrap_or_else(|| {
        eprintln!("Error: --name is required");
        std::process::exit(1);
    });
    let style_path = style.unwrap_or_else(|| {
        eprintln!("Error: --style is required");
        std::process::exit(1);
    });

    let style_src = PathBuf::from(&style_path);
    if !style_src.exists() {
        eprintln!("Error: style file not found: {}", style_src.display());
        std::process::exit(1);
    }

    // Copy style JSON into models directory
    let models_dir = find_models_dir();
    let token_name = sanitize_token_name(&name);
    let style_filename = format!("style_{}.json", token_name.to_lowercase());
    let style_dest = models_dir.join(&style_filename);
    std::fs::copy(&style_src, &style_dest).unwrap_or_else(|e| {
        eprintln!("Error copying style file: {e}");
        std::process::exit(1);
    });
    eprintln!("Copied style to: {}", style_dest.display());

    // Write registry entries
    unsafe {
        let clsid_braced = format!("{{{}}}", lightblue_sapi::sapi::ENGINE_CLSID_STR);
        let token_subkey = format!(
            "SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\{}",
            token_name
        );
        let attr_subkey = format!("{}\\Attributes", token_subkey);

        if reg_create_and_set(&token_subkey, &[
            (None, &name),
            (Some("CLSID"), &clsid_braced),
            (Some("411"), ""),
            (Some("StyleFile"), &style_filename),
        ]).is_err() {
            eprintln!("Error: Failed to write registry. Run as Administrator.");
            std::process::exit(1);
        }

        if reg_create_and_set(&attr_subkey, &[
            (Some("Language"), "40D"),
            (Some("Gender"), &gender),
            (Some("Name"), &name),
            (Some("Vendor"), "LightBlue"),
        ]).is_err() {
            eprintln!("Error: Failed to write registry attributes.");
            std::process::exit(1);
        }
    }

    eprintln!("Registered SAPI voice: \"{}\" (token: {})", name, token_name);
    eprintln!("The voice will appear in Windows Speech settings.");
}

// ---------------------------------------------------------------------------
// Unregister command
// ---------------------------------------------------------------------------

fn cmd_unregister(args: &[String]) {
    let mut name: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--name" {
            i += 1;
            name = args.get(i).cloned();
        }
        i += 1;
    }

    let name = name.unwrap_or_else(|| {
        eprintln!("Error: --name is required");
        std::process::exit(1);
    });

    let token_name = sanitize_token_name(&name);
    let token_subkey = format!(
        "SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\{}",
        token_name
    );

    unsafe {
        use windows::Win32::System::Registry::{RegDeleteTreeW, HKEY_LOCAL_MACHINE};
        let subkey_w: Vec<u16> = token_subkey.encode_utf16().chain(std::iter::once(0)).collect();
        let status = RegDeleteTreeW(HKEY_LOCAL_MACHINE, windows_core::PCWSTR(subkey_w.as_ptr()));
        if status.is_err() {
            eprintln!("Warning: Could not delete registry key (may not exist or need admin).");
        }
    }

    // Try to remove the style file
    let models_dir = find_models_dir();
    let style_filename = format!("style_{}.json", token_name.to_lowercase());
    let style_path = models_dir.join(&style_filename);
    if style_path.exists() {
        let _ = std::fs::remove_file(&style_path);
    }

    eprintln!("Unregistered voice: \"{}\"", name);
}

// ---------------------------------------------------------------------------
// List command
// ---------------------------------------------------------------------------

fn cmd_list() {
    unsafe {
        use windows::Win32::System::Registry::{
            RegOpenKeyExW, RegEnumKeyExW, RegQueryValueExW, RegCloseKey,
            HKEY, HKEY_LOCAL_MACHINE, KEY_READ, REG_SZ,
        };

        let base = "SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens";
        let base_w: Vec<u16> = base.encode_utf16().chain(std::iter::once(0)).collect();
        let mut hkey = HKEY::default();
        let status = RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            windows_core::PCWSTR(base_w.as_ptr()),
            0,
            KEY_READ,
            &mut hkey,
        );
        if status.is_err() {
            eprintln!("Cannot read voice registry.");
            return;
        }

        let clsid_expected = format!("{{{}}}", lightblue_sapi::sapi::ENGINE_CLSID_STR);
        let mut index = 0u32;
        loop {
            let mut name_buf = vec![0u16; 256];
            let mut name_len = name_buf.len() as u32;
            let status = RegEnumKeyExW(
                hkey,
                index,
                windows_core::PWSTR(name_buf.as_mut_ptr()),
                &mut name_len,
                None,
                windows_core::PWSTR::null(),
                None,
                None,
            );
            if status.is_err() {
                break;
            }
            index += 1;

            let token_name = String::from_utf16_lossy(&name_buf[..name_len as usize]);

            // Open this token and check CLSID
            let sub = format!("{}\\{}", base, token_name);
            let sub_w: Vec<u16> = sub.encode_utf16().chain(std::iter::once(0)).collect();
            let mut sub_hkey = HKEY::default();
            if RegOpenKeyExW(HKEY_LOCAL_MACHINE, windows_core::PCWSTR(sub_w.as_ptr()), 0, KEY_READ, &mut sub_hkey).is_err() {
                continue;
            }

            let clsid = reg_read_string(sub_hkey, "CLSID").unwrap_or_default();
            if clsid.eq_ignore_ascii_case(&clsid_expected) {
                let display = reg_read_string(sub_hkey, "").unwrap_or_else(|| token_name.clone());
                let style = reg_read_string(sub_hkey, "StyleFile").unwrap_or_else(|| "-".into());

                // Read gender from Attributes subkey
                let attr_sub = format!("{}\\Attributes", sub);
                let attr_w: Vec<u16> = attr_sub.encode_utf16().chain(std::iter::once(0)).collect();
                let mut attr_hkey = HKEY::default();
                let gender = if RegOpenKeyExW(HKEY_LOCAL_MACHINE, windows_core::PCWSTR(attr_w.as_ptr()), 0, KEY_READ, &mut attr_hkey).is_ok() {
                    let g = reg_read_string(attr_hkey, "Gender").unwrap_or_else(|| "?".into());
                    let _ = RegCloseKey(attr_hkey);
                    g
                } else {
                    "?".into()
                };

                println!("  {} [{}] style={}", display, gender, style);
            }

            let _ = RegCloseKey(sub_hkey);
        }
        let _ = RegCloseKey(hkey);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_models_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    if exe_dir.join("models").exists() {
        return exe_dir.join("models");
    }
    // Fallback: repo layout (target/release -> repo root)
    let repo_root = exe_dir.parent().and_then(|p| p.parent());
    match repo_root {
        Some(root) if root.join("models").exists() => root.join("models"),
        _ => {
            eprintln!("Error: Cannot find 'models' directory.");
            eprintln!("Looked in: {}", exe_dir.display());
            std::process::exit(1);
        }
    }
}

/// Convert a display name to a safe registry token name.
fn sanitize_token_name(name: &str) -> String {
    let mut token: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    // Prefix to avoid collisions
    if !token.starts_with("LightBlue") {
        token = format!("LightBlue_{token}");
    }
    token
}

unsafe fn reg_create_and_set(
    subkey: &str,
    values: &[(Option<&str>, &str)],
) -> Result<(), ()> {
    use windows::Win32::System::Registry::{
        RegCreateKeyExW, RegSetValueExW, RegCloseKey, HKEY, HKEY_LOCAL_MACHINE,
        KEY_WRITE, REG_OPTION_NON_VOLATILE, REG_SZ,
    };

    let subkey_w: Vec<u16> = subkey.encode_utf16().chain(std::iter::once(0)).collect();
    let mut hkey = HKEY::default();
    let status = RegCreateKeyExW(
        HKEY_LOCAL_MACHINE,
        windows_core::PCWSTR(subkey_w.as_ptr()),
        0,
        windows_core::PCWSTR::null(),
        REG_OPTION_NON_VOLATILE,
        KEY_WRITE,
        None,
        &mut hkey,
        None,
    );
    if status.is_err() {
        return Err(());
    }

    for (name, value) in values {
        let value_w: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
        let name_ptr = match name {
            Some(n) => {
                let nw: Vec<u16> = n.encode_utf16().chain(std::iter::once(0)).collect();
                // Need to keep nw alive — use a local scope trick
                let status = RegSetValueExW(
                    hkey,
                    windows_core::PCWSTR(nw.as_ptr()),
                    0,
                    REG_SZ,
                    Some(std::slice::from_raw_parts(
                        value_w.as_ptr() as *const u8,
                        value_w.len() * 2,
                    )),
                );
                if status.is_err() {
                    let _ = RegCloseKey(hkey);
                    return Err(());
                }
                continue;
            }
            None => windows_core::PCWSTR::null(),
        };
        let status = RegSetValueExW(
            hkey,
            name_ptr,
            0,
            REG_SZ,
            Some(std::slice::from_raw_parts(
                value_w.as_ptr() as *const u8,
                value_w.len() * 2,
            )),
        );
        if status.is_err() {
            let _ = RegCloseKey(hkey);
            return Err(());
        }
    }

    let _ = RegCloseKey(hkey);
    Ok(())
}

unsafe fn reg_read_string(hkey: windows::Win32::System::Registry::HKEY, name: &str) -> Option<String> {
    use windows::Win32::System::Registry::{RegQueryValueExW, REG_SZ};

    let name_w: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let mut buf = vec![0u8; 1024];
    let mut buf_len = buf.len() as u32;
    let mut reg_type = REG_SZ;
    let status = RegQueryValueExW(
        hkey,
        windows_core::PCWSTR(name_w.as_ptr()),
        None,
        Some(&mut reg_type),
        Some(buf.as_mut_ptr()),
        Some(&mut buf_len),
    );
    if status.is_err() || buf_len == 0 {
        return None;
    }
    let wide = std::slice::from_raw_parts(buf.as_ptr() as *const u16, (buf_len as usize) / 2);
    let end = wide.iter().position(|&c| c == 0).unwrap_or(wide.len());
    Some(String::from_utf16_lossy(&wide[..end]))
}
