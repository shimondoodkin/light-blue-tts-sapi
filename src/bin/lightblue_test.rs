//! Benchmark tool — runs multiple TTS syntheses to measure warmup vs steady-state.

use std::sync::Arc;
use std::time::Instant;

use lightblue_sapi::bridge::LightBlueSynthesizer;
use lightblue_sapi::sapi::TtsSynthesizer;
use lightblue_sapi::tts::TTSConfig;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    let phrases: &[&str] = &[
        // Short
        "שלום",
        "מה שלומך",
        // Medium
        "היום יום יפה מאוד והשמש זורחת בחוץ",
        // Long
        "בוקר טוב לכולם, היום אנחנו הולכים לדבר על הטכנולוגיה החדשה שמשנה את העולם שלנו",
        // Very long
        "החינוך הוא הכלי החשוב ביותר שיש לנו כדי לשנות את העולם. כל ילד וילדה ראויים לקבל חינוך איכותי שיאפשר להם לממש את הפוטנציאל שלהם ולתרום לחברה",
        // Repeat short to show cached performance
        "שלום",
        "מה שלומך",
    ];

    let models_dir = find_models_dir();
    let onnx_dir = models_dir.join("onnx");
    let config_json = models_dir.join("tts.json");
    let phonikud_model = models_dir.join("phonikud.onnx");
    let phonikud_tokenizer = models_dir.join("tokenizer.json");
    let style_json = models_dir.join("style.json");

    let style_json_path = if style_json.exists() {
        Some(style_json.to_string_lossy().into_owned())
    } else {
        None
    };

    // Parse optional CLI args: --steps N --cfg F
    let args: Vec<String> = std::env::args().collect();
    let mut steps: Option<u32> = None;
    let mut cfg_scale: Option<f32> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { i += 1; steps = args.get(i).and_then(|s| s.parse().ok()); }
            "--cfg" => { i += 1; cfg_scale = args.get(i).and_then(|s| s.parse().ok()); }
            _ => {}
        }
        i += 1;
    }

    let mut tts_config = TTSConfig::new(&onnx_dir, &config_json);
    tts_config.default_style_json = style_json_path.as_ref().map(PathBuf::from);
    if let Some(s) = steps { tts_config.steps = s; }
    if let Some(c) = cfg_scale { tts_config.cfg_scale = c; }

    eprintln!("=== Config: steps={}, cfg_scale={:.1} ===", tts_config.steps, tts_config.cfg_scale);
    let t_load = Instant::now();

    let synth = LightBlueSynthesizer::new(
        tts_config,
        &phonikud_model.to_string_lossy(),
        &phonikud_tokenizer.to_string_lossy(),
        style_json_path,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    eprintln!("Models loaded in {:.2?}\n", t_load.elapsed());

    let synth: Arc<dyn TtsSynthesizer> = Arc::new(synth);

    for (i, phrase) in phrases.iter().enumerate() {
        eprintln!("=== Run {} — \"{}\" ===", i + 1, phrase);
        let t = Instant::now();

        let (samples, sr) = synth.synthesize(phrase, None, None, false).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        });

        let duration = samples.len() as f64 / sr as f64;
        let elapsed = t.elapsed();
        eprintln!(
            "  synth: {:.2?}  audio: {:.2}s  RTF: {:.2}x\n",
            elapsed,
            duration,
            elapsed.as_secs_f64() / duration,
        );
    }
}

fn find_models_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    if exe_dir.join("models").exists() {
        return exe_dir.join("models");
    }
    let repo_root = exe_dir.parent().and_then(|p| p.parent());
    match repo_root {
        Some(root) if root.join("models").exists() => root.join("models"),
        _ => {
            eprintln!("Error: Cannot find 'models' directory.");
            std::process::exit(1);
        }
    }
}
