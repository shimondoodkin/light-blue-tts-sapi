//! Style JSON loading.
//!
//! The voice JSON files (e.g. male1.json) store tensor data as nested arrays
//! (e.g. `[[[0.1, 0.2, ...]]]`).  We flatten them to `Vec<f32>` and record
//! the dimensions.

use serde::Deserialize;
use std::path::Path;

/// A single tensor entry in the style JSON.
#[derive(Debug, Clone)]
pub struct StyleTensor {
    pub data: Vec<f32>,
    pub dims: Vec<usize>,
}

/// Style JSON file structure.
#[derive(Debug, Clone)]
pub struct StyleJson {
    pub style_ttl: StyleTensor,
    pub style_keys: StyleTensor,
    pub style_dp: StyleTensor,
}

impl StyleJson {
    /// Load a style JSON file from disk.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let data = std::fs::read_to_string(path)?;
        let raw: RawStyleJson = serde_json::from_str(&data)?;
        Ok(Self {
            style_ttl: parse_tensor(&raw.style_ttl)?,
            style_keys: parse_tensor(&raw.style_keys)?,
            style_dp: parse_tensor(&raw.style_dp)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal: parse arbitrarily-nested JSON arrays into flat f32 + dims
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawStyleJson {
    style_ttl: RawTensor,
    style_keys: RawTensor,
    style_dp: RawTensor,
}

#[derive(Deserialize)]
struct RawTensor {
    data: serde_json::Value,
    #[serde(default)]
    dims: Option<Vec<usize>>,
}

fn parse_tensor(raw: &RawTensor) -> Result<StyleTensor, Box<dyn std::error::Error + Send + Sync>> {
    let mut flat = Vec::new();
    let mut dims = Vec::new();
    flatten_value(&raw.data, &mut flat, &mut dims, 0);
    // If the file already had explicit dims, prefer those.
    if let Some(ref d) = raw.dims {
        dims = d.clone();
    }
    Ok(StyleTensor { data: flat, dims })
}

/// Recursively flatten nested JSON arrays, recording the size at each depth.
fn flatten_value(v: &serde_json::Value, out: &mut Vec<f32>, dims: &mut Vec<usize>, depth: usize) {
    match v {
        serde_json::Value::Array(arr) => {
            // Record this dimension's size (only on the first element at this depth).
            if dims.len() <= depth {
                dims.push(arr.len());
            }
            for item in arr {
                flatten_value(item, out, dims, depth + 1);
            }
        }
        serde_json::Value::Number(n) => {
            out.push(n.as_f64().unwrap_or(0.0) as f32);
        }
        _ => {}
    }
}
