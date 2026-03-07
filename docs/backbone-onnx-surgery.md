# Backbone ONNX Surgery

`backbone_keys.onnx` has `Shape -> Gather -> Unsqueeze` chains that ORT places on CPU, causing CPU<->GPU transfers. The surgery replaces them with GPU-compatible ops.

## How It Works

The model extracts sequence lengths via `Shape(tensor)[dim]` for positional encoding slicing (`arange(1000)[:seq_len]`). The script:

1. Finds Shape->Gather->Unsqueeze chains whose output feeds a Slice node
2. Classifies each as T_lat or T_text by tracing back to `latent_mask` or `text_emb`
3. Replaces them with `ReduceSum -> Cast -> Reshape` on the mask inputs (all-ones tensors where sum = dimension size)

All replacement ops run on CUDA EP. Model interface is unchanged (same 8 inputs).

## Usage

```bash
python scripts/backbone_surgery.py
python scripts/simplify_models_in_place.py
```
