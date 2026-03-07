# Backbone ONNX Model Surgery — Full GPU Execution

## Problem

When running `backbone_keys.onnx` with CUDA ExecutionProvider, ORT assigned 10 out
of 1253 nodes to CPUExecutionProvider. These were 5 chains of `Shape -> Gather -> Unsqueeze`
(15 nodes total, but ORT merges some) that extract dynamic tensor dimensions at runtime.

The `Shape` op is fundamentally CPU-only in ORT's CUDA EP — it queries tensor metadata
which lives in CPU memory. ORT's `fallback_cpu_capability.cc` intentionally places these
small ops on CPU, but this causes CPU<->GPU memory transfers that hurt performance.

## Root Cause

The model uses positional encoding with a pattern like:

```
arange(1000)[:seq_len]
```

Implemented in ONNX as:

```
Slice(const_data, start=0, end=Shape(tensor)[dim])
```

The 5 chains extracting dynamic dimensions:

| Chain | Source tensor | Gather index | Extracted dim | Output node |
|-------|-------------|-------------|---------------|-------------|
| 1 | `2Xx9zOUsUKuS` (shape `[1, T_text, 256]`) | 1 | T_text | `FJ7hKuMXo8J8` |
| 2 | `EFfUkCI5gisH` (shape `[1, 4, T_lat, 32]`) | 2 | T_lat | `EgUrDmpWH2JI` |
| 3 | `bf00fDA2R3tu` (shape `[1, 4, T_lat, 32]`) | 2 | T_lat | `Tkb5Fe560ZlY` |
| 4 | `wJi29v2dBOtT` (shape `[1, 4, T_lat, 32]`) | 2 | T_lat | `1nUPdBvudfuN` |
| 5 | `jgRlaVZqTpG6` (shape `[1, 4, T_lat, 32]`) | 2 | T_lat | `GvQhfOlBoGJk` |

## Solution

Two-step approach:

1. **Remove** the 15 CPU-only `Shape -> Gather -> Unsqueeze` nodes
2. **Derive** the dimension values inside the model from the mask inputs using
   GPU-compatible ops: `ReduceSum -> Cast -> Reshape`

The masks (`latent_mask` shape `[1,1,T_lat]` and `text_mask` shape `[1,1,T_text]`)
are all-ones tensors already passed to the model. Summing all elements gives the
dimension value. `ReduceSum`, `Cast`, and `Reshape` all run on CUDA EP.

No extra model inputs are needed — the model's interface is the same as the
original (8 inputs), minus the CPU-only ops.

### Surgery Script

See `scripts/backbone_surgery.py` for the standalone executable version.

```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

model = onnx.load("backbone_keys_orig.onnx")

# Step 1: Rewire chain outputs to new internal names
t_lat_outputs  = ['EgUrDmpWH2JI', 'Tkb5Fe560ZlY', '1nUPdBvudfuN', 'GvQhfOlBoGJk']
t_text_outputs = ['FJ7hKuMXo8J8']

for node in model.graph.node:
    for i, inp in enumerate(node.input):
        if inp in t_lat_outputs:
            node.input[i] = 't_lat_dim'
        elif inp in t_text_outputs:
            node.input[i] = 't_text_dim'

# Step 2: Remove orphaned Shape/Gather/Unsqueeze nodes (iterative cleanup)
all_chain_outputs = set(t_lat_outputs + t_text_outputs)
nodes_to_remove = []
for node in model.graph.node:
    if node.op_type in ('Shape', 'Gather', 'Unsqueeze'):
        if any(o in all_chain_outputs for o in node.output):
            nodes_to_remove.append(node)
            continue
        for other in model.graph.node:
            if other in nodes_to_remove:
                if any(o in other.input for o in node.output):
                    nodes_to_remove.append(node)
                    break

for _ in range(3):
    removed_inputs = set()
    for n in nodes_to_remove:
        removed_inputs.update(n.input)
    for node in model.graph.node:
        if node not in nodes_to_remove and node.op_type in ('Shape', 'Gather'):
            if any(o in removed_inputs for o in node.output):
                nodes_to_remove.append(node)

for node in nodes_to_remove:
    if node in model.graph.node:
        model.graph.node.remove(node)

# Step 3: Add GPU-compatible nodes to derive dims from masks
#   latent_mask [1,1,T_lat] -> ReduceSum(all axes) -> Cast(int64) -> Reshape([1])
#   text_mask   [1,1,T_text] -> ReduceSum(all axes) -> Cast(int64) -> Reshape([1])

axes_012 = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name='_axes_012')
model.graph.initializer.append(axes_012)
shape_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name='_shape_1')
model.graph.initializer.append(shape_1)

nodes = [
    helper.make_node('ReduceSum', ['latent_mask', '_axes_012'], ['_t_lat_f32'],
                     name='_reduce_lat', keepdims=0),
    helper.make_node('Cast', ['_t_lat_f32'], ['_t_lat_i64'],
                     name='_cast_lat', to=TensorProto.INT64),
    helper.make_node('Reshape', ['_t_lat_i64', '_shape_1'], ['t_lat_dim'],
                     name='_reshape_lat'),
    helper.make_node('ReduceSum', ['text_mask', '_axes_012'], ['_t_text_f32'],
                     name='_reduce_text', keepdims=0),
    helper.make_node('Cast', ['_t_text_f32'], ['_t_text_i64'],
                     name='_cast_text', to=TensorProto.INT64),
    helper.make_node('Reshape', ['_t_text_i64', '_shape_1'], ['t_text_dim'],
                     name='_reshape_text'),
]
for node in reversed(nodes):
    model.graph.node.insert(0, node)

onnx.checker.check_model(model)
onnx.save(model, "backbone_keys.onnx")
```

### Key Detail: ReduceSum Axes

The masks have shape `[1, 1, T]`. `ReduceSum` with `axes=[0,1,2]` and `keepdims=0`
produces a scalar (shape `()`), which `Reshape([1])` converts to shape `[1]`.

Using only `axes=[0,1]` would produce shape `(T,)` — a vector of T ones, not a
scalar — which cannot be reshaped to `[1]`.

## Result

- **Before:** 1253 nodes, 10 on CPUExecutionProvider
- **After:** 1244 nodes, ALL on CUDAExecutionProvider
- **No extra model inputs** — original 8-input interface preserved
- **Verification:** Bit-for-bit identical outputs (`max_diff = 0.00e+00`) across
  3 different input sizes

## Files

| File | Description |
|------|-------------|
| `models/onnx/backbone_keys.onnx` | Final fixed model (v2, active in production) |
| `models/onnx/backbone_keys_orig.onnx` | Original unmodified model (backup) |
| `models/onnx/backbone_keys_v1.onnx` | v1: used extra model inputs (superseded) |
| `models/onnx/backbone_keys_v2.onnx` | Copy of current production model |
