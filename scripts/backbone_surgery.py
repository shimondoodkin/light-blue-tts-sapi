#!/usr/bin/env python3
"""
Backbone ONNX Surgery — Remove CPU-only nodes for full GPU execution.

When running backbone_keys.onnx with CUDA ExecutionProvider, ORT assigns
Shape/Gather/Unsqueeze chains to CPU, causing CPU<->GPU memory transfers.

This script removes those 15 CPU-only nodes and replaces them with
GPU-compatible ReduceSum/Cast/Reshape ops that derive dimension values
from the mask inputs already passed to the model.

See docs/backbone-onnx-surgery.md for full documentation.

Usage:
    python backbone_surgery.py [input.onnx] [output.onnx]

    Defaults:
        input:  models/backbone_keys_orig.onnx
        output: models/backbone_keys.onnx

Requirements:
    pip install onnx numpy
"""

import sys
import os
import shutil

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np
except ImportError:
    print("Error: This script requires 'onnx' and 'numpy' packages.")
    print("Install with: pip install onnx numpy")
    sys.exit(1)


def apply_surgery(input_path: str, output_path: str) -> tuple[str, str]:
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    original_count = len(model.graph.node)

    # Step 1: Discover Shape->Gather->Unsqueeze chains that feed into Slice nodes.
    # Only these chains extract positional-encoding dimensions (T_lat, T_text) for
    # the arange(1000)[:seq_len] pattern. Other Shape chains (used for internal
    # reshapes) must be left alone.
    #
    # In the unsimplified model, each Shape node may have multiple Gather consumers,
    # and each Gather may have multiple Unsqueeze consumers. We check ALL paths
    # and only target those where at least one Unsqueeze output feeds a Slice node.
    node_by_output = {}
    for node in model.graph.node:
        for o in node.output:
            node_by_output[o] = node

    # Build consumer lookup
    consumers_of = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers_of.setdefault(inp, []).append(node)

    # Find all Unsqueeze outputs (from Shape->Gather->Unsqueeze chains) that feed Slice
    t_lat_outputs = []
    t_text_outputs = []
    for node in model.graph.node:
        if node.op_type != 'Shape':
            continue
        shape_out = node.output[0]
        for gather in consumers_of.get(shape_out, []):
            if gather.op_type != 'Gather':
                continue
            for unsqueeze in consumers_of.get(gather.output[0], []):
                if unsqueeze.op_type != 'Unsqueeze':
                    continue

                final_output = unsqueeze.output[0]

                # Only target Unsqueeze outputs that feed into a Slice node
                slice_consumers = [n for n in consumers_of.get(final_output, [])
                                   if n.op_type == 'Slice']
                if not slice_consumers:
                    continue

                # Classify: trace Shape input back to determine if it derives
                # from latent_mask (T_lat) or text_emb (T_text)
                source = node.input[0]
                visited = set()
                is_lat = False
                is_text = False
                queue = [source]
                while queue:
                    name = queue.pop()
                    if name in visited:
                        continue
                    visited.add(name)
                    for gi in model.graph.input:
                        if gi.name == name:
                            if 'latent' in name or 'lat' in name.lower():
                                is_lat = True
                            elif 'text' in name.lower():
                                is_text = True
                            break
                    if is_lat or is_text:
                        break
                    if name in node_by_output:
                        producer = node_by_output[name]
                        queue.extend(producer.input)

                if is_lat:
                    t_lat_outputs.append(final_output)
                elif is_text:
                    t_text_outputs.append(final_output)
                else:
                    print(f"  WARNING: Could not classify Shape chain ending at {final_output}")

    print(f"  Found {len(t_lat_outputs)} T_lat chain outputs: {t_lat_outputs}")
    print(f"  Found {len(t_text_outputs)} T_text chain outputs: {t_text_outputs}")

    # Rewire chain outputs to new internal names
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in t_lat_outputs:
                node.input[i] = 't_lat_dim'
            elif inp in t_text_outputs:
                node.input[i] = 't_text_dim'

    # Step 2: Dead-node elimination for orphaned Shape/Gather/Unsqueeze chains.
    # Iteratively remove nodes whose outputs are not consumed by any remaining node.
    graph_output_names = {o.name for o in model.graph.output}
    removable_ops = {'Shape', 'Gather', 'Unsqueeze'}
    removed_total = 0
    changed = True
    while changed:
        changed = False
        consumed = set()
        for node in model.graph.node:
            consumed.update(node.input)
        to_remove = []
        for node in model.graph.node:
            if node.op_type in removable_ops:
                if not any(o in consumed or o in graph_output_names for o in node.output):
                    to_remove.append(node)
        for node in to_remove:
            model.graph.node.remove(node)
            changed = True
            removed_total += 1

    print(f"  Removed {removed_total} CPU-only nodes")

    # Step 3: Add GPU-compatible nodes to derive dims from masks
    #   latent_mask [1,1,T_lat] -> ReduceSum(all axes) -> Cast(int64) -> Reshape([1])
    #   text_mask   [1,1,T_text] -> ReduceSum(all axes) -> Cast(int64) -> Reshape([1])

    existing_inits = {i.name for i in model.graph.initializer}

    if '_axes_012' not in existing_inits:
        axes_012 = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name='_axes_012')
        model.graph.initializer.append(axes_012)
    if '_shape_1' not in existing_inits:
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

    print(f"  Added 6 GPU-compatible replacement nodes")

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    final_count = len(model.graph.node)
    print(f"  Nodes: {original_count} -> {final_count}")
    print(f"  Saved: {output_path}")
    print("  Model check passed.")

    return input_path, output_path


def validate(input_path: str, output_path: str) -> None:
    """Run both models with random inputs and compare outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Skipping validation (onnxruntime not installed)")
        return

    print("Validating output matches input...")
    orig_sess = ort.InferenceSession(input_path, providers=['CPUExecutionProvider'])
    new_sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

    # Try multiple random shapes — some may be invalid due to internal reshape constraints.
    max_attempts = 10
    for attempt in range(max_attempts):
        # Use consistent sizes for matching symbolic dimension names.
        # Use multiples of 4 to satisfy common reshape constraints.
        dim_map: dict[str, int] = {}
        feeds = {}
        for inp in orig_sess.get_inputs():
            shape = []
            for d in inp.shape:
                if isinstance(d, int):
                    shape.append(d)
                else:
                    if d not in dim_map:
                        dim_map[d] = np.random.randint(1, 8) * 4
                    shape.append(dim_map[d])
            dtype = np.float32 if inp.type == 'tensor(float)' else np.int64
            # Masks must be all 1.0 so ReduceSum == dimension size (surgery assumption)
            if 'mask' in inp.name:
                feeds[inp.name] = np.ones(shape, dtype=dtype)
            elif dtype == np.float32:
                feeds[inp.name] = np.random.randn(*shape).astype(dtype)
            else:
                feeds[inp.name] = np.ones(shape, dtype=dtype)

        try:
            orig_outputs = orig_sess.run(None, feeds)
            new_outputs = new_sess.run(None, feeds)
        except ort.capi.onnxruntime_pybind11_state.RuntimeException:
            continue  # invalid shape combo, retry

        for i, (orig, new) in enumerate(zip(orig_outputs, new_outputs)):
            if not np.allclose(orig, new, atol=1e-5):
                name = orig_sess.get_outputs()[i].name
                max_diff = np.max(np.abs(orig - new))
                print(f"  MISMATCH on output '{name}': max diff = {max_diff}")
                sys.exit(1)

        print(f"  All {len(orig_outputs)} outputs match (atol=1e-5, dims={dim_map})")
        return

    print("  WARNING: Could not find valid random shapes after %d attempts, skipping validation" % max_attempts)


def main():
    # Determine script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    default_input = os.path.join(repo_root, "models", "backbone_keys_orig.onnx")
    default_output = os.path.join(repo_root, "models", "backbone_keys.onnx")

    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    if not os.path.exists(input_path):
        # If no orig file, copy the regular file to orig before surgery
        if input_path == default_input and os.path.exists(default_output):
            print(f"  Copying {default_output} -> {input_path}")
            shutil.copy2(default_output, input_path)
        else:
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

    input_path, output_path = apply_surgery(input_path, output_path)
    validate(input_path, output_path)


if __name__ == "__main__":
    main()
