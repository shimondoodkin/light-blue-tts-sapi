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
        input:  models/onnx/backbone_keys_orig.onnx
        output: models/onnx/backbone_keys.onnx

Requirements:
    pip install onnx numpy
"""

import sys
import os

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np
except ImportError:
    print("Error: This script requires 'onnx' and 'numpy' packages.")
    print("Install with: pip install onnx numpy")
    sys.exit(1)


def apply_surgery(input_path: str, output_path: str) -> None:
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    original_count = len(model.graph.node)

    # Step 1: Rewire chain outputs to new internal names
    t_lat_outputs = ['EgUrDmpWH2JI', 'Tkb5Fe560ZlY', '1nUPdBvudfuN', 'GvQhfOlBoGJk']
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

    print(f"  Removed {len(nodes_to_remove)} CPU-only nodes")

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

    print(f"  Added 6 GPU-compatible replacement nodes")

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    final_count = len(model.graph.node)
    print(f"  Nodes: {original_count} -> {final_count}")
    print(f"  Saved: {output_path}")
    print("  Model check passed.")


def main():
    # Determine script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    default_input = os.path.join(repo_root, "models", "onnx", "backbone_keys_orig.onnx")
    default_output = os.path.join(repo_root, "models", "onnx", "backbone_keys.onnx")

    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    apply_surgery(input_path, output_path)


if __name__ == "__main__":
    main()
