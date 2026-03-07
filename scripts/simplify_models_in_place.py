"""
simplify_models.py — Simplify all ONNX models in the models/ folder in-place using onnxsim.

Usage:
    python scripts/simplify_models.py
    python scripts/simplify_models.py --models-dir path/to/models
"""

import argparse
import sys
from pathlib import Path

try:
    import onnx
    import onnxsim
except ImportError:
    sys.exit(
        "[ERROR] onnx and onnxsim are required.\n"
        "  pip install onnx onnxsim\n"
        "  Or run: python scripts/build_onnxsim.py"
    )


def simplify_model(path: Path) -> bool:
    """Simplify a single ONNX model in-place. Returns True on success."""
    orig_size = path.stat().st_size
    print(f"  {path.name} ...", end=" ", flush=True)
    try:
        model = onnx.load(str(path))
    except Exception as e:
        print(f"FAILED to load: {e}")
        return False

    # Deduplicate metadata_props keys (some models have duplicates that onnxsim rejects)
    seen = set()
    unique_props = []
    for prop in model.metadata_props:
        if prop.key not in seen:
            seen.add(prop.key)
            unique_props.append(prop)
    if len(unique_props) < len(model.metadata_props):
        del model.metadata_props[:]
        model.metadata_props.extend(unique_props)

    try:
        model_sim, check = onnxsim.simplify(model)
    except Exception as e:
        print(f"FAILED: {e}")
        return False

    if not check:
        print("FAILED validation check")
        return False

    onnx.save(model_sim, str(path))

    orig_mb = orig_size / (1024 * 1024)
    simp_mb = path.stat().st_size / (1024 * 1024)
    print(f"OK  {orig_mb:.1f} MB -> {simp_mb:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Simplify all ONNX models in a folder")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="Directory containing .onnx files (default: models/)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    if not models_dir.is_dir():
        sys.exit(f"[ERROR] Directory not found: {models_dir}")

    onnx_files = sorted(models_dir.glob("*.onnx"))
    if not onnx_files:
        sys.exit(f"[ERROR] No .onnx files found in {models_dir}")

    print(f"Found {len(onnx_files)} ONNX model(s) in {models_dir}\n")

    succeeded = 0
    failed = 0

    for onnx_file in onnx_files:
        if onnx_file.name.endswith("_orig.onnx"):
            print(f"  {onnx_file.name} ... SKIPPED (_orig.onnx)")
            continue
        if simplify_model(onnx_file):
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed out of {len(onnx_files)} models.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
