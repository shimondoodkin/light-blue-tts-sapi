#!/usr/bin/env python3
"""
Download LightBlue TTS models from HuggingFace.

Usage:
    python download_models.py                     # download models
    python download_models.py --dir /path/to/app  # custom target directory
    python download_models.py --force             # re-download existing files

No pip dependencies required (stdlib only).
"""

import os
import sys
import urllib.request
import urllib.error
import argparse

# Files to download: (url, local_path)
FILES = [
    # TTS ONNX models
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/backbone_keys.onnx", "models/backbone_keys.onnx"),
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/text_encoder.onnx", "models/text_encoder.onnx"),
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/reference_encoder.onnx", "models/reference_encoder.onnx"),
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/vocoder.onnx", "models/vocoder.onnx"),
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/length_pred.onnx", "models/length_pred.onnx"),
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/length_pred_style.onnx", "models/length_pred_style.onnx"),
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/stats.npz", "models/stats.npz"),
    ("https://huggingface.co/notmax123/LightBlue/resolve/main/uncond.npz", "models/uncond.npz"),
    # Voice styles
    ("https://raw.githubusercontent.com/maxmelichov/Light-BlueTTS/main/voices/male1.json", "models/voices/male1.json"),
    ("https://raw.githubusercontent.com/maxmelichov/Light-BlueTTS/main/voices/female1.json", "models/voices/female1.json"),
    # Phonikud
    ("https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.onnx", "models/phonikud.onnx"),
    ("https://huggingface.co/dicta-il/dictabert-large-char-menaked/raw/main/tokenizer.json", "models/tokenizer.json"),
]


def download_file(url, dest_path, force=False):
    """Download a file with progress indication."""
    if not force and os.path.exists(dest_path):
        print(f"  [skip] {os.path.basename(dest_path)} (exists)")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    name = os.path.basename(dest_path)
    print(f"  [download] {name}...", end="", flush=True)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LightBlue-TTS-Downloader/1.0"})
        with urllib.request.urlopen(req) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None

            with open(dest_path, "wb") as f:
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

            size_mb = downloaded / (1024 * 1024)
            print(f" {size_mb:.1f}MB OK")
            return True

    except urllib.error.HTTPError as e:
        print(f" FAILED (HTTP {e.code})")
        return False
    except Exception as e:
        print(f" FAILED ({e})")
        return False


def find_base_dir():
    """Find the base directory (look for models/ subdir or use CWD)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    if os.path.exists(os.path.join(repo_root, "Cargo.toml")):
        return repo_root
    if os.path.exists(os.path.join(repo_root, "models")):
        return repo_root

    return os.getcwd()


def main():
    parser = argparse.ArgumentParser(description="Download LightBlue TTS models")
    parser.add_argument("--dir", help="Target directory (default: auto-detect)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files even if they exist")
    args = parser.parse_args()

    base_dir = args.dir or find_base_dir()
    base_dir = os.path.abspath(base_dir)

    print("=" * 50)
    print("LightBlue TTS Model Downloader")
    print("=" * 50)
    print(f"Target: {base_dir}")
    print()

    for subdir in ["models", "models/voices"]:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    success = 0
    failed = 0
    for url, rel_path in FILES:
        dest = os.path.join(base_dir, rel_path)
        if download_file(url, dest, args.force):
            success += 1
        else:
            failed += 1

    print()
    print(f"Done: {success} succeeded, {failed} failed")
    if failed > 0:
        print("Some downloads failed. Re-run with --force to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
