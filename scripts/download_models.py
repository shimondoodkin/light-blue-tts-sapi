#!/usr/bin/env python3
"""
Download LightBlue TTS models.

Two modes:
  1. Default: download pre-optimized models from GitHub releases
     (with backbone surgery already applied)
  2. --original: download original models from HuggingFace repos

Usage:
    python download_models.py                     # from GitHub releases
    python download_models.py --original          # from HuggingFace
    python download_models.py --dir /path/to/app  # custom target directory

No pip dependencies required (stdlib only).
"""

import os
import sys
import json
import urllib.request
import urllib.error
import argparse

# GitHub releases base URL (update owner/repo as needed)
GITHUB_OWNER = "shimondoodkin"
GITHUB_REPO = "lightblue-sapi"

# HuggingFace base URLs
HF_TTS = "https://huggingface.co/notmax123/LightBlue/resolve/main"
HF_PHONIKUD = "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main"

# Files to download from HuggingFace (original mode)
ORIGINAL_FILES = [
    # TTS ONNX models
    (f"{HF_TTS}/onnx/backbone_keys.onnx", "models/onnx/backbone_keys.onnx"),
    (f"{HF_TTS}/onnx/text_encoder.onnx", "models/onnx/text_encoder.onnx"),
    (f"{HF_TTS}/onnx/reference_encoder.onnx", "models/onnx/reference_encoder.onnx"),
    (f"{HF_TTS}/onnx/vocoder.onnx", "models/onnx/vocoder.onnx"),
    (f"{HF_TTS}/onnx/length_pred.onnx", "models/onnx/length_pred.onnx"),
    (f"{HF_TTS}/onnx/length_pred_style.onnx", "models/onnx/length_pred_style.onnx"),
    (f"{HF_TTS}/onnx/stats.npz", "models/onnx/stats.npz"),
    (f"{HF_TTS}/onnx/uncond.npz", "models/onnx/uncond.npz"),
    # TTS config
    (f"{HF_TTS}/tts.json", "models/tts.json"),
    # Voice styles (renamed to match app expectations)
    (f"{HF_TTS}/style.json", "models/voices/male1.json"),
    (f"{HF_TTS}/style_female.json", "models/voices/female1.json"),
    # Phonikud
    (f"{HF_PHONIKUD}/phonikud.onnx", "models/phonikud.onnx"),
    (f"{HF_PHONIKUD}/tokenizer.json", "models/tokenizer.json"),
]

# Files to download from GitHub releases (pre-optimized)
RELEASE_FILES = [
    "models/onnx/backbone_keys.onnx",
    "models/onnx/text_encoder.onnx",
    "models/onnx/reference_encoder.onnx",
    "models/onnx/vocoder.onnx",
    "models/onnx/length_pred.onnx",
    "models/onnx/length_pred_style.onnx",
    "models/onnx/stats.npz",
    "models/onnx/uncond.npz",
    "models/tts.json",
    "models/voices/male1.json",
    "models/voices/female1.json",
    "models/phonikud.onnx",
    "models/tokenizer.json",
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


def get_latest_release_tag():
    """Get the latest release tag from GitHub."""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LightBlue-TTS-Downloader/1.0"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("tag_name", "v1.0.0")
    except Exception:
        return "v1.0.0"


def download_from_releases(base_dir, force=False):
    """Download pre-optimized models from GitHub releases."""
    print("Fetching latest release info...")
    tag = get_latest_release_tag()
    print(f"Release: {tag}")
    print()

    base_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{tag}"

    success = 0
    failed = 0
    for rel_path in RELEASE_FILES:
        filename = rel_path.replace("/", "_")  # flat file naming in release assets
        url = f"{base_url}/{filename}"
        dest = os.path.join(base_dir, rel_path)
        if download_file(url, dest, force):
            success += 1
        else:
            failed += 1

    return success, failed


def download_from_original(base_dir, force=False):
    """Download original models from HuggingFace."""
    print("Downloading from HuggingFace (original models)...")
    print()

    success = 0
    failed = 0
    for url, rel_path in ORIGINAL_FILES:
        dest = os.path.join(base_dir, rel_path)
        if download_file(url, dest, force):
            success += 1
        else:
            failed += 1

    return success, failed


def find_base_dir():
    """Find the base directory (look for models/ subdir or use CWD)."""
    # If run from repo root or app install dir
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Prefer repo root if it has a models/ dir or Cargo.toml
    if os.path.exists(os.path.join(repo_root, "Cargo.toml")):
        return repo_root
    if os.path.exists(os.path.join(repo_root, "models")):
        return repo_root

    return cwd


def main():
    parser = argparse.ArgumentParser(description="Download LightBlue TTS models")
    parser.add_argument("--dir", help="Target directory (default: auto-detect)")
    parser.add_argument("--original", action="store_true",
                        help="Download original models from HuggingFace (not pre-optimized)")
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

    # Create directory structure
    for subdir in ["models/onnx", "models/voices"]:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    if args.original:
        success, failed = download_from_original(base_dir, args.force)
    else:
        success, failed = download_from_releases(base_dir, args.force)

    print()
    print(f"Done: {success} succeeded, {failed} failed")
    if failed > 0:
        print("Some downloads failed. Re-run with --force to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
