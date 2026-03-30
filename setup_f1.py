import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
HF_HOME = REPO_ROOT / "hf_download"

MODEL_REPOS = [
    "hunyuanvideo-community/HunyuanVideo",
    "lllyasviel/flux_redux_bfl",
    "lllyasviel/FramePack_F1_I2V_HY_20250503",
]

HUNYUAN_SUBFOLDERS = [
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "vae",
]

FLUX_REDUX_SUBFOLDERS = [
    "feature_extractor",
    "image_encoder",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install Python dependencies and pre-download the FramePack-F1 models."
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python package installation and only download model files.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model downloads and only install Python packages.",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu126",
        help="Index URL used for torch/torchvision/torchaudio installation.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(HF_HOME),
        help="Cache directory for Hugging Face downloads. Defaults to ./hf_download.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision to pin when downloading snapshots.",
    )
    return parser.parse_args()


def run(cmd, env=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def install_python_dependencies(args):
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            args.torch_index_url,
        ],
        env=env,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REPO_ROOT / "requirements.txt"),
            "huggingface_hub",
        ],
        env=env,
    )


def get_hf_env(args):
    env = os.environ.copy()
    env["HF_HOME"] = str(Path(args.cache_dir).resolve())
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    return env


def download_model_snapshots(args):
    from huggingface_hub import login, snapshot_download

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = {}

    if args.hf_token:
        login(token=args.hf_token)
        print("HF login ok.")

    for repo_id in MODEL_REPOS:
        print(f"Downloading {repo_id} ...")
        downloaded_paths[repo_id] = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            resume_download=True,
            revision=args.revision,
        )
        print(f"Cached at {downloaded_paths[repo_id]}")

    return downloaded_paths


def verify_downloads(downloaded_paths):
    expected_dirs = [
        Path(downloaded_paths["hunyuanvideo-community/HunyuanVideo"]) / subfolder
        for subfolder in HUNYUAN_SUBFOLDERS
    ] + [
        Path(downloaded_paths["lllyasviel/flux_redux_bfl"]) / subfolder
        for subfolder in FLUX_REDUX_SUBFOLDERS
    ] + [
        Path(downloaded_paths["lllyasviel/FramePack_F1_I2V_HY_20250503"])
    ]

    missing = [path for path in expected_dirs if not path.exists()]
    if missing:
        print("Some expected downloaded paths are missing:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("Verified downloaded model directories.")


def main():
    args = parse_args()

    os.environ["HF_HOME"] = str(Path(args.cache_dir).resolve())

    if not args.skip_python:
        install_python_dependencies(args)

    if not args.skip_models:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print(
                "huggingface_hub is not installed yet. Run without --skip-python first, "
                "or install it manually before downloading models."
            )
            raise

        downloaded_paths = download_model_snapshots(args)
        verify_downloads(downloaded_paths)

    print("Setup complete.")
    print(f"HF_HOME={Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()
