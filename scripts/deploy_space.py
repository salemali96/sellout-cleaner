"""Deploy the app to a free Hugging Face Docker Space (CPU basic), or check its status.

    UV_SYSTEM_CERTS=1 uv run --no-project --with huggingface_hub --with "truststore>=0.10" \
        python scripts/deploy_space.py [--name sellout-cleaner] [--private] [--dry-run | --status]

Login: the Hugging Face token is read by huggingface_hub from ~/.cache/huggingface/token (or HF_TOKEN). This script
never prints it. Only the files in ALLOW are uploaded, so .env, data and the venv can never be sent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = ["app.py", "pyproject.toml", "uv.lock", ".python-version", "Dockerfile", ".dockerignore", "README.md",
         "DEPLOY.md", ".env.example", ".streamlit/config.toml", "config/*.yaml", "src/**/*.py", "scripts/*.py"]
FORBIDDEN = (".env", "agent_memory.json")


def files_to_upload() -> list[str]:
    found = sorted({p.relative_to(ROOT).as_posix() for pat in ALLOW for p in ROOT.glob(pat) if p.is_file()})
    bad = [f for f in found if Path(f).name in FORBIDDEN or "__pycache__" in f]
    if bad:
        sys.exit(f"refusing to upload: {bad}")
    return found


def use_system_certs() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()  # corporate proxies with their own certificate
    except ImportError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sellout-cleaner")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the files that would be uploaded")
    ap.add_argument("--status", action="store_true", help="show the Space's build/run state")
    args = ap.parse_args()
    files = files_to_upload()
    if args.dry_run:
        print(*files, f"\n{len(files)} files", sep="\n")
        return 0

    use_system_certs()
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    repo = f"{user}/{args.name}"
    if args.status:
        rt = api.space_info(repo).runtime
        print(f"{repo}: {rt.stage}  hardware={rt.hardware}")
        return 0
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True,
                    space_hardware="cpu-basic")  # the free hardware
    api.upload_folder(folder_path=ROOT, repo_id=repo, repo_type="space", allow_patterns=files,
                      commit_message="Deploy Sellout Cleaner")
    print(f"uploaded {len(files)} files to https://huggingface.co/spaces/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
