#!/usr/bin/env python3
"""Download public datasets used for smoke tests and warm-start training."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("external_data"))
    parser.add_argument("--kaggle-ref", action="append", default=[])
    parser.add_argument("--hf-repo", action="append", default=[])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.kaggle_ref and not args.hf_repo:
        args.kaggle_ref = [
            "saikatpanda/cubes-craters-and-cylinder",
            "mrpoulpe/cubes-and-cylinders",
        ]
        args.hf_repo = ["0-ma/geometric-shapes"]

    for ref in args.kaggle_ref:
        target = args.out / ref.split("/", 1)[-1]
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "kaggle", "datasets", "download", "-d", ref,
                "-p", str(target), "--unzip",
            ],
            check=True,
        )

    if args.hf_repo:
        from huggingface_hub import snapshot_download

        for repo in args.hf_repo:
            target = args.out / repo.replace("/", "__")
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(target),
                local_dir_use_symlinks=False,
            )

    print(f"Downloaded public data under: {args.out.resolve()}")


if __name__ == "__main__":
    main()
