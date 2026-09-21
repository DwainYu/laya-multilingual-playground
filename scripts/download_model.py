#!/usr/bin/env python3
"""Download convaiinnovations/laya-multilingual from ModelScope into models/ and verify it."""

from __future__ import annotations

import sys

from laya_play import (
    HUGGINGFACE_URL,
    MODEL_ID,
    MODELSCOPE_URL,
    dir_size_mib,
    missing_files,
    model_dir,
)


def main() -> int:
    target = model_dir()
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model:   {MODEL_ID}")
    print(f"Source:  {MODELSCOPE_URL}")
    print(f"Target:  {target}\n")

    from modelscope import snapshot_download

    downloaded = snapshot_download(MODEL_ID, local_dir=str(target))
    print(f"snapshot_download returned: {downloaded}\n")

    missing = missing_files(target)
    if missing:
        print("ERROR: checkpoint is incomplete, missing required files:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(f"\nHugging Face mirror of the same checkpoint: {HUGGINGFACE_URL}", file=sys.stderr)
        return 1

    print("Checkpoint OK:")
    for path in sorted(target.rglob("*")):
        if path.is_file():
            print(f"  {str(path.relative_to(target)):<32}{path.stat().st_size / 1024**2:>9.2f} MiB")
    print(f"\nTotal: {dir_size_mib(target):.1f} MiB in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
