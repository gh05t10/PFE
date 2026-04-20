#!/usr/bin/env python3
"""
Write a reproducibility snapshot: git commit, env, fingerprints of manifests + code.

Outputs:
  - ``artifacts/pipeline_snapshots/snapshot_<UTC>.json``
  - ``artifacts/pipeline_snapshots/latest.json`` (copy of latest run)

Re-run after any important preprocessing or training step; commit snapshots + code to git to restore context.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline_snapshot import save_snapshot


def main() -> None:
    p = argparse.ArgumentParser(description="Record pipeline reproducibility snapshot.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional explicit output JSON path",
    )
    args = p.parse_args()
    base = Path(__file__).resolve().parent
    path = save_snapshot(base, args.out)
    print(f"Wrote {path}")
    print(f"Also updated {base / 'artifacts/pipeline_snapshots/latest.json'}")


if __name__ == "__main__":
    main()
