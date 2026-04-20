"""
Record reproducibility metadata: git revision, environment, and fingerprints of pipeline outputs.

Use after preprocessing / training to restore *which code + which artifacts* produced a result.
Large binaries (``.npz``, ``.pt``) are listed with size/mtime only; small JSON/TXT get SHA256.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Files above this size get mtime+size only (no full-file hash)
HASH_SIZE_LIMIT_BYTES = 5 * 1024 * 1024


def _git_rev(project_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"commit": None, "branch": None, "dirty": None, "error": None}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        st = subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True, stderr=subprocess.DEVNULL)
        out["dirty"] = len(st.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        out["error"] = str(e)
    return out


def _sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        n = 0
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
            if max_bytes is not None and n >= max_bytes:
                break
    return h.hexdigest()


def fingerprint_path(path: Path, project_root: Path) -> dict[str, Any]:
    try:
        rel = str(path.resolve().relative_to(project_root))
    except ValueError:
        rel = str(path)
    st = path.stat()
    rec: dict[str, Any] = {
        "path": rel,
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }
    suf = path.suffix.lower()
    if st.st_size <= HASH_SIZE_LIMIT_BYTES and suf in {".json", ".txt", ".md", ".csv"}:
        rec["sha256"] = _sha256_file(path)
    elif st.st_size > HASH_SIZE_LIMIT_BYTES:
        rec["sha256_prefix"] = _sha256_file(path, max_bytes=1 << 20) + "… (first 1MB only)"
    else:
        rec["sha256"] = _sha256_file(path)
    return rec


def collect_pipeline_files(project_root: Path) -> list[Path]:
    """Key artifacts under ``processed/chl_shallow`` + requirements + run scripts."""
    roots = [
        project_root / "processed" / "chl_shallow",
        project_root,
    ]
    patterns_json_txt = [
        "**/resample_meta.txt",
        "**/split_manifest.json",
        "**/scaler_params.json",
        "**/baseline_metrics.json",
        "**/train_config.json",
        "**/train_config_slide.json",
        "**/gru_eval_summary.json",
        "**/slide_eval_summary.json",
        "**/metrics.csv",
        "**/metrics_slide.csv",
        "**/README_eda.txt",
        "**/eda_summary.json",
        "**/windowed_*/*.json",
    ]
    extra = ["requirements.txt", "run_*.py", "src/**/*.py"]
    files: set[Path] = set()

    base = project_root / "processed" / "chl_shallow"
    if base.is_dir():
        for pat in patterns_json_txt:
            for p in base.glob(pat):
                if p.is_file():
                    files.add(p.resolve())

    req = project_root / "requirements.txt"
    if req.is_file():
        files.add(req.resolve())

    for p in project_root.glob("run_*.py"):
        files.add(p.resolve())
    for p in (project_root / "src").rglob("*.py"):
        if p.is_file():
            files.add(p.resolve())

    return sorted(files)


def collect_large_artifacts(project_root: Path) -> list[dict[str, Any]]:
    """NPZ / checkpoints: size + mtime only (no full hash)."""
    base = project_root / "processed" / "chl_shallow"
    if not base.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for pattern in ("**/*.npz", "**/*.pt"):
        for p in base.glob(pattern):
            if not p.is_file():
                continue
            st = p.stat()
            rel = str(p.relative_to(project_root))
            rows.append(
                {
                    "path": rel,
                    "size_bytes": st.st_size,
                    "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
    return sorted(rows, key=lambda x: x["path"])


def build_snapshot(project_root: Path | None = None) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd()).resolve()
    snap: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cwd": str(project_root),
        "env_PFE_RESAMPLE_FREQ": os.environ.get("PFE_RESAMPLE_FREQ"),
        "git": _git_rev(project_root),
        "tracked_files": [],
        "large_artifacts": collect_large_artifacts(project_root),
    }
    for fp in collect_pipeline_files(project_root):
        try:
            snap["tracked_files"].append(fingerprint_path(fp, project_root))
        except OSError:
            continue
    snap["tracked_files"].sort(key=lambda x: x["path"])
    return snap


def save_snapshot(project_root: Path, out_path: Path | None = None) -> Path:
    project_root = project_root.resolve()
    snap = build_snapshot(project_root)
    out_dir = project_root / "artifacts" / "pipeline_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"snapshot_{ts}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return out_path


__all__ = ["build_snapshot", "save_snapshot", "collect_pipeline_files"]
