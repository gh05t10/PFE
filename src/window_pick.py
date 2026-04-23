"""
When several ``windowed_L*_H*_P*_S*`` folders exist, pick the one with the **smallest stride**
(e.g. S1 before S48 — densest supervision, fair default for training).
"""

from __future__ import annotations

import re
from pathlib import Path


def stride_from_window_dir(p: Path) -> int:
    m = re.search(r"_S(\d+)$", p.name)
    if not m:
        return 10**9
    return int(m.group(1))


def pick_window_dir(norm_dir: Path, pattern: str = "windowed_L*_H*_P*_S*") -> Path:
    cand = sorted(Path(norm_dir).glob(pattern))
    if not cand:
        raise FileNotFoundError(f"No {pattern!r} under {norm_dir}")
    return min(cand, key=stride_from_window_dir)


__all__ = ["pick_window_dir", "stride_from_window_dir"]
