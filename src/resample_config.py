"""
Configurable resampling grid for buoy time series (PatchTST / soft-sensor experiments).

Change the default once in code, or override per run without editing files:
  - Environment: ``PFE_RESAMPLE_FREQ`` (e.g. ``30min``, ``10min``, ``15min``)
  - CLI: ``python run_unified_resample.py --freq 10min``

Any pandas-compatible offset string is allowed (see pandas ``freq`` / ``DateOffset`` docs).
"""

from __future__ import annotations

import os
import re

from pandas.tseries.frequencies import to_offset

DEFAULT_RESAMPLE_FREQ = "30min"
ENV_RESAMPLE_FREQ = "PFE_RESAMPLE_FREQ"


def validate_freq(freq: str) -> str:
    """Return *freq* if pandas accepts it as a DateOffset; else raise ValueError."""
    if not freq or not str(freq).strip():
        raise ValueError("resample frequency must be a non-empty string")
    s = str(freq).strip()
    try:
        to_offset(s)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid pandas frequency {s!r} — use e.g. '30min', '10min', '1H'.") from e
    return s


def get_resample_freq(*, cli: str | None = None, env: bool = True) -> str:
    """
    Resolve resample frequency: **cli** > **env** (``PFE_RESAMPLE_FREQ``) > **default**.

    Parameters
    ----------
    cli:
        Explicit override (e.g. from ``argparse``). Empty string is ignored.
    env:
        If True, read ``PFE_RESAMPLE_FREQ`` when *cli* is unset.
    """
    if cli is not None and str(cli).strip():
        return validate_freq(cli)
    if env:
        raw = os.environ.get(ENV_RESAMPLE_FREQ)
        if raw is not None and str(raw).strip():
            return validate_freq(raw)
    return validate_freq(DEFAULT_RESAMPLE_FREQ)


def freq_slug(freq: str) -> str:
    """Filesystem-safe token for folder names (e.g. ``30min``, ``10min``)."""
    s = validate_freq(freq)
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip()).strip("_")
    return slug or "custom"


__all__ = [
    "DEFAULT_RESAMPLE_FREQ",
    "ENV_RESAMPLE_FREQ",
    "validate_freq",
    "get_resample_freq",
    "freq_slug",
]
