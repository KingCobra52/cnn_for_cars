"""Provenance helpers: record what produced an artifact."""

from __future__ import annotations

import subprocess


def git_sha() -> str | None:
    """Return the current commit hash.

    Recorded alongside every cache entry and every training run, so an artifact can
    always be traced back to the code that made it.

    Returns:
        The full commit hash, or None outside a git checkout or if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
