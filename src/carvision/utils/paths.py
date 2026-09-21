"""Canonical filesystem locations.

Every path is resolved relative to the repository root, or to ``CARVISION_ROOT``
when set. Nothing in this package writes to an implicit working directory -- the
legacy notebook's ``plt.savefig('accuracy_comparison.png')`` landed wherever Colab
happened to be, which is why no figure was ever reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the project root.

    Honours ``CARVISION_ROOT`` when set, so tests and Spaces deployments can
    redirect every output without patching call sites. Otherwise walks up from
    this file to the directory holding ``pyproject.toml``, falling back to the
    package parent when the project is installed rather than checked out.
    """
    override = os.environ.get("CARVISION_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


def data_dir() -> Path:
    """Directory holding the downloaded dataset (gitignored)."""
    return repo_root() / "data"


def splits_dir() -> Path:
    """Directory holding the committed split CSVs."""
    return repo_root() / "data" / "splits"


def cache_dir() -> Path:
    """Directory holding the frozen-backbone embedding cache (gitignored)."""
    return repo_root() / "artifacts" / "embeddings"


def artifacts_dir() -> Path:
    """Directory containing generated experiment artifacts."""
    return repo_root() / "artifacts"


def runs_dir() -> Path:
    """Directory holding one subdirectory per training run (gitignored)."""
    return repo_root() / "artifacts" / "runs"


def figures_dir() -> Path:
    """Directory holding committed figures referenced by the docs."""
    return repo_root() / "docs" / "figures"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and its parents if absent, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
