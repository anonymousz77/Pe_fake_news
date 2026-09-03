"""Path resolution for the Pe_Fake_News_Dec data layer.

This module is the single source of truth for every filesystem location in the
project. **Nothing anywhere else may hardcode a path.** Import the constants or
helpers from here instead::

    from configs.paths import RAW, raw_dir, free_gb

PROJECT_ROOT is resolved, in order, from:

1. the ``PE_FAKE_NEWS_ROOT`` environment variable,
2. ``PE_FAKE_NEWS_ROOT`` in a ``.env`` file at the repo root,
3. the parent of this file's directory (i.e. the repo root itself).

The dataset register (``data/sources.yaml``) is read lazily on first use and
cached, never at import time: a malformed register should break the script that
actually needs it, not every import in the project.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

_THIS_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _THIS_DIR.parent

_ENV_VAR: Final[str] = "PE_FAKE_NEWS_ROOT"


def _resolve_project_root() -> Path:
    """Resolve PROJECT_ROOT from the environment, then .env, then this file."""
    raw = os.environ.get(_ENV_VAR)

    if not raw:
        try:
            from dotenv import dotenv_values
        except ImportError:  # python-dotenv is optional at runtime
            pass
        else:
            raw = dotenv_values(_REPO_ROOT / ".env").get(_ENV_VAR)

    root = Path(raw).expanduser().resolve() if raw else _REPO_ROOT

    if not root.is_dir():
        raise NotADirectoryError(
            f"{_ENV_VAR} points at {root}, which is not an existing directory."
        )
    return root


PROJECT_ROOT: Final[Path] = _resolve_project_root()

DATA: Final[Path] = PROJECT_ROOT / "data"
RAW: Final[Path] = DATA / "raw"
INTERIM: Final[Path] = DATA / "interim"
PROCESSED: Final[Path] = DATA / "processed"
SPLITS: Final[Path] = PROCESSED / "splits"
REPORTS: Final[Path] = DATA / "reports"

#: The evidence store: the time-filtered retrieval corpus of dated documents
#: that claims are checked against. Not documentation, not a dataset mirror.
#: Gitignored; expected to reach tens of GB.
BIBLE: Final[Path] = DATA / "bible"

#: Hand-written, git-tracked prose (data card, counts ledger, provenance).
DOCS: Final[Path] = PROJECT_ROOT / "docs"

SOURCES_YAML: Final[Path] = DATA / "sources.yaml"
MANIFEST: Final[Path] = DATA / "MANIFEST.sha256"

#: Directories ensure_tree() creates. BIBLE is included: it is ignored by git,
#: not absent from disk.
_TREE: Final[tuple[Path, ...]] = (
    DATA,
    RAW,
    INTERIM,
    PROCESSED,
    SPLITS,
    REPORTS,
    BIBLE,
    DOCS,
)


def ensure_tree() -> None:
    """Create every project directory. Idempotent."""
    for directory in _TREE:
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    """Load and cache data/sources.yaml, keyed by dataset name.

    Lazy by design: imported modules that never touch the register must not
    fail because the register is malformed.
    """
    import yaml

    if not SOURCES_YAML.is_file():
        raise FileNotFoundError(f"dataset register not found: {SOURCES_YAML}")

    with SOURCES_YAML.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    if not isinstance(document, dict) or "datasets" not in document:
        raise ValueError(f"{SOURCES_YAML} has no top-level 'datasets:' key")

    entries = document["datasets"]
    if not isinstance(entries, list):
        raise ValueError(f"{SOURCES_YAML}: 'datasets' must be a list")

    registry: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"{SOURCES_YAML}: entry {index} has no 'name'")
        name = entry["name"]
        if name in registry:
            raise ValueError(f"{SOURCES_YAML}: duplicate dataset name {name!r}")
        registry[name] = entry

    return registry


def dataset_names() -> tuple[str, ...]:
    """Every dataset name in the register, in file order."""
    return tuple(_registry())


def dataset(name: str) -> dict[str, Any]:
    """The register entry for ``name``.

    Raises:
        KeyError: if ``name`` is not in the register.
    """
    registry = _registry()
    if name not in registry:
        known = ", ".join(registry)
        raise KeyError(
            f"unknown dataset {name!r} — not in {SOURCES_YAML} (known: {known})"
        )
    return registry[name]


def raw_dir(name: str) -> Path:
    """The immutable raw directory for one dataset: ``data/raw/<name>``.

    The only sanctioned way to name a raw dataset directory. The name is checked
    against the register so a typo cannot silently create a stray tree.
    """
    dataset(name)  # raises KeyError naming the file and the unknown dataset
    return RAW / name


def free_gb(path: Path = PROJECT_ROOT) -> float:
    """Free space in GB on the volume holding ``path``.

    Check this before any acquisition: the project runs under a hard 160 GB cap.
    """
    return shutil.disk_usage(path).free / 1024**3
