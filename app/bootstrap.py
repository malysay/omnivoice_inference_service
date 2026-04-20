from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

from app.config import Settings


class OmniVoiceImportError(RuntimeError):
    """Raised when the omnivoice package cannot be imported."""


def _candidate_source_dirs(settings: Settings) -> Iterable[Path]:
    cwd = Path.cwd().resolve()
    defaults = [
        settings.source_path,
        cwd.parent / "OmniVoice",
        cwd.parent / "omnivoice",
        cwd / "OmniVoice",
        cwd / "omnivoice",
    ]
    seen: set[Path] = set()
    for candidate in defaults:
        if candidate is None:
            continue
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def ensure_omnivoice_importable(settings: Settings):
    try:
        return importlib.import_module("omnivoice")
    except ImportError:
        pass

    for source_dir in _candidate_source_dirs(settings):
        package_dir = source_dir / "omnivoice"
        if package_dir.is_dir() and (source_dir / "pyproject.toml").exists():
            sys.path.insert(0, str(source_dir))
            try:
                return importlib.import_module("omnivoice")
            except ImportError:
                continue

    raise OmniVoiceImportError(
        "Cannot import `omnivoice`. Install the package or set OMNIVOICE_SOURCE_DIR "
        "to the local OmniVoice repository root."
    )
