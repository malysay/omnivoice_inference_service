from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.service import OmniVoiceManager


def test_model_dir_must_be_directory(tmp_path: Path) -> None:
    bad_path = tmp_path / "omnivoice.py"
    bad_path.write_text("# not a model checkpoint\n")

    manager = OmniVoiceManager(Settings(model_dir=str(bad_path), preload_model=False))

    with pytest.raises(RuntimeError, match="must point to a model directory, not a file"):
        manager._load_model_sync()
