from __future__ import annotations

from pathlib import Path

import pytest

from app.model_assets import minimal_model_allow_patterns, validate_model_dir


def test_minimal_model_allow_patterns_contains_required_entries() -> None:
    patterns = minimal_model_allow_patterns()

    assert "config.json" in patterns
    assert "*.safetensors" in patterns
    assert "audio_tokenizer/**" in patterns


def test_validate_model_dir_requires_runtime_assets(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Missing required runtime assets"):
        validate_model_dir(tmp_path)


def test_validate_model_dir_accepts_minimal_runtime_snapshot(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "model.safetensors").write_text("weights\n")
    (tmp_path / "tokenizer.json").write_text("{}\n")
    (tmp_path / "audio_tokenizer").mkdir()
    (tmp_path / "audio_tokenizer" / "config.json").write_text("{}\n")

    validate_model_dir(tmp_path)
