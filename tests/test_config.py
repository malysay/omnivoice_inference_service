from __future__ import annotations

from app.config import Settings


def test_default_language_is_yakut() -> None:
    settings = Settings()
    assert settings.default_language == "sah"
