from __future__ import annotations

from pathlib import Path


MINIMAL_MODEL_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "*.model",
    "chat_template.jinja",
    "audio_tokenizer/**",
)


def minimal_model_allow_patterns() -> list[str]:
    return list(MINIMAL_MODEL_ALLOW_PATTERNS)


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise RuntimeError(
            "OMNIVOICE_MODEL_DIR must point to a model directory, not a file: "
            f"{model_dir}"
        )

    missing: list[str] = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not any(model_dir.glob("*.safetensors")):
        missing.append("at least one *.safetensors file")
    tokenizer_artifacts = [
        model_dir / "tokenizer.json",
        model_dir / "tokenizer.model",
        model_dir / "sentencepiece.bpe.model",
        model_dir / "vocab.json",
        model_dir / "vocab.txt",
    ]
    if not any(path.is_file() for path in tokenizer_artifacts):
        missing.append("text tokenizer files (for example tokenizer.json or tokenizer.model)")
    if not (model_dir / "audio_tokenizer").is_dir():
        missing.append("audio_tokenizer/")

    if missing:
        details = ", ".join(missing)
        raise RuntimeError(
            f"Model directory is incomplete: {model_dir}. Missing required runtime assets: {details}"
        )
