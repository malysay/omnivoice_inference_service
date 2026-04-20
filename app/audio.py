from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioDecodeError(ValueError):
    """Raised when input audio cannot be decoded."""


def decode_audio_b64(audio_b64: str) -> tuple[np.ndarray, int]:
    try:
        raw = base64.b64decode(audio_b64)
    except Exception as exc:  # noqa: BLE001
        raise AudioDecodeError("Invalid base64 audio payload.") from exc
    return decode_audio_bytes(raw)


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise AudioDecodeError("Unable to decode reference audio.") from exc

    if audio.size == 0:
        raise AudioDecodeError("Reference audio is empty.")

    audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def encode_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def sanitize_filename(filename: str | None, fallback_stem: str = "output") -> str:
    if not filename:
        return f"{fallback_stem}.wav"
    safe = "".join(ch for ch in filename if ch.isalnum() or ch in {"-", "_", "."}).strip(".")
    if not safe:
        safe = fallback_stem
    if not safe.lower().endswith(".wav"):
        safe += ".wav"
    return safe


def persist_wav(audio_bytes: bytes, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / sanitize_filename(filename)
    out_path.write_bytes(audio_bytes)
    return out_path
