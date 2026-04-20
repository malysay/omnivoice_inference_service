from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf

from app.audio import decode_audio_b64, encode_wav_bytes, sanitize_filename


def test_roundtrip_audio_codec() -> None:
    sample_rate = 24000
    audio = np.linspace(-0.2, 0.2, num=sample_rate, dtype=np.float32)
    wav_bytes = encode_wav_bytes(audio, sample_rate)
    encoded = base64.b64encode(wav_bytes).decode("utf-8")

    decoded, sr = decode_audio_b64(encoded)

    assert sr == sample_rate
    assert decoded.ndim == 1
    assert decoded.shape[0] == sample_rate


def test_sanitize_filename() -> None:
    assert sanitize_filename("yakut-demo") == "yakut-demo.wav"
    assert sanitize_filename("../../bad?.wav") == "bad.wav"
