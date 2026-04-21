from __future__ import annotations

from app.triton_client import build_infer_payload, decode_infer_response


def test_build_infer_payload_wraps_json_request() -> None:
    payload = build_infer_payload({"text": "hello", "language": "sah"})
    assert payload["inputs"][0]["name"] == "REQUEST_JSON"
    assert payload["inputs"][0]["datatype"] == "BYTES"
    assert "\"text\":\"hello\"" in payload["inputs"][0]["data"][0]


def test_decode_infer_response_extracts_wav_bytes() -> None:
    response = {
        "outputs": [
            {"name": "AUDIO_WAV", "data": [82, 73, 70, 70]},
            {"name": "SAMPLE_RATE", "data": [24000]},
            {"name": "ELAPSED_MS", "data": [123]},
        ]
    }
    audio_bytes, sample_rate, elapsed_ms = decode_infer_response(response)
    assert audio_bytes == b"RIFF"
    assert sample_rate == 24000
    assert elapsed_ms == 123
