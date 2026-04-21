from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from app.request_parsing import parse_synthesis_request_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton HTTP client for OmniVoice")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="omnivoice")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="sah")
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--ref-text", dest="ref_text", default=None)
    parser.add_argument("--ref-audio-b64", dest="reference_audio_b64", default=None)
    parser.add_argument("--num-step", dest="num_step", type=int, default=None)
    parser.add_argument("--guidance-scale", dest="guidance_scale", type=float, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    return parser.parse_args()


def build_infer_payload(request_data: dict[str, Any]) -> dict[str, Any]:
    request = parse_synthesis_request_payload(request_data)
    request_json = request.model_dump_json(exclude_none=True)
    return {
        "inputs": [
            {
                "name": "REQUEST_JSON",
                "shape": [1],
                "datatype": "BYTES",
                "data": [request_json],
            }
        ],
        "outputs": [
            {"name": "AUDIO_WAV"},
            {"name": "SAMPLE_RATE"},
            {"name": "ELAPSED_MS"},
            {"name": "FILENAME"},
            {"name": "SAVED_TO"},
        ],
    }


def decode_infer_response(payload: dict[str, Any]) -> tuple[bytes, int, int]:
    outputs = {item["name"]: item for item in payload.get("outputs", [])}
    audio_values = outputs["AUDIO_WAV"]["data"]
    sample_rate = int(outputs["SAMPLE_RATE"]["data"][0])
    elapsed_ms = int(outputs["ELAPSED_MS"]["data"][0])
    audio_bytes = bytes(audio_values)
    return audio_bytes, sample_rate, elapsed_ms


def main() -> None:
    args = parse_args()
    request_data = {
        "text": args.text,
        "language": args.language,
        "instruct": args.instruct,
        "ref_text": args.ref_text,
        "reference_audio_b64": args.reference_audio_b64,
        "num_step": args.num_step,
        "guidance_scale": args.guidance_scale,
        "speed": args.speed,
        "duration": args.duration,
    }
    payload = build_infer_payload(request_data)

    endpoint = f"{args.url.rstrip('/')}/v2/models/{args.model}/infer"
    response = httpx.post(endpoint, json=payload, timeout=600.0)
    response.raise_for_status()

    audio_bytes, sample_rate, elapsed_ms = decode_infer_response(response.json())
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)

    print(
        json.dumps(
            {
                "output": str(out_path),
                "sample_rate": sample_rate,
                "elapsed_ms": elapsed_ms,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
