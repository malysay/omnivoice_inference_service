from __future__ import annotations

import json
from typing import Any

from app.schemas import SynthesisRequest


def parse_synthesis_request_payload(payload: str | bytes | dict[str, Any]) -> SynthesisRequest:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")

    if isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = payload

    if not isinstance(data, dict):
        raise ValueError("Synthesis payload must be a JSON object.")

    return SynthesisRequest.model_validate(data)
