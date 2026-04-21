from __future__ import annotations

import pytest

from app.request_parsing import parse_synthesis_request_payload


def test_parse_synthesis_request_payload_from_json() -> None:
    request = parse_synthesis_request_payload('{"text":"hello","language":"sah"}')
    assert request.text == "hello"
    assert request.language == "sah"


def test_parse_synthesis_request_payload_requires_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_synthesis_request_payload('["not-an-object"]')
