#!/usr/bin/env bash
set -euo pipefail

curl -sS -X POST "http://127.0.0.1:8080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -o yakut.wav \
  -d '{
    "text": "Мин аатым Айаал. Бу OmniVoice көмөтүнэн үөскэтиллибит якут тылынан тест.",
    "language": "sah",
    "speed": 1.0,
    "num_step": 32
  }'
