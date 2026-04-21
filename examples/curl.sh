#!/usr/bin/env bash
set -euo pipefail

curl -sS -X POST "http://127.0.0.1:8000/v2/models/omnivoice/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "REQUEST_JSON",
        "shape": [1],
        "datatype": "BYTES",
        "data": [
          "{\"text\":\"Мин аатым Айаал. Бу OmniVoice көмөтүнэн үөскэтиллибит якут тылынан тест.\",\"language\":\"sah\",\"speed\":1.0,\"num_step\":32}"
        ]
      }
    ],
    "outputs": [
      {"name": "AUDIO_WAV"},
      {"name": "SAMPLE_RATE"},
      {"name": "ELAPSED_MS"},
      {"name": "FILENAME"},
      {"name": "SAVED_TO"}
    ]
  }'
