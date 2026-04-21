#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "== live =="
curl -fsS "$BASE_URL/v2/health/live"
echo

echo "== ready =="
curl -fsS "$BASE_URL/v2/health/ready"
echo

echo "== model ready =="
curl -fsS "$BASE_URL/v2/models/omnivoice/ready"
echo

echo "== model metadata =="
curl -fsS "$BASE_URL/v2/models/omnivoice"
echo
