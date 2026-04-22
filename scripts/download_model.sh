#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_REPO_ID="${HF_REPO_ID:-k2-fsa/OmniVoice}"
HF_REVISION="${HF_REVISION:-main}"
MODEL_DIR="${MODEL_DIR:-./models/omnivoice-runtime}"
HF_TOKEN="${HF_TOKEN:-}"
DOWNLOAD_MODE="${DOWNLOAD_MODE:-full}"
PIP_BIN="${PIP_BIN:-python -m pip}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "== OmniVoice model download =="
echo "repo: $HF_REPO_ID"
echo "revision: $HF_REVISION"
echo "output: $MODEL_DIR"
echo "mode: $DOWNLOAD_MODE"

$PIP_BIN install '.[download]'

cmd=(
  $PYTHON_BIN -m app.download_model
  --repo-id "$HF_REPO_ID"
  --revision "$HF_REVISION"
  --output-dir "$MODEL_DIR"
)

if [[ "$DOWNLOAD_MODE" == "full" ]]; then
  cmd+=(--full-snapshot)
elif [[ "$DOWNLOAD_MODE" != "minimal" ]]; then
  echo "Unsupported DOWNLOAD_MODE=$DOWNLOAD_MODE. Use 'full' or 'minimal'." >&2
  exit 1
fi

if [[ -n "$HF_TOKEN" ]]; then
  cmd+=(--token "$HF_TOKEN")
fi

"${cmd[@]}"

echo
echo "Model download completed."
echo "Set OMNIVOICE_MODEL_DIR=$(cd "$MODEL_DIR" && pwd) in .env"
