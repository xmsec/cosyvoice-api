#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${COSYVOICE_APP_DIR:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9233}"
MODELS_DIR="${MODELS_DIR:-./pretrained_models}"
OUTPUT_DIR="${OUTPUT_DIR:-./tmp}"
REFER_AUDIO_DIR="${REFER_AUDIO_DIR:-./reference_audio}"
VOICES_CONFIG="${VOICES_CONFIG:-./voices.example.json}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"
API_KEY="${COSYVOICE_API_KEY:-ppt-master-cosyvoice-local-key}"
PRELOAD_MODELS="${PRELOAD_MODELS:-tts}"
STRICT_VOICE_PROFILES="${STRICT_VOICE_PROFILES:-1}"
DISABLE_DOWNLOAD="${DISABLE_DOWNLOAD:-0}"
BACKGROUND="${BACKGROUND:-0}"
LOG_FILE="${LOG_FILE:-./tmp/cosyvoice-api.log}"

cd "$APP_DIR"

if [[ ! -f "api.py" ]]; then
  echo "error: api.py not found in $APP_DIR" >&2
  echo "Set COSYVOICE_APP_DIR to the directory that contains api.py." >&2
  exit 1
fi

if [[ ! -f "$VOICES_CONFIG" ]]; then
  echo "error: voices config not found: $VOICES_CONFIG" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$REFER_AUDIO_DIR"

cmd=(
  "$PYTHON_BIN" "api.py"
  --host "$HOST"
  --port "$PORT"
  --models-dir "$MODELS_DIR"
  --output-dir "$OUTPUT_DIR"
  --refer-audio-dir "$REFER_AUDIO_DIR"
  --voices-config "$VOICES_CONFIG"
  --api-key "$API_KEY"
)

if [[ -n "$PUBLIC_BASE_URL" ]]; then
  cmd+=(--public-base-url "$PUBLIC_BASE_URL")
fi

if [[ "$STRICT_VOICE_PROFILES" != "0" ]]; then
  cmd+=(--strict-voice-profiles)
fi

if [[ "$DISABLE_DOWNLOAD" != "0" ]]; then
  cmd+=(--disable-download)
fi

if [[ -n "$PRELOAD_MODELS" ]]; then
  read -r -a preload_array <<< "$PRELOAD_MODELS"
  cmd+=(--preload-models "${preload_array[@]}")
fi

echo "Starting CosyVoice API"
echo "  app_dir: $APP_DIR"
echo "  url: http://$HOST:$PORT"
echo "  voices_config: $VOICES_CONFIG"
echo "  reference_audio_dir: $REFER_AUDIO_DIR"
echo "  output_dir: $OUTPUT_DIR"
echo "  preload_models: ${PRELOAD_MODELS:-none}"

if [[ "$BACKGROUND" == "1" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  echo "  log_file: $LOG_FILE"
  nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
  echo "  pid: $!"
else
  exec "${cmd[@]}"
fi
