#!/usr/bin/env bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$ROOT/.venv-attack/bin/python" ] || { echo "attack_env.sh: $ROOT/.venv-attack missing; run make env-attack" >&2; exit 2; }
if [ -n "${AAC_ATTACK_HOME:-}" ]; then
 case "$AAC_ATTACK_HOME" in
 /*) ;;
 *) echo "attack_env.sh: AAC_ATTACK_HOME must be an absolute path" >&2; exit 2 ;;
 esac
 case "$AAC_ATTACK_HOME" in
 *[!A-Za-z0-9._/-]*) echo "attack_env.sh: AAC_ATTACK_HOME may hold only [A-Za-z0-9._/-]" >&2; exit 2 ;;
 esac
 [ -d "$AAC_ATTACK_HOME" ] || { echo "attack_env.sh: AAC_ATTACK_HOME is not a directory" >&2; exit 2; }
fi
case "${AAC_THREADS:-}" in
 '') ;;
 *[!0-9]*|0) echo "attack_env.sh: AAC_THREADS must be a positive integer" >&2; exit 2 ;;
esac
exec env -i \
 PATH="$ROOT/.venv-attack/bin:/usr/bin:/bin" \
 HOME="${AAC_ATTACK_HOME:-$HOME}" \
 ${AAC_ATTACK_HOME:+HF_DATASETS_CACHE=$AAC_ATTACK_HOME/.hf-datasets} \
 ${AAC_THREADS:+OMP_NUM_THREADS=$AAC_THREADS} \
 ${AAC_THREADS:+MKL_NUM_THREADS=$AAC_THREADS} \
 LANG=C.UTF-8 \
 LC_ALL=C.UTF-8 \
 HF_HOME="$ROOT/.hf" \
 HF_HUB_OFFLINE=1 \
 TRANSFORMERS_OFFLINE=1 \
 HF_HUB_DISABLE_TELEMETRY=1 \
 LAMBO_DIR="$ROOT/.hf/lambo" \
 TF_CPP_MIN_LOG_LEVEL=3 \
 CUDA_VISIBLE_DEVICES="${AAC_GPU:-}" \
 PYTHONPATH="$ROOT/third_party/BODEGA" \
 PYTHONNOUSERSITE=1 \
 PYTHONDONTWRITEBYTECODE=1 \
 PYTHONHASHSEED=0 \
 TOKENIZERS_PARALLELISM=false \
 "$@"
