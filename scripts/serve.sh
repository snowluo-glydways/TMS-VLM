#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for Qwen3.8-27B on ALL 4 GPUs (tensor parallel = 4).
#   scripts/serve.sh baseline   -> plain decoding
#   scripts/serve.sh mtp        -> built-in MTP speculative decoding (3 draft tokens); also used by JEV-style
# Only one server runs at a time (both want all 4 GPUs); always on port 8000.
set -euo pipefail
MODE=${1:?usage: serve.sh baseline|mtp}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/.venv/bin/activate"
MODEL=${MODEL:-$ROOT/models/Qwen3.8-27B}
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
TP=${TP:-4}

COMMON=(--served-model-name qwen3.8-27b --port "${PORT:-8000}"
        --tensor-parallel-size "$TP" --mm-encoder-tp-mode data
        --max-model-len 16384 --gpu-memory-utilization 0.90
        --enable-prefix-caching --reasoning-parser qwen3
        --limit-mm-per-prompt '{"image":1,"video":0}' --max-logprobs 20)

case "$MODE" in
  baseline) EXTRA=() ;;
  mtp)      EXTRA=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}') ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

exec vllm serve "$MODEL" "${COMMON[@]}" "${EXTRA[@]}"
