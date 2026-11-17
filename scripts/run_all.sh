#!/usr/bin/env bash
# Run all three methods end-to-end on 4 GPUs (one server at a time) and render videos.
#   STRIDE=15 CONC=1 scripts/run_all.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
source .venv/bin/activate
STRIDE=${STRIDE:-15}; CONC=${CONC:-1}
mkdir -p logs results

start_server() {
  pkill -f "[v]llm serve" || true; sleep 15
  nohup scripts/serve.sh "$1" > "logs/serve_$1.log" 2>&1 &
  until curl -sf localhost:8000/v1/models > /dev/null; do
    pgrep -f "[v]llm serve" > /dev/null || { echo "server $1 died, see logs/serve_$1.log"; exit 1; }
    sleep 5
  done
  echo "server $1 ready"
}

run() {  # method
  python -m tms_vlm.detect --method "$1" --stride "$STRIDE" --concurrency "$CONC" 2>&1 | tee "logs/detect_$1.log"
  python -m tms_vlm.render --method "$1"
}

[ -f results/baseline/summary.json ] || { start_server baseline; run baseline; }
start_server mtp
[ -f results/mtp/summary.json ] || run mtp
[ -f results/jev/summary.json ] || run jev
pkill -f "[v]llm serve" || true
python -m tms_vlm.compare
