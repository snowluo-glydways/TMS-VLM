#!/usr/bin/env bash
# Full V2X public test: 3 methods x {native, 2x upscale} on 4 GPUs (one vLLM server at a time), metrics + vis.
set -e
cd "$(dirname "$0")"; PY=../.venv/bin/python
wait_server() { sleep 20; until curl -sf localhost:8000/v1/models > /dev/null; do pgrep -f "[v]llm serve" > /dev/null || { echo "server died"; exit 1; }; sleep 5; done; }
stop_server() { pkill -f "[v]llm serve" || true; while pgrep -f "[v]llm serve" > /dev/null; do sleep 2; done; sleep 5; }
[ -f subset.json ] || $PY run_eval.py prepare --n 200
for srv in baseline mtp; do
  if [ "$srv" = baseline ]; then methods="baseline"; else methods="mtp jev"; fi
  need=0; for m in $methods; do for up in 1 2; do r=$m; [ $up = 2 ] && r=${m}_up2x; [ -s results/$r/predictions.jsonl ] && [ $(wc -l < results/$r/predictions.jsonl) -ge 200 ] || need=1; done; done
  [ $need = 1 ] || continue
  curl -sf localhost:8000/v1/models | grep -q . && [ "$(cat .server 2>/dev/null)" = $srv ] || { stop_server; setsid nohup ../scripts/serve.sh $srv > ../logs/serve_${srv}_v2x.log 2>&1 < /dev/null & echo $srv > .server; wait_server; }
  for m in $methods; do for up in 1 2; do
    r=$m; [ $up = 2 ] && r=${m}_up2x
    [ -s results/$r/predictions.jsonl ] && [ $(wc -l < results/$r/predictions.jsonl) -ge 200 ] && continue
    $PY run_eval.py detect --method $m --upscale $up > logs_$r.txt 2>&1
  done; done
done
stop_server; rm -f .server
for m in baseline mtp jev; do $PY run_eval.py vis --method $m; $PY run_eval.py vis --method $m --run ${m}_up2x; done
$PY run_eval.py evaluate
