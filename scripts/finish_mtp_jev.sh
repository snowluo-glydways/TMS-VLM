#!/usr/bin/env bash
# (re)run JEV on the running MTP server, then render vis/videos for mtp + jev and write the comparison
set -e
cd "$(dirname "$0")/.." && . .venv/bin/activate
python -m tms_vlm.detect --method jev --stride 15 --concurrency 1 > logs/detect_jev.log 2>&1
for m in mtp jev; do python -m tms_vlm.render --method $m --vis; python -m tms_vlm.render --method $m; done > logs/render_mtp_jev.log 2>&1
python -m tms_vlm.compare
