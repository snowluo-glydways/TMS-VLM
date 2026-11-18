# TMS-VLM — pedestrian 2D detection with Qwen3.8-27B

> **Live demo: https://snowluo-glydways.github.io/TMS-VLM/**
> 20 representative frames (baseline / MTP / JEV side by side) + latency charts for all 194 frames.
> **V2X public-dataset demo: https://snowluo-glydways.github.io/TMS-VLM/v2x/** (2x-upscaled runs, 200 images, accuracy vs ground truth + latency).
> Demo frames in the repo: [`docs/frames/baseline/`](docs/frames/baseline/), [`docs/frames/mtp/`](docs/frames/mtp/), [`docs/frames/jev/`](docs/frames/jev/) (full-res 3840x2160) and `docs/thumbs/<method>/` (640x360); V2X annotated images: `docs/v2x/vis/<baseline|mtp|jev>_up2x/` (all 200 per run, 1664x1664).

Pedestrian bounding-box detection on the TMS roadside camera log
`20261110_ATL175_DEV_MSF1_AP1_UPE_003_tms1000_calib`
(catalog artifact `a35726a9-c637-4ab2-bd09-366ba9e1c7fe`, 3840x2160, 15 fps, 193 s, 2902 frames),
using Qwen3.8-27B served by vLLM on **4x A100-80GB (tensor parallel = 4)**.

## Methods

| method | server | what the model outputs | confidence |
|---|---|---|---|
| `baseline` | plain vLLM | JSON `{"bbox_2d", "label", "confidence"}` per pedestrian (~300 tokens/frame) | self-reported by the VLM |
| `mtp` | vLLM + MTP speculative decoding (`{"method":"mtp","num_speculative_tokens":3}`) | same prompt and parsing as baseline | self-reported |
| `jev` (JEV-style + MTP) | MTP server | box coordinates only `[[x1,y1,x2,y2],...]` (~110 tokens) | each box gets a 1-token "is this a pedestrian? Yes/No" question; score = P(Yes)/(P(Yes)+P(No)) from the logprobs. Questions share the cached image prefix and run in parallel |

Coordinates are Qwen's 0-1000 relative scale. Thinking mode is disabled, greedy decoding.
Frames are resized to 1920 px wide before sending.

## Layout

```
data/camera_video.mp4          downloaded camera video
models/Qwen3.8-27B             HF weights
scripts/catalog_login.py       log in to catalog.tbdrobotics.com (Cognito hosted UI + PKCE)
scripts/serve.sh baseline|mtp  start vLLM on all 4 GPUs (port 8000)
scripts/run_all.sh             full pipeline: baseline -> mtp -> jev, render videos, comparison table
scripts/snapshot_check.sh      make / verify a manifest to prove a Thunder snapshot contains this project
tms_vlm/detect.py              detection (writes results/<method>/detections.jsonl + summary.json)
tms_vlm/render.py              annotated video / frames (results/<method>/annotated.mp4)
tms_vlm/compare.py             results/comparison.md
```

## Reproduce

```bash
cd ~/TMS-VLM
uv venv .venv --python 3.12 && . .venv/bin/activate
uv pip install vllm openai opencv-python-headless pillow numpy "huggingface_hub[hf_xet]"
sudo apt-get install -y ffmpeg fonts-dejavu-core
hf download Qwen/Qwen3.8-27B --local-dir models/Qwen3.8-27B
# data (needs catalog account):
CATALOG_USER=... CATALOG_PASS=... python scripts/catalog_login.py > .token.json   # then see scripts for the video URL
STRIDE=15 CONC=1 scripts/run_all.sh     # STRIDE=15 -> 1 fps; CONC=1 -> true single-frame latency
```

## Snapshot checklist (Thunder)

1. `scripts/snapshot_check.sh make` on this instance (writes `SNAPSHOT_MANIFEST.txt`).
2. On your laptop: `tnr snapshot create`, then `tnr snapshot list` until status is **READY**; the size should be roughly the disk usage (~70 GB with model + venv).
3. Create a test instance from the snapshot, run `cd ~/TMS-VLM && scripts/snapshot_check.sh verify` -> must print **ALL OK**.
4. Only then delete the original instance (and delete the test instance).
5. Additionally keep code in git and download `results/` locally — Thunder gives no durability guarantee for snapshots.

## Results (4x A100-80GB, TP=4, 194 frames = 1 fps over the 193 s log, concurrency 1)

| method | latency mean (s) | output tokens/frame | tokens/s | pedestrians/frame |
|---|---|---|---|---|
| baseline | 4.55 | 307 | 67.5 | 6.74 |
| mtp | 1.74 | 306 | 176.1 | 6.72 |
| jev (+MTP) | 1.31 | 135 | 158.9 (box stage) | 6.73 |

- MTP: same outputs as baseline (6.72 vs 6.74 pedestrians/frame), 2.6x faster.
- JEV: box-only output (assistant reply prefilled with `[[`, otherwise Qwen falls back to its verbose
  `{"bbox_2d", "label"}` format — see `results/_jev_v1_verbose_boxes`), plus parallel 1-token Yes/No verification.
  P(Yes) on real detections is 0.89-1.00; baseline self-reported confidence only takes ~16 coarse values (0.5, 0.8, 0.88, 0.9, 1.0 ...).

Outputs: `results/vis/<method>/frame_XXXXXX.jpg` (full-res annotated images, one per processed frame),
`results/<method>/annotated.mp4`, `results/<method>/detections.jsonl`, `results/comparison.md`.

## Demo page

**https://snowluo-glydways.github.io/TMS-VLM/** (GitHub Pages, served from `docs/`): 20 representative frames
with all three methods side by side, per-frame latency for all 194 frames, and the latency breakdown.
Rebuild with `.venv/bin/python scripts/make_demo.py` (picks the frames deterministically, copies them to
`docs/frames/<method>/`, writes `docs/data.json`).

## Public V2X dataset test

See [v2x_pubilc_test/README.md](v2x_pubilc_test/README.md): the same three methods on 200 images of the
Constellation roadside-camera dataset (accuracy: P/R/F1/AP for detection and 2D boxes; latency).
