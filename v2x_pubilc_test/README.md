# V2X public-dataset test — Qwen3.8-27B pedestrian 2D detection

The same three methods as the TMS pipeline (`../tms_vlm/detect.py`: **baseline**, **MTP**, **JEV-style + MTP**,
identical prompts, requests and parsing) are evaluated for detection accuracy, 2D-box accuracy and latency on a
public roadside (infrastructure-side) camera dataset.

## Dataset

**Constellation** — high-elevation roadside cameras at the COSMOS smart-intersection testbed (NYC).
[HF](https://huggingface.co/datasets/mehmetkeremturkcan/constellation_urban_intersection_dataset) ·
[arXiv 2404.16944](https://arxiv.org/abs/2404.16944) · license CC BY-NC-SA 3.0 (research only).

- Used: `constellation_pre_transformation.zip` (original camera view), **val split**, **200 images** chosen with a
  fixed random seed (`subset.json`), **942 GT pedestrians**. 832x832 images; pedestrians are small (median GT box ~35x32 px).
- GT: YOLO boxes, classes vehicle / pedestrian; **only pedestrians are evaluated**.
- Not used: TUMTraf / A9 (registration required, 3D labels), DAIR-V2X and Rope3D (application required).

## Runs

- `baseline`, `mtp`, `jev`: image sent as-is (exactly like the TMS pipeline).
- `*_up2x`: image enlarged 2x (bicubic) before sending — common small-object trick, reported separately.
- Hardware: 4x A100-80GB, vLLM tensor-parallel = 4, one image in flight (true per-image latency), warm-up excluded.

## Results (200 images, 942 pedestrians) — full tables in `results/comparison.md`

| run | AP@IoU0.5 | F1@IoU0.5 | AP@IoU0.3 | F1@IoU0.3 | R@center | F1@center | latency mean / p90 (s) | out tokens |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.057 | 0.151 | 0.385 | 0.512 | 0.613 | 0.589 | 3.33 / 6.30 | 232 |
| mtp | 0.050 | 0.143 | 0.346 | 0.497 | 0.610 | 0.577 | 1.22 / 2.09 | 241 |
| jev | 0.040 | 0.084 | 0.432 | 0.282 | 0.649 | 0.335 | 2.76 / 8.70 | 325 |
| baseline_up2x | 0.104 | 0.244 | 0.508 | 0.657 | 0.614 | 0.709 | 2.75 / 5.38 | 161 |
| mtp_up2x | 0.104 | 0.244 | 0.517 | 0.663 | 0.613 | 0.712 | 1.27 / 2.06 | 159 |
| jev_up2x | 0.103 | 0.222 | **0.552** | 0.537 | **0.645** | 0.582 | **1.20 / 1.81** | 113 |

`center` = a prediction counts as a hit if its center lies inside a GT box (independent of how wide boxes are drawn).

## Findings

1. **IoU-0.5 numbers are dominated by an annotation-convention mismatch, not by mislocalisation.**
   Constellation draws pedestrians as roughly square boxes (~2x the person's width); Qwen draws tight body boxes:
   predicted width = 0.47-0.52x GT, height = 0.96-0.98x GT, center offset ~0. Even a perfectly placed tight box then
   has IoU ~0.5, so AP75 ~ 0. With IoU>=0.3 or center-hit matching, recall is ~0.53-0.65 and precision up to ~0.85.
2. **Small objects are the real limit.** ~35-40% of pedestrians (~30 px) are missed in every run. Enlarging the image
   2x mostly removes false positives (precision@center 0.57 -> 0.84) and stabilises output; recall barely changes.
3. **MTP**: same accuracy as baseline (differences are within greedy-decoding noise), **2.2-2.7x lower latency**.
4. **JEV**: the P(Yes) score ranks boxes best (highest AP@IoU0.3 / AP@center in both settings), and filtering at
   score >= 0.5 brings its F1 to the level of baseline (table C). But on the native 832-px images the box-only output
   sometimes falls into repetition loops (22/200 images with >30 boxes, up to 113), which inflates latency (p90 8.7 s)
   and false positives. With 2x upscaling this mostly disappears (3/200) and JEV becomes the **fastest** run
   (1.20 s mean, 113 output tokens).

## Files

```
run_eval.py            prepare / detect / evaluate / vis
run_all.sh             full run (switches the 4-GPU vLLM server baseline -> mtp automatically)
subset.json            the 200 evaluated images
results/<run>/predictions.jsonl, metrics.json
results/comparison.md  tables A (standard IoU0.5), B (convention-robust), C (score filtering), D (latency)
vis/<run>/*.jpg        annotated images: predictions colored as in the TMS vis (green >=0.80, yellow 0.60-0.80,
                       orange <0.60, '#N 0.xx'), ground truth as thin cyan boxes
data/raw/              downloaded zips;  data/constellation_pre/  extracted val split
```

Reproduce: `./run_all.sh` (from this folder; needs `../models/Qwen3.8-27B` and `../.venv`).
