# Qwen3.8-27B pedestrian detection on Constellation (V2X roadside cameras), images upscaled 2x

**Setup.** We evaluated on **Constellation**, a public V2X roadside-perception dataset. Its images come from
high-elevation infrastructure cameras at the COSMOS smart-intersection testbed in New York City
([Hugging Face](https://huggingface.co/datasets/mehmetkeremturkcan/constellation_urban_intersection_dataset),
[arXiv:2404.16944](https://arxiv.org/abs/2404.16944); CC BY-NC-SA 3.0).
- Images: we used the original camera view (the pre-perspective-transformation release) and sampled **200 images from
  the validation split** with a fixed random seed. Each image is 832 x 832 and was upscaled 2x before being sent to the model.
- Ground truth: the dataset's human-annotated 2D bounding boxes, with two classes (vehicle, pedestrian). Only the
  pedestrian class is evaluated, which gives **942 GT pedestrians**. Pedestrians are small: the median GT box is about 35 x 32 px.
- Methods: all three methods (baseline, MTP, JEV-style + MTP) use the same prompts and pipeline as the TMS camera test.
- Hardware and latency: inference ran on **4x A100-80GB with vLLM tensor parallelism of 4 (TP=4)**, with **one image
  in flight**, so the reported latency is the true end-to-end time for a single image. A warm-up request was excluded.

| Method | Precision | Recall | F1 | AP | Mean IoU | Latency mean (s) | Latency p90 (s) |
|---|---|---|---|---|---|---|---|
| Baseline | 0.839 | 0.614 | 0.709 | 0.566 | 0.456 | 2.75 | 5.38 |
| MTP | **0.849** | 0.613 | **0.712** | 0.572 | 0.458 | 1.27 | 2.06 |
| JEV-style + MTP | 0.530 | **0.645** | 0.582 | **0.616** | **0.464** | **1.20** | **1.81** |

Notes:
- How hits are counted: a prediction is a hit when its center falls inside a ground-truth box. The usual IoU >= 0.5
  rule doesn't fit this dataset, because its pedestrian boxes are drawn about 2x wider than the person.
- Mean IoU: average overlap between the model's box and the ground-truth box, over hits only.
- MTP: same accuracy as baseline, about 2.2x faster.
- JEV-style: best AP (ranking by confidence) and the fastest. It returns more boxes overall; keeping only boxes with
  score >= 0.5 raises its F1 to 0.704.
- Full metrics: comparison.md
