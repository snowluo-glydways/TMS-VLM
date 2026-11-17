"""Accuracy + latency of the three Qwen3.8-27B pedestrian-detection methods on a public V2X roadside dataset.

Dataset: Constellation (COSMOS smart-intersection testbed, high-elevation roadside cameras, NYC),
https://huggingface.co/datasets/mehmetkeremturkcan/constellation_urban_intersection_dataset (CC BY-NC-SA 3.0).
We use the original camera view (constellation_pre_transformation.zip), val split, a fixed random subset.
GT: YOLO boxes, classes {0: vehicle, 1: pedestrian}; only pedestrians are evaluated.

The prompts / requests / parsing are exactly those of ../tms_vlm/detect.py (baseline, mtp, jev).

  python run_eval.py prepare --n 200            # pick subset -> subset.json
  python run_eval.py detect --method baseline   # needs ../scripts/serve.sh baseline  (mtp/jev: serve.sh mtp)
  python run_eval.py evaluate                   # metrics for every method that has predictions
  python run_eval.py vis --method jev           # annotated images -> vis/<method>/
"""
import argparse, asyncio, json, random, sys, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tms_vlm.detect import detect_baseline, detect_jev, encode_frame  # noqa: E402
from tms_vlm import render  # noqa: E402

DATA = HERE / "data" / "constellation_pre"
PED = 1


def load_gt(img_path):
    lab = DATA / "labels" / "val" / (Path(img_path).stem + ".txt")
    boxes = []
    for line in open(lab):
        p = line.split()
        if len(p) == 5 and int(float(p[0])) == PED:
            x, y, w, h = map(float, p[1:])
            boxes.append([(x - w / 2) * 1000, (y - h / 2) * 1000, (x + w / 2) * 1000, (y + h / 2) * 1000])
    return boxes


# ---------------------------------------------------------------- prepare
def prepare(args):
    imgs = sorted(p.name for p in (DATA / "images" / "val").glob("*.jpg"))
    random.Random(args.seed).shuffle(imgs)
    sub = sorted(imgs[: args.n])
    n_ped = sum(len(load_gt(DATA / "images" / "val" / f)) for f in sub)
    json.dump({"seed": args.seed, "images": sub}, open(HERE / "subset.json", "w"), indent=1)
    print(f"subset: {len(sub)} images, {n_ped} GT pedestrians -> subset.json")


# ---------------------------------------------------------------- detect
async def detect(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY", timeout=600)
    model = (await client.models.list()).data[0].id
    fn = detect_jev if args.method == "jev" else detect_baseline
    sub = json.load(open(HERE / "subset.json"))["images"]
    run = args.method if args.upscale == 1 else f"{args.method}_up{args.upscale:g}x"
    out = HERE / "results" / run

    def load(name):
        im = cv2.imread(str(DATA / "images" / "val" / name))
        if args.upscale != 1:  # optional: enlarge small roadside images before sending (boxes are relative 0-1000)
            im = cv2.resize(im, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)
        return encode_frame(im, args.max_side)

    out.mkdir(parents=True, exist_ok=True)
    f_out = open(out / "predictions.jsonl", "w")
    # warm-up (not timed): first request compiles / allocates
    await fn(client, model, load(sub[0]), 2048)
    for i, name in enumerate(sub):
        img = load(name)
        t0 = time.perf_counter()
        dets, raw, ntok, nprompt, t_gen = await fn(client, model, img, 2048)
        lat = time.perf_counter() - t0
        rec = {"image": name, "latency": round(lat, 3), "gen_latency": round(t_gen, 3), "completion_tokens": ntok,
               "prompt_tokens": nprompt, "detections": dets, "raw": raw}
        f_out.write(json.dumps(rec) + "\n"); f_out.flush()
        print(f"[{run}] {i + 1:3d}/{len(sub)} {name} pred={len(dets)} gt={len(load_gt(DATA / 'images' / 'val' / name))} "
              f"lat={lat:5.2f}s tok={ntok}", flush=True)


# ---------------------------------------------------------------- evaluate
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(preds, gts, thr):
    """Greedy matching in descending score order (COCO/VOC style). Returns per-pred (score, tp, iou).
    thr = IoU threshold, or "center": a prediction hits a GT box if its center lies inside it
    (insensitive to box-drawing convention; nearest-center unused GT is taken)."""
    used, res = set(), []
    for p in sorted(preds, key=lambda d: -d["score"]):
        b = p["box"]
        if thr == "center":
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            cand = [(abs(cx - (g[0] + g[2]) / 2) + abs(cy - (g[1] + g[3]) / 2), j) for j, g in enumerate(gts)
                    if j not in used and g[0] <= cx <= g[2] and g[1] <= cy <= g[3]]
            if cand:
                j = min(cand)[1]; used.add(j); res.append((p["score"], 1, iou(b, gts[j])))
            else:
                res.append((p["score"], 0, 0.0))
            continue
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j not in used:
                v = iou(b, g)
                if v > best:
                    best, bj = v, j
        if best >= thr:
            used.add(bj); res.append((p["score"], 1, best))
        else:
            res.append((p["score"], 0, best))
    return res


def prf(res, n_gt, min_score=0.0):
    r = [x for x in res if x[0] >= min_score]
    tp = sum(x[1] for x in r); P = tp / len(r) if r else 0.0; R = tp / n_gt if n_gt else 0.0
    return round(P, 4), round(R, 4), round(2 * P * R / (P + R), 4) if P + R else 0.0


def ap(all_res, n_gt):
    """VOC all-point interpolated AP. Ties in score keep model output order (stable sort)."""
    if not all_res:
        return 0.0
    r = sorted(all_res, key=lambda x: -x[0])
    tp = np.cumsum([x[1] for x in r]); fp = np.cumsum([1 - x[1] for x in r])
    rec = tp / max(n_gt, 1); prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(args):
    rows = []
    order = {"baseline": 0, "mtp": 1, "jev": 2}
    runs = sorted((p.parent.name for p in (HERE / "results").glob("*/predictions.jsonl")),
                  key=lambda r: ("_up" in r, order.get(r.split("_")[0], 9)))
    for m in runs:
        recs = [json.loads(l) for l in open(HERE / "results" / m / "predictions.jsonl")]
        n_gt = 0; R = {k: [] for k in ["0.5", "0.75", "0.3", "center"]}; res_ap = {}; cnt_err = []
        wr, hr = [], []
        for r in recs:
            gts = load_gt(DATA / "images" / "val" / r["image"]); n_gt += len(gts)
            for k in R:
                R[k] += match(r["detections"], gts, "center" if k == "center" else float(k))
            for t in np.arange(0.5, 0.96, 0.05):
                res_ap.setdefault(round(t, 2), []).extend(match(r["detections"], gts, t))
            cnt_err.append(abs(len(r["detections"]) - len(gts)))
            for d in r["detections"]:  # size bias vs best-overlapping GT
                g = max(gts, key=lambda g: iou(d["box"], g), default=None)
                if g is not None and iou(d["box"], g) >= 0.1:
                    wr.append((d["box"][2] - d["box"][0]) / (g[2] - g[0])); hr.append((d["box"][3] - d["box"][1]) / (g[3] - g[1]))
        lats = np.array([r["latency"] for r in recs])
        row = {"method": m, "images": len(recs), "gt_peds": n_gt, "pred_peds": len(R["0.5"])}
        for k, name in [("0.5", "IoU0.5"), ("0.3", "IoU0.3"), ("center", "center")]:
            P_, R_, F_ = prf(R[k], n_gt); row[f"P@{name}"], row[f"R@{name}"], row[f"F1@{name}"] = P_, R_, F_
            row[f"F1@{name},score>=0.5"] = prf(R[k], n_gt, 0.5)[2]
            row[f"AP@{name}"] = round(ap(R[k], n_gt), 4)
        row.update({"AP75": round(ap(R["0.75"], n_gt), 4),
                    "AP50:95": round(float(np.mean([ap(v, n_gt) for v in res_ap.values()])), 4),
                    "mean_IoU_center_hits": round(float(np.mean([x[2] for x in R["center"] if x[1]])), 4),
                    "pred/GT_width": round(float(np.median(wr)), 2) if wr else None,
                    "pred/GT_height": round(float(np.median(hr)), 2) if hr else None,
                    "count_MAE": round(float(np.mean(cnt_err)), 3),
                    "imgs_with_>30_preds": sum(len(r["detections"]) > 30 for r in recs),
                    "latency_mean_s": round(float(lats.mean()), 3), "latency_p50_s": round(float(np.median(lats)), 3),
                    "latency_p90_s": round(float(np.percentile(lats, 90)), 3),
                    "output_tokens_mean": round(float(np.mean([r["completion_tokens"] for r in recs])), 1),
                    "tokens_per_s": round(float(np.mean([r["completion_tokens"] / r["gen_latency"] for r in recs])), 1)})
        rows.append(row)
        (HERE / "results" / m / "metrics.json").write_text(json.dumps(row, indent=2))

    def table(title, cols):
        md = f"### {title}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        return md + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows) + "\n\n"
    md = table("A. Standard detection metrics (IoU >= 0.5)",
               ["method", "gt_peds", "pred_peds", "P@IoU0.5", "R@IoU0.5", "F1@IoU0.5", "AP@IoU0.5", "AP75", "AP50:95"])
    md += table("B. Box-convention-robust metrics (GT boxes are ~square, ~2x wider than the person)",
                ["method", "P@IoU0.3", "R@IoU0.3", "F1@IoU0.3", "AP@IoU0.3", "P@center", "R@center", "F1@center",
                 "AP@center", "mean_IoU_center_hits", "pred/GT_width", "pred/GT_height"])
    md += table("C. Score filtering (keep boxes with score >= 0.5) and counting",
                ["method", "F1@IoU0.5,score>=0.5", "F1@IoU0.3,score>=0.5", "F1@center,score>=0.5", "count_MAE",
                 "imgs_with_>30_preds"])
    md += table("D. Latency (4x A100 TP=4, 1 image in flight)",
                ["method", "latency_mean_s", "latency_p50_s", "latency_p90_s", "output_tokens_mean", "tokens_per_s"])
    (HERE / "results" / "comparison.md").write_text(md)
    print(md)


# ---------------------------------------------------------------- vis
def vis(args):
    from PIL import Image, ImageDraw
    args.run = args.run or args.method
    f = HERE / "results" / args.run / "predictions.jsonl"
    out = HERE / "vis" / args.run
    out.mkdir(parents=True, exist_ok=True)
    titles = {"baseline": "Qwen3.8-27B", "mtp": "Qwen3.8-27B +MTP", "jev": "Qwen3.8-27B (jev-style) +MTP"}
    for r in map(json.loads, open(f)):
        img = cv2.imread(str(DATA / "images" / "val" / r["image"]))
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        gts = load_gt(DATA / "images" / "val" / r["image"])
        tp = sum(x[1] for x in match(r["detections"], gts, 0.5))
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); W, H = pil.size
        d = ImageDraw.Draw(pil, "RGBA")
        for g in gts:  # GT: thin cyan
            d.rectangle([g[0] / 1000 * W, g[1] / 1000 * H, g[2] / 1000 * W, g[3] / 1000 * H], outline=(0, 255, 255), width=1)
        pil = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        # predictions, HUD and legend with the same style / colors / thresholds as the TMS vis
        rec = {"detections": r["detections"], "latency": r["latency"]}
        run_tag = "" if args.run == args.method else f" [{args.run}]"
        hud = [f"{titles[args.method]}{run_tag} | pedestrian detection | {r['image']}",
               f"pred: {len(r['detections'])}  GT: {len(gts)}  TP@IoU0.5: {tp}  latency: {r['latency']:.2f}s",
               "thin cyan box = ground-truth pedestrian (Constellation V2X roadside)"]
        frame = render.draw(pil, rec, 0, 1.0, args.method, hud=hud)
        cv2.imwrite(str(out / r["image"]), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {out}")


def main():
    ap_ = argparse.ArgumentParser()
    sp = ap_.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); p.add_argument("--n", type=int, default=200); p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("detect"); p.add_argument("--method", choices=["baseline", "mtp", "jev"], required=True)
    p.add_argument("--port", type=int, default=8000); p.add_argument("--max-side", type=int, default=1920)
    p.add_argument("--upscale", type=float, default=1.0, help="1 = same as TMS pipeline; 2 = enlarge image 2x first")
    sp.add_parser("evaluate")
    p = sp.add_parser("vis"); p.add_argument("--method", choices=["baseline", "mtp", "jev"], required=True)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--run", default=None, help="results/<run> to draw, e.g. jev_up2x (default = method)")
    a = ap_.parse_args()
    if a.cmd == "detect":
        asyncio.run(detect(a))
    else:
        {"prepare": prepare, "evaluate": evaluate, "vis": vis}[a.cmd](a)


if __name__ == "__main__":
    main()
