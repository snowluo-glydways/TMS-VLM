"""Build the V2X demo page data (docs/v2x/): the 2x-upscaled runs only.

Copies vis/<run>/*.jpg of baseline_up2x / mtp_up2x / jev_up2x to docs/v2x/vis/<run>/, makes 400 px
thumbnails, and writes docs/v2x/data.json with metrics + per-image stats (GT count, predictions,
center-hit / IoU>=0.3 true positives, latency, tokens).

usage (from this folder): ../.venv/bin/python make_demo.py
"""
import json
import shutil

from PIL import Image

from run_eval import DATA, HERE, load_gt, match

RUNS = ["baseline_up2x", "mtp_up2x", "jev_up2x"]
OUT = HERE.parent / "docs" / "v2x"


def main():
    preds = {r: {x["image"]: x for x in map(json.loads, open(HERE / f"results/{r}/predictions.jsonl"))} for r in RUNS}
    images = json.load(open(HERE / "subset.json"))["images"]
    metrics = {r: json.load(open(HERE / f"results/{r}/metrics.json")) for r in RUNS}

    rows = []
    for name in images:
        gts = load_gt(DATA / "images" / "val" / name)
        row = {"image": name, "gt": len(gts)}
        for r in RUNS:
            p = preds[r][name]
            dets = p["detections"]
            row[r] = {"n": len(dets),
                      "tp_c": sum(x[1] for x in match(dets, gts, "center")),
                      "tp_3": sum(x[1] for x in match(dets, gts, 0.3)),
                      "lat": p["latency"], "gen": p.get("gen_latency", p["latency"]),
                      "tok": p["completion_tokens"]}
        rows.append(row)

    for r in RUNS:
        (OUT / "vis" / r).mkdir(parents=True, exist_ok=True)
        (OUT / "thumbs" / r).mkdir(parents=True, exist_ok=True)
        for name in images:
            src = HERE / "vis" / r / name
            shutil.copy2(src, OUT / "vis" / r / name)
            im = Image.open(src).convert("RGB")
            im.thumbnail((400, 400), Image.LANCZOS)
            im.save(OUT / "thumbs" / r / name, quality=80)

    (OUT / "data.json").write_text(json.dumps({"runs": RUNS, "metrics": metrics, "images": rows}))
    print(f"{len(rows)} images x {len(RUNS)} runs -> {OUT}")


if __name__ == "__main__":
    main()
