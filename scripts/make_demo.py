"""Build the GitHub Pages demo (docs/): 20 representative frames x 3 methods + latency data.

Selection (deterministic): every frame where the three methods disagree on the pedestrian
count, the scene extremes (sparsest / densest / closest person / widest spread / lowest JEV
P(Yes)), then farthest-point sampling over (count, nearest box height, #near people,
horizontal spread, time) until 20 frames.

usage: .venv/bin/python scripts/make_demo.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["baseline", "mtp", "jev"]
N_DEMO = 20
OUT = ROOT / "docs"


def load():
    d = {}
    for m in METHODS:
        d[m] = {}
        for line in open(ROOT / f"results/{m}/detections.jsonl"):
            r = json.loads(line)
            d[m][r["frame"]] = r
    return d


def features(d, f):
    dets = d["jev"][f]["detections"]
    hs = [b["box"][3] - b["box"][1] for b in dets] or [0]
    xs = [(b["box"][0] + b["box"][2]) / 2 for b in dets] or [0]
    ns = [len(d[m][f]["detections"]) for m in METHODS]
    return dict(n=len(dets), maxh=max(hs), near=sum(h > 150 for h in hs),
                xspread=max(xs) - min(xs), disagree=max(ns) - min(ns),
                minp=min([b["score"] for b in dets] or [1.0]))


def select(d):
    fr = sorted(d["baseline"])
    F = {f: features(d, f) for f in fr}
    sel = [f for f in fr if F[f]["disagree"] > 0]
    sel += [max(fr, key=lambda f: F[f]["maxh"]), min(fr, key=lambda f: F[f]["n"]),
            max(fr, key=lambda f: F[f]["n"]), min(fr, key=lambda f: F[f]["minp"]),
            max(fr, key=lambda f: F[f]["xspread"])]
    sel = list(dict.fromkeys(sel))
    X = np.array([[F[f]["n"] / 9, F[f]["maxh"] / 600, F[f]["near"] / 3,
                   F[f]["xspread"] / 500, f / fr[-1]] for f in fr])
    idx = {f: i for i, f in enumerate(fr)}
    while len(sel) < N_DEMO:
        S = X[[idx[s] for s in sel]]
        dist = [np.min(np.linalg.norm(S - X[i], axis=1)) for i in range(len(fr))]
        sel.append(fr[int(np.argmax(dist))])
    return sorted(sel), F


def tags(F, f, extremes):
    t = []
    if F[f]["disagree"]:
        t.append("methods disagree")
    t += [name for name, ef in extremes.items() if ef == f]
    if F[f]["near"] >= 2:
        t.append("people near camera")
    elif F[f]["near"] == 0:
        t.append("all far away")
    return t


def main():
    d = load()
    sel, F = select(d)
    fr = sorted(d["baseline"])
    extremes = {"sparsest": min(fr, key=lambda f: F[f]["n"]),
                "most crowded": max(fr, key=lambda f: F[f]["n"]),
                "closest person": max(fr, key=lambda f: F[f]["maxh"]),
                "widest spread": max(fr, key=lambda f: F[f]["xspread"]),
                "lowest JEV P(Yes)": min(fr, key=lambda f: F[f]["minp"])}

    for m in METHODS:
        (OUT / "frames" / m).mkdir(parents=True, exist_ok=True)
        (OUT / "thumbs" / m).mkdir(parents=True, exist_ok=True)
        for f in sel:
            name = f"frame_{f:06d}.jpg"
            src = ROOT / "results/vis" / m / name
            shutil.copy2(src, OUT / "frames" / m / name)
            im = Image.open(src)
            im.resize((640, 360), Image.LANCZOS).save(OUT / "thumbs" / m / name, quality=82)

    summary = {m: json.load(open(ROOT / f"results/{m}/summary.json")) for m in METHODS}
    for m in METHODS:
        lat = np.array([d[m][f]["latency"] for f in fr])
        gen = np.array([d[m][f].get("gen_latency", d[m][f]["latency"]) for f in fr])
        summary[m].update(latency_p90_s=round(float(np.percentile(lat, 90)), 3),
                          latency_min_s=round(float(lat.min()), 3),
                          latency_max_s=round(float(lat.max()), 3),
                          gen_latency_mean_s=round(float(gen.mean()), 3))
    data = {
        "methods": METHODS,
        "summary": summary,
        "timeline": [{"frame": f, "t": d["baseline"][f]["t"],
                      **{m: d[m][f]["latency"] for m in METHODS},
                      **{f"{m}_n": len(d[m][f]["detections"]) for m in METHODS}} for f in fr],
        "demo": [{"frame": f, "t": d["baseline"][f]["t"], "tags": tags(F, f, extremes),
                  **{m: {"n": len(d[m][f]["detections"]), "latency": d[m][f]["latency"],
                         "gen_latency": d[m][f].get("gen_latency", d[m][f]["latency"]),
                         "tokens": d[m][f]["completion_tokens"],
                         "scores": [b["score"] for b in d[m][f]["detections"]]}
                     for m in METHODS}} for f in sel],
    }
    (OUT / "data.json").write_text(json.dumps(data))
    print("selected", sel)


if __name__ == "__main__":
    main()
