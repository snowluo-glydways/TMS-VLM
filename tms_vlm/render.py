"""Render detections.jsonl onto the camera video (or export single annotated frames).

Every output frame shows the boxes of the most recent processed frame at or before it.
  python -m tms_vlm.render --method baseline              -> results/baseline/annotated.mp4
  python -m tms_vlm.render --method jev --frames 256 2792 -> results/jev/frame_000256.jpg ...
"""
import argparse, json, subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLES = {"baseline": "Qwen3.8-27B | pedestrian detection",
          "mtp": "Qwen3.8-27B +MTP | pedestrian detection",
          "jev": "Qwen3.8-27B (jev-style) +MTP | pedestrian detection"}
GREEN, YELLOW, ORANGE = (0, 230, 0), (255, 220, 0), (255, 140, 0)
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]


def font(size):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def color_and_label(method, score):
    # Same levels / thresholds / colors for every method; only the score source differs (see legend).
    c = GREEN if score >= 0.80 else YELLOW if score >= 0.60 else ORANGE
    return c, f"{score:.2f}"


def legend_lines(method):
    head = "#N  = pedestrian number N in this frame (model output order; not the same person across frames)"
    if method == "jev":
        src = ["0.xx = P(Yes) that the box is a pedestrian. How it is computed: for each box the model is asked",
               "   \"Is the region [x1, y1, x2, y2] a pedestrian? Answer Yes or No\" (same image, 1 output token);",
               "   0.xx = P(Yes) / (P(Yes) + P(No)), from that token's probabilities (not text written by the model)"]
    else:
        src = ["0.xx = model confidence that the box is a pedestrian (0-1, self-reported by the VLM)"]
    return ["Legend   label = '#N 0.xx'", head, *src,
            [("box color:  ", None), ("green", GREEN), (" >= 0.80    ", None), ("yellow", YELLOW),
             (" 0.60-0.80    ", None), ("orange", ORANGE), (" < 0.60", None)]]


def draw(frame_bgr, rec, frame_idx, fps, method, hud=None):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    W, H = img.size
    u = W / 2000.0  # layout unit, tuned on 2000-px-wide output
    d = ImageDraw.Draw(img, "RGBA")
    f_lbl, f_hud, f_leg = font(round(20 * u)), font(round(28 * u)), font(round(20 * u))

    dets = rec["detections"] if rec else []
    placed = []  # label rectangles already drawn; move new labels up to avoid overlap
    def hits(r):
        return any(r[0] < p[2] and p[0] < r[2] and r[1] < p[3] and p[1] < r[3] for p in placed)
    for i, det in enumerate(dets, 1):
        x1, y1, x2, y2 = [v / 1000.0 * s for v, s in zip(det["box"], (W, H, W, H))]
        c, lab = color_and_label(method, det["score"])
        d.rectangle([x1, y1, x2, y2], outline=c, width=max(2, round(3 * u)))
        text = f"#{i} {lab}"
        tb = d.textbbox((0, 0), text, font=f_lbl)
        tw, th = tb[2] - tb[0] + 6 * u, tb[3] - tb[1] + 8 * u
        tx = min(x1, W - tw)
        ty = y1 - th if y1 - th >= 0 else y1
        while hits((tx, ty, tx + tw, ty + th)) and ty - th >= 0:
            ty -= th + 2 * u
        if ty + th < y1 - 1:  # leader line from moved label down to its box
            d.line([x1 + 1, ty + th, x1 + 1, y1], fill=c, width=max(1, round(2 * u)))
        placed.append((tx, ty, tx + tw, ty + th))
        d.rectangle([tx, ty, tx + tw, ty + th], fill=c)
        d.text((tx + 3 * u, ty + 2 * u - tb[1]), text, font=f_lbl, fill=(0, 0, 0))

    # HUD
    lat = f"{rec['latency']:.2f}s" if rec else "-"
    hud = hud or [TITLES[method], f"frame {frame_idx}  t={frame_idx / fps:.2f}s  pedestrians: {len(dets)}  latency: {lat}"]
    lh = 40 * u
    hw = max(d.textlength(s, font=f_hud) for s in hud) + 24 * u
    d.rectangle([0, 0, hw, lh * len(hud) + 14 * u], fill=(30, 30, 30, 190))
    for k, s in enumerate(hud):
        d.text((12 * u, 8 * u + k * lh), s, font=f_hud, fill=(255, 255, 255))

    # legend
    lines = legend_lines(method)
    lh = 32 * u
    def width(l):
        return sum(d.textlength(t, font=f_leg) for t, _ in l) if isinstance(l, list) else d.textlength(l, font=f_leg)
    lw = max(width(l) for l in lines) + 24 * u
    top = H - lh * len(lines) - 14 * u
    d.rectangle([0, top, lw, H], fill=(30, 30, 30, 190))
    for k, l in enumerate(lines):
        x, y = 10 * u, top + 8 * u + k * lh
        for t, c in (l if isinstance(l, list) else [(l, None)]):
            d.text((x, y), t, font=f_leg, fill=c or (255, 255, 255))
            x += d.textlength(t, font=f_leg)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(TITLES), required=True)
    ap.add_argument("--video", default="data/camera_video.mp4")
    ap.add_argument("--results", default=None, help="dir with detections.jsonl (default results/<method>)")
    ap.add_argument("--width", type=int, default=1920, help="output width")
    ap.add_argument("--frames", type=int, nargs="*", help="only export these frames as JPEGs")
    ap.add_argument("--vis", action="store_true",
                    help="export every processed frame as a JPEG to results/vis/<method>/ (full resolution)")
    args = ap.parse_args()

    rdir = Path(args.results or f"results/{args.method}")
    recs = {r["frame"]: r for r in map(json.loads, open(rdir / "detections.jsonl"))}
    keys = sorted(recs)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    W, H = args.width, round(H0 * args.width / W0 / 2) * 2

    def latest(i):
        import bisect
        j = bisect.bisect_right(keys, i) - 1
        return recs[keys[j]] if j >= 0 else None

    if args.vis:
        vdir = Path("results/vis") / args.method
        vdir.mkdir(parents=True, exist_ok=True)
        want, i = set(keys), 0
        while want:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                cv2.imwrite(str(vdir / f"frame_{i:06d}.jpg"), draw(fr, recs[i], i, fps, args.method),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                want.discard(i)
            i += 1
        print(f"{len(keys)} images -> {vdir}")
        return

    if args.frames:
        for fi in args.frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            out = draw(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), latest(fi), fi, fps, args.method)
            p = rdir / f"frame_{fi:06d}.jpg"; cv2.imwrite(str(p), out); print(p)
        return

    last = keys[-1] if keys else 0
    out_path = rdir / "annotated.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "fast",
                           "-crf", "23", "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    i = 0
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        ff.stdin.write(draw(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), latest(i), i, fps, args.method).tobytes())
        i += 1
    ff.stdin.close(); ff.wait()
    print(out_path)


if __name__ == "__main__":
    main()
