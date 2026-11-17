"""Pedestrian 2D-box detection on the TMS camera video with Qwen3.8-27B served by vLLM.

Methods
  baseline : one request per frame; model outputs JSON boxes + self-reported confidence. Plain server.
  mtp      : identical prompt/pipeline, sent to the server started with MTP speculative decoding.
  jev      : (JEV-style, on the MTP server) model outputs box coordinates only; every box then gets a
             one-token "is this a pedestrian?" question and confidence = P(Yes) / (P(Yes) + P(No))
             from the token logprobs. A frame's questions share the image prefix (prefix cache) and
             run in parallel.

Output: results/<run>/detections.jsonl, one line per processed frame.
"""
import argparse, asyncio, base64, json, math, re, time
from pathlib import Path

import cv2
from openai import AsyncOpenAI

PORTS = {"baseline": 8000, "mtp": 8000, "jev": 8000}  # one 4-GPU server at a time

PROMPT_BASELINE = (
    "Detect every pedestrian (any person on foot: standing, walking, running, partly visible or far away) "
    "in this traffic-camera image. Return a JSON array where each element is "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "pedestrian", "confidence": c}. '
    "Coordinates are relative to the image on a 0-1000 scale (top-left 0,0). "
    "c is your confidence (0-1) that the box contains a pedestrian. "
    "Return [] if there are none. Output only the JSON array."
)
PROMPT_JEV_BOXES = (
    "Detect every pedestrian (any person on foot: standing, walking, running, partly visible or far away) "
    "in this traffic-camera image. Output only a JSON array of boxes [[x1, y1, x2, y2], ...] "
    "with coordinates on a 0-1000 scale relative to the image (top-left 0,0). Output [] if none."
)
PROMPT_JEV_VERIFY = (
    "Look at the region with 0-1000 relative coordinates [{x1}, {y1}, {x2}, {y2}] in this image. "
    "Is that region a pedestrian (a person on foot)? Answer with exactly one word: Yes or No."
)

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1}


def encode_frame(frame, max_side):
    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def parse_json_array(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # salvage complete objects / boxes from truncated output
        objs = re.findall(r"\{[^{}]*\}", m.group(0))
        if objs:
            out = []
            for o in objs:
                try:
                    out.append(json.loads(o))
                except json.JSONDecodeError:
                    pass
            return out
        return [json.loads(b) for b in re.findall(r"\[\s*-?\d+[^\[\]]*\]", m.group(0)[1:])]


def clean_box(b):
    try:
        x1, y1, x2, y2 = [max(0.0, min(1000.0, float(v))) for v in b[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1), round(y1), round(x2), round(y2)]


def image_msg(img_url, text):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img_url}},
        {"type": "text", "text": text}]}]


async def detect_baseline(client, model, img_url, max_tokens):
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model, messages=image_msg(img_url, PROMPT_BASELINE), temperature=0.0,
        max_tokens=max_tokens, extra_body=NO_THINK)
    raw = r.choices[0].message.content or ""
    dets = []
    for d in parse_json_array(raw):
        if not isinstance(d, dict):
            continue
        box = clean_box(d.get("bbox_2d", d.get("bbox", [])))
        if box is None:
            continue
        try:
            conf = float(d.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        dets.append({"box": box, "score": round(conf, 4)})
    return dets, raw, r.usage.completion_tokens, r.usage.prompt_tokens, time.perf_counter() - t0


def p_yes_from_logprobs(top):
    yes = no = 0.0
    for t in top:
        tok = t.token.strip().lower()
        if tok == "yes":
            yes += math.exp(t.logprob)
        elif tok == "no":
            no += math.exp(t.logprob)
    return yes / (yes + no) if yes + no > 0 else 0.0


async def detect_jev(client, model, img_url, max_tokens):
    t0 = time.perf_counter()
    # Prefill the answer with "[[" so the model can only continue a bare coordinate list
    # (without it, Qwen falls back to its verbose {"bbox_2d":..., "label":...} grounding format).
    msgs = image_msg(img_url, PROMPT_JEV_BOXES) + [{"role": "assistant", "content": "[["}]
    r = await client.chat.completions.create(
        model=model, messages=msgs, temperature=0.0, max_tokens=max_tokens,
        extra_body={**NO_THINK, "continue_final_message": True, "add_generation_prompt": False})
    raw = "[[" + (r.choices[0].message.content or "")
    t_gen = time.perf_counter() - t0
    boxes = []
    for b in parse_json_array(raw):
        if isinstance(b, dict):
            b = b.get("bbox_2d", b.get("bbox", []))
        cb = clean_box(b) if isinstance(b, list) else None
        if cb and cb not in boxes:
            boxes.append(cb)

    async def verify(box):
        q = PROMPT_JEV_VERIFY.format(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
        v = await client.chat.completions.create(
            model=model, messages=image_msg(img_url, q), temperature=0.0, max_tokens=1,
            logprobs=True, top_logprobs=20, extra_body=NO_THINK)
        return p_yes_from_logprobs(v.choices[0].logprobs.content[0].top_logprobs)

    scores = await asyncio.gather(*(verify(b) for b in boxes))
    dets = [{"box": b, "score": round(s, 4)} for b, s in zip(boxes, scores)]
    return dets, raw, r.usage.completion_tokens, r.usage.prompt_tokens, t_gen


def iter_frames(video, stride, start, end):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok or (end is not None and idx >= end):
            break
        if idx >= start and (idx - start) % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx, idx / fps, frame
        idx += 1
    cap.release()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "mtp", "jev"], required=True)
    ap.add_argument("--video", default="data/camera_video.mp4")
    ap.add_argument("--out", default=None, help="output dir (default results/<method>)")
    ap.add_argument("--stride", type=int, default=15, help="process every Nth frame (15 = 1 fps)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="frames in flight; 1 = measure true single-frame latency")
    ap.add_argument("--max-side", type=int, default=1920, help="resize longest image side before sending")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    port = args.port or PORTS[args.method]
    client = AsyncOpenAI(base_url=f"http://{args.host}:{port}/v1", api_key="EMPTY", timeout=600)
    model = (await client.models.list()).data[0].id
    out = Path(args.out or f"results/{args.method}")
    out.mkdir(parents=True, exist_ok=True)
    fn = detect_jev if args.method == "jev" else detect_baseline
    sem = asyncio.Semaphore(args.concurrency)
    f_out = open(out / "detections.jsonl", "w")
    tasks = []

    async def run(idx, t, img_url):
        async with sem:
            t0 = time.perf_counter()
            dets, raw, ntok, nprompt, t_gen = await fn(client, model, img_url, args.max_tokens)
            lat = time.perf_counter() - t0
        rec = {"frame": idx, "t": round(t, 3), "latency": round(lat, 3), "completion_tokens": ntok,
               "prompt_tokens": nprompt, "gen_latency": round(t_gen, 3), "detections": dets, "raw": raw}
        f_out.write(json.dumps(rec) + "\n"); f_out.flush()
        print(f"[{args.method}] frame {idx:5d} t={t:7.2f}s peds={len(dets)} lat={lat:6.2f}s "
              f"tok={ntok} ({ntok / t_gen:5.1f} tok/s)", flush=True)

    t_all = time.perf_counter()
    for idx, t, frame in iter_frames(args.video, args.stride, args.start, args.end):
        img = encode_frame(frame, args.max_side)
        await sem.acquire(); sem.release()  # back-pressure: don't decode the whole video up front
        tasks.append(asyncio.create_task(run(idx, t, img)))
    await asyncio.gather(*tasks)
    f_out.close()

    recs = [json.loads(l) for l in open(out / "detections.jsonl")]
    lats = sorted(r["latency"] for r in recs)
    toks = sum(r["completion_tokens"] for r in recs)
    summary = {
        "method": args.method, "frames": len(recs), "stride": args.stride, "concurrency": args.concurrency,
        "max_side": args.max_side, "wall_s": round(time.perf_counter() - t_all, 1),
        "latency_mean_s": round(sum(lats) / len(lats), 3), "latency_p50_s": lats[len(lats) // 2],
        "completion_tokens_mean": round(toks / len(recs), 1),
        "decode_tok_per_s_mean": round(sum(r["completion_tokens"] / r["gen_latency"] for r in recs) / len(recs), 1),
        "pedestrians_mean": round(sum(len(r["detections"]) for r in recs) / len(recs), 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
