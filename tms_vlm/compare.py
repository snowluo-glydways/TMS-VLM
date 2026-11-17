"""Print a comparison table of results/*/summary.json and write results/comparison.md."""
import json
from pathlib import Path

rows = []
for m in ["baseline", "mtp", "jev"]:
    p = Path("results") / m / "summary.json"
    if p.exists():
        rows.append(json.loads(p.read_text()))
cols = [("method", "method"), ("frames", "frames"), ("latency_mean_s", "latency mean (s)"),
        ("latency_p50_s", "latency p50 (s)"), ("completion_tokens_mean", "output tokens/frame"),
        ("decode_tok_per_s_mean", "tokens/s (1st request)"), ("pedestrians_mean", "pedestrians/frame")]
lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
for r in rows:
    lines.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
md = "\n".join(lines)
Path("results/comparison.md").write_text(md + "\n")
print(md)
