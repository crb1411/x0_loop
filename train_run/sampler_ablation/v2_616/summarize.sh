#!/usr/bin/env bash
# Collect finished v2_616 sampler-ablation results into one table.
set -euo pipefail
OUT_BASE="./runs/sampler_ablation/v2_616"

uv run python - "$OUT_BASE" <<'PY'
import glob, json, os, sys
base = sys.argv[1]
latest = {}
for f in sorted(glob.glob(os.path.join(base, "*", "gen_eval_metrics_*.jsonl"))):
    exp = os.path.basename(os.path.dirname(f))
    last = None
    with open(f, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    if not last:
        continue
    prev = latest.get(exp)
    if prev is not None and os.path.getmtime(prev[0]) > os.path.getmtime(f):
        continue
    latest[exp] = (f, last)
rows = []
for exp, (_path, last) in latest.items():
    m = last.get("metrics", last)
    rows.append((
        exp, last.get("sampler"), last.get("steps"), last.get("guidance_scale"),
        m.get("frechet_inception_distance"), m.get("inception_score_mean"),
        m.get("kernel_inception_distance_mean"), m.get("precision"), m.get("recall"),
    ))
if not rows:
    print(f"no results yet under {base}")
    sys.exit(0)
rows.sort(key=lambda r: (r[4] is None, r[4] if r[4] is not None else 1e9))
hdr = ("experiment", "sampler", "steps", "cfg", "FID", "IS", "KID", "P", "R")
print(f"{hdr[0]:<28}{hdr[1]:<10}{hdr[2]:>5} {hdr[3]:>4}  {hdr[4]:>8} {hdr[5]:>6} {hdr[6]:>9} {hdr[7]:>5} {hdr[8]:>5}")
def fmt(v, n): return ("%.*f" % (n, v)) if isinstance(v, (int, float)) else "-"
for r in rows:
    print(f"{r[0]:<28}{str(r[1]):<10}{str(r[2]):>5} {str(r[3]):>4}  "
          f"{fmt(r[4],2):>8} {fmt(r[5],3):>6} {fmt(r[6],5):>9} {fmt(r[7],3):>5} {fmt(r[8],3):>5}")
PY
