#!/usr/bin/env bash
# Collect finished v5_616 sampler-ablation results into one table.
set -euo pipefail
OUT_BASE="./runs/sampler_ablation/v5_616"

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
    sched = last.get("guidance_schedule")
    if isinstance(sched, dict):
        name = str(sched.get("name", ""))
        power = sched.get("power", None)
        sched_label = name if power is None else f"{name}:p{power}"
    elif sched:
        sched_label = str(sched)
    else:
        sched_label = "constant"
    rows.append((
        exp, last.get("sampler"), last.get("steps"), last.get("guidance_scale"), sched_label,
        m.get("frechet_inception_distance"), m.get("inception_score_mean"),
        m.get("kernel_inception_distance_mean"), m.get("precision"), m.get("recall"),
    ))
if not rows:
    print(f"no results yet under {base}")
    sys.exit(0)
rows.sort(key=lambda r: (r[5] is None, r[5] if r[5] is not None else 1e9))
hdr = ("experiment", "sampler", "steps", "cfg", "schedule", "FID", "IS", "KID", "P", "R")
print(f"{hdr[0]:<34}{hdr[1]:<10}{hdr[2]:>5} {hdr[3]:>4}  {hdr[4]:<18} {hdr[5]:>8} {hdr[6]:>6} {hdr[7]:>9} {hdr[8]:>5} {hdr[9]:>5}")
def fmt(v, n): return ("%.*f" % (n, v)) if isinstance(v, (int, float)) else "-"
for r in rows:
    print(f"{r[0]:<34}{str(r[1]):<10}{str(r[2]):>5} {str(r[3]):>4}  {r[4]:<18} "
          f"{fmt(r[5],2):>8} {fmt(r[6],3):>6} {fmt(r[7],5):>9} {fmt(r[8],3):>5} {fmt(r[9],3):>5}")
PY
