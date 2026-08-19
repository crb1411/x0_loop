from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("gen_eval_metrics_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "frechet_inception_distance" in row and "error" not in row:
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank x0loop-v2 checkpoints by fixed-noise FID.")
    parser.add_argument("run_root", nargs="?", default="runs/x0loop_v2")
    parser.add_argument("--samples", type=int, default=5000, help="Compare rows with this sample count.")
    parser.add_argument("--best-only", action="store_true", help="Print only the best checkpoint path.")
    args = parser.parse_args()
    root = Path(args.run_root)

    ranked: list[tuple[float, str, int, int, Path]] = []
    if list(root.glob("gen_eval_metrics_*.jsonl")):
        branch_dirs = [root]
    else:
        branch_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for branch_dir in branch_dirs:
        for row in _rows(branch_dir):
            step = int(row["step"])
            fid = float(row["frechet_inception_distance"])
            samples = int(row["num_samples"])
            if samples != args.samples:
                continue
            checkpoint = branch_dir / "checkpoints" / f"ckpt_step_{step:08d}.pt"
            ranked.append((fid, branch_dir.name, step, samples, checkpoint))

    if not ranked:
        raise SystemExit(f"no successful {args.samples}-sample FID rows found below {root}")

    ranked.sort()
    if args.best_only:
        best_checkpoint = ranked[0][-1]
        if not best_checkpoint.is_file():
            raise SystemExit(f"best checkpoint is missing: {best_checkpoint}")
        print(best_checkpoint)
        return

    print("fid\tbranch\tstep\tsamples\tcheckpoint")
    for fid, branch, step, samples, checkpoint in ranked:
        marker = "" if checkpoint.is_file() else " [missing]"
        print(f"{fid:.6f}\t{branch}\t{step}\t{samples}\t{checkpoint}{marker}")


if __name__ == "__main__":
    main()
