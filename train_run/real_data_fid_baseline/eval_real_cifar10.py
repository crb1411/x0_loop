from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import CIFAR10

from x0loop.training.generative_eval import _torch_fidelity_load_compat


EDM_CIFAR10_FID_REF_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
EDM_CIFAR10_INPUT2_ALIASES = {
    "edm",
    "edm-cifar10",
    "edm-cifar10-train",
    "edm-cifar10-32x32",
    "edm-cifar10-32x32.npz",
}
EDM_ROOT = Path("/data/seek/aigc/edm")


def _parse_args():
    p = argparse.ArgumentParser(description="Export real CIFAR10 train subsets and compute torch-fidelity metrics.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--input2", default="cifar10-train")
    p.add_argument("--fid-statistics-file", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--keep-images", action="store_true")
    return p.parse_args()


def _resolve_metric_reference(args) -> tuple[str | None, str | None, str]:
    input2 = str(args.input2) if args.input2 is not None else ""
    fid_statistics_file = args.fid_statistics_file

    if input2 in EDM_CIFAR10_INPUT2_ALIASES:
        fid_statistics_file = _cached_fid_ref(
            EDM_CIFAR10_FID_REF_URL,
            cache_dir=Path(args.cache_root) if args.cache_root else Path(args.out_dir) / "fid_refs",
        )
        return None, fid_statistics_file, input2

    if input2.startswith(("http://", "https://")) or input2.endswith(".npz"):
        fid_statistics_file = _cached_fid_ref(
            input2,
            cache_dir=Path(args.cache_root) if args.cache_root else Path(args.out_dir) / "fid_refs",
        )
        return None, fid_statistics_file, input2

    return input2, fid_statistics_file, input2


def _cached_fid_ref(ref: str, *, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if ref.startswith(("http://", "https://")):
        dest = cache_dir / Path(ref).name
        if not dest.exists():
            print(f"[real_fid] download fid ref: {ref} -> {dest}", flush=True)
            urllib.request.urlretrieve(ref, dest)
        return str(dest)
    return str(Path(ref).expanduser().resolve())


def _save_subset(dataset: CIFAR10, indices: list[int], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for out_idx, ds_idx in enumerate(indices):
        image, label = dataset[ds_idx]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image from CIFAR10, got {type(image).__name__}")
        image.save(out_dir / f"real_{out_idx:06d}_y{int(label)}.png")


def _random_indices(dataset: CIFAR10, *, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if n <= len(dataset):
        return rng.sample(range(len(dataset)), n)
    return [rng.randrange(len(dataset)) for _ in range(n)]


def _balanced_indices(dataset: CIFAR10, *, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset.targets):
        by_class.setdefault(int(label), []).append(idx)
    classes = sorted(by_class)
    per_class, rem = divmod(n, len(classes))
    out: list[int] = []
    for pos, cls in enumerate(classes):
        k = per_class + (1 if pos < rem else 0)
        pool = by_class[cls]
        if k <= len(pool):
            out.extend(rng.sample(pool, k))
        else:
            out.extend(pool)
            out.extend(rng.choice(pool) for _ in range(k - len(pool)))
    rng.shuffle(out)
    return out


def _class_counts(dataset: CIFAR10, indices: list[int]) -> dict[str, int]:
    counts = {str(i): 0 for i in range(10)}
    for idx in indices:
        counts[str(int(dataset.targets[idx]))] += 1
    return counts


def _calculate_metrics(input1: Path, *, args) -> dict:
    from torch_fidelity import calculate_metrics

    input2, fid_statistics_file, input2_label = _resolve_metric_reference(args)
    if input2_label in EDM_CIFAR10_INPUT2_ALIASES:
        return _calculate_edm_fid(input1, fid_statistics_file=fid_statistics_file, args=args)

    use_fid_ref_only = input2 is None and fid_statistics_file is not None
    kwargs = {
        "input1": str(input1),
        "datasets_root": args.dataset_root,
        "datasets_download": bool(args.download),
        "cuda": bool(args.cuda),
        "isc": not use_fid_ref_only,
        "fid": True,
        "kid": not use_fid_ref_only,
        "ppl": False,
        "prc": not use_fid_ref_only,
        "verbose": False,
        "cache": True,
    }
    if input2 is not None:
        kwargs["input2"] = input2
    if fid_statistics_file is not None:
        kwargs["fid_statistics_file"] = fid_statistics_file
    if args.cache_root:
        kwargs["cache_root"] = args.cache_root
    with _torch_fidelity_load_compat():
        return calculate_metrics(**kwargs)


def _calculate_edm_fid(input1: Path, *, fid_statistics_file: str | None, args) -> dict:
    if fid_statistics_file is None:
        raise ValueError("EDM FID requires fid_statistics_file")
    if not (EDM_ROOT / "fid.py").exists():
        raise FileNotFoundError(f"EDM fid.py not found under {EDM_ROOT}")

    cmd = [
        sys.executable,
        "fid.py",
        "calc",
        "--images",
        str(input1),
        "--ref",
        str(fid_statistics_file),
        "--num",
        str(args.num_samples),
        "--seed",
        str(args.seed),
    ]
    print(f"[real_fid] edm fid: cd {EDM_ROOT} && {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EDM_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        cmd,
        cwd=str(EDM_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"EDM fid.py failed with exit code {proc.returncode}")

    fid = None
    for line in reversed(proc.stdout.splitlines()):
        try:
            fid = float(line.strip())
            break
        except ValueError:
            continue
    if fid is None:
        raise RuntimeError("Could not parse FID from EDM fid.py output")
    return {"frechet_inception_distance": fid}


def _run_one(name: str, indices: list[int], dataset: CIFAR10, *, args) -> dict:
    out_root = Path(args.out_dir)
    image_dir = out_root / name / "images"
    print(f"[real_fid] export {name}: n={len(indices)} -> {image_dir}", flush=True)
    _save_subset(dataset, indices, image_dir)
    input2, fid_statistics_file, input2_label = _resolve_metric_reference(args)
    ref_desc = f"fid_statistics_file={fid_statistics_file}" if input2 is None else f"input2={input2}"
    print(f"[real_fid] metrics {name}: input1={image_dir} {ref_desc}", flush=True)
    metrics = _calculate_metrics(image_dir, args=args)
    row = {
        "experiment": name,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "num_samples": len(indices),
        "input1": str(image_dir),
        "input2": input2_label,
        "resolved_input2": input2,
        "fid_statistics_file": fid_statistics_file,
        "class_counts": _class_counts(dataset, indices),
        "metrics": metrics,
    }
    row.update({k: metrics[k] for k in (
        "frechet_inception_distance",
        "inception_score_mean",
        "kernel_inception_distance_mean",
        "precision",
        "recall",
        "f_score",
    ) if k in metrics})
    with open(out_root / "real_data_metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    if not args.keep_images:
        shutil.rmtree(image_dir)
        row["input1"] = None
    return row


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dataset = CIFAR10(root=args.dataset_root, train=True, download=bool(args.download), transform=None)
    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be > 0, got {args.num_samples}")

    experiments = [
        ("random_train_50k", _random_indices(dataset, n=args.num_samples, seed=args.seed)),
        ("balanced_train_50k", _balanced_indices(dataset, n=args.num_samples, seed=args.seed)),
    ]
    rows = [_run_one(name, indices, dataset, args=args) for name, indices in experiments]

    print("experiment              FID      IS       KID      P      R", flush=True)
    for row in rows:
        print(
            f"{row['experiment']:<22} "
            f"{row.get('frechet_inception_distance', float('nan')):7.3f} "
            f"{row.get('inception_score_mean', float('nan')):7.3f} "
            f"{row.get('kernel_inception_distance_mean', float('nan')):8.5f} "
            f"{row.get('precision', float('nan')):6.3f} "
            f"{row.get('recall', float('nan')):6.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
