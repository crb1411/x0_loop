"""Frozen-generator readiness test for the Cycle 04 distribution critic.

This diagnostic never updates the generator.  It first materializes a fixed,
class-matched real/fake dataset from the declared EMA Heun inference kernel,
then trains a fresh copy of the Cycle 03 discriminator and measures held-out
AUROC/accuracy.  The discriminator stored in the generator checkpoint is also
evaluated on the same split, which separates critic capacity from adversarial
co-adaptation.

Example:
    CUDA_VISIBLE_DEVICES=6 uv run python -m experiments.x0loop_v2.diagnose_critic_readiness \
      --checkpoint runs/x0loop_v2_from_scratch/cycle03/terminal-gan/checkpoints/ckpt_step_00015000.pt \
      --out runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from x0loop.infer import _load_model
from x0loop.losses.adversarial import discriminator_loss, r1_penalty
from x0loop.models.discriminator import build_x0_discriminator
from x0loop.training.factories import build_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a frozen generator's critic readiness.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--num-train", type=int, default=8192)
    parser.add_argument("--num-val", type=int, default=2048)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--critic-batch-size", type=int, default=64)
    parser.add_argument("--critic-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def binary_auroc(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> float:
    """Exact rank AUROC with tie handling; real is the positive class."""

    real = real_logits.detach().float().flatten().cpu()
    fake = fake_logits.detach().float().flatten().cpu()
    scores = torch.cat([real, fake])
    labels = torch.cat([torch.ones_like(real), torch.zeros_like(fake)])
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + 1 + end)
        start = end
    n_pos = float(real.numel())
    n_neg = float(fake.numel())
    rank_sum = float(ranks[labels.bool()].sum())
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0)


def _balanced_labels(count: int, num_classes: int, *, seed: int) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    labels = torch.arange(count, dtype=torch.long).remainder(num_classes)
    generator = torch.Generator().manual_seed(seed)
    return labels[torch.randperm(count, generator=generator)]


def _materialize_real(dataset: object, labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    targets = torch.as_tensor(getattr(dataset, "targets"), dtype=torch.long)
    class_pools = [torch.nonzero(targets == cls, as_tuple=False).flatten() for cls in range(int(targets.max()) + 1)]
    generators = [torch.Generator().manual_seed(seed + 104729 * cls) for cls in range(len(class_pools))]
    permutations = [pool[torch.randperm(pool.numel(), generator=generators[cls])] for cls, pool in enumerate(class_pools)]
    offsets = [0 for _ in class_pools]
    images: list[torch.Tensor] = []
    for label in labels.tolist():
        offset = offsets[label]
        if offset >= permutations[label].numel():
            raise ValueError(f"Not enough real examples for class {label}")
        image, actual_label = dataset[int(permutations[label][offset])]
        if int(actual_label) != label:
            raise AssertionError("class-balanced real selection produced a mismatched label")
        offsets[label] += 1
        images.append(image)
    return torch.stack(images).to(torch.float16)


@torch.inference_mode()
def _materialize_fake(
    *,
    model: torch.nn.Module,
    model_cfg: object,
    process: torch.nn.Module,
    labels: torch.Tensor,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    gen_eval = cfg.get("gen_eval", {}) or {}
    steps = int(gen_eval.get("steps", 20))
    sampler = str(gen_eval.get("sampler", "heun"))
    guidance_scale = float(gen_eval.get("guidance_scale", 2.2))
    guidance_schedule = gen_eval.get("guidance_schedule")
    if sampler.lower() != "heun" or guidance_schedule is not None:
        raise ValueError("Cycle 04 readiness is registered for Heun with a constant CFG scale")
    num_classes = int(model_cfg.num_classes)
    batches: list[torch.Tensor] = []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, labels.numel(), batch_size):
            cond = labels[start : start + batch_size].to(device)
            null_cond = torch.full_like(cond, num_classes)
            result = process.sample(
                model=model,
                steps=steps,
                shape=(cond.shape[0], model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
                device=device,
                dtype=torch.float32,
                return_trace=False,
                cond=cond,
                null_cond=null_cond,
                guidance_scale=guidance_scale,
                guidance_schedule=guidance_schedule,
                sampler=sampler,
            )
            batches.append(result["x"].detach().cpu().to(torch.float16))
    return torch.cat(batches, dim=0)


def _prepare_dataset(args: argparse.Namespace, cfg: dict, device: torch.device, path: Path) -> dict:
    total = args.num_train + args.num_val
    labels = _balanced_labels(total, int(cfg["model"]["num_classes"]), seed=args.seed + 11)
    dataset = build_dataset(cfg, train=True)
    real = _materialize_real(dataset, labels, seed=args.seed + 23)
    model, model_cfg, process = _load_model(cfg, args.checkpoint, device)
    model.eval()
    started = time.perf_counter()
    fake = _materialize_fake(
        model=model,
        model_cfg=model_cfg,
        process=process,
        labels=labels,
        cfg=cfg,
        device=device,
        batch_size=args.sample_batch_size,
        seed=args.seed + 37,
    )
    duration = time.perf_counter() - started
    payload = {
        "real": real,
        "fake": fake,
        "labels": labels,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("step", 0)),
        "sampling": {
            "steps": int((cfg.get("gen_eval", {}) or {}).get("steps", 20)),
            "sampler": str((cfg.get("gen_eval", {}) or {}).get("sampler", "heun")),
            "guidance_scale": float((cfg.get("gen_eval", {}) or {}).get("guidance_scale", 2.2)),
            "duration_s": duration,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    del model, process
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


@torch.inference_mode()
def _evaluate(
    discriminator: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    discriminator.eval()
    real_logits: list[torch.Tensor] = []
    fake_logits: list[torch.Tensor] = []
    for start in range(0, labels.numel(), batch_size):
        cond = labels[start : start + batch_size].to(device)
        t = torch.zeros(cond.shape[0], device=device)
        real_logits.append(discriminator(real[start : start + batch_size].to(device).float(), t, cond).cpu())
        fake_logits.append(discriminator(fake[start : start + batch_size].to(device).float(), t, cond).cpu())
    real_logit = torch.cat(real_logits).float()
    fake_logit = torch.cat(fake_logits).float()
    return {
        "auroc": binary_auroc(real_logit, fake_logit),
        "accuracy": float(0.5 * ((real_logit > 0).float().mean() + (fake_logit < 0).float().mean())),
        "real_accuracy": float((real_logit > 0).float().mean()),
        "fake_accuracy": float((fake_logit < 0).float().mean()),
        "real_logit_mean": float(real_logit.mean()),
        "fake_logit_mean": float(fake_logit.mean()),
        "logit_margin": float(real_logit.mean() - fake_logit.mean()),
    }


def _load_cycle03_discriminator(cfg: dict, checkpoint: dict, device: torch.device) -> torch.nn.Module | None:
    state = checkpoint.get("discriminator")
    if state is None:
        return None
    discriminator = build_x0_discriminator(cfg).to(device)
    discriminator.load_state_dict(state, strict=True)
    return discriminator


def _train_fresh_critic(
    args: argparse.Namespace,
    cfg: dict,
    payload: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(args.seed + 101)
    random.seed(args.seed + 101)
    discriminator = build_x0_discriminator(cfg).to(device)
    dc = cfg.get("discriminator", {}) or {}
    betas = dc.get("betas", [0.0, 0.99])
    optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(dc.get("lr", 2e-4)),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(dc.get("weight_decay", 0.0)),
        fused=device.type == "cuda",
    )
    adversarial = cfg.get("adversarial", {}) or {}
    r1 = adversarial.get("r1", {}) or {}
    r1_gamma = float(r1.get("gamma", 0.0))
    r1_interval = int(r1.get("interval", 16))
    loss_name = str(adversarial.get("loss", "hinge")).lower()
    n_train = int(payload["num_train"])
    train_real = payload["real"][:n_train]
    train_fake = payload["fake"][:n_train]
    train_labels = payload["labels"][:n_train]
    val_real = payload["real"][n_train:]
    val_fake = payload["fake"][n_train:]
    val_labels = payload["labels"][n_train:]
    generator = torch.Generator().manual_seed(args.seed + 211)
    history: list[dict[str, Any]] = []
    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for step in range(1, args.critic_steps + 1):
        indices = torch.randint(n_train, (args.critic_batch_size,), generator=generator)
        cond = train_labels[indices].to(device)
        t = torch.zeros(cond.shape[0], device=device)
        use_r1 = r1_gamma > 0.0 and r1_interval > 0 and step % r1_interval == 0
        real = train_real[indices].to(device).float().requires_grad_(use_r1)
        fake = train_fake[indices].to(device).float()
        discriminator.train()
        optimizer.zero_grad(set_to_none=True)
        real_logits = discriminator(real, t, cond)
        fake_logits = discriminator(fake, t, cond)
        per_loss, _, _ = discriminator_loss(real_logits, fake_logits, loss=loss_name)
        loss = per_loss.mean()
        penalty = torch.zeros((), device=device)
        if use_r1:
            penalty = r1_penalty(real_logits, real).mean()
            loss = loss + 0.5 * r1_gamma * r1_interval * penalty
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.critic_steps:
            train_metrics = _evaluate(
                discriminator,
                train_real,
                train_fake,
                train_labels,
                device=device,
                batch_size=max(args.critic_batch_size, 256),
            )
            val_metrics = _evaluate(
                discriminator,
                val_real,
                val_fake,
                val_labels,
                device=device,
                batch_size=max(args.critic_batch_size, 256),
            )
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "r1": float(penalty.detach()),
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            print(json.dumps(_json_ready(row), sort_keys=True), flush=True)
            if val_metrics["auroc"] > best_auc:
                best_auc = val_metrics["auroc"]
                best_state = {key: value.detach().cpu().clone() for key, value in discriminator.state_dict().items()}
    duration = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("critic training produced no evaluation state")
    discriminator.load_state_dict(best_state)
    best = max(history, key=lambda row: row["val"]["auroc"])
    return discriminator, {
        "history": history,
        "best": best,
        "duration_s": duration,
        "critic_config": asdict(discriminator.cfg),
        "optimizer": {
            "lr": float(dc.get("lr", 2e-4)),
            "betas": [float(betas[0]), float(betas[1])],
            "weight_decay": float(dc.get("weight_decay", 0.0)),
        },
        "r1": {"gamma": r1_gamma, "interval": r1_interval},
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    cycle = result.get("cycle03_critic")
    random_init = result["random_init_critic"]
    fresh = result["fresh_critic"]["best"]
    lines = [
        "# Cycle 04 critic readiness",
        "",
        f"Frozen generator: `{result['checkpoint']}` (step {result['checkpoint_step']})",
        "",
        f"Fixed samples: train={result['num_train']}, held-out={result['num_val']}; "
        f"Heun-{result['sampling']['steps']}, CFG={result['sampling']['guidance_scale']}, seed={result['seed']}.",
        "",
        "| critic | held-out AUROC | held-out accuracy | logit margin |",
        "|---|---:|---:|---:|",
        f"| random init | {random_init['auroc']:.6f} | {random_init['accuracy']:.6f} | {random_init['logit_margin']:.6f} |",
    ]
    if cycle is not None:
        lines.append(
            f"| Cycle 03 co-trained | {cycle['auroc']:.6f} | {cycle['accuracy']:.6f} | {cycle['logit_margin']:.6f} |"
        )
    lines.append(
        f"| frozen-G fresh critic (best step {fresh['step']}) | {fresh['val']['auroc']:.6f} | "
        f"{fresh['val']['accuracy']:.6f} | {fresh['val']['logit_margin']:.6f} |"
    )
    gap = fresh["train"]["auroc"] - fresh["val"]["auroc"]
    lines.extend(
        [
            "",
            f"Best fresh-critic train/held-out AUROC gap: {gap:.6f}.",
            "",
            "Readiness is not a generator-quality result and does not count as one of the next three trainings.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.num_train <= 0 or args.num_val <= 0:
        raise ValueError("num-train and num-val must both be positive")
    if args.critic_steps <= 0 or args.eval_every <= 0:
        raise ValueError("critic-steps and eval-every must both be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint.get("config") or {})
    if not cfg:
        raise ValueError("checkpoint has no embedded config")
    dataset_path = out / "fixed_terminal_dataset.pt"
    if args.force_prepare or not dataset_path.is_file():
        payload = _prepare_dataset(args, cfg, device, dataset_path)
    else:
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        expected = (args.num_train, args.num_val, args.seed, str(checkpoint_path))
        actual = (
            int(payload["num_train"]),
            int(payload["num_val"]),
            int(payload["seed"]),
            str(payload["checkpoint"]),
        )
        if actual != expected:
            raise ValueError(f"existing fixed dataset metadata mismatch: expected={expected}, actual={actual}")

    n_train = int(payload["num_train"])
    val_real = payload["real"][n_train:]
    val_fake = payload["fake"][n_train:]
    val_labels = payload["labels"][n_train:]
    torch.manual_seed(args.seed + 101)
    random_discriminator = build_x0_discriminator(cfg).to(device)
    # torch.nn spectral_norm initializes its power-iteration vectors lazily in
    # training mode.  Prime them once so the random baseline has finite,
    # interpretable logits; this does not update any learned parameter.
    random_discriminator.train()
    with torch.no_grad():
        prime_n = min(args.critic_batch_size, val_labels.numel())
        prime_cond = val_labels[:prime_n].to(device)
        prime_real = val_real[:prime_n].to(device).float()
        prime_t = torch.zeros(prime_n, device=device)
        # A single power iteration per spectral-normalized layer is not enough
        # for a deep random network: multiplicative scale error can still make
        # logits enormous. Twenty parameter-free iterations stabilize the
        # baseline while leaving all learned weights untouched.
        for _ in range(20):
            random_discriminator(prime_real, prime_t, prime_cond)
    random_metrics = _evaluate(
        random_discriminator,
        val_real,
        val_fake,
        val_labels,
        device=device,
        batch_size=max(args.critic_batch_size, 256),
    )
    cycle03_discriminator = _load_cycle03_discriminator(cfg, checkpoint, device)
    cycle03_metrics = None
    if cycle03_discriminator is not None:
        cycle03_metrics = _evaluate(
            cycle03_discriminator,
            val_real,
            val_fake,
            val_labels,
            device=device,
            batch_size=max(args.critic_batch_size, 256),
        )
    fresh_discriminator, fresh_result = _train_fresh_critic(args, cfg, payload, device)
    torch.save(
        {
            "discriminator": fresh_discriminator.state_dict(),
            "config": cfg,
            "source_checkpoint": str(checkpoint_path),
            "best": fresh_result["best"],
        },
        out / "fresh_critic_best.pt",
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["checkpoint_step"]),
        "seed": int(payload["seed"]),
        "num_train": n_train,
        "num_val": int(payload["num_val"]),
        "sampling": payload["sampling"],
        "dataset": str(dataset_path),
        "random_init_critic": random_metrics,
        "cycle03_critic": cycle03_metrics,
        "fresh_critic": fresh_result,
    }
    result_path = out / "critic_readiness.json"
    result_path.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(_json_ready(result), out / "critic_readiness.md")
    print(f"wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
