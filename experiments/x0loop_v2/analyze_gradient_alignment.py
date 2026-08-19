"""Measure fresh/terminal parameter-gradient alignment without updating a model.

Cycle 03 controlled the adversarial gradient at the model-output tensor.  This
tool follows both losses through the shared JiT backbone and reports the actual
parameter-space cosine and norm ratio, for both the co-trained Cycle 03 critic
and the frozen-generator readiness critic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch

from x0loop.core.time_sampling import build_time_sampler
from x0loop.losses.adversarial import generator_loss
from x0loop.losses.spec import build_loss
from x0loop.models.denoiser import Denoiser
from x0loop.models.discriminator import build_x0_discriminator
from x0loop.models.factory import build_model
from x0loop.training.context import ModelContext
from x0loop.training.engine import compute_forward_batch
from x0loop.training.factories import build_augment, build_process, build_schedule
from x0loop.utils.checkpoint import _adapt_ema_state_to_model, _load_model_state_with_fallback
from x0loop.utils.ema import EMA


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fresh/terminal gradient alignment.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--readiness-critic", required=True)
    parser.add_argument("--fixed-dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--target-output-ratio", type=float, default=0.10)
    parser.add_argument("--scale-max", type=float, default=10.0)
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


def _parameter_group(name: str) -> str:
    clean = name.replace("_orig_mod.", "")
    if clean.startswith("process."):
        return "process"
    if ".blocks." in clean:
        block = int(clean.split(".blocks.", 1)[1].split(".", 1)[0])
        if block <= 3:
            return "blocks_00_03"
        if block <= 7:
            return "blocks_04_07"
        return "blocks_08_11"
    if ".final_layer." in clean:
        return "final_layer"
    if any(token in clean for token in ("time_mlp", "label_emb")):
        return "conditioning"
    return "input_embedding"


def gradient_pair_metrics(
    named_fresh: Iterable[tuple[str, torch.Tensor | None]],
    named_aux: Iterable[tuple[str, torch.Tensor | None]],
    *,
    scale: float,
) -> dict[str, Any]:
    """Aggregate gradient cosine/energy without flattening the 60M-param model."""

    grouped: dict[str, dict[str, torch.Tensor | float | int]] = {}
    total_dot = torch.zeros((), dtype=torch.float64)
    total_fresh_sq = torch.zeros((), dtype=torch.float64)
    total_aux_sq = torch.zeros((), dtype=torch.float64)
    negative_tensors = 0
    active_tensors = 0
    negative_numel = 0
    active_numel = 0
    aux_map = dict(named_aux)
    for name, fresh in named_fresh:
        aux = aux_map.get(name)
        if fresh is None or aux is None:
            continue
        fresh_cpu = fresh.detach().float()
        aux_cpu = aux.detach().float()
        dot = (fresh_cpu * aux_cpu).sum().double().cpu()
        fresh_sq = fresh_cpu.square().sum().double().cpu()
        aux_sq = aux_cpu.square().sum().double().cpu()
        total_dot += dot
        total_fresh_sq += fresh_sq
        total_aux_sq += aux_sq
        active_tensors += 1
        active_numel += fresh.numel()
        if float(dot) < 0.0:
            negative_tensors += 1
            negative_numel += fresh.numel()
        group_name = _parameter_group(name)
        group = grouped.setdefault(
            group_name,
            {
                "dot": torch.zeros((), dtype=torch.float64),
                "fresh_sq": torch.zeros((), dtype=torch.float64),
                "aux_sq": torch.zeros((), dtype=torch.float64),
                "numel": 0,
            },
        )
        group["dot"] += dot
        group["fresh_sq"] += fresh_sq
        group["aux_sq"] += aux_sq
        group["numel"] += fresh.numel()

    eps = 1e-24

    def summarize(dot: torch.Tensor, fresh_sq: torch.Tensor, aux_sq: torch.Tensor, numel: int) -> dict[str, float | int]:
        fresh_norm = float(fresh_sq.sqrt())
        aux_norm = float(aux_sq.sqrt())
        cosine = float(dot / (fresh_sq * aux_sq).sqrt().clamp_min(eps))
        scaled_ratio = float(scale) * aux_norm / max(fresh_norm, 1e-12)
        combined_sq = fresh_sq + 2.0 * float(scale) * dot + float(scale) ** 2 * aux_sq
        combined_cosine = float(
            (fresh_sq + float(scale) * dot)
            / (fresh_sq.sqrt() * combined_sq.clamp_min(eps).sqrt()).clamp_min(eps)
        )
        return {
            "numel": int(numel),
            "fresh_norm": fresh_norm,
            "aux_norm": aux_norm,
            "cosine": cosine,
            "scaled_aux_to_fresh_norm": scaled_ratio,
            "fresh_vs_combined_cosine": combined_cosine,
        }

    result: dict[str, Any] = summarize(total_dot, total_fresh_sq, total_aux_sq, active_numel)
    result.update(
        {
            "active_tensors": active_tensors,
            "negative_dot_tensor_fraction": negative_tensors / max(active_tensors, 1),
            "negative_dot_numel_fraction": negative_numel / max(active_numel, 1),
            "groups": {
                name: summarize(
                    values["dot"],
                    values["fresh_sq"],
                    values["aux_sq"],
                    int(values["numel"]),
                )
                for name, values in sorted(grouped.items())
            },
        }
    )
    return result


def _load_training_model(
    cfg: dict,
    checkpoint: dict,
    device: torch.device,
) -> tuple[Denoiser, object, ModelContext, EMA]:
    net, model_cfg = build_model(cfg["model"])
    net = net.to(device)
    schedule = build_schedule(cfg)
    process = build_process(cfg, schedule).to(device)
    loss_fn = build_loss(cfg["loss"], schedule)
    time_sampler = build_time_sampler(cfg, schedule)
    denoiser = Denoiser(
        net,
        process=process,
        loss_fn=loss_fn,
        time_sampler=time_sampler,
        time_condition_jitter=cfg.get("time_condition_jitter"),
        model_conditioning=cfg.get("model_conditioning"),
    ).to(device)
    _load_model_state_with_fallback(denoiser, checkpoint["model"], strict=True)
    ema = EMA(denoiser, decay=float((cfg.get("train", {}) or {}).get("ema_decay", 0.996)))
    ema.load_state_dict(_adapt_ema_state_to_model(denoiser, checkpoint["ema"]))
    model_ctx = ModelContext(
        model=net,
        model_cfg=model_cfg,
        use_fsdp=False,
        fsdp_mode="none",
        precision="bf16" if device.type == "cuda" else "fp32",
        use_ddp=False,
        distributed_mode="none",
    )
    return denoiser, process, model_ctx, ema


def _load_critics(
    cfg: dict,
    checkpoint: dict,
    readiness_path: Path,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    critics: dict[str, torch.nn.Module] = {}
    cycle_state = checkpoint.get("discriminator")
    if cycle_state is None:
        raise ValueError("generator checkpoint has no co-trained discriminator")
    cycle = build_x0_discriminator(cfg).to(device)
    cycle.load_state_dict(cycle_state, strict=True)
    critics["cycle03_cotrained"] = cycle
    readiness = torch.load(readiness_path, map_location="cpu", weights_only=False)
    fresh = build_x0_discriminator(cfg).to(device)
    fresh.load_state_dict(readiness["discriminator"], strict=True)
    critics["frozen_g_fresh"] = fresh
    for critic in critics.values():
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    return critics


def _one_batch(
    *,
    cfg: dict,
    checkpoint_step: int,
    batch_index: int,
    x0: torch.Tensor,
    labels: torch.Tensor,
    denoiser: Denoiser,
    process: object,
    model_ctx: ModelContext,
    ema: EMA,
    critics: dict[str, torch.nn.Module],
    target_output_ratio: float,
    scale_max: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    runtime = SimpleNamespace(device=device, world_size=1, rank=0)
    augment, augment_mode = build_augment(cfg)
    torch.manual_seed(seed + batch_index * 1009)
    fwd = compute_forward_batch(
        cfg=cfg,
        runtime=runtime,
        model_ctx=model_ctx,
        denoiser=denoiser,
        process=process,
        augment=augment,
        augment_mode=augment_mode,
        x0=x0,
        y=labels,
        use_label_cond=True,
        step=checkpoint_step + batch_index,
        clean_loop_cfg=None,
        clean_loop_bank=None,
        ema=ema,
    )
    if fwd.adv_fake is None or fwd.adv_output is None or fwd.adv_t is None:
        raise RuntimeError("terminal adversarial payload was not constructed")
    params = [(name, parameter) for name, parameter in denoiser.named_parameters() if parameter.requires_grad]
    names = [name for name, _ in params]
    tensors = [parameter for _, parameter in params]
    fresh_output_grad = torch.autograd.grad(fwd.loss, fwd.out, retain_graph=True)[0]
    fresh_param = torch.autograd.grad(fwd.loss, tensors, retain_graph=True, allow_unused=True)
    fresh_named = list(zip(names, fresh_param, strict=True))
    result: dict[str, Any] = {
        "batch_index": batch_index,
        "fresh_loss": float(fwd.loss.detach()),
        "fresh_output_grad_norm": float(fresh_output_grad.detach().float().norm()),
        "terminal_fake_rms": float(fwd.adv_fake.detach().float().square().mean().sqrt()),
        "critics": {},
    }
    for critic_index, (critic_name, critic) in enumerate(critics.items()):
        logits = critic(fwd.adv_fake.float(), fwd.adv_t, fwd.adv_cond)
        g_loss = generator_loss(logits, loss=str((cfg.get("adversarial", {}) or {}).get("loss", "hinge"))).mean()
        aux_output_grad = torch.autograd.grad(g_loss, fwd.adv_output, retain_graph=True)[0]
        aux_output_norm = float(aux_output_grad.detach().float().norm())
        scale = min(
            target_output_ratio * result["fresh_output_grad_norm"] / max(aux_output_norm, 1e-12),
            scale_max,
        )
        aux_param = torch.autograd.grad(
            g_loss,
            tensors,
            retain_graph=critic_index < len(critics) - 1,
            allow_unused=True,
        )
        result["critics"][critic_name] = {
            "g_loss": float(g_loss.detach()),
            "fake_logit_mean": float(logits.detach().mean()),
            "aux_output_grad_norm": aux_output_norm,
            "output_scale": scale,
            "output_scaled_ratio": scale * aux_output_norm / max(result["fresh_output_grad_norm"], 1e-12),
            "parameters": gradient_pair_metrics(
                fresh_named,
                list(zip(names, aux_param, strict=True)),
                scale=scale,
            ),
        }
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    critics = rows[0]["critics"].keys()
    result: dict[str, Any] = {"num_batches": len(rows), "critics": {}}
    scalar_keys = (
        "g_loss",
        "fake_logit_mean",
        "aux_output_grad_norm",
        "output_scale",
        "output_scaled_ratio",
    )
    param_keys = (
        "cosine",
        "scaled_aux_to_fresh_norm",
        "fresh_vs_combined_cosine",
        "negative_dot_tensor_fraction",
        "negative_dot_numel_fraction",
    )
    for critic in critics:
        critic_result: dict[str, Any] = {}
        for key in scalar_keys:
            values = torch.tensor([row["critics"][critic][key] for row in rows], dtype=torch.float64)
            critic_result[key] = {"mean": float(values.mean()), "std": float(values.std(unbiased=False))}
        for key in param_keys:
            values = torch.tensor([row["critics"][critic]["parameters"][key] for row in rows], dtype=torch.float64)
            critic_result[f"parameter_{key}"] = {
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
            }
        group_names = rows[0]["critics"][critic]["parameters"]["groups"].keys()
        critic_result["groups"] = {}
        for group in group_names:
            critic_result["groups"][group] = {}
            for key in ("cosine", "scaled_aux_to_fresh_norm", "fresh_vs_combined_cosine"):
                values = torch.tensor(
                    [row["critics"][critic]["parameters"]["groups"][group][key] for row in rows],
                    dtype=torch.float64,
                )
                critic_result["groups"][group][key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(unbiased=False)),
                }
        result["critics"][critic] = critic_result
    return result


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cycle 04 fresh/terminal gradient alignment",
        "",
        f"Checkpoint: `{result['checkpoint']}`; batches={result['num_batches']}, "
        f"batch_size={result['batch_size']}, target output ratio={result['target_output_ratio']:.3f}.",
        "",
        "| critic | output ratio | parameter ratio | parameter cosine | fresh vs combined cosine | negative-dot numel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in result["aggregate"]["critics"].items():
        lines.append(
            f"| {name} | {values['output_scaled_ratio']['mean']:.6f} | "
            f"{values['parameter_scaled_aux_to_fresh_norm']['mean']:.6f} | "
            f"{values['parameter_cosine']['mean']:.6f} | "
            f"{values['parameter_fresh_vs_combined_cosine']['mean']:.6f} | "
            f"{values['parameter_negative_dot_numel_fraction']['mean']:.6f} |"
        )
    lines.extend(["", "## Layer groups", ""])
    for name, values in result["aggregate"]["critics"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| group | scaled aux/fresh norm | cosine | fresh vs combined cosine |",
                "|---|---:|---:|---:|",
            ]
        )
        for group, metrics in values["groups"].items():
            lines.append(
                f"| {group} | {metrics['scaled_aux_to_fresh_norm']['mean']:.6f} | "
                f"{metrics['cosine']['mean']:.6f} | {metrics['fresh_vs_combined_cosine']['mean']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "Output-space control and parameter-space control are reported separately. "
            "No optimizer step is taken by this diagnostic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    readiness_path = Path(args.readiness_critic)
    dataset_path = Path(args.fixed_dataset)
    for path in (checkpoint_path, readiness_path, dataset_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("batch-size and num-batches must be positive")
    if not (0.0 < args.target_output_ratio <= 1.0):
        raise ValueError("target-output-ratio must be in (0,1]")
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint.get("config") or {})
    if not cfg:
        raise ValueError("checkpoint has no embedded config")
    cfg.setdefault("adversarial", {})["enabled"] = True
    cfg["adversarial"]["fake_space"] = "terminal_x0"
    cfg["adversarial"]["weight"] = 1.0
    cfg["adversarial"]["start_step"] = 0
    cfg.setdefault("clean_loop", {})["enabled"] = False
    denoiser, process, model_ctx, ema = _load_training_model(cfg, checkpoint, device)
    denoiser.train()
    critics = _load_critics(cfg, checkpoint, readiness_path, device)
    fixed = torch.load(dataset_path, map_location="cpu", weights_only=False)
    needed = args.batch_size * args.num_batches
    if int(fixed["num_train"]) < needed:
        raise ValueError(f"fixed dataset has {fixed['num_train']} train samples but {needed} are required")
    rows: list[dict[str, Any]] = []
    checkpoint_step = int(checkpoint.get("step", 0))
    for batch_index in range(args.num_batches):
        start = batch_index * args.batch_size
        x0 = fixed["real"][start : start + args.batch_size]
        labels = fixed["labels"][start : start + args.batch_size]
        row = _one_batch(
            cfg=cfg,
            checkpoint_step=checkpoint_step,
            batch_index=batch_index,
            x0=x0,
            labels=labels,
            denoiser=denoiser,
            process=process,
            model_ctx=model_ctx,
            ema=ema,
            critics=critics,
            target_output_ratio=args.target_output_ratio,
            scale_max=args.scale_max,
            device=device,
            seed=args.seed,
        )
        rows.append(_json_ready(row))
        print(json.dumps(_json_ready(row), sort_keys=True), flush=True)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "readiness_critic": str(readiness_path),
        "fixed_dataset": str(dataset_path),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "target_output_ratio": args.target_output_ratio,
        "scale_max": args.scale_max,
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gradient_alignment.json").write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(_json_ready(result), out / "gradient_alignment.md")
    print(f"wrote {out / 'gradient_alignment.json'}", flush=True)


if __name__ == "__main__":
    main()
