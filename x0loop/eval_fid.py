"""Standalone generative-metric (FID/IS/KID/...) eval from a single checkpoint.

The model architecture, process and dataset are rebuilt from the config embedded
in the checkpoint, so only an override YAML (sampler / steps / guidance / num
samples) and the checkpoint path are needed. Each process evaluates on one GPU:

    CUDA_VISIBLE_DEVICES=0 python -m x0loop.eval_fid \
        --ckpt /path/ckpt_step_00100000.pt \
        --eval-config train_run/sampler_ablation/<exp>/eval.yaml \
        --set logging.out_dir=runs/sampler_ablation/<exp>
"""

from __future__ import annotations

import argparse

import torch

from x0loop.core.config import _deep_merge_dict, _load_yaml
from x0loop.models.denoiser import Denoiser
from x0loop.train import _apply_set_overrides
from x0loop.training.context import ResumeState
from x0loop.training.factories import build_model_context, build_process, build_schedule, init_runtime
from x0loop.training.generative_eval import _cfg as parse_gen_cfg
from x0loop.training.generative_eval import _run_generative_eval
from x0loop.utils.checkpoint import _load_model_state_with_fallback, _replace_state_dict_prefix, _strip_state_dict_prefix
from x0loop.utils.ema import EMA


def _apply_prefix_transform(state_dict: dict, name: str) -> dict:
    """Re-apply the key transform that the model loader used (e.g. compile's
    `net._orig_mod.` prefix) to the EMA shadow so its keys match the model."""
    if name == "none":
        return state_dict
    if "->" in name:
        old, new = name.split("->")
        return _replace_state_dict_prefix(state_dict, old, new)
    return _strip_state_dict_prefix(state_dict, name)


def _build_denoiser(cfg: dict, model: torch.nn.Module, process: torch.nn.Module) -> Denoiser:
    """Rebuild the training-time wrapper, including its time conditioning."""
    return Denoiser(
        model,
        process=process,
        model_conditioning=cfg.get("model_conditioning", None),
        solver_correction=cfg.get("solver_correction", None),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Standalone FID/IS/KID eval from a checkpoint.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to ckpt_step_*.pt")
    p.add_argument("--eval-config", type=str, required=True, help="Override YAML (gen_eval/sampler/steps/cfg + runtime).")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--tag", type=str, default="eval")
    return p.parse_args()


def main():
    args = parse_args()

    # Single load: the checkpoint carries the training config, model and EMA.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base_cfg = dict(ckpt.get("config") or {})
    if not base_cfg:
        raise ValueError(f"Checkpoint {args.ckpt} has no embedded 'config'; cannot rebuild the model.")

    cfg = _deep_merge_dict(base_cfg, _load_yaml(args.eval_config))
    _apply_set_overrides(cfg, args.overrides)

    runtime = init_runtime(cfg)
    schedule = build_schedule(cfg)
    process = build_process(cfg, schedule).to(runtime.device)
    model_ctx = build_model_context(cfg, runtime)
    denoiser = _build_denoiser(cfg, model_ctx.model, process)

    use_ema = bool(cfg.get("train", {}).get("use_ema", True)) and ("ema" in ckpt)
    ema = EMA(model=denoiser, decay=float(cfg.get("train", {}).get("ema_decay", 0.999))) if use_ema else None

    # strict=True so the loader actively finds the right prefix remap (e.g.
    # compile's `net._orig_mod.` -> `net.`); strict=False would silently load
    # nothing and leave the net at random init.
    info = _load_model_state_with_fallback(denoiser, ckpt["model"], strict=True)
    if ema is not None:
        # The EMA shadow was captured with the same (compiled) keys, so apply the
        # transform the model loader used before copying EMA weights in.
        shadow = _apply_prefix_transform(ckpt["ema"]["shadow"], info["prefix"])
        ema.load_state_dict({"decay": ckpt["ema"]["decay"], "shadow": shadow})
    step = int(ckpt.get("step", 0))
    resume = ResumeState(start_epoch=0, global_step=step, run_step=0, ckpt_mode="full")

    gen_cfg = parse_gen_cfg(cfg)
    if runtime.is_main:
        missing = getattr(info, "missing_keys", None) if info is not None else None
        runtime.logger.log_text(
            f"[eval_fid] ckpt={args.ckpt} step={step} use_ema={use_ema} "
            f"sampler={gen_cfg['sampler']} steps={gen_cfg['steps']} cfg={gen_cfg['guidance_scale']} "
            f"num_samples={gen_cfg['num_samples']} missing_keys={missing}"
        )
    _run_generative_eval(
        cfg=cfg, gen_cfg=gen_cfg, model=denoiser, runtime=runtime,
        model_ctx=model_ctx, process=process, ema=ema, resume=resume, tag=args.tag,
    )
    runtime.logger.close()


if __name__ == "__main__":
    main()
