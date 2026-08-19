from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionMatchingConfig:
    enabled: bool = False
    objective: str = "inception_kid"
    weight: float = 1.0
    start_step: int = 10000
    warmup_steps: int = 1000
    batch_ratio: float = 0.0625
    gradient_ratio: float = 0.10
    scale_max: float = 10.0
    degree: int = 3
    terminal_steps: int = 20
    terminal_sampler: str = "heun"
    terminal_guidance_scale: float = 2.2
    suffix_steps: int = 4


def build_distribution_matching_config(cfg: dict) -> DistributionMatchingConfig:
    raw = cfg.get("distribution_matching", {}) or {}
    terminal = raw.get("terminal", {}) or {}
    gen_eval = cfg.get("gen_eval", {}) or {}
    result = DistributionMatchingConfig(
        enabled=bool(raw.get("enabled", False)),
        objective=str(raw.get("objective", "inception_kid")).lower(),
        weight=float(raw.get("weight", 1.0)),
        start_step=int(raw.get("start_step", 10000)),
        warmup_steps=int(raw.get("warmup_steps", 1000)),
        batch_ratio=float(raw.get("batch_ratio", 0.0625)),
        gradient_ratio=float(raw.get("gradient_ratio", 0.10)),
        scale_max=float(raw.get("scale_max", 10.0)),
        degree=int(raw.get("degree", 3)),
        terminal_steps=int(terminal.get("steps", gen_eval.get("steps", 20))),
        terminal_sampler=str(terminal.get("sampler", gen_eval.get("sampler", "heun"))).lower(),
        terminal_guidance_scale=float(
            terminal.get("guidance_scale", gen_eval.get("guidance_scale", 2.2))
        ),
        suffix_steps=int(terminal.get("suffix_steps", 4)),
    )
    if result.objective != "inception_kid":
        raise ValueError(
            "distribution_matching.objective must be inception_kid for the registered experiment, "
            f"got {result.objective!r}"
        )
    if result.weight < 0.0:
        raise ValueError(f"distribution_matching.weight must be non-negative, got {result.weight}")
    if result.start_step < 0 or result.warmup_steps < 0:
        raise ValueError("distribution_matching start/warmup steps must be non-negative")
    if not 0.0 < result.batch_ratio <= 1.0:
        raise ValueError(
            f"distribution_matching.batch_ratio must be in (0,1], got {result.batch_ratio}"
        )
    if not 0.0 <= result.gradient_ratio <= 1.0:
        raise ValueError(
            f"distribution_matching.gradient_ratio must be in [0,1], got {result.gradient_ratio}"
        )
    if result.scale_max <= 0.0 or result.degree <= 0:
        raise ValueError("distribution_matching scale_max and degree must be positive")
    if result.terminal_sampler != "heun":
        raise ValueError("distribution_matching terminal sampler must be heun")
    if result.terminal_steps <= 0 or not 1 <= result.suffix_steps <= result.terminal_steps:
        raise ValueError(
            "distribution_matching terminal requires steps>0 and 1<=suffix_steps<=steps, "
            f"got steps={result.terminal_steps}, suffix_steps={result.suffix_steps}"
        )
    return result


def distribution_matching_weight(config: DistributionMatchingConfig, step: int) -> float:
    if (not config.enabled) or config.weight <= 0.0 or step < config.start_step:
        return 0.0
    if config.warmup_steps <= 0:
        return float(config.weight)
    progress = min(max((step - config.start_step + 1) / float(config.warmup_steps), 0.0), 1.0)
    return float(config.weight) * progress
