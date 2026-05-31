from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"pattern not found: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    train_path = Path("x0loop/train.py")
    text = train_path.read_text()

    if "from x0loop.core.time_sampling import build_time_sampler" not in text:
        text = replace_once(
            text,
            "from x0loop.core.schedules import TimeSchedule\n",
            "from x0loop.core.schedules import TimeSchedule\nfrom x0loop.core.time_sampling import build_time_sampler\n",
            label="import build_time_sampler",
        )

    if "    time_sampler: object\n" not in text:
        text = replace_once(
            text,
            "@dataclass\nclass TrainComponents:\n    schedule: TimeSchedule\n    process: object\n",
            "@dataclass\nclass TrainComponents:\n    schedule: TimeSchedule\n    time_sampler: object\n    process: object\n",
            label="TrainComponents.time_sampler",
        )

    if "time_sampler = build_time_sampler(cfg, schedule)" not in text:
        text = replace_once(
            text,
            "def build_train_components(cfg: dict, model_ctx: ModelContext, runtime: RuntimeContext) -> TrainComponents:\n    schedule = build_schedule(cfg)\n    process = build_process(cfg, schedule)\n    loss_fn = _build_loss(cfg[\"loss\"], schedule)\n",
            "def build_train_components(cfg: dict, model_ctx: ModelContext, runtime: RuntimeContext) -> TrainComponents:\n    schedule = build_schedule(cfg)\n    time_sampler = build_time_sampler(cfg, schedule)\n    process = build_process(cfg, schedule)\n    loss_fn = _build_loss(cfg[\"loss\"], schedule)\n",
            label="build_train_components.time_sampler",
        )

    if "[time_sampler]" not in text:
        text = replace_once(
            text,
            "    if runtime.is_main:\n        atom_descs = \", \".join(repr(a) for a in loss_fn.atoms)\n        runtime.logger.log_text(f\"[loss] {atom_descs}\")\n",
            "    if runtime.is_main:\n        atom_descs = \", \".join(repr(a) for a in loss_fn.atoms)\n        runtime.logger.log_text(f\"[loss] {atom_descs}\")\n        runtime.logger.log_text(f\"[time_sampler] {cfg.get('time_sampler', {'name': 'legacy'})}\")\n",
            label="time_sampler logging",
        )

    if "        time_sampler=time_sampler,\n" not in text:
        text = replace_once(
            text,
            "    return TrainComponents(\n        schedule=schedule,\n        process=process,\n",
            "    return TrainComponents(\n        schedule=schedule,\n        time_sampler=time_sampler,\n        process=process,\n",
            label="TrainComponents return time_sampler",
        )

    text = text.replace(
        "    t = components.schedule.sample_t(bsz, device=runtime.device)\n",
        "    t = components.time_sampler.sample(bsz, device=runtime.device)\n",
    )

    train_path.write_text(text)

    yaml_paths = [
        Path("train_run/configs/cifar10/cifar10_dit_flow_train_x0.yaml"),
        Path("train_run/configs/cifar10/cifar10_dit_flow_train_x0_weighted.yaml"),
    ]
    for path in yaml_paths:
        if not path.exists():
            continue
        y = path.read_text()
        if "time_sampler:" not in y:
            if "schedule:\n  mode: flow\n" in y:
                y = y.replace(
                    "schedule:\n  mode: flow\n",
                    "schedule:\n  mode: flow\n\ntime_sampler:\n  name: uniform_continuous\n",
                    1,
                )
            elif "schedule:\n  mode: diffusion\n" in y:
                y = y.replace(
                    "schedule:\n  mode: diffusion\n",
                    "schedule:\n  mode: diffusion\n\ntime_sampler:\n  name: uniform_discrete\n  num_steps: 1000\n",
                    1,
                )
        y = y.replace("outer_weight: x0", "outer_weight: target")
        y = y.replace("outer_weight_power: 0.5", "outer_weight_power: 2.0")
        y = y.replace("out_dir_base: runs/cifar10_flow_weighted_x0", "out_dir_base: runs/cifar10_flow_weighted_target")
        path.write_text(y)


if __name__ == "__main__":
    main()
