from __future__ import annotations

import math
import os

import torch
from PIL import Image, ImageDraw

from x0loop.core.image_normalization import image_to_display_minus_one_one
from x0loop.training.context import LoopConfig, ResumeState, RuntimeContext
from x0loop.training.sampling import build_sample_label_names


def _enabled(cfg: dict) -> bool:
    obs_cfg = (cfg.get("trainset_observe", {}) or cfg.get("trainset_observation", {}) or {})
    logging_cfg = cfg.get("logging", {}) or {}
    return bool(obs_cfg.get("enabled", logging_cfg.get("trainset_observe_enabled", False)))


def _cfg_value(cfg: dict, key: str, default):
    obs_cfg = (cfg.get("trainset_observe", {}) or cfg.get("trainset_observation", {}) or {})
    logging_cfg = cfg.get("logging", {}) or {}
    return obs_cfg.get(key, logging_cfg.get(f"trainset_observe_{key}", default))


def _should_run(cfg: dict, *, loop_cfg: LoopConfig, resume: ResumeState, epoch: int, micro_step: int) -> bool:
    if not _enabled(cfg) or resume.global_step <= 0:
        return False
    every_steps = int(_cfg_value(cfg, "every_steps", 0) or 0)
    if every_steps > 0:
        return resume.global_step % every_steps == 0

    every_epochs = int(_cfg_value(cfg, "every_epochs", 0) or 0)
    if every_epochs <= 0:
        return False
    return micro_step == 0 and (epoch + 1) % every_epochs == 0


def _label_text(label_id: int | None, cfg: dict, label_names: tuple[str, ...] | None) -> str:
    if label_id is None:
        return "n/a"
    null_id = int((cfg.get("model", {}) or {}).get("num_classes", -1))
    if label_id == null_id:
        return "null"
    if label_names is not None and 0 <= label_id < len(label_names):
        return label_names[label_id]
    return str(label_id)


def _flat_label_ids(label: torch.Tensor | None, n: int) -> list[int | None]:
    if label is None:
        return [None] * n
    values = label.detach().cpu().flatten().tolist()
    return [int(v) for v in values[:n]]


def _image_to_uint8_rgb(x: torch.Tensor, *, cfg: dict, scale: int) -> Image.Image:
    x = image_to_display_minus_one_one(x.detach().float().cpu(), cfg)
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")
    c, h, w = x.shape
    if c == 1:
        x = x.repeat(3, 1, 1)
    elif c >= 3:
        x = x[:3]
    else:
        pad = torch.zeros((3 - c, h, w), dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    arr = x.permute(1, 2, 0).contiguous().numpy()
    image = Image.fromarray(arr)
    if scale > 1:
        nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        image = image.resize((w * scale, h * scale), resample=nearest)
    return image


def _first_n(x: torch.Tensor | None, n: int) -> torch.Tensor | None:
    if x is None:
        return None
    return x.detach()[:n]


def _save_pair_grid(
    *,
    cfg: dict,
    out_path: str,
    source: str,
    input_name: str,
    input_tensor: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    label: torch.Tensor | None,
    cond: torch.Tensor | None,
    age: torch.Tensor | None,
    pred_x0: torch.Tensor | None,
    step: int,
) -> None:
    n = int(input_tensor.shape[0])
    if n <= 0:
        return
    cols = int(_cfg_value(cfg, "cols", 5) or 5)
    scale = int(_cfg_value(cfg, "scale", 3) or 3)
    gap = int(_cfg_value(cfg, "gap", 6) or 6)
    text_h = int(_cfg_value(cfg, "text_h", 30) or 30)
    label_names = build_sample_label_names(cfg)

    img0 = _image_to_uint8_rgb(input_tensor[0], cfg=cfg, scale=scale)
    img_w, img_h = img0.size
    cell_w = max(img_w, 118)
    cell_h = img_h + text_h
    entries = [(input_name, input_tensor), ("x0", x0)]
    if pred_x0 is not None:
        entries.append(("x0_hat", pred_x0))
    rows_per_group = len(entries)
    sample_rows = int(math.ceil(n / cols))
    rows = sample_rows * rows_per_group
    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas_h = rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    label_ids = _flat_label_ids(label, n)
    cond_ids = _flat_label_ids(cond, n)
    t_values = t.detach().float().cpu().flatten().tolist()
    age_values = age.detach().float().cpu().flatten().tolist() if age is not None else None

    for i in range(n):
        group_row = i // cols
        col = i % cols
        for row_offset, (name, tensor_batch) in enumerate(entries):
            tensor = tensor_batch[i]
            row = group_row * rows_per_group + row_offset
            x_left = gap + col * (cell_w + gap)
            y_top = gap + row * (cell_h + gap)
            image = _image_to_uint8_rgb(tensor, cfg=cfg, scale=scale)
            canvas.paste(image, (x_left + (cell_w - image.size[0]) // 2, y_top))
            label_txt = _label_text(label_ids[i], cfg, label_names)
            cond_txt = _label_text(cond_ids[i], cfg, label_names)
            t_name = "t1" if source == "bank" else "t"
            meta = f"{source} {name} y={label_txt} cond={cond_txt}"
            time_meta = f"{t_name}={float(t_values[i]):.3f}"
            if source == "bank" and age_values is not None:
                time_meta += f" age={int(round(float(age_values[i])))}"
            draw.text((x_left, y_top + img_h + 1), meta, fill=(0, 0, 0))
            draw.text((x_left, y_top + img_h + 15), time_meta, fill=(0, 0, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def run_trainset_observation_if_due(
    *,
    cfg: dict,
    runtime: RuntimeContext,
    loop_cfg: LoopConfig,
    resume: ResumeState,
    epoch: int,
    micro_step: int,
    observe: dict | None,
) -> None:
    if not runtime.is_main or not observe:
        return
    if not _should_run(cfg, loop_cfg=loop_cfg, resume=resume, epoch=epoch, micro_step=micro_step):
        return

    num = int(_cfg_value(cfg, "num", 10) or 10)
    out_dir_name = str(_cfg_value(cfg, "dir", "trainset_observe"))
    out_dir = os.path.join(runtime.out_dir, out_dir_name)
    saved: list[str] = []

    fresh = observe.get("fresh") or {}
    fresh_input = _first_n(fresh.get("input"), num)
    fresh_x0 = _first_n(fresh.get("x0"), num)
    fresh_t = _first_n(fresh.get("t"), num)
    fresh_label = _first_n(fresh.get("label"), num)
    fresh_cond = _first_n(fresh.get("cond"), num)
    fresh_pred_x0 = _first_n(fresh.get("pred_x0"), num)
    if fresh_input is not None and fresh_x0 is not None and fresh_t is not None:
        path = os.path.join(out_dir, f"step_{resume.global_step:08d}_fresh.png")
        _save_pair_grid(
            cfg=cfg,
            out_path=path,
            source="fresh",
            input_name="xt",
            input_tensor=fresh_input,
            x0=fresh_x0,
            t=fresh_t,
            label=fresh_label,
            cond=fresh_cond,
            age=None,
            pred_x0=fresh_pred_x0,
            step=resume.global_step,
        )
        saved.append(path)

    bank = observe.get("bank") or {}
    bank_input = _first_n(bank.get("input"), num)
    bank_x0 = _first_n(bank.get("x0"), num)
    bank_t = _first_n(bank.get("t"), num)
    bank_label = _first_n(bank.get("label"), num)
    bank_cond = _first_n(bank.get("cond"), num)
    bank_age = _first_n(bank.get("age"), num)
    bank_pred_x0 = _first_n(bank.get("pred_x0"), num)
    if bank_input is not None and bank_x0 is not None and bank_t is not None:
        path = os.path.join(out_dir, f"step_{resume.global_step:08d}_bank.png")
        _save_pair_grid(
            cfg=cfg,
            out_path=path,
            source="bank",
            input_name="xt1_hat",
            input_tensor=bank_input,
            x0=bank_x0,
            t=bank_t,
            label=bank_label,
            cond=bank_cond,
            age=bank_age,
            pred_x0=bank_pred_x0,
            step=resume.global_step,
        )
        saved.append(path)

    if saved:
        runtime.logger.log_text(f"[trainset_observe] saved step={resume.global_step} files={', '.join(saved)}")
