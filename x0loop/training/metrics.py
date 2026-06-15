from __future__ import annotations

import torch
import torch.distributed as dist

from x0loop.core.process_base import BaseProcess, ForwardBatch
from x0loop.core.schedules import TimeSchedule
from x0loop.losses.atomic import CompositeLoss, regress


def endpoint_loss_label(process: BaseProcess) -> str:
    return "z" if hasattr(process, "mu_data") else "eps"


def bucket_losses(t: torch.Tensor, per_example_loss: torch.Tensor) -> dict[str, float]:
    edges = [0.0, 0.1, 0.3, 0.7, 1.0]
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (t >= lo) & (t <= hi)
        else:
            mask = (t >= lo) & (t < hi)
        key = f"loss_t{lo}_{hi}".replace(".", "p")
        if mask.any():
            out[key] = float(per_example_loss[mask].mean().item())
        else:
            out[key] = 0.0
    return out



def compute_tbin_sums(
    t: torch.Tensor,
    per_example_loss: torch.Tensor,
    per_example_weight: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = t.detach().float().clamp(0.0, 1.0)
    loss = per_example_loss.detach().to(torch.float64)
    weight = per_example_weight.detach().to(torch.float64)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=t.device)
    idx = torch.bucketize(t, edges[1:-1], right=False)

    counts = torch.bincount(idx, minlength=num_bins).to(torch.float64)
    sum_loss = torch.zeros(num_bins, device=t.device, dtype=torch.float64)
    sum_weight = torch.zeros(num_bins, device=t.device, dtype=torch.float64)
    sum_loss.scatter_add_(0, idx, loss)
    sum_weight.scatter_add_(0, idx, weight)
    return counts, sum_weight, sum_loss


def compute_tbin_value_sum(
    t: torch.Tensor,
    per_example_value: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = t.detach().float().clamp(0.0, 1.0)
    value = per_example_value.detach().to(torch.float64)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=t.device)
    idx = torch.bucketize(t, edges[1:-1], right=False)

    counts = torch.bincount(idx, minlength=num_bins).to(torch.float64)
    sum_value = torch.zeros(num_bins, device=t.device, dtype=torch.float64)
    sum_value.scatter_add_(0, idx, value)
    return counts, sum_value



def format_tbin_summary(
    edges: torch.Tensor,
    counts: torch.Tensor,
    avg_a: torch.Tensor,
    avg_w: torch.Tensor,
    avg_eps: torch.Tensor,
    avg_x0: torch.Tensor,
    avg_v: torch.Tensor,
    extra_avgs: dict[str, torch.Tensor] | None = None,
    endpoint_label: str = "eps",
) -> str:
    parts = []
    n = counts.numel()
    for i in range(n):
        left = float(edges[i].item())
        right = float(edges[i + 1].item())
        close = "]" if i == n - 1 else ")"
        cnt = int(counts[i].item())
        fields = [
            f"n={cnt}",
            f"a={float(avg_a[i].item()):.4g}",
            f"w={float(avg_w[i].item()):.4g}",
            f"l{endpoint_label}={float(avg_eps[i].item()):.4g}",
            f"lx0={float(avg_x0[i].item()):.4g}",
            f"lv={float(avg_v[i].item()):.4g}",
        ]
        for key, values in (extra_avgs or {}).items():
            fields.append(f"{key}={float(values[i].item()):.4g}")
        parts.append(f"[{left:.2f},{right:.2f}{close}: {', '.join(fields)}")
    return " | ".join(parts)


class TimeBinAccumulator:
    def __init__(self, *, num_bins: int, device: torch.device):
        self.num_bins = int(num_bins)
        self.edges = torch.linspace(0.0, 1.0, self.num_bins + 1, device=device, dtype=torch.float64)
        self.counts = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_alpha = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_weight = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_eps = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_x0 = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_v = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.endpoint_label = "eps"
        self.extra_counts: dict[str, torch.Tensor] = {}
        self.extra_sums: dict[str, torch.Tensor] = {}

    def update_extra(self, *, t: torch.Tensor, values: dict[str, torch.Tensor] | None) -> None:
        if not values:
            return
        for key, value in values.items():
            if value is None:
                continue
            if value.ndim > 1:
                value = value.view(value.shape[0], -1).mean(dim=1)
            c, s = compute_tbin_value_sum(t, value, num_bins=self.num_bins)
            if key not in self.extra_counts:
                self.extra_counts[key] = torch.zeros_like(self.counts)
                self.extra_sums[key] = torch.zeros_like(self.sum_eps)
            self.extra_counts[key] += c
            self.extra_sums[key] += s

    def update(
        self,
        *,
        schedule: TimeSchedule,
        process: BaseProcess,
        loss_fn: CompositeLoss,
        fb: ForwardBatch,
        out: torch.Tensor,
        extra_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        t = fb.t.detach()
        out_d = out.detach()
        self.endpoint_label = endpoint_loss_label(process)

        # Per-example unweighted MSE for terminal endpoint/x0/v (diagnostic, always MSE regardless of training formula).
        eps_u = regress("mse", process.eps_from_output(fb.xt, t, out_d, aux={}), process.eps_target(fb).detach())
        x0_u = regress("mse", process.x0_from_output(fb.xt, t, out_d, aux={}), process.x0_target(fb).detach())
        v_u = regress("mse", process.v_from_output(fb.xt, t, out_d, aux={}), process.v_target(fb).detach())
        if hasattr(process, "mu_data") and getattr(process, "predict_mudata", False):
            lmu = regress(
                "mse",
                process.mudata_from_output(fb.xt, t, out_d, aux={}),
                process.mudata_target(fb).detach(),
            )
            extra_values = {**(extra_values or {}), "lmu": lmu}

        # Prefer the actual outer objective weight; fall back to explicit per-space atom weight.
        if getattr(loss_fn, "outer_weight_fn", None) is not None:
            w = loss_fn.outer_weight(fb, x0_u).float()
        else:
            ref_atom = next((a for a in loss_fn.atoms if a.weight_fn is not None), None)
            if ref_atom is not None:
                w = ref_atom.weight_fn(t, None)
                if w.ndim > 1:
                    w = w.view(w.shape[0], -1).mean(dim=1)
                w = w.float()
            else:
                w = torch.ones(t.shape[0], device=t.device, dtype=torch.float32)

        alpha_t = schedule.alpha(t)
        alpha_t = alpha_t.detach().float()
        if alpha_t.ndim > 1:
            alpha_t = alpha_t.view(alpha_t.shape[0], -1).mean(dim=1)

        c, sw, sl_eps = compute_tbin_sums(t, eps_u, w, num_bins=self.num_bins)
        _, _, sl_x0 = compute_tbin_sums(t, x0_u, w, num_bins=self.num_bins)
        _, _, sl_v = compute_tbin_sums(t, v_u, w, num_bins=self.num_bins)
        _, sa = compute_tbin_value_sum(t, alpha_t, num_bins=self.num_bins)

        self.counts += c
        self.sum_alpha += sa
        self.sum_weight += sw
        self.sum_eps += sl_eps
        self.sum_x0 += sl_x0
        self.sum_v += sl_v
        self.update_extra(t=t, values=extra_values)

    def summary(self, *, is_distributed: bool) -> str:
        rc = self.counts.clone()
        rsa = self.sum_alpha.clone()
        rsw = self.sum_weight.clone()
        rse = self.sum_eps.clone()
        rsxl = self.sum_x0.clone()
        rsvl = self.sum_v.clone()
        extra_counts = {k: v.clone() for k, v in self.extra_counts.items()}
        extra_sums = {k: v.clone() for k, v in self.extra_sums.items()}
        if is_distributed and dist.is_available() and dist.is_initialized():
            for t in (rc, rsa, rsw, rse, rsxl, rsvl):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            for key in extra_counts:
                dist.all_reduce(extra_counts[key], op=dist.ReduceOp.SUM)
                dist.all_reduce(extra_sums[key], op=dist.ReduceOp.SUM)

        denom = rc.clamp_min(1.0)
        extra_avgs = {
            key: extra_sums[key] / extra_counts[key].clamp_min(1.0)
            for key in extra_sums
        }
        return format_tbin_summary(
            self.edges, rc,
            rsa / denom, rsw / denom,
            rse / denom, rsxl / denom, rsvl / denom,
            extra_avgs,
            self.endpoint_label,
        )

    def reset(self) -> None:
        for t in (self.counts, self.sum_alpha, self.sum_weight, self.sum_eps, self.sum_x0, self.sum_v):
            t.zero_()
        for t in (*self.extra_counts.values(), *self.extra_sums.values()):
            t.zero_()
