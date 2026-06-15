import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class PatchToy(nn.Module):
    def __init__(self, mode="conv1x1", dim=512, bottleneck=128, depth=8):
        super().__init__()
        self.mode = mode
        self.proj1 = nn.Conv2d(3, bottleneck, kernel_size=4, stride=4)
        if mode == "conv1x1":
            self.proj2 = nn.Conv2d(bottleneck, dim, kernel_size=1)
        elif mode == "linear":
            self.proj2 = nn.Linear(bottleneck, dim)
        else:
            raise ValueError("mode must be conv1x1 or linear")
        blocks = []
        for _ in range(depth):
            blocks += [
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(dim, 3 * 32 * 32)

    def forward(self, x):
        x = self.proj1(x)
        if self.mode == "conv1x1":
            x = self.proj2(x).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
            x = self.proj2(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.head(x)


def setup():
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["conv1x1", "linear"], default="conv1x1")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--batch-per-rank", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    local_rank, rank, world = setup()
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = PatchToy(mode=args.mode, depth=args.depth).to(device)
    if args.compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(args.batch_per_rank, 3, 32, 32, device=device)
    y = torch.randn(args.batch_per_rank, 3 * 32 * 32, device=device)
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.dtype)

    start = None
    for step in range(1, args.steps + 1):
        if step == args.warmup + 1:
            torch.cuda.synchronize()
            start = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        if autocast_dtype is None:
            pred = model(x)
            loss = (pred - y).square().mean()
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                pred = model(x)
                loss = (pred - y).square().mean()
        loss.backward()
        opt.step()

        if rank == 0 and step % 50 == 0:
            torch.cuda.synchronize()
            if start is not None:
                elapsed = time.perf_counter() - start
                done = step - args.warmup
                img_s = done * args.batch_per_rank * world / elapsed
            else:
                img_s = 0.0
            print(
                "step=%d img_s=%.1f loss=%.6f mode=%s compile=%s dtype=%s python=%s torch=%s"
                % (
                    step,
                    img_s,
                    float(loss.detach().cpu()),
                    args.mode,
                    args.compile,
                    args.dtype,
                    sys.version.split()[0],
                    torch.__version__,
                ),
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
