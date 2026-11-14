"""Measure per-sample-gradient throughput and peak VRAM for the tail model on this GPU."""
import sys
import time

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from triguard.backbone import load_backbone
from triguard.model import ModelCfg, TailModel

dev = "cuda"
bb = load_backbone(sys.argv[1] if len(sys.argv) > 1 else "dinov2_vits14", "cpu")
tail = TailModel(bb.blocks, bb.norm, 8, 8, ModelCfg()).to(dev)
p = tail.init_params(0)
T, D = 257 if "dinov2" in bb.name else 197, bb.dim


def loss1(tp, x, y):
    return F.cross_entropy(tail.logits(tp, x), y.expand(x.shape[0]))


f = vmap(grad(loss1), in_dims=(None, 0, 0))
for P, K in [(8, 4), (16, 4), (24, 4), (32, 4), (64, 1)]:
    try:
        torch.cuda.reset_peak_memory_stats()
        X = torch.randn(P, K, T, D, device=dev)
        Y = torch.randint(0, 8, (P,), device=dev)
        for amp in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                f(p, X, Y)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    f(p, X, Y)
                torch.cuda.synchronize()
            dt = (time.time() - t0) / 5
            print(f"P={P} K={K} amp={amp}: {dt*1000:.0f} ms/chunk  {P*K/dt:.0f} view-passes/s  "
                  f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"P={P} K={K}: OOM")
        torch.cuda.empty_cache()
