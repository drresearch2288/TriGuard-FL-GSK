import time

import torch

from triguard.backbone import load_backbone
from triguard.model import ModelCfg, TailModel
from triguard.psg import per_sample_grads

dev = "cuda"
bb = load_backbone("dinov2_vits14", "cpu")
tail = TailModel(bb.blocks, bb.norm, 8, 8, ModelCfg()).to(dev)
tail.use_sdpa = True
p = tail.init_params(0)
names = list(p)
for P, K in [(8, 4), (16, 4), (32, 4), (48, 4)]:
    torch.cuda.reset_peak_memory_stats()
    try:
        X = torch.randn(P, K, 257, 384, device=dev)
        Y = torch.randint(0, 8, (P,), device=dev)
        per_sample_grads(tail, p, names, X, Y)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            per_sample_grads(tail, p, names, X, Y)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 5
        print(f"P={P} K={K}: {P*K/dt:.0f} view-passes/s  peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    except torch.cuda.OutOfMemoryError:
        print("OOM", P)
        torch.cuda.empty_cache()
