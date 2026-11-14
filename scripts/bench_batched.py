"""Upper bound: plain batched fwd+bwd (no per-sample grads), with and without SDPA."""
import time

import torch
import torch.nn.functional as F

from triguard.backbone import load_backbone
from triguard.model import ModelCfg, TailModel

dev = "cuda"
bb = load_backbone("dinov2_vits14", "cpu")
tail = TailModel(bb.blocks, bb.norm, 8, 8, ModelCfg()).to(dev)
p = {k: v.requires_grad_(True) for k, v in tail.init_params(0).items()}
for N in (32, 64, 128):
    X = torch.randn(N, 257, 384, device=dev)
    Y = torch.randint(0, 8, (N,), device=dev)
    for sdpa in (False, True):
        tail.use_sdpa = sdpa
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(6):
                if i == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(tail.logits(p, X), Y)
                torch.autograd.grad(loss, list(p.values()))
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 5
            print(f"N={N} sdpa={sdpa}: {N/dt:.0f} view-passes/s peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"N={N} sdpa={sdpa}: OOM")
            torch.cuda.empty_cache()
