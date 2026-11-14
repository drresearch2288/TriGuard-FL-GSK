import torch

from triguard.backbone import load_backbone
from triguard.model import ModelCfg, TailModel
from triguard.psg import per_sample_grads
from triguard import dp

dev = "cuda"
bb = load_backbone("dinov2_vits14", "cpu")
tail = TailModel(bb.blocks, bb.norm, 8, 3, ModelCfg()).to(dev)
tail.use_sdpa = True
p = tail.init_params(0)
names = list(p)
for step in range(6):
    X = torch.randn(8, 2, 257, 384, device=dev)
    Y = torch.randint(0, 3, (8,), device=dev)
    g = per_sample_grads(tail, p, names, X, Y)
    acc = {}
    dp.clip_and_accumulate(g, 1.0, acc)
    print(step, f"{torch.cuda.memory_allocated()/1e9:.3f} GB", {k: (v.requires_grad, v.grad_fn is not None) for k, v in list(g.items())[:2]})
