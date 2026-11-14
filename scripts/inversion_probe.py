"""Quick probe: which gradient-inversion settings actually reconstruct images (no DP, ResNet-18 GN, batch 8)?"""
import os
import sys
import time

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from triguard.audit import _target_images, _tv
from triguard.cnn import make_model

dev = "cuda"
torch.manual_seed(0)
model = make_model(8, os.environ.get("PROBE_NORM", "gn")).to(dev).eval()
mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
params = list(model.parameters())
import os

BS = int(os.environ.get("PROBE_BS", 8))
rows, imgs = _target_images(BS)
x, y = imgs.to(dev), torch.as_tensor(rows.y.values, device=dev)
obs = [g.detach() for g in torch.autograd.grad(F.cross_entropy(model((x - mean) / std), y), params)]
cfgs = {"A_current": dict(signed=False, tv=1e-4, lr=0.1), "B_signed": dict(signed=True, tv=1e-4, lr=0.1),
        "C_signed_tv": dict(signed=True, tv=1e-2, lr=0.1), "D_layercos": dict(signed=True, tv=1e-2, lr=0.1, layer=True), "E_lr01": dict(signed=True, tv=1e-2, lr=0.01), "F_lr003": dict(signed=True, tv=1e-2, lr=0.003), "G_lr01_lowtv": dict(signed=True, tv=1e-3, lr=0.01), "H_lr001": dict(signed=True, tv=1e-2, lr=0.001), "I_lr001_tv": dict(signed=True, tv=1e-1, lr=0.001), "J_lr0003": dict(signed=True, tv=1e-2, lr=0.0003)}
which = sys.argv[1:] or list(cfgs)
iters = 2000
for name in which:
    c = cfgs[name]
    img = (torch.rand_like(x) * 0.2 + 0.4).requires_grad_(True)
    opt = torch.optim.Adam([img], lr=c["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * f) for f in (0.375, 0.625, 0.875)], 0.1)
    t0 = time.time()
    for it in range(1, iters + 1):
        g = torch.autograd.grad(F.cross_entropy(model((img - mean) / std), y), params, create_graph=True)
        num = sum((a * b).sum() for a, b in zip(g, obs))
        den = torch.sqrt(sum((a**2).sum() for a in g)) * torch.sqrt(sum((b**2).sum() for b in obs))
        if c.get("layer"):
            match = torch.stack([1 - F.cosine_similarity(a.flatten(), b.flatten(), dim=0) for a, b in zip(g, obs)]).mean()
        else:
            match = 1 - num / den
        loss = match + c["tv"] * _tv(img)
        opt.zero_grad()
        loss.backward()
        if c["signed"]:
            img.grad.sign_()
        opt.step()
        sched.step()
        with torch.no_grad():
            img.clamp_(0, 1)
        if it in (250, 500, 1000, 2000):
            s = ssim(img.detach(), x, data_range=1.0).item()
            print(f"{name} it={it} cos_loss={1 - float(num / den):.4f} SSIM={s:.3f} {time.time() - t0:.0f}s", flush=True)
