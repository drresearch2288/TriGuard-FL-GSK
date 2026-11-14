"""Stage 6: PrivAudit — one-run canary audit, LiRA membership inference, gradient inversion.

Site 1 of the plan (BCN) is site id 0 here.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom, norm
from sklearn.metrics import roc_auc_score, roc_curve

from triguard.cache import Cache
from triguard.paths import RUNS
from triguard.utils import Logger, device, load_json, save_json

TARGET_SITE = 0


# ====================================================================== one-run audit (Steinke et al. 2023)
def eps_lower_bound(r: int, v: int, conf: float = 0.95) -> float:
    """Largest eps such that v correct out of r guesses would be implausible (p < 1-conf) under eps-DP.

    Under eps-DP each guess is correct with prob <= e^eps / (1 + e^eps) (Steinke, Nasr, Jagielski 2023, Thm 1;
    the delta term is negligible here: delta ~ 1e-4 per site and r <= 200)."""
    if r == 0:
        return 0.0
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = math.exp(mid) / (1 + math.exp(mid))
        if binom.sf(v - 1, r, p) < 1 - conf:  # P[X >= v] too small -> eps must be larger
            lo = mid
        else:
            hi = mid
    return lo


def canary_audit(jid: str, eps: float, m: int = 1000, seed: int = 0, k_guess=(25, 50, 100, 200)):
    from triguard.fl import FLTrainer
    from triguard.jobrun import make_fl_cfg
    from triguard.experiments import method_cfg

    rng = np.random.default_rng(seed + 12345)
    can = Cache("isic2020_canary")
    cdf = can.df.head(m)
    secret = rng.random(len(cdf)) < 0.5  # S_i ~ Bernoulli(1/2)
    inc = cdf[secret]
    args = method_cfg("triguard")
    args.update(dataset="isic2019", eps=float(eps), seed=seed, rounds=50,
                extra_train={"dataset": "isic2020_canary", "rows": inc.aug_row.astype(int).tolist(),
                             "labels": inc.y.astype(int).tolist()})
    tr = FLTrainer(make_fl_cfg(jid, args))
    tr.run()
    site = tr.sites[0]
    # loss of every canary (in or out) under the target site's final model, on the clean view
    with torch.no_grad():
        x = can.get_clean(cdf.cache_row.values).to(tr.dev).float()
        lg = torch.cat([tr.model.logits(site.params, x[i : i + 128]) for i in range(0, len(x), 128)])
        loss = F.cross_entropy(lg, torch.as_tensor(cdf.y.values, device=tr.dev), reduction="none").cpu().numpy()
    order = np.argsort(loss)  # low loss -> guess IN
    res = {"eps_theory": eps, "m": len(cdf), "n_in": int(secret.sum()), "guesses": {}}
    for k in k_guess:
        g_in, g_out = order[:k], order[-k:]
        v = int(secret[g_in].sum() + (~secret[g_out]).sum())
        r = 2 * k
        res["guesses"][k] = {"r": r, "correct": v, "precision": v / r, "eps_lb_95": eps_lower_bound(r, v)}
    res["eps_emp"] = res["guesses"][100]["eps_lb_95"]  # pre-registered k+ = k- = 100
    res["eps_emp_max_over_k"] = max(g["eps_lb_95"] for g in res["guesses"].values())
    res["auc_in_vs_out"] = float(roc_auc_score(secret, -loss))
    save_json(res, RUNS / jid / "audit_result.json")
    print(res)


# ====================================================================== LiRA (Carlini et al. 2022)
N_SHADOWS = 8  # deliverable scope (plan: 16); every sample is IN for exactly half of the shadows


def lira_membership(n: int, n_shadows: int = N_SHADOWS, seed: int = 0) -> np.ndarray:
    """keep[s, i]: sample i in shadow s; every sample is IN for exactly half of the shadows."""
    rng = np.random.default_rng(seed)
    r = rng.random((n_shadows, n)).argsort(0)
    return r < n_shadows // 2


def lira_shadow(jid: str, eps: float, shadow: int, rounds: int = 30):
    from triguard.experiments import method_cfg
    from triguard.fl import FLTrainer
    from triguard.jobrun import make_fl_cfg

    cache = Cache("isic2019")
    df = cache.df
    tgt = df[(df.split == "train") & (df.site == TARGET_SITE)].sort_values("cache_row")
    keep = lira_membership(len(tgt))[shadow]
    mask = np.ones(len(df), bool)
    mask[tgt.cache_row.values[~keep]] = False
    out = RUNS / jid
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_mask.npy", mask)
    args = method_cfg("triguard")
    args.update(dataset="isic2019", eps=float(eps), seed=1000 + shadow, rounds=rounds, train_mask=str(out / "train_mask.npy"))
    tr = FLTrainer(make_fl_cfg(jid, args))
    tr.run()
    lg = tr.predict(tr.sites[0].params, tgt.cache_row.values)
    np.savez_compressed(out / "lira_logits.npz", logits=lg, y=tgt.y.values, keep=keep, rows=tgt.cache_row.values)


def _phi(logits, y):
    lg = logits - logits.max(1, keepdims=True)
    p = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
    py = np.clip(p[np.arange(len(y)), y], 1e-12, 1 - 1e-12)
    return np.log(py) - np.log(1 - py)


def lira_score(jid: str, eps: float, shadows: list[str]):
    Z = [np.load(RUNS / s / "lira_logits.npz") for s in shadows]
    y = Z[0]["y"]
    phi = np.stack([_phi(z["logits"], y) for z in Z])  # [S, n]
    keep = np.stack([z["keep"] for z in Z])  # [S, n]
    S = len(Z)
    scores, labels = [], []
    for t in range(S):  # each shadow in turn is the target; the rest are reference models
        ref = np.arange(S) != t
        pin = np.where(keep[ref], phi[ref], np.nan)
        pout = np.where(~keep[ref], phi[ref], np.nan)
        mu_in, mu_out = np.nanmean(pin, 0), np.nanmean(pout, 0)
        # global variances are more stable with few shadows (Carlini et al., "fixed variance" variant)
        sd_in, sd_out = np.nanstd(pin), np.nanstd(pout)
        s = norm.logpdf(phi[t], mu_in, sd_in + 1e-6) - norm.logpdf(phi[t], mu_out, sd_out + 1e-6)
        scores.append(s)
        labels.append(keep[t])
    s, l = np.concatenate(scores), np.concatenate(labels)
    fpr, tpr, _ = roc_curve(l, s)
    res = {"eps": eps, "auc": float(roc_auc_score(l, s)), "tpr_at_0.1%fpr": float(np.interp(1e-3, fpr, tpr)),
           "tpr_at_1%fpr": float(np.interp(1e-2, fpr, tpr)), "n_pairs": int(len(l)), "n_shadows": S}
    grid = np.logspace(-5, 0, 200)
    res["roc"] = {"fpr": grid.tolist(), "tpr": np.interp(grid, fpr, tpr).tolist()}
    save_json(res, RUNS / jid / "audit_result.json")
    print({k: v for k, v in res.items() if k != "roc"})


# ====================================================================== gradient inversion
def _image_metrics(rec: torch.Tensor, orig: torch.Tensor):
    import lpips
    from pytorch_msssim import ssim

    rec, orig = rec.clamp(0, 1), orig.clamp(0, 1)
    s = ssim(rec, orig, data_range=1.0, size_average=False)
    mse = ((rec - orig) ** 2).flatten(1).mean(1)
    psnr = 10 * torch.log10(1 / mse.clamp(min=1e-10))
    fn = lpips.LPIPS(net="alex", verbose=False).to(rec.device)
    lp = fn(rec * 2 - 1, orig * 2 - 1).flatten()
    return s.cpu().numpy(), psnr.cpu().numpy(), lp.detach().cpu().numpy()


def _target_images(n_images):
    import torchvision.transforms as T

    from triguard.data import isic2019

    df = Cache("isic2019").df
    rows = df[(df.split == "test") & (df.site == TARGET_SITE)].sort_values("cache_row").head(n_images)
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    imgs = torch.stack([tf(isic2019.load_image(r)) for r in rows.itertuples()])
    return rows, imgs


def _tv(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def inversion(jid: str, eps: float, n_images: int = 200, batch: int = 8, iters_grad: int = 400, iters_img: int = 400,
              seed: int = 0):
    """Attack on one shared-adapter update (batch 8) of a trained TriGuard model at site 0.

    Stage 1: optimise block-8 tokens so their shared-B gradient matches the observed (noisy) update
             (cosine distance, labels known to the attacker: worst case).
    Stage 2: frozen-block feature inversion: optimise an image whose blocks 1-8 output matches those tokens (+ TV)."""
    from triguard.backbone import load_backbone
    from triguard.experiments import eps_tag
    from triguard.model import ModelCfg, TailModel

    log = Logger(RUNS / jid / "train.log")
    dev = device()
    torch.manual_seed(seed)
    src = RUNS / f"isic2019__triguard__eps{eps_tag(eps)}__s0"
    params = torch.load(src / "params.pt")[TARGET_SITE]
    led = {r["site"]: r for r in load_json(src / "train_result.json")["ledger"]}[TARGET_SITE]
    cfgd = load_json(src / "train_result.json")["cfg"]
    sigma, C = (led["sigma"], cfgd["clip"]) if math.isfinite(eps) else (0.0, cfgd["clip"])
    bb = load_backbone("dinov2_vits14", dev)
    tail = TailModel(bb.blocks, bb.norm, 8, 8, ModelCfg(**cfgd["model"])).to(dev)
    p = {k: v.to(dev) for k, v in params.items()}
    shared = [k for k in p if k.startswith("s.")]
    mean = torch.tensor(bb.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(bb.std, device=dev).view(1, 3, 1, 1)
    rows, imgs = _target_images(n_images)
    ys = torch.as_tensor(rows.y.values, device=dev)
    recs = []

    def shared_grad(tok, y, create_graph):
        tp = {k: (v.detach().requires_grad_(True) if k in shared else v.detach()) for k, v in p.items()}
        loss = F.cross_entropy(tail.logits(tp, tok), y)
        return torch.autograd.grad(loss, [tp[k] for k in shared], create_graph=create_graph)

    for b0 in range(0, len(imgs), batch):
        x = imgs[b0 : b0 + batch].to(dev)
        y = ys[b0 : b0 + batch]
        with torch.no_grad():
            tok_true = bb.forward_frozen((x - mean) / std, 8)
        # observed update: per-sample clipped sum + DP noise (as DP-SGD would release it)
        per = []
        for i in range(len(x)):
            g = shared_grad(tok_true[i : i + 1], y[i : i + 1], False)
            nrm = torch.sqrt(sum((gi**2).sum() for gi in g))
            f = min(1.0, C / float(nrm + 1e-6)) if sigma > 0 else 1.0
            per.append([gi * f for gi in g])
        obs = [sum(pp[j] for pp in per) for j in range(len(shared))]
        if sigma > 0:
            obs = [o + torch.randn_like(o) * sigma * C for o in obs]
        obs = [o / len(x) for o in obs]
        # ---- stage 1: token matching
        tok = torch.randn_like(tok_true) * tok_true.std() + tok_true.mean()
        tok.requires_grad_(True)
        opt = torch.optim.Adam([tok], lr=0.05)
        for it in range(iters_grad):
            g = shared_grad(tok, y, True)
            num = sum((a * b).sum() for a, b in zip(g, obs))
            den = torch.sqrt(sum((a**2).sum() for a in g)) * torch.sqrt(sum((b**2).sum() for b in obs)) + 1e-12
            loss = 1 - num / den
            opt.zero_grad()
            loss.backward()
            opt.step()
        # ---- stage 2: feature inversion through frozen blocks 1-8
        img = torch.full_like(x, 0.5).requires_grad_(True)
        opt2 = torch.optim.Adam([img], lr=0.05)
        tgt = tok.detach()
        for it in range(iters_img):
            t = bb.tokens(((img.clamp(0, 1)) - mean) / std)
            for blk in bb.blocks[:8]:
                t = blk(t)
            loss = F.mse_loss(t, tgt) + 0.05 * _tv(img)
            opt2.zero_grad()
            loss.backward()
            opt2.step()
        recs.append(img.detach().clamp(0, 1).cpu())
        log(f"inverted {b0 + len(x)}/{len(imgs)} (stage-1 cos dist {float(loss):.3f})")
    rec = torch.cat(recs)
    s, ps, lp = _image_metrics(rec.to(dev), imgs.to(dev))
    torch.save({"rec": rec[:16], "orig": imgs[:16]}, RUNS / jid / "examples.pt")
    res = {"eps": eps, "ssim": float(s.mean()), "psnr": float(ps.mean()), "lpips": float(lp.mean()), "n": len(rec),
           "ssim_all": s.tolist(), "sigma": sigma, "clip": C}
    save_json(res, RUNS / jid / "audit_result.json")
    print({k: v for k, v in res.items() if k != "ssim_all"})


def inversion_cnn(jid: str, n_images: int = 200, batch: int = 8, iters: int = 1000, seed: int = 0):
    """Reference leak: FedAvg full fine-tuning of ResNet-18 (GroupNorm) without DP, batch-8 update (Geiping et al.)."""
    from triguard.cnn import make_model

    log = Logger(RUNS / jid / "train.log")
    dev = device()
    torch.manual_seed(seed)
    model = make_model(8).to(dev)
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    params = [q for q in model.parameters()]
    rows, imgs = _target_images(n_images)
    ys = torch.as_tensor(rows.y.values, device=dev)
    recs = []
    for b0 in range(0, len(imgs), batch):
        x, y = imgs[b0 : b0 + batch].to(dev), ys[b0 : b0 + batch]
        obs = [g.detach() for g in torch.autograd.grad(F.cross_entropy(model((x - mean) / std), y), params)]
        img = torch.full_like(x, 0.5).requires_grad_(True)
        opt = torch.optim.Adam([img], lr=0.1)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.375), int(iters * 0.625), int(iters * 0.875)], 0.1)
        for it in range(iters):
            g = torch.autograd.grad(F.cross_entropy(model((img.clamp(0, 1) - mean) / std), y), params, create_graph=True)
            num = sum((a * b).sum() for a, b in zip(g, obs))
            den = torch.sqrt(sum((a**2).sum() for a in g)) * torch.sqrt(sum((b**2).sum() for b in obs))
            loss = 1 - num / den + 1e-4 * _tv(img)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        recs.append(img.detach().clamp(0, 1).cpu())
        log(f"inverted {b0 + len(x)}/{len(imgs)}")
    rec = torch.cat(recs)
    s, ps, lp = _image_metrics(rec.to(dev), imgs.to(dev))
    torch.save({"rec": rec[:16], "orig": imgs[:16]}, RUNS / jid / "examples.pt")
    save_json({"eps": math.inf, "model": "resnet18_gn_fullft", "ssim": float(s.mean()), "psnr": float(ps.mean()),
               "lpips": float(lp.mean()), "n": len(rec)}, RUNS / jid / "audit_result.json")
