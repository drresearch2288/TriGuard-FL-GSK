"""The full TriGuard-FL experiment matrix as prioritised jobs.

Each job: id, kind, resource (gpu | cpu | net), priority (lower runs first), deps, args, est_h (laptop GPU hours).
Running `python -m triguard.experiments` writes runs/jobs.json; the queue runner executes it.
"""
from __future__ import annotations

import json
import math

from triguard.paths import RUNS

EPS = [1.0, 3.0, 8.0, math.inf]
SEEDS = [0]  # outsourced deliverable: one end-to-end seed for every experiment (incl. baselines)
# Only what the plan's tables/figures report: Tables 13, 14, 19 (Fig 11) show eps = 3 only.
GEN_EPS = [3.0]  # Camelyon17, MedMNIST+
ABL_EPS = [3.0]  # ablations
VIEW_PASSES_PER_S = 268  # measured on RTX 3050 Laptop 4 GB (fast per-sample grads, bf16)


def eps_tag(e):
    return "inf" if not math.isfinite(e) else f"{e:g}"


# ------------------------------------------------------------------ methods (FLCfg overrides)
def method_cfg(m: str) -> dict:
    M = {
        # TriGuard-FL training core = FedLoRA (shared A and B, federated head, sample-size weights). The contribution is
        # the stack around it: DP-SCP, per-patient ledger, PrivAudit. RE aggregation and DP-gated personalisation lost
        # accuracy at eps = 3 (TriGuard-D: 0.381 vs 0.468) and are ablations a1 / a2 / a8.
        "triguard": dict(model=dict(train_shared_A=True, local=False, head_federated=True), agg="fedavg",
                         final_head_epochs=0.0),
        "b1_local": dict(model=dict(), agg="none", final_head_epochs=1.0),
        "b2_central": dict(model=dict(), agg="none", centralised=True, final_head_epochs=1.0),
        "b4_probe": dict(linear_probe=True, agg="fedavg", final_head_epochs=0.0),
        "b5_fedlora": dict(model=dict(train_shared_A=True, local=False, head_federated=True), agg="fedavg",
                           final_head_epochs=0.0),
        "b6_ffalora": dict(model=dict(local=False, head_federated=True), agg="fedavg", final_head_epochs=0.0),
        "b7_personal": dict(model=dict(train_shared_A=True, local=True), agg="fedavg", final_head_epochs=1.0),
    }
    TG = M["triguard"]
    M.update({  # ablations of TriGuard-FL (one change each); a7 / a8 are the earlier full designs
        "a1_re": {**TG, "agg": "re"},  # random-effects (DerSimonian-Laird) aggregation
        "a2_gated": {**TG, "model": dict(train_shared_A=True, local=True, head_federated=True),
                     "final_head_epochs": 1.0, "personal_gate": 1.0},  # + DP-SNR-gated private adapter / head
        "a4_k1": {**TG, "views": 1},
        "a5_imagenet": {**TG, "backbone": "vit_s_in"},
        "a6_biomedclip": {**TG, "backbone": "biomedclip", "phys_batch": 6},
        "a7_allpersonal": dict(model=dict(), agg="re", final_head_epochs=1.0),  # plan: private head at every site
        "a8_triguardD": dict(model=dict(head_federated=True), agg="re", final_head_epochs=1.0, personal_gate=1.0),
    })
    # a3 (frozen A) is exactly B6 FFA-LoRA: its run is reused through an alias job (see build()).
    d = dict(M[m])
    d["method"] = m
    return d


DATASET_TRAIN = {  # train-set sizes after the calib split, used for cost estimates
    "isic2019": 14900, "isic2020": 4800, "camelyon17": 48000,
    **{f: 4800 for f in ["pathmnist", "bloodmnist", "organamnist", "octmnist", "pneumoniamnist", "retinamnist"]},
}


def est_hours(ds, cfg, rounds=50, frac=1.0):
    views = cfg.get("views", 4 if ds != "camelyon17" else 1)
    if cfg.get("linear_probe"):
        return 0.15
    mult = {"biomedclip": 2.6, "vit_s_in": 0.75}.get(cfg.get("backbone", ""), 1.0)
    vp = DATASET_TRAIN[ds] * frac * views * (rounds + 1)
    return 1.1 * mult * vp / VIEW_PASSES_PER_S / 3600 + 0.05


class Plan:
    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def add(self, id, kind, resource, priority, args=None, deps=(), est_h=0.0):
        self.jobs[id] = dict(id=id, kind=kind, resource=resource, priority=float(priority), args=args or {},
                             deps=[d for d in deps], est_h=float(est_h))
        return id

    def alias(self, jid, src, prio):
        """A run whose config + seed equal another run's: copy that run's outputs, then evaluate as usual."""
        self.add(jid, "alias", "cpu", prio, {"source": src}, [src], 0)
        self.add(f"eval__{jid}", "eval", "cpu", prio, {"run": jid}, [jid], 0)
        return jid

    def train(self, ds, method, eps, seed, prio, deps, extra=None, eval_prio=None, rounds=50, tag=None):
        cfg = method_cfg(method)
        cfg.update(dataset=ds, eps=eps, seed=seed, rounds=rounds)
        if ds == "camelyon17":
            cfg.setdefault("views", 1)
        if extra:
            cfg.update(extra)
        bb = cfg.get("backbone", "dinov2_vits14")
        jid = f"{ds}__{tag or method}__eps{eps_tag(eps)}__s{seed}"
        cdeps = list(deps) + [f"cache__{ds}__{bb}"]
        if ds != "isic2020" and not cfg.get("no_tuning"):
            cdeps.append("tune__select")
        self.add(jid, "train", "gpu", prio, cfg, cdeps, est_hours(ds, cfg, rounds))
        self.add(f"eval__{jid}", "eval", "cpu", eval_prio if eval_prio is not None else prio, {"run": jid}, [jid], 0)
        return jid


def build() -> dict:
    P = Plan()
    # ---------------------------------------------------------------- data (network + CPU + GPU caching)
    P.add("dl__isic2019", "download", "net", 0, {"step": "isic2019"})
    P.add("dl__heart", "download", "net", 0, {"step": "heart"})
    P.add("dl__backbones", "download", "net", 0.5, {"step": "backbones"})
    P.add("dl__isic2020", "download", "net", 1, {"step": "isic2020"})
    P.add("dl__pad_ufes", "download", "net", 2, {"step": "pad_ufes"})
    P.add("dl__camelyon17", "download", "net", 3, {"step": "camelyon17"})
    for i, f in enumerate(["bloodmnist", "pneumoniamnist", "retinamnist", "organamnist", "octmnist", "pathmnist"]):
        P.add(f"dl__{f}", "download", "net", 4 + 0.1 * i, {"step": "medmnist", "flag": f})
    P.add("prep__isic2019", "prepare", "cpu", 0, {"dataset": "isic2019"}, ["dl__isic2019"])
    # ISIC 2020 near-duplicate removal compares against the preprocessed ISIC 2019 images
    P.add("prep__isic2020", "prepare", "cpu", 1, {"dataset": "isic2020"}, ["dl__isic2020", "prep__isic2019"])
    P.add("prep__pad_ufes20", "prepare", "cpu", 2, {"dataset": "pad_ufes20"}, ["dl__pad_ufes"])
    P.add("prep__camelyon17", "prepare", "cpu", 3, {"dataset": "camelyon17"}, ["dl__camelyon17"])

    def cache(ds, bb="dinov2_vits14", prio=0, deps=(), **kw):
        return P.add(f"cache__{ds}__{bb}", "cache", "gpu", prio, dict(dataset=ds, backbone=bb, **kw),
                     list(deps) + ["dl__backbones"], est_h=kw.pop("est", 0.3))

    cache("isic2019", prio=0, deps=["prep__isic2019"], views=4)
    cache("isic2020", prio=0.9, deps=["prep__isic2020"], views=4)
    cache("isic2020_canary", prio=4.9, deps=["prep__isic2020"], views=4)
    cache("pad_ufes20", prio=2, deps=["prep__pad_ufes20"], views=0)
    cache("camelyon17", prio=7.9, deps=["prep__camelyon17"], views=1, max_train=60000, max_eval=20000, disk_gb=32)
    for f in ["bloodmnist", "pneumoniamnist", "retinamnist", "organamnist", "octmnist", "pathmnist"]:
        cache(f, prio=8.4, deps=[f"dl__{f}"], views=4, max_train=6000, max_eval=2000, disk_gb=8)
    cache("isic2019", "vit_s_in", prio=6, deps=["prep__isic2019"], views=4, disk_gb=14)
    cache("isic2019", "biomedclip", prio=6, deps=["prep__isic2019"], views=4, disk_gb=27)

    # ---------------------------------------------------------------- M4 check: one timed DP run
    P.train("isic2019", "triguard", 3.0, 99, 0.95, [], extra={"rounds": 2, "no_tuning": True}, rounds=2, tag="timing")

    # ---------------------------------------------------------------- hyperparameter search on public ISIC 2020 proxy
    # v1 (C x lr grid + rank, server momentum 0.9) showed: clip and rank barely matter, lr dominates -- but v1 ran with
    # the diverging server momentum, so v2 re-searches lr only (C = 2.0, rank 8 from v1) with the fixed server step.
    tune_ids = []
    for lr in (5e-4, 1e-3, 2e-3, 3e-3):
        jid = f"tune2__lr{lr}"
        cfg = method_cfg("triguard")
        cfg.update(dataset="isic2020", eps=3.0, seed=0, rounds=30, clip=2.0, lr=lr, model={"rank": 8, "alpha": 16.0})
        P.add(jid, "train", "gpu", 1, cfg, ["cache__isic2020__dinov2_vits14"], est_hours("isic2020", cfg, 30))
        tune_ids.append(jid)
    P.add("tune__select", "tune_select", "cpu", 1.1, {"runs": tune_ids, "stage": 2}, tune_ids)

    # ---------------------------------------------------------------- P1 main table (Fed-ISIC2019)
    main = ["triguard", "b6_ffalora", "b5_fedlora", "b7_personal", "b1_local", "b2_central", "b4_probe"]
    for seed in SEEDS:
        for ei, e in enumerate([3.0, 1.0, 8.0, math.inf]):
            for mi, m in enumerate(main):
                base = 2 if seed == 0 else (4 if seed <= 2 else 10)
                if m == "triguard":  # same config + seed as B5: reuse that run instead of training it twice
                    P.alias(f"isic2019__triguard__eps{eps_tag(e)}__s{seed}", f"isic2019__b5_fedlora__eps{eps_tag(e)}__s{seed}",
                            base + 0.1 * ei)
                    continue
                P.train("isic2019", m, e, seed, base + 0.1 * ei + 0.01 * mi, [])
    # B3: DP-FedAvg full fine-tune ResNet-18 (GroupNorm), raw images
    for seed in SEEDS:
        for e in EPS:
            jid = f"isic2019__b3_resnet__eps{eps_tag(e)}__s{seed}"
            P.add(jid, "cnn", "gpu", 3 if seed == 0 else 10, dict(dataset="isic2019", eps=e, seed=seed, rounds=50),
                  ["prep__isic2019", "cache__isic2019__dinov2_vits14", "tune__select"], 1.6)
            P.add(f"eval__{jid}", "eval", "cpu", 3, {"run": jid}, [jid])

    # ---------------------------------------------------------------- ablations (Table 19 / Fig 11 report eps = 3 only)
    for seed in SEEDS:
        for e in ABL_EPS:
            pr = (6 if seed == 0 else 9) + (0.05 if e == 1.0 else 0)
            for a in ["a1_re", "a2_gated", "a4_k1", "a5_imagenet", "a6_biomedclip", "a7_allpersonal", "a8_triguardD"]:
                P.train("isic2019", a, e, seed, pr, [])
            P.alias(f"isic2019__a3_frozenA__eps{eps_tag(e)}__s{seed}", f"isic2019__b6_ffalora__eps{eps_tag(e)}__s{seed}", pr)
    P.add("clean__isic2019__vit_s_in", "cleanup", "cpu", 9.5, {"cache": ["isic2019", "vit_s_in"]},
          [f"isic2019__a5_imagenet__eps{eps_tag(e)}__s{s}" for e in ABL_EPS for s in SEEDS])
    P.add("clean__isic2019__biomedclip", "cleanup", "cpu", 9.5, {"cache": ["isic2019", "biomedclip"]},
          [f"isic2019__a6_biomedclip__eps{eps_tag(e)}__s{s}" for e in ABL_EPS for s in SEEDS])

    # ---------------------------------------------------------------- PrivAudit (site 1 = BCN = site id 0)
    for e in EPS:
        et = eps_tag(e)
        P.add(f"audit__canary__eps{et}", "canary", "gpu", 5, {"eps": e, "m": 1000, "seed": 0},
              ["cache__isic2019__dinov2_vits14", "cache__isic2020_canary__dinov2_vits14", "tune__select"],
              est_hours("isic2019", {}) * 1.05)
        P.add(f"audit__inversion__eps{et}", "inversion", "gpu", 5.5, {"eps": e, "n_images": 200, "batch": 8, "lr_img": 0.003},
              [f"isic2019__triguard__eps{et}__s0"], 1.5)
        shadows = []
        for s in range(8):  # LiRA shadows per eps (must equal audit.N_SHADOWS)
            jid = f"lira__eps{et}__shadow{s}"
            P.add(jid, "lira_shadow", "gpu", 9 + (0.1 if e in (8.0, 1.0) else 0), {"eps": e, "shadow": s, "rounds": 30},
                  ["cache__isic2019__dinov2_vits14", "tune__select"], est_hours("isic2019", {}, 30, frac=0.73))
            shadows.append(jid)
        P.add(f"audit__lira__eps{et}", "lira_score", "cpu", 9, {"eps": e, "shadows": shadows}, shadows)
    P.add("audit__inversion__fedavg_full_nodp", "inversion_cnn", "gpu", 5.5, {"n_images": 200, "batch": 8},
          ["cache__isic2019__dinov2_vits14"], 1.0)

    # ---------------------------------------------------------------- P4 PAD-UFES-20 (external device, test only)
    for m in ["triguard", "b6_ffalora"]:
        src = f"isic2019__{m}__eps3__s0"
        P.add(f"pad__{m}", "pad_transfer", "gpu", 3.5, {"source": src, "recal_n": 200},
              [src, f"eval__{src}", "cache__pad_ufes20__dinov2_vits14"], 0.1)

    # ---------------------------------------------------------------- P5 Fed-Heart-Disease (tabular, CPU)
    for seed in SEEDS:
        for e in EPS:
            for m in ["dpfedavg_lr", "re_dpscp"]:
                P.add(f"heart__{m}__eps{eps_tag(e)}__s{seed}", "heart", "cpu", 2.5, {"method": m, "eps": e, "seed": seed},
                      ["dl__heart"])

    # ---------------------------------------------------------------- P2 Camelyon17-WILDS (unseen hospital)
    for seed in SEEDS:  # Table 13 reports eps = 3 and methods B3, B4, B6, TriGuard
        for e in GEN_EPS:
            for m in ["triguard", "b6_ffalora", "b4_probe"]:
                P.train("camelyon17", m, e, seed, (8 if seed == 0 else 11) + (0 if e == 3.0 else 0.2), [])
            jid = f"camelyon17__b3_resnet__eps{eps_tag(e)}__s{seed}"
            P.add(jid, "cnn", "gpu", 8.5 if seed == 0 else 11, dict(dataset="camelyon17", eps=e, seed=seed, rounds=50),
                  ["cache__camelyon17__dinov2_vits14", "tune__select"], 1.5)
            P.add(f"eval__{jid}", "eval", "cpu", 8, {"run": jid}, [jid])

    # ---------------------------------------------------------------- P3 MedMNIST+ (20 Dirichlet clients)
    for f in ["bloodmnist", "pneumoniamnist", "retinamnist", "organamnist", "octmnist", "pathmnist"]:
        users = []
        for seed in SEEDS:  # Table 14 reports eps = 3
            for e in GEN_EPS:
                for m in ["triguard", "b6_ffalora"]:
                    users.append(P.train(f, m, e, seed, (8.5 if seed == 0 else 11.5) + (0 if e == 3.0 else 0.2), []))
                jid = f"{f}__b3_resnet__eps{eps_tag(e)}__s{seed}"
                P.add(jid, "cnn", "gpu", 8.6 if seed == 0 else 11.5, dict(dataset=f, eps=e, seed=seed, rounds=50),
                      [f"cache__{f}__dinov2_vits14", "tune__select"], 0.6)
                P.add(f"eval__{jid}", "eval", "cpu", 8.6, {"run": jid}, [jid])
                users.append(jid)
        P.add(f"clean__{f}", "cleanup", "cpu", 12, {"cache": [f, "dinov2_vits14"]}, users)

    # ---------------------------------------------------------------- reporting
    P.add("report__final", "report", "cpu", 99, {}, [j for j in P.jobs if j.startswith("eval__")])
    return P.jobs


def write():
    jobs = build()
    (RUNS / "jobs.json").write_text(json.dumps(jobs, indent=1, default=str))
    gpu = sum(j["est_h"] for j in jobs.values() if j["resource"] == "gpu")
    print(f"{len(jobs)} jobs, estimated GPU hours: {gpu:.0f} (~{gpu/24:.1f} days)")
    return jobs


if __name__ == "__main__":
    write()
