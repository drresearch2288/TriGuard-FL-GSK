# TriGuard-FL — runbook

Everything (downloads, preprocessing, feature caching, training, evaluation, audits, tables, figures) runs from
one priority queue. You normally only need the four commands below (run them from `D:\trail3\TriGuard`).

| What | Command |
|---|---|
| Start / resume the queue (safe after a crash or reboot) | `.venv\Scripts\python.exe -m triguard.queue run` |
| Progress, ETA, running jobs, failures | `.venv\Scripts\python.exe -m triguard.queue status` (also written to `runs\STATUS.md` every minute) |
| Stop gracefully (running jobs finish, nothing new starts) | `.venv\Scripts\python.exe -m triguard.queue stop` |
| Re-run a failed job | `.venv\Scripts\python.exe -m triguard.queue retry <job id>` |
| Live dashboard (progress, GPU/CPU use, temperatures) | `.venv\Scripts\python.exe -m triguard.dashboard` → open http://127.0.0.1:8765 |
| Rebuild all tables + figures from finished runs | `.venv\Scripts\python.exe -m triguard.report` → `results\RESULTS.md` |

## Shutting down and resuming
- You can shut down at any time. Training runs save a checkpoint every 2 rounds and continue from it; downloads
  continue from the partial file. At most ~2 rounds (≈ 10–15 min) of GPU work are lost per shutdown.
- After switching on: open PowerShell in `D:\trail3\TriGuard` and run
  `$env:PYTHONPATH="D:\trail3\TriGuard"; .venv\Scripts\python.exe -m triguard.queue run`
  (keep the window open; minimise it). Jobs that were running are restarted automatically.

## Keep the laptop running
- Plugged in, **sleep = Never** and **lid close = Do nothing** while plugged in (Settings → System → Power).
- Good airflow / a cooling pad: the GPU runs 24/7 for weeks.
- Do not run other GPU programs (games, local LLMs): the card has only 4 GB.

## Order of work (priority)
1. Data: ISIC 2019 → ISIC 2020 subset → PAD-UFES-20 → Camelyon17 → MedMNIST+ (network ≈ 0.5–1 MB/s is the bottleneck on day 1).
2. Timing run, then hyper-parameter search on the public ISIC 2020 proxy (11 runs, no privacy cost).
3. Main Fed-ISIC2019 table for every method and ε (ε = 3 first) → conformal / ledger / fairness evaluated automatically on CPU.
4. B3 ResNet baseline, PAD-UFES transfer, Fed-Heart (already done), canary audit, gradient inversion.
5. Ablations, Camelyon17, MedMNIST+, LiRA shadows.

Scope of this deliverable: **one seed (seed 0) for every experiment, all baselines included** (`SEEDS` in
`triguard/experiments.py`; add seeds there later and re-run `python -m triguard.experiments` — finished jobs are kept).

## Where things are
- `runs\<job id>\` — logs (`stdout.log`, `train.log`), `train_result.json`, `eval_result.json`, `preds.npz`, `params.pt`
- `runs\STATUS.md` — live status; `runs\history.jsonl` — measured job durations
- `results\RESULTS.md`, `results\tables\*.csv`, `results\figures\*.png` — paper tables / figures (measured)

## Deviations from the written plan (forced by the 4 GB GPU, 16 GB RAM, ~1 MB/s internet, 108 GB disk)
- Per-sample gradients use an exact closed-form LoRA/head method (hooks + flash attention, bf16) instead of vmap — 16 % faster, verified equal to vmap in tests. Physical batch 16 samples × 4 views.
- Shared-adapter optimiser is AdamW (as in the plan). Random-effects aggregation compares per-step effects
  Y_k = Δ_k / T_k (T_k = local steps this round). Their relative precision is the plan's DP formula, the variance of the
  site's averaged noisy clipped gradient v_k = σ_k² C² / (B_k² T_k), and one common scale factor matched to the
  (already-private) Adam state puts v_k in update units for Cochran's Q / τ². Pure post-processing.
  First version (superseded 5 Nov, run kept in `runs/_superseded_re_v1/`): v_k was the Adam-space noise variance summed
  over raw, unnormalised updates. That penalised sites that take more steps, so the largest site (54 % of the data) got
  10 % of the weight and τ² stayed 0. Result at ε = 3: mean BA 0.366 vs FedLoRA's 0.468. Runs made with the first
  version (Kaggle: TriGuard ε = 1 in main1, ablations a2–a5 in batch2) are rerun.
- ISIC 2020 proxy: 6,600-image subset (all malignant + benign) split into 6 synthetic sites with Fed-ISIC2019's size ratios; canaries are a further 1,000 patient-disjoint images. Fetched from inside the remote zip (no 23 GB download).
- MedMNIST+: ≤ 6,000 train and ≤ 2,000 test images per subset (keeps the plan's cost ratio); Camelyon17: 60k-patch training subset, K = 1 augmented view (disk).
- Calibration budget for ε ≠ 3 keeps the plan's 0.3 : 0.2 : 2.5 split, scaled so the RDP-composed calibration ε equals the total ε.
- DP-SCP slack covers the noise in both the histograms and the released label counts, so Theorem 1's guarantee holds.
  As first implemented (plan's 100 uniform bins, slack on the cumulative sum from below) the slack was γ ≈ 144
  calibration ranks at ε = 3, which made every set the full label set (8/8 classes). Now: (i) the bound is taken on the
  upper tail (few noisy bins; the count noise enters scaled by α), (ii) a public 12-bin grid, geometric in 1 − score
  (APS scores pile up near 1), (iii) Mondrian cells chosen from the released noisy counts only: classes too rare to
  carry the DP slack share one cell (clustered conformal), and if that cell is still too small they use the all-class
  cell. All post-processing, no extra privacy. ISIC ε = 3, α = 0.1: coverage 0.94, set size 3.8, worst-site 0.92,
  worst-class 0.83 (vs. non-private FCP 0.90 / 3.1 / 0.88 / 0.76). Covered by `tests/test_conformal.py`.
- **TriGuard-D (5 Nov, superseded 6 Nov — now ablation A8 and, for the gate alone, A2): personalisation gated by DP noise.** The plan gives every site a private head and adapter.
  Under DP the small sites' heads learn nothing: at ε = 3, site 5 (n = 286) predicted only nevus (BA 0.25) in every
  per-site-head method (TriGuard v1, B1, B7), while the shared-head baselines reached 0.34–0.62. TriGuard now trains a
  federated head (with random-effects aggregation). A site keeps its private adapter and a private head fine-tune only
  if B_k / (σ_k √d_head) ≥ 1, i.e. its averaged clipped head gradient has a per-coordinate SNR of at least 1. The rule
  uses public quantities only and has no tuned threshold. Shared-only sites skip the head fine-tune, so their σ is
  recalibrated for fewer steps. ISIC: ε = 1 → site 0 personal; ε = 3 and 8 → sites 0–3. The plan's always-personal
  design is ablation a7, and a8 has no private parts at all. Ablations a1–a6 are one-change variants of TriGuard-D.
- **TriGuard-FL training core = FedLoRA (6 Nov).** TriGuard-D at ε = 3 reached mean BA 0.381 (worst site 0.283),
  below FedLoRA (0.468) and FFA-LoRA (0.416). Two causes: (i) τ² stayed 0, so random-effects weights reduce to DP
  inverse-variance weights, which grow with n²: the largest site got 79 % of the weight (FedAvg: 54 %) and sites 4–5
  lost 0.1 BA; (ii) private adapters hurt mid-size sites (site 1: 0.37 → 0.28). TriGuard-FL now trains exactly like
  FedLoRA (shared A and B, federated head, sample-size weights). Its contribution is the stack around training: DP-SCP,
  the per-patient ledger and PrivAudit. On Fed-ISIC2019 the TriGuard training runs are therefore the B5 runs (same
  config and seed; `alias` jobs copy them instead of training twice), so the accuracy rows of B5 and TriGuard coincide.
  Ablations are one-change variants of this core: A1 + RE aggregation, A2 + DP-gated private adapter/head, A3 frozen
  A (= B6, reused), A4 K = 1, A5 ImageNet ViT-S, A6 BiomedCLIP; A7 is the plan's original design and A8 TriGuard-D
  (its finished run). Canary ε = 3 / 8 and LiRA ε = 3 had already run with the earlier TriGuard configuration and are
  kept; their DP accounting (σ, q, steps, δ) is the same, only the trained parameters differ. Remaining audits use the
  FedLoRA core. Superseded runs: `runs/_superseded_D/`.
- **DP conformal baselines (7 Nov), after the novelty check.** U5: pooled quantile by the exponential mechanism
  (Lu et al. 2023, appendix C.2) with a tail correction; it needs an exact distributed quantile, so a trusted aggregator.
  U6: the same per class at ε/2, no merging of rare classes. U7: DP-FedCP (Plassier et al. 2023), simplified (full
  participation, one local step, plain pinball subgradient on -log(1 - score), weights capped at 20), composed over
  T × classes × sites releases. All get DP-SCP's histogram budget. `efficiency` in eval_result.json sweeps nominal α so
  methods can be read at equal set size (Table 12b). This is descriptive: the operating point comes from the test set.
  ISIC ε = 3 trial at DP-SCP's set size (about 4.0): worst-site / worst-class coverage DP-SCP 0.911 / 0.850,
  non-private FCP 0.929 / 0.869, U5 0.928 / 0.868, U7 0.842 / 0.776, U6 0.803 / 0.788. So DP-SCP's higher worst-group
  coverage at α = 0.1 comes from over-coverage, not from a better method; it leads only among methods that do not trust
  the aggregator.
- **B3 redone (7 Nov).** The first B3 runs scored 0.244 at ε = ∞, close to chance. Causes: GroupNorm discards the
  pretrained normalisation, and per-sample gradients were scaled by the batch size before clipping (Opacus'
  loss_reduction default did not match the summed loss; no effect on the privacy guarantee). B3 now uses BatchNorm frozen
  at its public ImageNet statistics. Search on the public ISIC 2020 proxy, 30 rounds, macro-AUC: ε = ∞ GroupNorm 0.761,
  frozen lr 0.05 0.851, frozen lr 0.5 diverged; ε = 3 GroupNorm 0.786, frozen lr 0.05 0.781, lr 0.2 0.754, lr 0.01 0.653.
  Frozen BN at lr 0.05 is used for every ε (it ties at ε = 3 and wins at ε = ∞). Old runs: `runs/_superseded_b3_gn/`.
- **Camelyon17 (7 Nov).** WILDS' archive link returns a 64 KB stub, so the data come from the Kaggle copy
  `arinaromashkina/wilds-camelyon17-v1`; the worker checks row and hospital counts and a sample of patches against the
  WILDS release. Split corrected to the official one (train centers 0, 3, 4; val 1; test 2). Features stay on Kaggle.
- Server momentum 0 (plan: FedAdam-style momentum 0.9 with server step 1). With momentum 0.9 the effective server step
  compounds ~10x; in a sanity check the non-private linear probe's balanced accuracy fell from 0.45 to 0.22 over 50 rounds,
  while without momentum it rose to 0.54 and stayed stable.
- Class imbalance: all methods train with the logit-adjusted loss (Menon et al. 2021). Class priors come from the DP noisy
  calibration label counts that DP-SCP releases anyway (ε_label, already in the ledger); they are released once and reused.
- Hyper-parameters selected by macro-AUC on the public ISIC 2020 proxy (BA was 0.5 for every setting there: majority class).
  Stage 1 (with the old momentum) fixed C = 2.0, rank 8; stage 2 (momentum 0) lr grid {5e-4, 1e-3, 2e-3, 3e-3}
  selected lr = 2e-3 (macro-AUC 0.763 vs 0.759 / 0.749 / 0.737).
- LiRA uses 8 shadow federated runs per ε (plan: 16), each sample IN for exactly 4 of them; Table 17 / Fig 8 are produced in full, the AUC / TPR estimates are somewhat noisier.
- Camelyon17, MedMNIST+ and the ablations are run at ε = 3 only, the only budget their tables (13, 14, 19) report.
- Calibration splits are lesion-disjoint (ISIC 2019 has lesion ids, not patient ids), slide-disjoint (Camelyon) and patient-disjoint (PAD, ISIC 2020).
- **Gradient inversion redone (8 Nov).** The first audit (signed Adam, lr 0.1, GroupNorm ResNet) reached SSIM 0.004:
  the step was far too large for pixels in [0, 1]. Now: ResNet-18 with frozen BatchNorm, full fine-tuning, no DP,
  batch 8, 2,000 iterations, Adam lr 1e-3, TV weight 0.1 (chosen on 8 probe images; best of 7 settings). Over 200
  images: SSIM 0.192, PSNR 13.1 dB, LPIPS 0.92. A flat image of each target's own mean colour scores SSIM 0.596, so
  the attack recovers less structure than that trivial baseline: at batch 8 it does not reconstruct images even
  without DP, and the audit cannot credit DP with stopping it. Old run: `runs/_superseded_inversion_v1/`.
- **B3 at ε = ∞ (8 Nov).** The non-private B3 run on ISIC diverged to NaN (plain SGD, lr 0.05, no per-sample clipping;
  site 0 takes 8 steps per round). The non-private path now clips the mini-batch gradient norm at the same C = 2.0;
  the DP runs are unchanged. NaN run kept in `runs/_superseded_b3_nan/`.
- **Index-only caches (8 Nov).** The MedMNIST feature caches were deleted for disk space, index included. Their
  index.csv is rebuilt deterministically by `cache.rebuild_index_only` (same subsampling and seed) and matches
  the row ids saved by every finished MedMNIST run.
- **TriGuard inversion audit fixed (8 Nov).** Stage 2 (feature inversion) called `backbone.tokens`, which runs under
  no_grad, so the image never moved from its flat grey start: all four ε gave the same SSIM 0.594. Stage 2 now keeps
  the gradient. Stage-2 lr chosen on 16 images at ε = ∞: 0.05 → SSIM 0.064, 0.01 → 0.232, 0.003 → 0.423 (LPIPS 1.05 /
  0.96 / 0.95); flat-colour baseline 0.615. The results also store `ssim_baseline`. Old runs:
  `runs/_superseded_inversion_lora_nograd/`.
- **Camelyon17 conformal (8 Nov).** Evaluation took calibration sites only among test sites, so Camelyon (calibrate on
  0/3/4, test on unseen 2) had no conformal results and Table 13 was empty. Now every site with calibration data is used;
  the only other runs affected are RetinaMNIST (one calibration-only site).
