# TriGuard-FL — measured results

All numbers below are **measured** from the runs in `runs/` (mean ± SD over seeds; n = seeds finished). '—' = not finished yet. Regenerate any time with `python -m triguard.report`.

### Table 10. Balanced accuracy (%) on the Fed-ISIC2019 test set (pooled)

| Method | BA ε=1 | BA ε=3 | BA ε=8 | BA ε=∞ | Macro-AUC ε=3 |
|---|---|---|---|---|---|
| B1 Local only | 38.1 (n=1) | 45.8 (n=1) | 55.5 (n=1) | 68.6 (n=1) | 0.874 (n=1) |
| B3 DP-FedAvg full FT (ResNet-18) | 25.6 (n=1) | 27.0 (n=1) | 27.6 (n=1) | 53.5 (n=1) | 0.781 (n=1) |
| B4 DP linear probe | 30.6 (n=1) | 35.2 (n=1) | 39.9 (n=1) | 54.5 (n=1) | 0.837 (n=1) |
| B5 FedLoRA + DP-SGD | 32.3 (n=1) | 41.1 (n=1) | 49.2 (n=1) | 65.5 (n=1) | 0.872 (n=1) |
| B6 FFA-LoRA + DP | 31.7 (n=1) | 39.0 (n=1) | 49.0 (n=1) | 64.5 (n=1) | 0.863 (n=1) |
| B7 Personal LoRA + DP | 36.0 (n=1) | 46.8 (n=1) | 56.7 (n=1) | 68.3 (n=1) | 0.876 (n=1) |
| TriGuard-FL (proposed) | 32.3 (n=1) | 41.1 (n=1) | 49.2 (n=1) | 65.5 (n=1) | 0.872 (n=1) |
| B2 Centralised pooled | 41.3 (n=1) | 52.4 (n=1) | 57.4 (n=1) | 68.6 (n=1) | 0.902 (n=1) |



### Table 11. Per-site balanced accuracy (%) at ε = 3

| Method | BCN | Vidir-MM | Vidir-Mod | Rosendahl | MSK | Vienna | Worst site | n |
|---|---|---|---|---|---|---|---|---|
| B5 FedLoRA + DP-SGD | 41.5 | 38.9 | 54.3 | 38.4 | 45.8 | 61.7 | 38.4 | 1 |
| B6 FFA-LoRA + DP | 39.4 | 37.3 | 52.5 | 39.8 | 46.5 | 34.0 | 34.0 | 1 |
| B7 Personal LoRA + DP | 47.6 | 31.6 | 36.9 | 41.0 | 58.7 | 24.7 | 24.7 | 1 |
| B1 Local only | 44.5 | 23.5 | 45.3 | 32.0 | 51.1 | 25.0 | 23.5 | 1 |
| TriGuard-FL (proposed) | 41.5 | 38.9 | 54.3 | 38.4 | 45.8 | 61.7 | 38.4 | 1 |



### Table 12. Coverage and efficiency at α = 0.10 on top of the ε = 3 TriGuard model

| Method | Private? | Marginal cov. | Worst-site cov. | Worst-class cov. | MEL cov. | Set size | P(cov ≥ 1−α) | ECE | n |
|---|---|---|---|---|---|---|---|---|---|
| U1 Local split CP | yes | 0.903 | 0.841 | 0.791 | 0.807 | 3.18 | 0.85 | 0.102 | 1 |
| U2 FCP (Lu et al.) | no | 0.901 | 0.885 | 0.787 | 0.798 | 3.11 | 0.75 | 0.102 | 1 |
| U3 Naive DP histogram CP | yes | 0.913 | 0.896 | 0.804 | 0.817 | 3.31 | 1.00 | 0.102 | 1 |
| U4 Label-shift FCP (Plassier et al.) | no | 0.902 | 0.886 | 0.789 | 0.811 | 3.19 | 0.75 | 0.102 | 1 |
| U5 DP FCP, exponential mechanism (Lu et al.) | yes | 0.902 | 0.886 | 0.789 | 0.799 | 3.13 | 0.85 | 0.102 | 1 |
| U6 Per-class exponential mechanism | yes | 0.920 | 0.887 | 0.895 | 0.949 | 5.52 | 1.00 | 0.102 | 1 |
| U7 DP-FedCP (Plassier et al.) | yes | 0.959 | 0.921 | 0.867 | 0.931 | 5.12 | 1.00 | 0.102 | 1 |
| DP-SCP without shift check | yes | 0.939 | 0.903 | 0.837 | 0.964 | 3.85 | 1.00 | 0.103 | 1 |
| DP-SCP (proposed) | yes | 0.947 | 0.904 | 0.856 | 0.967 | 4.01 | 1.00 | 0.103 | 1 |



### Table 12b. Coverage at equal set size (curves read at DP-SCP's set size for α = 0.10)

| ε | Method | Private? | Set size | Coverage | Worst-site cov. | Worst-class cov. |
|---|---|---|---|---|---|---|
| 1 | DP-SCP (proposed) | yes | 5.00 | 0.953 | 0.913 | 0.876 |
| 1 | U2 FCP (Lu et al.) | no | 5.00 | 0.958 | 0.907 | 0.863 |
| 1 | U5 DP FCP, exponential mechanism (Lu et al.) | yes | 5.00 | 0.957 | 0.902 | 0.851 |
| 1 | U6 Per-class exponential mechanism | yes | 5.00 | 0.787 | 0.735 | 0.749 |
| 1 | U7 DP-FedCP (Plassier et al.) | yes | 5.00 | – | – | – |
| 3 | DP-SCP (proposed) | yes | 3.98 | 0.949 | 0.911 | 0.850 |
| 3 | U2 FCP (Lu et al.) | no | 3.98 | 0.941 | 0.929 | 0.869 |
| 3 | U5 DP FCP, exponential mechanism (Lu et al.) | yes | 3.98 | 0.941 | 0.928 | 0.868 |
| 3 | U6 Per-class exponential mechanism | yes | 3.98 | 0.834 | 0.803 | 0.788 |
| 3 | U7 DP-FedCP (Plassier et al.) | yes | 3.98 | 0.911 | 0.842 | 0.776 |
| 8 | DP-SCP (proposed) | yes | 3.43 | 0.941 | 0.900 | 0.892 |
| 8 | U2 FCP (Lu et al.) | no | 3.43 | 0.938 | 0.910 | 0.857 |
| 8 | U5 DP FCP, exponential mechanism (Lu et al.) | yes | 3.43 | 0.938 | 0.911 | 0.858 |
| 8 | U6 Per-class exponential mechanism | yes | 3.43 | 0.880 | 0.846 | 0.850 |
| 8 | U7 DP-FedCP (Plassier et al.) | yes | 3.43 | 0.915 | 0.862 | 0.815 |
| ∞ | DP-SCP (proposed) | yes | 3.00 | 0.952 | 0.913 | 0.847 |
| ∞ | U2 FCP (Lu et al.) | no | 3.00 | 0.961 | 0.932 | 0.886 |
| ∞ | U5 DP FCP, exponential mechanism (Lu et al.) | yes | 3.00 | 0.961 | 0.932 | 0.886 |
| ∞ | U6 Per-class exponential mechanism | yes | 3.00 | 0.947 | 0.896 | 0.860 |
| ∞ | U7 DP-FedCP (Plassier et al.) | yes | 3.00 | 0.959 | 0.925 | 0.899 |



### Table 13. Camelyon17-WILDS (binary tumour vs normal, α = 0.10)

| Method (ε=3) | Center 1 (val) BA | Center 2 (unseen test) BA | Coverage on center 2 | Set size |
|---|---|---|---|---|
| B3 DP-FedAvg full FT (ResNet-18) | 87.7 (n=1) | 89.1 (n=1) | 0.930 | 1.21 |
| B4 DP linear probe | 80.1 (n=1) | 89.9 (n=1) | 0.957 | 1.29 |
| B6 FFA-LoRA + DP | 84.5 (n=1) | 92.7 (n=1) | 0.951 | 1.18 |
| TriGuard-FL (proposed) | 86.2 (n=1) | 93.6 (n=1) | 0.944 | 1.12 |



### Table 14. MedMNIST+ 224×224, 20 clients, ε = 3 (balanced accuracy %)

| Subset | DP-FedAvg full FT | FFA-LoRA + DP | TriGuard-FL | TriGuard-FL coverage |
|---|---|---|---|---|
| pathmnist | 67.5 (n=1) | 65.7 (n=1) | 65.0 (n=1) | 0.982 |
| bloodmnist | 60.1 (n=1) | 33.4 (n=1) | 34.3 (n=1) | 0.978 |
| organamnist | 44.8 (n=1) | 18.9 (n=1) | 20.5 (n=1) | 0.994 |
| octmnist | 46.8 (n=1) | 37.4 (n=1) | 38.7 (n=1) | 0.902 |
| pneumoniamnist | 86.1 (n=1) | 84.8 (n=1) | 86.5 (n=1) | 0.976 |
| retinamnist | 20.0 (n=1) | 20.0 (n=1) | 20.0 (n=1) | 1.000 |



### Table 15. Transfer to PAD-UFES-20 smartphone images (6 mapped classes, ε = 3)

| Setting | Balanced acc. (%) | Coverage (α=0.10) | Set size |
|---|---|---|---|
| B6 FFA-LoRA + DP, ISIC thresholds | 39.8 | 0.948 | 4.55 |
| TriGuard-FL (proposed), ISIC thresholds | 39.4 | 0.936 | 4.63 |
| TriGuard-FL (proposed) + 200-image private recalibration | 39.4 | 1.000 | 6.00 |



### Table 16. Fed-Heart-Disease (non-image records)

| Method | ε | Mean AUC | Worst-hospital AUC | Coverage | Worst-site coverage |
|---|---|---|---|---|---|
| DP-FedAvg (logistic regression) + local CP | 1 | 0.893 ± 0.006 (n=5) | 0.828 ± 0.004 (n=5) | 0.944 ± 0.004 (n=5) | 0.910 ± 0.010 (n=5) |
| DP-FedAvg (logistic regression) + local CP | 3 | 0.898 ± 0.003 (n=5) | 0.829 ± 0.003 (n=5) | 0.947 ± 0.002 (n=5) | 0.914 ± 0.006 (n=5) |
| DP-FedAvg (logistic regression) + local CP | 8 | 0.898 ± 0.001 (n=5) | 0.827 ± 0.002 (n=5) | 0.947 ± 0.002 (n=5) | 0.915 ± 0.007 (n=5) |
| DP-FedAvg (logistic regression) + local CP | ∞ | 0.901 ± 0.001 (n=5) | 0.834 ± 0.001 (n=5) | 0.950 ± 0.004 (n=5) | 0.922 ± 0.009 (n=5) |
| RE aggregation + DP-SCP (C2–C4) | 1 | 0.894 ± 0.006 (n=5) | 0.828 ± 0.006 (n=5) | 1.000 ± 0.000 (n=5) | 1.000 ± 0.000 (n=5) |
| RE aggregation + DP-SCP (C2–C4) | 3 | 0.899 ± 0.003 (n=5) | 0.828 ± 0.003 (n=5) | 1.000 ± 0.000 (n=5) | 1.000 ± 0.000 (n=5) |
| RE aggregation + DP-SCP (C2–C4) | 8 | 0.900 ± 0.001 (n=5) | 0.827 ± 0.002 (n=5) | 1.000 ± 0.000 (n=5) | 1.000 ± 0.000 (n=5) |
| RE aggregation + DP-SCP (C2–C4) | ∞ | 0.901 ± 0.001 (n=5) | 0.834 ± 0.001 (n=5) | 0.986 ± 0.003 (n=5) | 0.981 ± 0.003 (n=5) |



### Table 17. Empirical privacy on site 1 (BCN)

| ε (theory) | LiRA AUC | TPR at 0.1% FPR | One-run ε lower bound (95%) | Inversion SSIM | PSNR | LPIPS |
|---|---|---|---|---|---|---|
| ∞ (no DP) | 0.550 | 0.45% | 1.55 | 0.404 | 13.4 | 0.949 |
| 8 | 0.476 | 0.07% | 0.06 | 0.408 | 13.4 | 0.946 |
| 3 | 0.475 | 0.05% | 0.00 | 0.409 | 13.4 | 0.944 |
| 1 | 0.477 | 0.05% | 0.00 | 0.411 | 13.4 | 0.944 |

Full fine-tuning without DP (FedAvg, ResNet-18 frozen BN, batch 8): inversion SSIM 0.192, PSNR 13.1, LPIPS 0.919.

### Table 18. Subgroup results for TriGuard-FL on Fed-ISIC2019

| ε | Attribute | Subgroup | BA | Coverage |
|---|---|---|---|---|
| ∞ | sex | female | 64.3 | 0.953 |
| ∞ | sex | male | 65.9 | 0.952 |
| ∞ | sex | Largest gap | 1.6 | 0.001 |
| ∞ | age | 40-60 | 66.1 | 0.945 |
| ∞ | age | <40 | 64.6 | 0.964 |
| ∞ | age | >60 | 62.7 | 0.956 |
| ∞ | age | Largest gap | 3.4 | 0.019 |
| 3 | sex | female | 41.2 | 0.944 |
| 3 | sex | male | 41.3 | 0.943 |
| 3 | sex | Largest gap | 0.1 | 0.002 |
| 3 | age | 40-60 | 38.2 | 0.942 |
| 3 | age | <40 | 30.8 | 0.937 |
| 3 | age | >60 | 42.3 | 0.948 |
| 3 | age | Largest gap | 11.5 | 0.011 |



### Table 19. Ablation on Fed-ISIC2019 (ε = 3)

| Variant | BA | Worst-site BA | Δ vs full |
|---|---|---|---|
| Full TriGuard-FL | 41.1 (n=1) | 38.4 (n=1) | – |
| A1 + random-effects aggregation | 41.4 (n=1) | 36.1 (n=1) | +0.2 |
| A2 + DP-gated private adapter / head | 45.1 (n=1) | 31.3 (n=1) | +4.0 |
| A3 frozen A (= B6 FFA-LoRA) | 39.0 (n=1) | 34.0 (n=1) | -2.1 |
| A4 − augmented cache (K = 1) | 40.0 (n=1) | 37.5 (n=1) | -1.1 |
| A5 ImageNet ViT-S instead of DINOv2 | 33.4 (n=1) | 27.1 (n=1) | -7.8 |
| A6 BiomedCLIP ViT-B instead of DINOv2 | 30.5 (n=1) | 28.7 (n=1) | -10.6 |
| A7 plan design (private head + adapter at every site, RE) | 45.8 (n=1) | 25.0 (n=1) | +4.6 |
| A8 TriGuard-D (RE + gated personalisation, frozen A) | 42.6 (n=1) | 28.3 (n=1) | +1.4 |



### Table 19b. Ablation on Fed-ISIC2019 (ε = 1)

| Variant | BA | Worst-site BA | Δ vs full |
|---|---|---|---|
| Full TriGuard-FL | 32.3 (n=1) | 29.9 (n=1) | – |



### Per-patient privacy ledger (TriGuard-FL, ε = 3, seed 0)

| Site | n_train | q | σ | steps | δ | C | ε_train | ε_label | ε_temp | ε_cal | ε_calib (RDP) | ε per patient |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BCN | 8014 | 0.128 | 3.398 | 400 | 4.0e-05 | 2.0 | 2.997 | 0.354 | 0.236 | 2.951 | 2.998 | 2.998 |
| Vidir-MM | 2530 | 0.250 | 4.292 | 200 | 1.4e-04 | 2.0 | 2.997 | 0.353 | 0.236 | 2.946 | 3.006 | 3.006 |
| Vidir-Mod | 2162 | 0.250 | 4.243 | 200 | 1.7e-04 | 2.0 | 2.997 | 0.354 | 0.236 | 2.946 | 3.006 | 3.006 |
| Rosendahl | 1446 | 0.250 | 4.116 | 200 | 2.6e-04 | 2.0 | 2.996 | 0.353 | 0.236 | 2.945 | 3.006 | 3.006 |
| MSK | 526 | 0.250 | 3.779 | 200 | 8.0e-04 | 2.0 | 2.998 | 0.352 | 0.235 | 2.934 | 3.016 | 3.016 |
| Vienna | 286 | 0.250 | 3.569 | 200 | 1.6e-03 | 2.0 | 2.997 | 0.351 | 0.234 | 2.927 | 3.023 | 3.023 |



### Statistics: TriGuard-FL vs best baseline

| ε | Best baseline | ΔBA (pp, mean over seeds) | seeds | Wilcoxon p | Bootstrap 95% CI (seed 0, pp) |
|---|---|---|---|---|---|
| 1 | B7 Personal LoRA + DP | -3.73 | 1 | n<5 | [-5.89, -1.67] |
| 3 | B7 Personal LoRA + DP | -5.63 | 1 | n<5 | [-8.49, -2.75] |
| 8 | B7 Personal LoRA + DP | -7.42 | 1 | n<5 | [-10.30, -4.51] |
| ∞ | B7 Personal LoRA + DP | -2.86 | 1 | n<5 | [-5.57, -0.14] |



### Table 20. Measured compute

| Job type | Runs | Hours each (mean) | Hours total |
|---|---|---|---|
| isic2019/b3_resnet | 9 | 1.71 | 15.4 |
| audit/inversion | 10 | 1.09 | 10.9 |
| isic2019/b2_central | 4 | 2.47 | 9.9 |
| isic2019/b7_personal | 4 | 2.42 | 9.7 |
| audit/canary | 4 | 2.38 | 9.5 |
| isic2019/b6_ffalora | 4 | 2.34 | 9.3 |
| isic2019/b1_local | 4 | 2.3 | 9.2 |
| lira/epsinf | 16 | 0.56 | 9.0 |
| isic2019/b5_fedlora | 4 | 2.15 | 8.6 |
| isic2019/triguard | 6 | 1.26 | 7.5 |
| lira/eps8 | 8 | 0.85 | 6.8 |
| lira/eps3 | 8 | 0.78 | 6.3 |
| lira/eps1 | 8 | 0.64 | 5.1 |
| camelyon17/b3_resnet | 1 | 2.9 | 2.9 |
| isic2019/a6_biomedclip | 1 | 2.73 | 2.7 |
| eval/isic2019 | 53 | 0.05 | 2.6 |
| isic2019/a5_imagenet | 2 | 1.31 | 2.6 |
| isic2019/a2_gated | 1 | 1.9 | 1.9 |
| isic2019/a7_allpersonal | 1 | 1.87 | 1.9 |
| isic2019/a1_fedavg | 1 | 1.84 | 1.8 |
| camelyon17/triguard | 1 | 1.83 | 1.8 |
| camelyon17/b6_ffalora | 1 | 1.73 | 1.7 |
| dl/isic2020 | 1 | 1.36 | 1.4 |
| isic2019/a1_re | 1 | 1.4 | 1.4 |
| isic2019/a2_nolocal | 1 | 1.29 | 1.3 |
| bloodmnist/b3_resnet | 2 | 0.58 | 1.2 |
| organamnist/b3_resnet | 2 | 0.58 | 1.2 |
| octmnist/b3_resnet | 2 | 0.58 | 1.2 |
| pathmnist/b3_resnet | 2 | 0.59 | 1.2 |
| prep/isic2020 | 1 | 0.97 | 1.0 |
| pneumoniamnist/b3_resnet | 2 | 0.46 | 0.9 |
| cache/camelyon17 | 1 | 0.75 | 0.8 |
| organamnist/b6_ffalora | 1 | 0.45 | 0.5 |
| cache/isic2019 | 3 | 0.18 | 0.5 |
| octmnist/triguard | 1 | 0.47 | 0.5 |
| pathmnist/b6_ffalora | 1 | 0.45 | 0.5 |
| organamnist/triguard | 1 | 0.48 | 0.5 |
| isic2019/a4_k1 | 1 | 0.45 | 0.5 |
| pathmnist/triguard | 1 | 0.48 | 0.5 |
| bloodmnist/b6_ffalora | 1 | 0.46 | 0.5 |
| bloodmnist/triguard | 1 | 0.48 | 0.5 |
| octmnist/b6_ffalora | 1 | 0.45 | 0.4 |
| pneumoniamnist/triguard | 1 | 0.38 | 0.4 |
| pneumoniamnist/b6_ffalora | 1 | 0.36 | 0.4 |
| cooldown/1793814395 | 1 | 0.26 | 0.3 |
| tune/rank4 | 1 | 0.3 | 0.3 |
| tune/rank16 | 1 | 0.29 | 0.3 |
| tune2/lr0.002 | 1 | 0.3 | 0.3 |
| tune2/lr0.003 | 1 | 0.31 | 0.3 |
| tune2/lr0.001 | 1 | 0.3 | 0.3 |
| tune2/lr0.0005 | 1 | 0.31 | 0.3 |
| heart/re_dpscp | 24 | 0.01 | 0.3 |
| isic2019/b4_probe | 4 | 0.08 | 0.3 |
| heart/dpfedavg_lr | 24 | 0.01 | 0.3 |
| retinamnist/b3_resnet | 2 | 0.15 | 0.3 |
| tune/C0.5_lr0.001_r8 | 1 | 0.29 | 0.3 |
| tune/C2.0_lr0.002_r8 | 1 | 0.3 | 0.3 |
| tune/C2.0_lr0.001_r8 | 1 | 0.29 | 0.3 |
| tune/C1.0_lr0.005_r8 | 1 | 0.29 | 0.3 |
| tune/C1.0_lr0.002_r8 | 1 | 0.29 | 0.3 |
| tune/C1.0_lr0.001_r8 | 1 | 0.3 | 0.3 |
| tune/C0.5_lr0.005_r8 | 1 | 0.29 | 0.3 |
| tune/C0.5_lr0.002_r8 | 1 | 0.29 | 0.3 |
| tune/C2.0_lr0.005_r8 | 1 | 0.29 | 0.3 |
| dl/pad_ufes | 1 | 0.25 | 0.2 |
| camelyon17/b4_probe | 1 | 0.18 | 0.2 |
| isic2019/timing | 1 | 0.23 | 0.2 |
| prep/pad_ufes20 | 1 | 0.15 | 0.2 |
| dl/isic2019 | 1 | 0.16 | 0.2 |
| cache/pneumoniamnist | 1 | 0.08 | 0.1 |
| cache/retinamnist | 1 | 0.05 | 0.1 |
| cache/pathmnist | 1 | 0.11 | 0.1 |
| cache/bloodmnist | 1 | 0.1 | 0.1 |
| cache/isic2020 | 1 | 0.05 | 0.1 |
| cache/octmnist | 1 | 0.09 | 0.1 |
| cache/organamnist | 1 | 0.09 | 0.1 |
| eval/organamnist | 4 | 0.04 | 0.1 |
| eval/bloodmnist | 4 | 0.02 | 0.1 |
| eval/pathmnist | 4 | 0.03 | 0.1 |
| retinamnist/triguard | 1 | 0.14 | 0.1 |
| prep/isic2019 | 1 | 0.05 | 0.1 |
| eval/retinamnist | 4 | 0.02 | 0.1 |
| eval/pneumoniamnist | 4 | 0.02 | 0.1 |
| eval/octmnist | 4 | 0.02 | 0.1 |
| retinamnist/b6_ffalora | 1 | 0.12 | 0.1 |
| dl/pathmnist | 1 | 0.05 | 0.0 |
| eval/camelyon17 | 4 | 0.01 | 0.0 |
| dl/retinamnist | 1 | 0.01 | 0.0 |
| clean/octmnist | 1 | 0.01 | 0.0 |
| clean/pathmnist | 1 | 0.01 | 0.0 |
| clean/organamnist | 1 | 0.01 | 0.0 |
| clean/pneumoniamnist | 1 | 0.01 | 0.0 |
| clean/retinamnist | 1 | 0.01 | 0.0 |
| dl/backbones | 1 | 0.03 | 0.0 |
| dl/bloodmnist | 1 | 0.02 | 0.0 |
| cache/isic2020_canary | 1 | 0.04 | 0.0 |
| audit/lira | 4 | 0.01 | 0.0 |
| cache/pad_ufes20 | 1 | 0.02 | 0.0 |
| dl/heart | 1 | 0.01 | 0.0 |
| dl/octmnist | 1 | 0.03 | 0.0 |
| dl/organamnist | 1 | 0.02 | 0.0 |
| dl/pneumoniamnist | 1 | 0.01 | 0.0 |
| clean/bloodmnist | 1 | 0.01 | 0.0 |
| clean/isic2019 | 2 | 0.0 | 0.0 |
| isic2019/a3_frozenA | 1 | 0.01 | 0.0 |
| pad/b6_ffalora | 1 | 0.02 | 0.0 |
| pad/triguard | 1 | 0.02 | 0.0 |
| report/final | 2 | 0.01 | 0.0 |
| prep/camelyon17 | 1 | 0.0 | 0.0 |
| tune/select_cl | 1 | 0.0 | 0.0 |
| tune/select | 2 | 0.0 | 0.0 |



## Figures

![fig02_privacy_utility](figures/fig02_privacy_utility.png)
![fig03_convergence](figures/fig03_convergence.png)
![fig04_per_site](figures/fig04_per_site.png)
![fig05_site_coverage](figures/fig05_site_coverage.png)
![fig06_coverage_alpha_size_eps](figures/fig06_coverage_alpha_size_eps.png)
![fig07_reliability](figures/fig07_reliability.png)
![fig08_lira_roc](figures/fig08_lira_roc.png)
![fig09_empirical_eps](figures/fig09_empirical_eps.png)
![fig10_inversion_ssim](figures/fig10_inversion_ssim.png)
![fig11_ablation](figures/fig11_ablation.png)
![fig12_cost](figures/fig12_cost.png)
![fig_extra_inversion_examples](figures/fig_extra_inversion_examples.png)
![fig_extra_re_weights](figures/fig_extra_re_weights.png)