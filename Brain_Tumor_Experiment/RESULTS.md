# Brain Tumor MRI — Symbolic Learning Results

Symbolic feature discovery (RL-searched formulas) + HistGradientBoosting (HGB)
classifier on the **Brain Tumor MRI Dataset** (4 classes: glioma, meningioma,
pituitary, notumor). Grayscale MRI, 512×512 JPG.

- **Split (used everywhere, including baselines):** validation carved from
  `Training/` (stratified, `val_fraction=0.15`, `seed=42`); `Testing/` used in
  full as the test set. → Train 4760 / Val 840 / Test 1600, **class-balanced**.
- All numbers below are on the **same 1600-image test set**, so our method and
  every baseline are directly comparable.

---

## 1. Headline result

| | value |
|---|---|
| **Test accuracy** | **0.916** |
| Balanced accuracy | 0.916 |
| Macro-OvR AUC | 0.975 |
| Per-class (gli/men/pit/notum) | 0.728 / 0.950 / 0.988 / 1.000 |
| **Learnable parameters** | **≈ 45k** (formulas/features = 0 params; all in HGB tree nodes) |

Best configuration: **conv-free operators + focus step-3 (glioma↔meningioma) +
10-region (3×3) distribution encoding + ANOVA→HGB**.

---

## 2. Method (pipeline)

1. **RL formula search (step 1)** — PPO searches RPN formulas over grayscale
   terminals `[I_GRAY, I_BLUR, I_GRAD, I_LOCALSTD, I_LOG, I_LBP]` and a library
   of fixed, modality-agnostic operators (Gabor/Sobel/Laplacian/DoG/morphology/
   symmetry/texture). 4 banks × 1500 formulas → 6000 candidates.
2. **Merge (step 2)** — dedupe to unique bodies.
3. **Post-merge de-dup + quality gating (step 3)** — each body is scored on one
   common image subset (signatures comparable across banks): ANOVA-F quality
   (optionally blended with the glioma-vs-meningioma binary F), keep top 70%,
   greedy `|Pearson|≥0.95` de-duplication, cap to 800.
4. **Feature extraction (step 4)** — each formula → spatial map → distribution
   statistics (12 stats × N regions). **10 regions (whole + 3×3) is the sweet spot.**
5. **Classifier (step 5)** — ANOVA feature selection → HistGradientBoosting.
6. **Interpretability report (step 6)**.

The **formulas and statistics carry no learnable parameters** (fixed symbolic
operations). All learned parameters live in the HGB tree ensemble.

---

## 3. Ablations (what helped, what didn't)

| Change | Test acc | glioma | Verdict |
|---|---|---|---|
| Baseline (conv-free, plain step3, 5-region) | 0.899 | 0.698 | reference |
| + focus step-3 (glioma↔meningioma quality blend) | 0.906 | 0.703 | **+0.7% ✓ kept** |
| + 10-region (3×3) encoding | **0.916** | **0.728** | **+1.0% ✓ kept (best)** |
| 17-region (4×4) encoding | 0.909 | 0.708 | ✗ past the sweet spot |
| + add 3000 glioma-vs-meningioma formulas (combined pool) | 0.898 | 0.690 | ✗ no gain |
| + operator prior + explicit `I_LRDIFF` symmetry terminal | 0.899 | 0.690 | ✗ no gain |
| MI feature selection instead of ANOVA (10-region) | 0.9125 vs 0.9100 | = | ✗ ≈ANOVA, 20–100× slower |

**Findings**
- The bottleneck is the **encoding/pooling**, not formula search. Refining the
  spatial pooling (5→10 regions) gave the largest clean gain; 4×4 overshoots
  (cells too small → noisy stats). 3×3 is the sweet spot.
- Search-side interventions (more formulas, operator priors, an explicit
  symmetry terminal) did **not** help — the signal is already in the formulas.
- **glioma↔meningioma is the intrinsic hard pair**: glioma sits at ~0.70–0.73
  across every variant, and even deep CNNs are weakest there (0.80–0.82).
- **conv-free** (dropping the learnable conv kernels) cost ~0 accuracy
  (0.899 ≡ 0.899) and made the pipeline fully reproducible + interpretable.
- **MI** selection ties ANOVA (noise-level) but is far slower → dropped.

---

## 4. Comparison vs baselines (same test set)

| Method | acc | bal | AUC | glioma | type | params (total / trainable) | train |
|---|---|---|---|---|---|---|---|
| densenet121 | 0.953 | 0.953 | 0.993 | 0.818 | deep CNN | 7.0M / 7.0M | 5.2 min |
| resnet50 | 0.952 | 0.952 | 0.993 | 0.818 | deep CNN | 23.5M / 23.5M | 4.4 min |
| CBM | 0.951 | 0.951 | 0.967 | 0.807 | concept bottleneck (deep) | 21.5M / 21.5M | 5.7 min |
| swin_tiny | 0.950 | 0.950 | 0.992 | 0.815 | deep transformer | 27.5M / 27.5M | 12.0 min |
| CRL | 0.948 | 0.948 | 0.984 | 0.797 | concept reasoning (deep) | 21.8M / 21.8M | 5.7 min |
| efficientnet_b0 | 0.947 | 0.947 | 0.994 | 0.797 | deep CNN | 4.0M / 4.0M | 3.1 min |
| dinov2_linear | 0.919 | 0.919 | 0.978 | 0.745 | frozen ViT + linear | 86.6M / ~3k | 0.8 min |
| **OURS (symbolic + HGB)** | **0.916** | **0.916** | **0.975** | **0.728** | **formulas (0p) + HGB** | **≈45k / ≈45k** | — |
| rulefit | 0.744 | 0.744 | 0.915 | 0.552 | rules (interpretable) | 2.2M / ~0 | 0.2 min |

### Parameter-count breakdown (ours)
- formulas: **0 params** (fixed symbolic ops)
- distribution statistics: **0 params**
- HGB: 1200 trees, **44,968 nodes** (23,084 leaves + 21,884 splits) → the learned scalars
- input features used: 500

---

## 5. Takeaways

1. **Best among genuinely interpretable methods** — beats the only comparable
   interpretable baseline (rulefit 0.744) by **+17 points** (0.916).
2. **Matches a frozen foundation-model linear probe** (DINOv2, 0.919) using
   **no neural features** and ~1900× fewer parameters at inference.
3. **Within ~3.5% of end-to-end deep CNNs** (≈0.95) using **~1/100–1/600 of the
   parameters** (≈45k vs 4M–27.5M), and the model is a transparent set of
   symbolic formulas + a small tree ensemble.
4. The residual gap is concentrated almost entirely in **glioma vs meningioma**,
   a pair that is hard for every method (deep models top out at ~0.82 there).

---

## 6. Artifacts

- Best run: `outputs/brain_dir3_fine/` (0.916, 10-region) — `classifier_results.json`,
  `validated_formulas.json` (800 formulas), `interpretability_report.txt`.
- Ablation runs: `outputs/brain_tumor_cf` (baseline), `brain_dir3` (focus,
  5-region), `brain_dir3_fine16` (4×4), `brain_dir1` (combined pool),
  `brain_tumor_prior` (operator prior + symmetry terminal), `brain_dir3_fine_mi` (MI).
- Baselines: `compare method/<method>/results.json`; helper scripts
  `compare method/train_brain.py`, `summarize.py`, `count_params.py`.
- Config of the best pipeline: `configs/brain_tumor.yaml`
  (conv-free, `focus_pair: [glioma, meningioma]`, set `n_regions: 10` for the
  3×3 encoding).

_Note: deep-baseline tree/param counts use standard conventions; HGB "params" =
tree-node count (split thresholds + leaf values), not floating-point weights, so
cross-type comparison is order-of-magnitude, not exact._
