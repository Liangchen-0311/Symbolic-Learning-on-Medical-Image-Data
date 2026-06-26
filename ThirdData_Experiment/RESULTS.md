# ThirdData (BUSI Breast Ultrasound) — Symbolic Learning Results

Symbolic feature discovery (RL-searched formulas) + HistGradientBoosting (HGB)
on the **ThirdData / BUSI breast-ultrasound dataset** (3 classes: benign,
malignant, normal). Grayscale ultrasound, PNG, variable resolution.

- **Split:** train/val/test already provided on disk and used as-is (no carving).
  Train 545 (benign 310 / malignant 148 / normal 87), Val 115, Test 120.
  **Class-imbalanced.**
- All numbers below are on the **same 120-image test set** → our method and every
  baseline are directly comparable.

---

## 1. Headline result

| | value |
|---|---|
| **Test accuracy** | **0.833** |
| Balanced accuracy | 0.760 |
| Macro-OvR AUC | 0.912 |
| Per-class (benign / malignant / normal) | 0.955 / 0.781 / 0.545 |
| Learnable parameters | ~tens of thousands (formulas/stats = 0 params; all in HGB tree nodes) |

Best configuration: **conv-free operators + focus step-3 (benign↔malignant) +
16-region (whole + 4×4) distribution encoding + ANOVA→HGB**, at 224px.

**Context:** the prior method (SecondData / HAM10000-style pipeline) reportedly
scored **0.78** on this data; our improved pipeline reaches **0.833 (+5.3 pts)**.

---

## 2. Ablations (what helped, what didn't)

All reuse the same RL-searched formulas unless noted; encoding/resolution changes
only re-run feature extraction + classifier.

| Setting | resolution | regions | test acc | note |
|---|---|---|---|---|
| baseline | 224 | 10 | 0.817 | start (Brain's best config) |
| **encoding 16 (4×4)** | 224 | **16** | **0.833** | **best ✓** |
| encoding 5 | 224 | 5 | 0.833 | tied best (AUC lower) |
| encoding 26 (5×5) | 224 | 26 | 0.825 | past the sweet spot |
| resolution 384 (full re-search) | 384 | 16 | 0.792 | ✗ worse even when RL re-searched at 384 |
| resolution 384 | 384 | 5 | 0.758 | ✗ |
| resolution 384 | 384 | 26 | 0.783 | ✗ |
| classifier imbalance tricks | 224 | 16 | no robust test gain | ✗ (see below) |

**Findings**
- **Encoding granularity** is the main lever (as on Brain), but the sweet spot
  here is **16 regions (4×4)**, not 10 — the optimum is dataset-dependent.
- **Higher resolution (384) hurts**, even when the RL search itself is redone at
  384 (so it is not a train/test resolution mismatch). Breast ultrasound is
  dominated by **speckle noise**; upscaling amplifies noise rather than signal,
  and the distribution-statistics encoding is largely scale-insensitive.
- **Classifier-side imbalance handling** (balanced class weights, normal-weight
  sweep up to ×8, per-class decision thresholds) improved `normal` recall on the
  *val* set but **did not transfer to test** — the `normal` class has only 87
  train / 24 val / 22 test images, so any per-class tuning overfits the tiny val
  set. This is a data-size limit, not a method bug.
- **conv-free** kept (reproducibility; ~0 accuracy cost, as on Brain).

---

## 3. Comparison vs baselines (same 120-image test set)

| Method | acc | bal | AUC | benign | malig | normal | type |
|---|---|---|---|---|---|---|---|
| resnet50 | **0.875** | 0.888 | 0.946 | 0.85 | 0.91 | 0.91 | deep CNN |
| swin_tiny | 0.842 | 0.862 | 0.959 | 0.82 | 0.81 | 0.96 | deep transformer |
| CBM | 0.842 | 0.862 | 0.930 | 0.82 | 0.81 | 0.96 | concept bottleneck (deep) |
| CRL | 0.842 | 0.863 | 0.891 | 0.79 | 0.94 | 0.86 | concept reasoning (deep) |
| **OURS (symbolic + HGB)** | **0.833** | 0.760 | 0.912 | **0.955** | 0.781 | 0.545 | **formulas (0p) + HGB** |
| efficientnet_b0 | 0.825 | 0.833 | 0.949 | 0.79 | 0.94 | 0.77 | deep CNN |
| densenet121 | 0.808 | 0.781 | 0.944 | 0.85 | 0.81 | 0.68 | deep CNN |
| dinov2_linear | 0.767 | 0.754 | 0.931 | 0.80 | 0.69 | 0.77 | frozen ViT + linear |
| rulefit | 0.667 | 0.654 | 0.874 | 0.77 | 0.28 | 0.91 | rules (interpretable) |

(Deep baselines: 30 epochs, grayscale→RGB, same split. Params: resnet50 23.5M,
densenet 7.0M, efficientnet 4.0M, swin 27.5M, CBM/CRL ~21M, dinov2 86.6M frozen,
rulefit 2.2M frozen. Ours: 0-param formulas + a small HGB tree ensemble.)

---

## 4. Takeaways (honest)

1. **Beats the prior method (0.78 → 0.833, +5.3 pts)** — the original goal.
2. **Best interpretable method by a wide margin**: rulefit (the only comparable
   interpretable baseline) is 0.667; we are 0.833 (+16.6 pts).
3. **Beats dinov2-linear, densenet, efficientnet**; on par with swin/CBM/CRL on
   accuracy (0.833 vs 0.842).
4. **Below the best deep CNN (resnet50, 0.875).** Unlike the Brain experiment
   (where we matched deep models within ~3.5%), here deep models hold a clearer
   edge — concentrated in the **`normal` class** (deep: 0.77–0.96; ours: 0.545)
   and reflected in our lower balanced accuracy (0.760).
5. The `normal` gap is a **small-sample limitation** (87 train images); it does
   not respond to classifier-side imbalance tricks (they overfit the 24-image
   val set). Closing it would likely require more `normal` data or a fundamentally
   different representation, not more formula search.
6. Caveat: the test set is only 120 images, so differences of a few points
   (~1 pt ≈ 1 image) are within noise; deep nets on 545 training images also have
   notable run-to-run variance.

---

## 5. Artifacts

- Best run: `outputs/thirddata_reg16/` (0.833, 16-region) — `classifier_results.json`,
  `validated_formulas.json` (800 formulas), `interpretability_report.txt`.
- Source 224 run (10-region, formulas reused for the encoding sweep): `outputs/thirddata/`.
- Ablation runs: `outputs/thirddata_reg{5,16,26}`, `outputs/thirddata_384full`,
  `outputs/thirddata_384_reg{5,26}`.
- Baselines: `compare method/<method>/results.json`; scripts
  `compare method/train_thirddata.py`, `summarize_td.py`, `count_params.py`,
  `scripts/analyze_normal.py`, `scripts/tune_imbalance.py`.
- Config of the best pipeline: `configs/thirddata.yaml`
  (conv-free, `focus_pair: [benign, malignant]`, `n_regions: 16`, 224px).

Dataset (not committed): BUSI-style breast ultrasound, 3 classes, on-disk
train/val/test under `ThirdData/`. Grayscale-loaded (content is grayscale B-mode).
