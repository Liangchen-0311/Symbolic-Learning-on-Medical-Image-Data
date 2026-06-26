# COVIDx CT-3A — Symbolic Learning Results

Symbolic feature discovery (RL-searched formulas) + HistGradientBoosting (HGB)
on **COVIDx CT-3A** chest-CT slices (3 classes: normal / pneumonia / COVID-19).
Grayscale CT, 512×512, loaded at 224px.

- **Dataset is huge** (~425k images / 63 GB) and heavily skewed (train COVID
  294k vs normal 36k). We use a **class-balanced symlink subset** (no copying):
  train 3000/class (9000), val 800/class (2400), test 1500/class (4500).
- All numbers on the same balanced 4500-image test set → directly comparable to
  the baselines below (so acc == balanced acc here).

---

## 1. Headline result

| | value |
|---|---|
| **Test accuracy** | **0.895** |
| Balanced accuracy | 0.895 |
| Macro-OvR AUC | 0.979 |
| Per-class (normal / pneumonia / covid) | 0.917 / 0.930 / 0.837 |
| Learnable parameters | ~tens of thousands (formulas/stats = 0 params; all in HGB tree nodes) |

Config: conv-free operators + focus step-3 (pneumonia↔covid) + 16-region
(whole + 4×4) distribution encoding + ANOVA→HGB, grayscale, 224px.
Pipeline: 6000 formulas → step-3 de-dup/quality → 800 → step-4 (192/body) → HGB.

---

## 2. Comparison vs baselines (same balanced test set)

| Method | acc | AUC | normal | pneu | covid | type |
|---|---|---|---|---|---|---|
| swin_tiny | **0.956** | 0.996 | 0.976 | 0.973 | 0.921 | deep transformer |
| CBM | 0.949 | 0.990 | 0.971 | 0.945 | 0.930 | concept bottleneck (deep) |
| densenet121 | 0.945 | 0.993 | 0.967 | 0.961 | 0.908 | deep CNN |
| resnet50 | 0.937 | 0.991 | 0.953 | 0.962 | 0.897 | deep CNN |
| efficientnet_b0 | 0.936 | 0.989 | 0.935 | 0.956 | 0.916 | deep CNN |
| CRL | 0.909 | 0.968 | 0.939 | 0.951 | 0.837 | concept reasoning (deep) |
| **OURS (symbolic + HGB)** | **0.895** | 0.979 | 0.917 | 0.930 | 0.837 | **formulas (0p) + HGB** |
| dinov2_linear | 0.820 | 0.948 | 0.869 | 0.909 | 0.681 | frozen ViT + linear |
| rulefit | 0.718 | 0.874 | 0.907 | 0.855 | 0.391 | rules (interpretable) |

(Deep baselines: 20 epochs, grayscale→RGB, same balanced subset. Ours: 0-param
formulas + small HGB ensemble.)

---

## 3. Takeaways

1. **Strong, well-balanced result**: 0.895 with all three classes 0.84–0.93 (no
   majority-class inflation, thanks to the balanced subset).
2. **Best among interpretable methods**: beats rulefit (0.718, +17.7 pts) and the
   frozen DINOv2 linear probe (0.820, +7.5 pts).
3. **Within ~6 pts of the best deep model** (swin 0.956). Smaller gap than on
   HAM10000, slightly larger than on Brain/ThirdData.
4. **`covid` is the shared hard class** (ours 0.837; even deep models 0.84–0.93)
   — COVID vs non-COVID pneumonia overlap is a known clinical difficulty, not a
   weakness specific to our method. AUC 0.979 shows the information is largely
   present; the gap is in the decision boundary.
5. **Saturation / scaling note**: a 96px / 15-iter / 25-formula smoke already hit
   0.866; the full 800-formula / 224px run reached 0.895 — i.e. 32× more formulas
   bought only ~3 pts. Consistent with the "few formulas already near the ceiling,
   adding more barely helps" pattern (motivating the residual-reward experiment).

---

## 4. Artifacts

- Best run: `outputs/covidx/` — `classifier_results.json`, `validated_formulas.json`
  (800 formulas), `interpretability_report.txt`.
- Baselines: `compare method/<method>/results.json`; scripts
  `compare method/train_covidx.py`, `summarize_covidx.py`.
- Config: `configs/covidx.yaml` (conv-free, focus pneumonia↔covid, n_regions 16, 224px).
- Balanced subset: `COVIDx_CT_subset/{train,val,test}/<class>/` (symlinks into the
  original `COVIDx CT/3A_images/`; built from the official manifests, seed 42).

Dataset (not committed): COVIDx CT-3A (Gunraj et al.); the raw 63GB image set and
the symlink subset are not in git. Manifests in `COVIDx CT/*.txt`.
