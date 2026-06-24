# Brain Tumor MRI — Symbolic Learning Experiment

Symbolic feature discovery (RL-searched formulas) + HistGradientBoosting (HGB)
classifier on the **Brain Tumor MRI Dataset** (4 classes: glioma, meningioma,
pituitary, notumor). Full results, ablations and baseline comparison are in
[`RESULTS.md`](RESULTS.md).

**Headline:** test accuracy **0.916** (AUC 0.975) with **≈45k learnable
parameters** (formulas/statistics carry 0 params; everything learned lives in
the HGB tree nodes) — best among interpretable methods, on par with a frozen
DINOv2 linear probe, within ~3.5% of deep CNNs at ~1/100 of the parameters.

## Dataset (not committed)

The raw images are **not** stored in this repo (public dataset + size). Download
the **Brain Tumor MRI Dataset** (Msoud Nickparvar, Kaggle) and place it as:

```
Brain Tumor MRI Dataset/
  Training/{glioma,meningioma,pituitary,notumor}/*.jpg
  Testing/ {glioma,meningioma,pituitary,notumor}/*.jpg
```

Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Then set `dataset_options.data_dir` in `configs/brain_tumor.yaml` to that path.

## Not committed (regenerable / large)

- `outputs/**/features.npz` — extracted feature matrices (up to 4.4 GB each),
  regenerate with step 4. Excluded via `.gitignore`.
- `*.pth` model checkpoints / `compare method/**/best_weights.pth`.

Committed under `outputs/` are the small artifacts: `validated_formulas.json`
(discovered formulas), `classifier_results.json` (metrics), interpretability
reports, and the trained HGB `*.pkl` classifiers.

## Run

```bash
python experiments/run_brain_tumor_pipeline.py \
    --config configs/brain_tumor.yaml --start_step 0 --end_step 6
```

Best config: conv-free operators, focus step-3 (glioma↔meningioma), 10-region
(3×3) distribution encoding, ANOVA→HGB. Baselines: `compare method/train_brain.py`.
