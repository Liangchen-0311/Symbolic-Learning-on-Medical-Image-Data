# ThirdData (BUSI Breast Ultrasound) — Symbolic Learning Experiment

Symbolic feature discovery (RL-searched formulas) + HistGradientBoosting on the
**BUSI breast-ultrasound dataset** (3 classes: benign / malignant / normal).
Full results, ablations and baseline comparison in [`RESULTS.md`](RESULTS.md).

**Headline:** test accuracy **0.833** (AUC 0.912) with ~tens of thousands of
learnable parameters (formulas/statistics carry 0 params; everything learned is
in the HGB tree nodes). This beats the prior method (0.78, +5.3 pts) and the only
comparable interpretable baseline rulefit (0.667, +16.6 pts); it is on par with
swin/CBM/CRL and below the best deep CNN (resnet50, 0.875) — the gap is mostly in
the small `normal` class (87 train images). Pipeline reused from the Brain Tumor
experiment with the grayscale engine; best config = conv-free + focus step-3
(benign↔malignant) + 16-region (4×4) encoding at 224px.

## Dataset (not committed)

BUSI-style breast ultrasound, already split on disk:

```
ThirdData/
  train/{benign,malignant,normal}/*.png
  val/  {benign,malignant,normal}/*.png
  test/ {benign,malignant,normal}/*.png
```

Source: BUSI — Breast Ultrasound Images Dataset (Al-Dhabyani et al., 2020),
e.g. https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
Set `dataset_options.data_dir` in `configs/thirddata.yaml` to that path.
Images are loaded as grayscale (ultrasound is grayscale B-mode).

## Not committed (regenerable / large)

- `outputs/**/features.npz` (regenerate with step 4), `*.pth`, `compare method/**/best_weights.pth`.

Committed under `outputs/`: small artifacts — `validated_formulas.json`,
`classifier_results.json`, interpretability reports, trained HGB `*.pkl`.

## Run

```bash
python experiments/run_thirddata_pipeline.py \
    --config configs/thirddata.yaml --start_step 0 --end_step 6
```

Baselines: `compare method/train_thirddata.py` (resnet50/densenet121/
efficientnet_b0/swin_tiny/dinov2_linear/rulefit/CBM/CRL); summary with
`compare method/summarize_td.py`.
