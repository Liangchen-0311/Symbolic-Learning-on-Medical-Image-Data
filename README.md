# Fracture Symbolic v3 — Symbolic Feature Discovery for Bone Fracture Classification

A reinforcement learning framework that discovers interpretable symbolic formulas for bone fracture classification on the HBFMID dataset (10 fracture types). A PPO agent searches over tensor operator sequences (edge detection, morphology, multiscale, Gabor, etc.) to construct feature extraction programs, which are then encoded and fed to ensemble classifiers.

## Results

| Method | Test Accuracy | Test AUC |
|--------|:------------:|:--------:|
| **HGB + MI + Sample Weight (best)** | **95.85%** | **0.9988** |
| Stacking + MI | 95.85% | 0.9973 |
| SVM (RBF) | 93.55% | 0.9981 |
| HGB + MI | 93.09% | 0.9982 |
| HGB + Sample Weight | 92.17% | 0.9959 |
| Stacking | 91.71% | 0.9950 |
| HGB Baseline | 89.40% | 0.9957 |
| MLP | 89.40% | 0.9912 |
| KNN | 88.94% | 0.9903 |
| Hierarchical | 85.25% | 0.8949 |
| LR (L2) | 75.12% | 0.9623 |
| Linear | 72.35% | 0.9503 |

- **Dataset**: HBFMID — 1,539 X-ray images, 10 fracture types
- **Split**: Stratified 70/15/15 (1,105 train / 217 val / 217 test)
- **Features**: 625 validated symbolic formulas → 70,000-dim encoded feature vector

## Pipeline Overview

```
Step 0: Dataset validation & stratified split (split_indices.npz)
Step 1: RL formula discovery (5 banks, PPO, 128×128 resolution)
Step 2: Merge & deduplicate formulas across banks
Step 3: Full-resolution validation (640×640, two-phase prefilter)
Step 4: Feature extraction + encoding (distribution stats + Fisher Vector + kernel map)
Step 5: Train ensemble classifiers (13 methods)
Step 6: Evaluation + interpretability report
```

### Multi-Bank Strategy

Each bank targets a different feature family:

| Bank | Focus | Max Depth | Sequence Length |
|------|-------|:---------:|:--------------:|
| 0 | Short-edge | 4 | 8 |
| 1 | Cross-channel | 6 | 15 |
| 2 | Texture-morphology | 5 | 12 |
| 3 | Multiscale-symmetry | 6 | 15 |
| 4 | Deep-composite | 7 | 18 |

### Operators

98 base tensor operators + 20 fracture-specific operators, including:
- **Edge**: `edge_x`, `edge_y`, `edge_mag`, `edge_orient`, `laplacian`, `corner_harris`
- **Morphology**: `dilate`, `opening`, `closing`, `tophat`
- **Filter banks**: `gabor_0/45/90`, `dog`, `local_std_5x5`, `local_contrast`, `lbp_like`
- **Multiscale**: `downsample_2x/4x`, `stride_pool_4`, `high_freq`, `low_freq`
- **Arithmetic**: `add`, `subtract`, `multiply`, `div`, `relu`, `sigmoid`, `pow2`, `sqrt_abs`

## Requirements

- Python 3.12
- CUDA 12.8 (GPU required for RL search)
- PyTorch 2.11+

### Install

```bash
# Option 1: Conda (recommended, exact reproduction)
conda env create -f environment.yml
conda activate symbol

# Option 2: pip
pip install -r requirements.txt
```

## Dataset Setup

Download the HBFMID (Human Bone Fractures Multi-modal Image Dataset) from [Roboflow](https://universe.roboflow.com/ahmedmagdy/human-bone-fractures-multi-modal-image-dataset-hbfmid) and place it under:

```
fracture_symbolic_v3/
└── datasets/
    └── Human Bone Fractures Multi-modal Image Dataset (HBFMID)/
        └── Bone Fractures Detection/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── valid/
            │   ├── images/
            │   └── labels/
            └── test/
                ├── images/
                └── labels/
```

The 10 fracture classes:

| ID | Class |
|----|-------|
| 0 | Comminuted |
| 1 | Greenstick |
| 2 | Healthy |
| 3 | Linear |
| 4 | Oblique Displaced |
| 5 | Oblique |
| 6 | Segmental |
| 7 | Spiral |
| 8 | Transverse Displaced |
| 9 | Transverse |

## Configuration

Edit [configs/fracture_v3_expanded.yaml](configs/fracture_v3_expanded.yaml):

```yaml
# Key parameters
dataset_options:
  data_dir: /path/to/HBFMID/Bone Fractures Detection
  resolution_quick: 128    # RL search resolution
  resolution_full: 640     # Validation & feature extraction resolution
  num_classes: 10

training:
  iterations: 5000         # PPO iterations per bank
  episodes_per_iteration: 30
  learning_rate: 0.0003

strategy:
  feature_bank_size: 1000  # Max formulas per bank
  min_accuracy_threshold: 0.08
  correlation_threshold: 0.85

multi_bank:
  enabled: true
  num_banks: 5

classifier:
  use_smote: true
  use_class_weight: true
  epochs: 50
```

## Running the Pipeline

### Full Pipeline (Step 0 → Step 6)

```bash
# Using run.sh
bash run.sh train          # GPU 0 (default)
bash run.sh train 1         # GPU 1

# Or directly
python experiments/run_fracture_pipeline.py --config configs/fracture_v3_expanded.yaml --gpu 0
```

### Run Specific Steps

```bash
# Start from Step 5 (skip RL search, reuse existing features.npz)
python experiments/run_fracture_pipeline.py --config configs/fracture_v3_expanded.yaml --start_step 5

# Run v6 advanced ensemble (Step 5-6 only, requires Step 0-4 completed)
python experiments/run_fracture_pipeline_v6.py --config configs/fracture_v3_expanded.yaml --start_step 5
```

### Background Run (Long Training)

```bash
nohup python experiments/run_fracture_pipeline.py \
    --config configs/fracture_v3_expanded.yaml \
    --gpu 0 \
    > outputs/train.log 2>&1 &

# Monitor
tail -f outputs/train.log
```

### Inference

```bash
# Predict on test set
bash run.sh predict

# Predict single image
bash run.sh predict_image path/to/xray.jpg

# Predict on a directory
bash run.sh predict_dir path/to/images/
```

## Output Structure

```
outputs/fracture_v3_expanded/
├── split_indices.npz          # Step 0: Stratified split indices
├── dataset_stats.json         # Step 0: Dataset statistics
├── phase1/
│   ├── bank_0/                # Step 1: RL formulas per bank
│   │   ├── feature_bank.json
│   │   └── checkpoints/
│   ├── bank_1/
│   └── ...
├── merged_formulas.json       # Step 2: 5000 merged formulas
├── validated_formulas.json    # Step 3: 625 validated formulas
├── features.npz               # Step 4: Encoded features (70,000-dim)
├── classifier_results.json    # Step 5: Base classifier results
├── v6/
│   ├── classifier_results_v6.json   # Step 5: v6 ensemble results
│   ├── best_classifier_v6.pkl       # Best trained model
│   └── test_predictions_v6.json     # Step 6: Test predictions
└── baseline_comparison/       # Baseline comparison results
```

## Project Structure

```
fracture_symbolic_v3/
├── configs/
│   └── fracture_v3_expanded.yaml       # Main configuration
├── src/
│   ├── data/
│   │   └── fracture_loader.py           # HBFMID dataset loader
│   ├── models/
│   │   └── policy_agent.py              # PPO policy network
│   ├── rl/
│   │   ├── ppo_trainer.py              # PPO training loop
│   │   ├── fracture_environment.py     # RL environment
│   │   ├── rpn_grammar_mask.py         # RPN grammar masking
│   │   └── reward.py                   # Reward computation
│   ├── symbolic/
│   │   ├── tensor_operators.py         # 98 base operators
│   │   ├── fracture_operators.py       # 20 fracture-specific operators
│   │   ├── tensor_evaluator.py         # Formula evaluation engine
│   │   ├── feature_bank.py             # Feature bank management
│   │   ├── large_feature_bank.py       # Large-scale feature bank
│   │   └── feature_encoding.py         # Fisher Vector + distribution stats
│   └── utils/
│       ├── evaluation_metrics.py       # Balanced accuracy, AUC
│       └── visualization.py           # Probe visualization
├── experiments/
│   ├── run_fracture_pipeline.py        # Main pipeline (Step 0-6)
│   ├── run_fracture_pipeline_v6.py     # v6: Advanced ensemble
│   └── baseline_comparison.py          # Baseline comparison
├── scripts/
│   ├── predict_fracture.py             # Inference script
│   ├── resplit_dataset.py             # Dataset re-splitting
│   └── evaluate_all_models.py         # Model evaluation
├── run.sh                              # Convenience script
├── requirements.txt
└── environment.yml
```

## Key Features

- **RPN Grammar Masking**: Ensures generated formulas are syntactically valid
- **Two-Phase Validation**: Low-resolution prefilter (128×128) → full-resolution verification (640×640)
- **Correlation-Based Deduplication**: Removes redundant formulas (threshold=0.85)
- **Medical Interpretability Bonus**: Rewards formulas using medically relevant operators
- **Adaptive Accuracy Threshold**: Starts low, increases over training to encourage exploration
- **Checkpoint Resume**: Automatically resumes from latest valid checkpoint

## Citation

If you use this code, please cite the HBFMID dataset:

```bibtex
@dataset{hbfmid,
  title={Human Bone Fractures Multi-modal Image Dataset (HBFMID)},
  author={Magdy, Ahmed},
  publisher={Roboflow Universe},
  url={https://universe.roboflow.com/ahmedmagdy/human-bone-fractures-multi-modal-image-dataset-hbfmid}
}
```

## License

This project is for research purposes. The HBFMID dataset has its own license — please refer to the Roboflow page for details.
