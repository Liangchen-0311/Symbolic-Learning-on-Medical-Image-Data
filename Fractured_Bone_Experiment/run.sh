#!/bin/bash
# Bone Fracture Symbolic Learning Pipeline (v3 — Expanded Feature Discovery)
# v3: 放宽相关性门槛、扩大特征银行容量、丰富编码、GPU选择
# Usage:
#   bash run.sh train          # Run full pipeline with GPU 0 (default)
#   bash run.sh train 1        # Run full pipeline with GPU 1
#   bash run.sh predict        # Predict on test set
#   bash run.sh predict_image xray.jpg   # Predict single image
#   bash run.sh predict_dir /path/to/    # Predict on image directory

set -e

GPU_ID="${2:-0}"
CONFIG="configs/fracture_v3_expanded.yaml"

case "$1" in
    train)
        echo "=== Running v3 Full Pipeline on GPU ${GPU_ID} ==="
        python experiments/run_fracture_pipeline.py --config "$CONFIG" --gpu "$GPU_ID"
        ;;
    predict)
        echo "=== Predicting on Test Set ==="
        python scripts/predict_fracture.py --config "$CONFIG" --gpu "$GPU_ID"
        ;;
    predict_image)
        if [ -z "$2" ]; then
            echo "Usage: bash run.sh predict_image <image_path> [gpu_id]"
            exit 1
        fi
        IMAGE_PATH="$2"
        GPU_ID="${3:-0}"
        echo "=== Predicting Single Image: $IMAGE_PATH ==="
        python scripts/predict_fracture.py --config "$CONFIG" --gpu "$GPU_ID" --image "$IMAGE_PATH"
        ;;
    predict_dir)
        if [ -z "$2" ]; then
            echo "Usage: bash run.sh predict_dir <image_directory> [gpu_id]"
            exit 1
        fi
        DIR_PATH="$2"
        GPU_ID="${3:-0}"
        echo "=== Predicting Image Directory: $DIR_PATH ==="
        python scripts/predict_fracture.py --config "$CONFIG" --gpu "$GPU_ID" --image_dir "$DIR_PATH"
        ;;
    *)
        echo "Bone Fracture Symbolic Learning Pipeline (v3 — Expanded)"
        echo ""
        echo "Usage:"
        echo "  bash run.sh train [gpu_id]           Run full pipeline (default GPU 0)"
        echo "  bash run.sh predict [gpu_id]         Predict on test set"
        echo "  bash run.sh predict_image <path> [gpu_id]  Predict single image"
        echo "  bash run.sh predict_dir <dir> [gpu_id]     Predict on image directory"
        echo ""
        echo "Examples:"
        echo "  bash run.sh train          # Train on GPU 0"
        echo "  bash run.sh train 1        # Train on GPU 1"
        ;;
esac
