#!/bin/bash
# Sequentially train all comparison baselines on Brain Tumor MRI (card 1).
cd "$(dirname "$0")"
METHODS="resnet50 densenet121 efficientnet_b0 swin_tiny dinov2_linear rulefit CBM CRL"
for m in $METHODS; do
  echo "########## $m ##########"
  CUDA_VISIBLE_DEVICES=1 python train_brain.py --method "$m" --epochs 30 --gpu 0 2>&1 | grep -vE "Warning|warn|B/s"
done
echo ALL_DONE
