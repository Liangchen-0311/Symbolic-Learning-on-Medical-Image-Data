#!/bin/bash
cd "/home/ET/lctan/Symbolic-Learning/Symbolic-Learning-on-Medical-Image-Data/COVIDx_Experiment/compare method"
GPU="${1:-3}"
for m in resnet50 densenet121 efficientnet_b0 swin_tiny dinov2_linear rulefit CBM CRL; do
  echo "########## $m ##########"
  CUDA_VISIBLE_DEVICES=$GPU python train_covidx.py --method "$m" --epochs 20 --gpu 0 2>&1 | grep -vE "Warning|warn|B/s"
done
echo ALL_DONE
