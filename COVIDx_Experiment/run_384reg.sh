#!/bin/bash
cd /home/ET/lctan/Symbolic-Learning/Symbolic-Learning-on-Medical-Image-Data/ThirdData_Experiment
for r in 5 26; do
  echo ===== 384_reg$r =====
  CUDA_VISIBLE_DEVICES=1 python -u experiments/run_thirddata_pipeline.py --config configs/thirddata_384_reg$r.yaml --start_step 4 --end_step 6 2>&1 | grep -vE "batch/s|body/s|it/s|added"
done
echo ALL_DONE
