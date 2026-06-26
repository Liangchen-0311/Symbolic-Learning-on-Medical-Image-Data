#!/bin/bash
cd /home/ET/lctan/Symbolic-Learning/Symbolic-Learning-on-Medical-Image-Data/ThirdData_Experiment
for v in reg5 reg16 reg26 res384; do
  echo ========== $v ==========
  CUDA_VISIBLE_DEVICES=1 python -u experiments/run_thirddata_pipeline.py --config configs/thirddata_$v.yaml --start_step 4 --end_step 6 2>&1 | grep -vE "batch/s|body/s|it/s|added"
done
echo ALL_DONE
