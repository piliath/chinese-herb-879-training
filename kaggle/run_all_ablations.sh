#!/bin/bash
# Phase C 消融实验: 在 AutoDL 上顺序跑 7 个配置
# 用法:
#   PYTHON=/root/miniconda3/bin/python bash run_all_ablations.sh   (指定解释器)
#   nohup env PYTHON=/root/miniconda3/bin/python bash run_all_ablations.sh > run_all_nohup.log 2>&1 &
PYTHON="${PYTHON:-python}"
set -e
mkdir -p logs
for cfg in base no_weighted weak_mixup weak_aug two_stage logit_adjust base_448; do
  echo "===== $cfg start $(date '+%F %T') =====" | tee -a logs/run_all.log
  $PYTHON -u train_ablation.py --config "$cfg" --epochs 25 2>&1 | tee "logs/${cfg}.txt" | grep -E "^\[epoch|two_stage|===== "
  echo "===== $cfg done $(date '+%F %T') =====" | tee -a logs/run_all.log
done
echo "ALL_DONE $(date '+%F %T')" | tee -a logs/run_all.log
