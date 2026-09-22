#!/bin/bash
# 一键查看 Phase C 消融训练进度 (AutoDL 上运行)
cd /root/autodl-tmp/herbal || exit 1
echo "=== 当前运行的配置 ==="
ps aux | grep 'train_ablatio[n]' | grep -v grep | awk '{print $2, $NF, $NF-1}' | head -2
ps aux | grep 'train_ablatio[n]' | grep -v grep | sed 's/.*train_ablation.py/  python train_ablation.py/' | head -2
echo
echo "=== 各配置完成情况 (epoch 数) ==="
for cfg in base no_weighted weak_mixup weak_aug two_stage logit_adjust base_448; do
  f="logs/${cfg}.txt"
  if [ -f "$f" ]; then
    done_ep=$(grep -c '^\[epoch' "$f" 2>/dev/null)
    last=$(grep '^\[epoch' "$f" 2>/dev/null | tail -1)
    if [ -n "$last" ]; then
      echo "  $cfg: $done_ep/25 轮完成 | $last"
    else
      echo "  $cfg: 正在跑第 $(($(grep -c '^\[epoch' /dev/null)+1)) 轮"
    fi
  else
    echo "  $cfg: 未开始"
  fi
done
echo
echo "=== 运行日志尾部 ==="
tail -4 logs/run_all.log 2>/dev/null
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
