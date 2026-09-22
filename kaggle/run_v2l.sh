#!/bin/bash
# ConvNeXt-V2-L 趋势测试 (15轮, 单配置)
source /etc/network_turbo 2>/dev/null || true  # AutoDL 学术加速, 下载 HF 权重
cd /root/autodl-tmp/herbal || exit 1
mkdir -p logs
/root/miniconda3/bin/python -u train_ablation.py --config v2l 2>&1 | tee logs/v2l.txt | grep -E "^\[epoch|=====|two_stage"
