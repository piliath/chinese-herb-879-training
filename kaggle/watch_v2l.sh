#!/bin/bash
# V2-L 趋势测试守护: 每小时进度邮件 + 完成邮件 + 自动关机
cd /root/autodl-tmp/herbal || exit 1
PY=/root/miniconda3/bin/python
CHART=/root/autodl-tmp/herbal/progress_chart.png
echo "[$(date '+%F %T')] v2l watcher started (hourly email + final + shutdown)"

send_progress() {
  n=$(grep -c '^\[epoch' logs/v2l.txt 2>/dev/null); n=${n:-0}
  last=$(grep '^\[epoch' logs/v2l.txt 2>/dev/null | tail -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  $PY make_chart.py 2>/dev/null || true
  $PY notify_email.py "[V2-L趋势测试] ${n}/15 轮" "时间: $(date '+%F %T')
完成轮数: ${n}/15
GPU利用率: ${gpu:-N/A}
最新: ${last:-训练中}
(附件: 训练曲线)" "$CHART" 2>/dev/null || echo "[email] 进度邮件失败"
}

last_hour=""
while true; do
  if pgrep -f 'train_ablation.py --config v2[l]' > /dev/null 2>&1; then
    now=$(date '+%Y-%m-%d %H')
    if [ "$now" != "$last_hour" ]; then
      last_hour="$now"
      send_progress
    fi
    sleep 300
  else
    sleep 300
    if pgrep -f 'train_ablation.py --config v2[l]' > /dev/null 2>&1; then
      continue
    fi
    break
  fi
done

n=$(grep -c '^\[epoch' logs/v2l.txt 2>/dev/null); n=${n:-0}
last=$(grep '^\[epoch' logs/v2l.txt 2>/dev/null | tail -1)
$PY make_chart.py 2>/dev/null || true
$PY notify_email.py "[完成] V2-L 趋势测试结束 (${n}/15 轮)" "最后: ${last}
结果目录: /root/autodl-tmp/herbal/runs_ablation/v2l/
(附件: 训练曲线)" "$CHART" 2>/dev/null || echo "[email] 完成邮件失败"

echo "[$(date '+%F %T')] 尝试自动关机"
curl -s -m 10 -X POST http://172.17.0.1:8080/api/instance/shutdown \
  && echo "shutdown OK" \
  || echo "shutdown FAILED - 请手动停止实例"
