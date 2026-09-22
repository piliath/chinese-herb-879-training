#!/bin/bash
# 训练管家守护进程:
#   1) 每小时(整点)发一封进度邮件到 163
#   2) 训练全部结束后发汇总邮件
#   3) 最后自动关机停止计费
cd /root/autodl-tmp/herbal || exit 1
PY=/root/miniconda3/bin/python
CONFIGS="base no_weighted weak_mixup weak_aug two_stage logit_adjust base_448"
REPORTED_FILE="reported_configs.txt"
[ -f "$REPORTED_FILE" ] || : > "$REPORTED_FILE"
echo "[$(date '+%F %T')] daemon started (hourly email + per-config + final + shutdown)"

# 检测新完成的配置 -> 发"参数+结果"邮件
check_config_done() {
  for cfg in $CONFIGS; do
    if grep -qx "$cfg" "$REPORTED_FILE"; then continue; fi
    n=$(grep -c '^\[epoch' "logs/${cfg}.txt" 2>/dev/null); n=${n:-0}
    if [ "$n" -ge 25 ]; then
      echo "$cfg" >> "$REPORTED_FILE"
      echo "[$(date '+%F %T')] $cfg 完成, 发送参数+结果邮件"
      $PY report_config.py "$cfg" 2>/dev/null || echo "[email] $cfg 完成邮件发送失败"
    fi
  done
}

build_stats() {
  total=0; summary=""
  for cfg in $CONFIGS; do
    n=$(grep -c '^\[epoch' "logs/${cfg}.txt" 2>/dev/null); n=${n:-0}
    total=$((total + n))
    last=$(grep '^\[epoch' "logs/${cfg}.txt" 2>/dev/null | tail -1)
    [ -n "$last" ] && summary="$summary
${cfg}: $(echo "$last" | sed 's/\[epoch [0-9]*\/25\]/ep/')"
  done
}

send_hourly() {
  cur=$(ps aux | grep 'train_ablatio[n]' | grep -v grep | sed 's/.*--config //' | awk '{print $1}' | head -1)
  build_stats
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  $PY make_chart.py 2>/dev/null || true
  $PY notify_email.py "[进度] 中草药消融 ${total}/175 轮 (当前:${cur:-?})" \
    "时间: $(date '+%F %T')
正在运行: ${cur:-未检测到训练进程}
完成轮数: ${total}/175
GPU利用率: ${gpu:-N/A}
各配置最新一轮:
${summary}
(附件: 训练曲线图)" \
    /root/autodl-tmp/herbal/progress_chart.png 2>/dev/null || echo "[email] 进度邮件发送失败"
}

send_final() {
  build_stats
  if [ "$total" -ge 175 ]; then
    subj="[完成] 中草药消融训练全部完成 (7/7)"
  else
    subj="[WARN] 中草药消融训练中断 ($total/175 轮)"
  fi
  $PY make_chart.py 2>/dev/null || true
  $PY notify_email.py "$subj" "训练已结束。完成轮数: ${total}/175。各配置最新一轮:
${summary}
结果目录: /root/autodl-tmp/herbal/runs_ablation/
(附件: 最终训练曲线图)" \
    /root/autodl-tmp/herbal/progress_chart.png 2>/dev/null || echo "[email] 完成邮件发送失败"
}

last_hour=""
while true; do
  check_config_done
  if pgrep -f 'run_all_ablations[.]sh' > /dev/null 2>&1; then
    now=$(date '+%Y-%m-%d %H')
    if [ "$now" != "$last_hour" ]; then
      last_hour="$now"
      send_hourly
    fi
    sleep 300
  else
    # 未检测到训练: 等5分钟再确认, 避免瞬时误判
    sleep 300
    if pgrep -f 'run_all_ablations[.]sh' > /dev/null 2>&1; then
      continue
    fi
    break
  fi
done

check_config_done
send_final
echo "[$(date '+%F %T')] 汇总邮件已发, 准备自动关机"
curl -s -m 10 -X POST http://172.17.0.1:8080/api/instance/shutdown \
  && echo "shutdown OK" \
  || echo "shutdown FAILED - 请手动去 AutoDL 控制台停止实例"
