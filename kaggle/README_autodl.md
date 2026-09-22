# Phase C 消融实验 — AutoDL 运行说明

在 AutoDL(3090/4090, 24GB 显存)上跑 7 组 ConvNeXt-B 消融。清洗后的数据集用 ModelScope 传到服务器,脚本直接跑。

## 需要上传的文件

1. **`dataset_kaggle_clean.zip`** (8.8GB, 用 ModelScope 传, 比 AutoDL 网页上传快)
   - 内含 `dataset_kaggle_clean/train`(145,537 张)、`valid`(16,244 张)、`conflict_holdout`(5,236 张同图异标签隔离图)
2. **脚本**:
   - `train_ablation.py` (训练主脚本)
   - `run_all_ablations.sh` (一键跑 7 组)
   - `summarize_ablations.py` (结果汇总)
   - `clean_dataset.py` + `aliases.json` (可选, 备查)

## 服务器上操作

```bash
# 1. 解压数据集 (放任意工作目录, 解压后应有 dataset_kaggle_clean/ 目录)
unzip dataset_kaggle_clean.zip

# 2. 确认环境 (需要 pytorch+cuda、timm、torchvision、numpy、tqdm)
conda activate <你的环境>
python -c "import torch, timm; print(torch.__version__, torch.cuda.get_device_name(0))"

# 3. 跑全部消融 (建议 nohup 后台, 7 组约 1.5~2 天 @4090 / 2~3 天 @3090)
nohup bash run_all_ablations.sh > run_all_nohup.log 2>&1 &

# 查看进度
tail -f run_all_nohup.log
# 或
tail -f logs/base.txt

# 4. 全部跑完后汇总
python summarize_ablations.py --json summary.json
```

## 7 组消融清单

| config | 改动 |
|---|---|
| `base` | 当前配方(加权采样 + 强增强 + mixup 1.0), 作为干净切分上的新基线 |
| `no_weighted` | 取消加权采样 → 实例采样 |
| `weak_mixup` | MixUp/CutMix prob 1.0 → 0.1 |
| `weak_aug` | RandAugment m7→m4, RandomErase 0.2→0.05 |
| `two_stage` | 前 10 轮冻结 backbone 只训分类头, 之后全量微调 |
| `logit_adjust` | 实例采样 + 关闭 mixup + Logit Adjustment Loss |
| `base_448` | 输入尺寸 448 (batch16/accum4) |

## 输出

每个配置写入 `runs_ablation/<config>/`:
- `train_results.json` (含 best_val_top1, best_epoch, 每轮 history)
- `config.json`
- `best_convnext_base.pth` (最优权重)

> 默认 40 epoch、batch32×accum2(448 用 batch16×accum4)。若想缩短时间可加 `--epochs 30`。
> 预训练权重首次运行时自动从网络下载(约 340MB), AutoDL 有网, 无需处理。

## 注意事项

- 脚本在 **Linux 上无中文字库问题**(绘图已移除, 全部文本/JSON 输出)。
- 每个配置是单变量隔离(除 `logit_adjust` 按方案组合了"实例采样+关mixup+LA")。
- 结果拿回本地后, 用 `diagnose_results.py` 可以对新基线做同样的逐类诊断。
