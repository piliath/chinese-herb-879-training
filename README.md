# chinese-herb-879-training

**[English](README_EN.md) · 中文**

**Kaggle 879 类中草药细粒度图像分类 —— 训练代码 · 训练过程 · 训练结果**

> 面向超大规模（879 类）中草药图像识别的研究性训练项目。系统考察**细粒度分类的精度上限及其制约因素**，全过程遵循干净评估协议（无泄漏切分、无测试集污染）。
>
> 技术栈：PyTorch · timm · Ultralytics · AutoDL（RTX 3090/4090）云端训练

![提升路径](figs/improvement_ladder.png)

---

## 📊 关键结果

| 阶段 | 指标 | 数值 |
|:---|:---|:---|
| 数据清洗 | 脏切分基线 → 干净基线 | 74.87% → **76.34%**（+1.47） |
| 训练配方消融 | 7 组配方极差 | 仅 **0.85pt**（最优 = 去加权采样 **76.49%**） |
| 模型集成 | 7 模型 logits 平均 | **77.46%**（+0.97 over 最优单模型） |
| 容量验证 | ConvNeXt-V2-L 单模型（197M，15 轮） | **77.49%**（单模型即超 7 模型集成） |
| 标签合并评估 | 按"药材组"合并同物异名 | **78.31%**（30 类 15 组，58.35% → 84.68%） |
| Kaggle 公开榜参照 | V2-L 干净验证 | **0.77490**（对应 2025-05 高分区） |

> 完整原始数据见 [`results/`](results/)：8 组训练 `train_results.json`（含逐轮 history）、`ensemble_result.json`、`merge_eval_result.json`。

---

## ⬇️ 模型权重下载（ModelScope）

训练权重单文件均超出 GitHub 100MB 限制，**不入库**，统一发布在 ModelScope：

**发布页：https://www.modelscope.cn/models/Lihuahua2022/chinese-herb-879-training**

879 类研究模式权重：

| 权重文件 | 大小 | 说明 |
|:---|:---|:---|
| `ConvNeXt-B_879类_no_weighted_best.pth` | 354 MB | **消融最优配方**（去加权采样，76.49%） |
| `ConvNeXt-V2-L_879类_best.pth` | 791 MB | **容量验证**（197M，单模型 77.49%） |
| `training_artifacts/ConvNeXt-B_879类_脏切分.pth` | 354 MB | 脏切分基线（74.87%） |
| `training_artifacts/ViT-Base_879类_patch16_384.pth` | 347 MB | 对比模型（74.00%） |
| `training_artifacts/EfficientNetV2-M_879类.pth` | 214 MB | 对比模型（50.78%） |
| `training_artifacts/YOLOv8l-cls_879类_best.pt` | 299 MB | YOLO 最大尺度（未收敛下界 66.28%） |
| `training_artifacts/YOLOv8m-cls_879类_best.pt` | 34 MB | 66.57% |
| `training_artifacts/YOLOv8s-cls_879类_best.pt` | 12.5 MB | 64.68% |
| `training_artifacts/YOLOv8n-cls_879类_best.pt` | 5.2 MB | 61.17% |

数据集：

- 879 类（Kaggle chinese-medicine-image）：[ModelScope 数据集](https://www.modelscope.cn/datasets/Lihuahua2022/Chinese_medicine_image)
- 47 类（herbal-eoeld V5）：[ModelScope 数据集](https://www.modelscope.cn/datasets/Lihuahua2022/zhongcaoyao)

---

## 🧩 训练代码

完整训练/评估代码位于 [`kaggle/`](kaggle/)，可在 AutoDL 云端或本地 GPU 直接运行。

```
kaggle/
├── train_convnext.py          # ConvNeXt-B 训练（879 类基准 / V2-L 容量验证）
├── train_vit.py               # ViT-Base 训练
├── train_efficientnet.py      # EfficientNetV2-M 训练
├── train_yolov8{n,s,m,l}.py   # YOLOv8-cls 四个尺度（Ultralytics）
├── train_ablation.py          # Phase C 7 组配方消融主脚本
├── clean_dataset.py           # Phase A 数据清洗（MD5 冲突隔离 / 无泄漏切分）
├── check_duplicates.py        # 同图异标签冲突检查
├── split_dataset.py           # 切分工具
├── download_kaggle.py         # kagglehub 数据集下载
├── diagnose_results.py        # 逐类诊断（per-class / 混淆对 / 分桶）
├── analysis_plots.py          # 拒识 Risk-Coverage / ECE 校准 / t-SNE
├── dinov2_probe.py            # DINOv2 冻结特征线性探针对照
├── ensemble_eval.py           # 7 模型 logits 集成评估
├── merge_eval.py              # 同物异名"药材组"合并评估
├── summarize_ablations.py     # 消融结果汇总
├── alias_analysis.py          # 同物异名/产地变体标签分析（↔ aliases.json）
├── ssh_helper.py              # AutoDL 训练进程管理辅助
├── notify_email.py            # 训练完成邮件通知
├── run_all_ablations.sh       # 一键跑 7 组消融（nohup 后台）
├── run_v2l.sh / watch_v2l.sh  # V2-L 训练与监控
├── watch_and_shutdown.sh / status.sh  # 训练进度监控 / 自动关停
└── README_autodl.md           # AutoDL 云端运行说明
```

**技术路线：数据清洗 → 配方消融 → 模型集成 → 容量验证 → 标签合并评估**

1. **数据清洗（Phase A）**：16.7 万张图片 MD5 全量比对 → 隔离同图异标签冲突图 5,236 张（2,522 组）→ 消除跨切分泄漏 1,017 张 → 按"图组"重建无泄漏 90/10 切分（train 145,537 / valid 16,244 / 879 类）。
2. **配方消融（Phase C）**：固定 ConvNeXt-B，统一协议（25 轮、batch32×grad_accum2、warmup5、EMA、余弦退火），单变量对照 7 组配方。
3. **模型集成（Phase D）**：7 个消融 checkpoint logits 平均 → 77.46%，证明不同配方错误互补。
4. **容量验证**：ImageNet-22K 预训练 ConvNeXt-V2-L（197M）微调 15 轮 → 77.49%；DINOv2 冻结特征线性探针仅 71.24%，证明该任务高度依赖领域微调。
5. **标签合并评估**：区分"可合并的同物异名/产地变体"与"不可合并的相似异药"，按药材组评估 → 78.31%。

---

## 📈 训练过程

**7 模型收敛对比**（完整训练历史，同切分、同配方口径）：

![7 模型收敛对比](figs/model_7models_accuracy_comparison_879.png)

**879 类训练曲线**（干净切分上各模型验证 Top-1 随轮次变化）：

![训练曲线](figs/training_curves_879.png)

> 关键发现：YOLO 族参数量放大 13.8 倍（2.72M→37.48M）精度仅 +5.40（且 n→s→m 增量递减），而同一共同预算（≤25 轮）下 ConvNeXt-B 与 YOLO 族存在 **+10.06 点**的架构断层 —— 细粒度识别对骨干网络的**表征容量与预训练分布**敏感，显式分类头尺度并非瓶颈。

---

## 🏆 训练结果

### 3.1 模型选型（879 类，7 模型对照）

| 模型 | 参数量 | 训练轮次 | 最高 Top-1 | 最优轮次 | 共同预算(≤25轮) |
|:---|:---:|:---:|:---:|:---:|:---:|
| YOLOv8n-cls | 2.72M | 40 | 61.17% | 39 | 59.30% |
| YOLOv8s-cls | 6.36M | 40 | 64.68% | 35 | 63.57% |
| YOLOv8m-cls | 17.05M | 40 | 66.57% | 36 | 65.34% |
| YOLOv8l-cls \* | 37.48M | 25（中止） | 66.28% | 25 | 66.28% |
| EfficientNetV2-M | 52.98M | 40 | 50.78% | 21 | 50.78% |
| ViT-Base | 86.77M | 40 | 74.00% | 40 | 73.54% |
| **ConvNeXt-Base** | 88.47M | 25 | **76.34%** | 20 | **76.34%** |

\* YOLOv8l-cls 按 40 轮配置训练但在第 25 轮中止，末轮学习率仍为 3.5e-4，精度持续上升，66.28% 为其**未收敛下界**。

### 3.2 训练配方消融（固定 ConvNeXt-B，干净切分，25 轮）

| 配置 | 关键改动 | 最高 Top-1 | 最优轮次 |
|:---|:---|:---:|:---:|
| `base` | 基线（加权采样 + RandAugment m7 + MixUp 1.0，384px） | 76.34% | 20 |
| `no_weighted` | 取消加权采样 → 实例采样 | **76.49%** ✔ | 22 |
| `base_448` | 输入尺寸 448px（batch16×accum4） | 76.42% | 23 |
| `weak_aug` | RandAugment m7→m4，RandomErase 0.2→0.05 | 76.27% | 23 |
| `weak_mixup` | MixUp/CutMix prob 1.0→0.1 | 76.25% | 21 |
| `logit_adjust` | 实例采样 + 关 MixUp + Logit Adjustment Loss | 76.08% | 17 |
| `two_stage` | 前 10 轮冻结 backbone 只训分类头 | 75.64% | 25 |

![配方消融](figs/ablation_879.png)

> 7 组配方极差仅 0.85pt：在干净切分 + 精心配平的默认配方下，配方层面的调整对最终精度的影响有限，**瓶颈不在训练配方**。

### 3.3 集成 · 容量验证 · 标签合并

| 方案 | Top-1 | 说明 |
|:---|:---:|:---|
| 7 模型 logits 集成 | 77.46% | +0.97 over 最优单模型，错误互补 |
| **ConvNeXt-V2-L（197M，15 轮）** | **77.49%** | 单模型即超 7 模型集成 |
| DINOv2 冻结特征线性探针 | 71.24% | 冻结特征不足以支撑 879 类 |
| 按"药材组"合并评估 | **78.31%** | 30 类 15 组，涉及类 58.35% → 84.68% |

同物异名/产地变体合并（部分示例，完整见 `results/merge_eval_result.json`）：

| 药材组 | 样本数 | 原始正确 → 合并正确 |
|:---|:---:|:---:|
| 淫羊藿 / 巫山淫羊藿 | 33 | 15 → 30 |
| 丹参 / 甘肃丹参 | 32 | 14 → 28 |
| 续断 / 川续断 | 30 | 14 → 26 |
| 丝瓜络 / 粤丝瓜络 | 34 | 23 → 32 |
| 七叶一枝花 / 重楼 | 39 | 22 → 35 |
| 葛根 / 粉葛 | 39 | 24 → 34 |

---

## 🖥 训练环境

| 项 | 配置 |
|:---|:---|
| GPU | AutoDL RTX 3090 / 4090（24GB）；本地 RTX 5070 推理验证 |
| 框架 | PyTorch 2.9 + timm 1.0 + Ultralytics 8.4 |
| 优化器 | AdamW + warmup(5) + 余弦退火 + EMA |
| 正则化 | DropPath 0.2 / weight_decay 0.05 / 标签平滑 0.1 / RandAugment |
| 有效批大小 | 64（batch32 × grad_accum2；448px 时 batch16 × accum4） |

---

## 🚀 快速开始

```bash
# 1. 环境
conda create -n herb python=3.10
pip install -r requirements.txt        # 先装 CUDA 匹配的 torch/torchvision

# 2. 数据（约 16.7 万张）
python kaggle/download_kaggle.py       # kagglehub 下载原始数据集
python kaggle/clean_dataset.py         # 清洗 → dataset_kaggle_clean（9.1GB）

# 3. 训练（GPU ≥ 24GB 显存）
python kaggle/train_convnext.py --epochs 25                      # 879 类基准 ConvNeXt-B
bash kaggle/run_all_ablations.sh                                 # 7 组配方消融
python kaggle/train_convnext.py --model convnextv2_large --epochs 15 --img_size 384   # V2-L

# 4. 评估与分析
python kaggle/diagnose_results.py --dir results_ablation/runs_ablation/no_weighted
python kaggle/ensemble_eval.py
python kaggle/merge_eval.py
python kaggle/analysis_plots.py
```

云端批量训练（AutoDL）与训练监控脚本的使用见 [`kaggle/README_autodl.md`](kaggle/README_autodl.md)。

---

## 📄 复现口径说明

- **无泄漏切分**：MD5 按"图组"隔离同图异标签，任何两张相同图片不会同时出现在 train 与 valid。
- **同配方可比**：7 模型对照均采用默认配方单次训练；共同预算口径（≤25 轮）用于跨模型比较，避免"训练到 40 轮"与"只训练到 25 轮"混比。
- **不去重评估**：验证指标均直接在原始类标签上计算，不做标签去重。
- 训练曲线、逐轮历史等原始证据见 [`results/runs_ablation/*/train_results.json`](results/runs_ablation)。

## 📜 数据许可

Kaggle chinese-medicine-image 竞赛数据集，使用须遵守 Kaggle 竞赛条款；47 类 herbal-eoeld V5 数据集来自 Roboflow 公开数据集。权重发布在 ModelScope，仅供学习研究使用。