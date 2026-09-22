# chinese-herb-879-training

**[English](README_EN.md) · 中文**

**879 类中草药细粒度图像识别：精度上限在哪里，又卡在什么问题上？**

> 一段完整的训练研究记录：数据清洗 → 配方消融 → 模型集成 → 容量验证 → 标签合并。
> 16.7 万张图片、879 个药材类别，在云端 GPU（AutoDL RTX 3090/4090）上训练，把干净评估协议下的最高精度从 74.87% 一路推到 **78.31%**。

---

## 一、这是一个什么问题？

中药材识别看起来像标准的图像分类——"给一张图，告诉我是哪种药"。但比起"猫 vs 狗"，它有三个天然的难点：

- **类别多且长得像**：879 个类别，很多药材外形高度相似（山药和薯蓣、天麻和仙人掌切片），普通人根本分不清；
- **同物异名**：同一味药在数据集里被拆成多个类——丹参又叫甘肃丹参，石斛有铁皮石斛等变体。模型要把"同一种药"学会分成"两个类"，本身就是额外负担；
- **相似异药**：不同药材又可能长得很像——玄参和苦玄参，其实是两种不同的药，不允许混。

更要命的是**原始数据本身不干净**：数据集里存在"同一张图被贴上不同类名"的标注冲突。如果直接拿去训练，模型会"作弊"——它可能只是记住了某张图，而不是学会了分辨药材，实验结论也因此失真。

所以这个项目表面上是"训练一个 879 类的分类模型"，实际上在回答两个更深刻的问题：

> **① 把数据和训练方法都规范化之后，模型精度能到达哪里？**
> **② 卡住精度的瓶颈，到底是数据、是训练配方，还是模型架构本身？**

## 二、项目背景

本仓库是我们课程项目《基于深度学习的中草药检测系统》的**研究扩展**。主系统面向 47 类常用药材，做成了一个可用的桌面应用（识别、模拟考试、错题本等）；而这里（879 类研究模式）负责把精度推到"研究上限"：

用开源竞赛数据集（[Kaggle chinese-medicine-image](https://www.kaggle.com/competitions/chinese-medicine-image)，16.7 万张 / 879 类），围绕四到五条实验主线进行了系统探索，并全程遵循**干净评估协议**——无泄漏切分、不污染测试集、不在验证集上做任何"手动调优"。

## 三、核心成果一览

![提升路径](figs/improvement_ladder.png)

这张图是整个研究的缩影：从脏切分基线 74.87% 出发，每做一步规范化或改进，精度上一个台阶，最终在按药材组合并评估的口径下达到 78.31%。

| 阶段 | 指标 | 数值 | 这一行的意思 |
|:---|:---|:---|:---|
| 数据清洗 | 脏切分 → 干净基线 | 74.87% → **76.34%**（+1.47） | 没动任何模型，光修好数据泄漏/同图异标签，就涨了 1.47 个点 |
| 训练配方消融 | 7 组配方极差 | 仅 **0.85pt** | 配方怎么调，最多差 0.85 个点——瓶颈不在训练配方 |
| 模型集成 | 7 模型 logits 平均 | **77.46%**（+0.97 over 最优单模型） | 让 7 个"各有盲点"的模型投票，比最好单点还高 0.97 |
| 容量验证 | ConvNeXt-V2-L 单模型（197M，15 轮） | **77.49%** | 一个参数翻到 197M 的大模型，单干就超过 7 模型集成 |
| 标签合并评估 | 按"药材组"合并同物异名 | **78.31%** | 数据标签本身就在压精度，把标注口径对齐药材语义就找回来 0.83 |
| Kaggle 公开榜参照 | V2-L 干净验证 | **0.77490** | 对标公开榜上 2025-05 的高分区 |

> 完整原始数据在 [`results/`](results/)：8 组训练的逐轮 `train_results.json`（含每轮 history）、`ensemble_result.json`、`merge_eval_result.json`。

---

## 四、五步研究路线

一句话技术路线：**数据清洗 → 配方消融 → 模型集成 → 容量验证 → 标签合并评估**。每一步都先问一个明确的问题，再动手。

### 第 1 步 · 数据清洗：先让数据说真话（Phase A）

**回答什么问题**：数据里的"作弊项"，会不会让结论失真？

**做了什么**：对 16.7 万张图片做 MD5 全量比对——同一张图贴了不同类名的，隔离；同一张图同时出现在训练集和验证集的，剔除。最后按"图组"（同一图片的所有副本为一个整体）重建 90/10 无泄漏切分。

**结果**：隔离同图异标签冲突 **5,236 张**（2,522 组）；消除跨切分泄漏 **1,017 张**；得到训练集 **145,537 张**、验证集 **16,244 张**、**879 类**。

**意味着什么**：干净切分后，同一个 ConvNeXt-B 基线的精度从 74.87% 涨到 **76.34%**（+1.47）。这一步的收益属于"白捡"——**模型一行没改，数据质量就是第一道坎**。

### 第 2 步 · 配方消融：训练细节能改变多少？（Phase C）

**回答什么问题**：把默认训练配方调到最优，能挤出多少精度？

**做了什么**：固定同一个模型（ConvNeXt-B）和统一协议（25 轮、batch32×grad_accum2、warmup5、EMA、余弦退火），每次只改**一个**变量，做 7 组单变量对照：去加权采样 / 448 分辨率 / 减增强 / mixup 0.1 / Logit Adjustment / 两阶段微调。

**结果**：最优配方是**去加权采样 76.49%**，最差 75.64%，7 组极差只有 **0.85pt**。

**意味着什么**：在数据干净、默认配方已经调平的前提下，配方层面的调整最多也就 0.85 个点——**瓶颈不在训练配方**，继续在配方上打转收益很小。

![配方消融](figs/ablation_879.png)

### 第 3 步 · 模型集成：让模型互相纠错（Phase D）

**回答什么问题**：不同配方训练出的模型，它们的错误能不能互补？

**做了什么**：把第 2 步的 7 个消融检查点的 logits 做平均，相当于 7 个模型"投票"。

**结果**：**77.46%**，比最优单模型（76.49%）高 0.97。

**意味着什么**：不同配方确实让模型学到了不同的盲点，集成能把它们缝起来。同时也暗示：**单模型精度才是真正的天花板**，集成是在有限的地基上搬砖。

### 第 4 步 · 容量验证：是不是模型不够大？

**回答什么问题**：把模型规模推上去，精度能突破吗？

**做了什么**：用 ImageNet-22K 预训练的 ConvNeXt-V2-L（197M 参数）微调 15 轮；对照组用 DINOv2 冻结特征直接接线性分类头。

**结果**：V2-L 单模型 **77.49%**——一个模型就打过了 7 模型集成（77.46%）；而 DINOv2 冻结特征的线性探针只有 **71.24%**。

**意味着什么**：① 模型容量管用，更大的模型能突破集成上限；② "冻结通用特征 + 现学个分类头"在这个任务上不够——879 类细粒度识别**高度依赖对领域数据的微调**。

### 第 5 步 · 标签合并：同物异名损失了多少？

**回答什么问题**："同一个药材被拆成几个类"这件事本身，压了模型多少精度？

**做了什么**：把可合并的同物异名/产地变体（丹参 ↔ 甘肃丹参、石斛 ↔ 铁皮石斛等，共 **15 组 30 类**）按药材组合并后再评估；同时把不可合并的相似异药（玄参 ↔ 苦玄参）明确区分开。

**结果**：合并后精度 **78.31%**；涉及合并的 30 个类，精度从 **58.35% 涨到 84.68%**。

**意味着什么**：数据集的标注口径本身就是一道隐形的"精度天花板"。把标签定义对齐到"药材语义"上，能带来肉眼可见的找回——这也反过来解释了第 1 步的数据清洗为什么值得做。

### 训练过程一览

下面两张图记录了模型收敛的全过程，是上面结论的直接证据：

![7 模型收敛对比](figs/model_7models_accuracy_comparison_879.png)

![训练曲线](figs/training_curves_879.png)

> **看图须知**：图里有一条竖直虚线（第 25 轮）。所有跨模型结论都按"第 25 轮及以前"的共同预算口径比较，避免"训练到 40 轮的模型"欺负"只训到 25 轮的模型"。
> **一个值得玩味的细节**：YOLO 家族参数量放大 13.8 倍（2.72M→37.48M），40 轮口径下精度只涨了 +5.40，且 n→s→m 增量递减；而同预算下 ConvNeXt-B 对整个 YOLO 家族是 **+10.06 点**的架构断层。细粒度识别吃的是骨干网络的**表征容量与预训练分布**，把分类头做再大也没用。

---

## 五、模型权重下载（ModelScope）

训练好的权重单文件都超过 GitHub 100MB 的限制，**不入库**，统一发布在 ModelScope：

**发布页：https://www.modelscope.cn/models/Lihuahua2022/chinese-herb-879-training**

879 类研究模式权重：

| 权重文件 | 大小 | 说明 |
|:---|:---|:---|
| `ConvNeXt-B_879类_no_weighted_best.pth` | 354 MB | **消融最优配方**（去加权采样，76.49%）|
| `ConvNeXt-V2-L_879类_best.pth` | 791 MB | **容量验证**（197M，单模型 77.49%）|
| `training_artifacts/ConvNeXt-B_879类_脏切分.pth` | 354 MB | 脏切分基线（74.87%）|
| `training_artifacts/ViT-Base_879类_patch16_384.pth` | 347 MB | 对比模型（74.00%）|
| `training_artifacts/EfficientNetV2-M_879类.pth` | 214 MB | 对比模型（50.78%）|
| `training_artifacts/YOLOv8l-cls_879类_best.pt` | 299 MB | YOLO 最大尺度（未收敛下界 66.28%）|
| `training_artifacts/YOLOv8m-cls_879类_best.pt` | 34 MB | 66.57% |
| `training_artifacts/YOLOv8s-cls_879类_best.pt` | 12.5 MB | 64.68% |
| `training_artifacts/YOLOv8n-cls_879类_best.pt` | 5.2 MB | 61.17% |

数据集：

- 879 类：[Kaggle 竞赛原始数据集](https://www.kaggle.com/competitions/chinese-medicine-image)（16.7 万张 / 879 类）· [ModelScope 镜像](https://www.modelscope.cn/datasets/Lihuahua2022/Chinese_medicine_image)
- 47 类（herbal-eoeld V5）：[ModelScope 数据集](https://www.modelscope.cn/datasets/Lihuahua2022/zhongcaoyao)

---

## 六、训练代码与快速开始

### 代码结构

`kaggle/` 里是全部训练与评估代码，可以在 AutoDL 云端或本地 GPU 直接跑。每个脚本后面标注了"它是干嘛的"：

```
kaggle/
├── train_convnext.py          # 训练 ConvNeXt-B（879 类基准，也用它训 V2-L 做容量验证）
├── train_vit.py               # 训练 ViT-Base
├── train_efficientnet.py      # 训练 EfficientNetV2-M
├── train_yolov8{n,s,m,l}.py   # 训练 YOLOv8-cls 四个尺度（Ultralytics）
├── train_ablation.py          # 第 2 步的 7 组配方消融主脚本
├── clean_dataset.py           # 第 1 步的数据清洗（MD5 冲突隔离 / 无泄漏切分）
├── check_duplicates.py        # 同图异标签冲突检查
├── split_dataset.py           # 切分工具
├── download_kaggle.py         # 用 kagglehub 下载原始数据集
├── diagnose_results.py        # 逐类诊断（per-class / 混淆对 / 分桶），看模型哪里分不清
├── analysis_plots.py          # 拒识 Risk-Coverage / ECE 校准 / t-SNE
├── dinov2_probe.py            # 第 4 步的 DINOv2 冻结特征线性探针对照
├── ensemble_eval.py           # 第 3 步的 7 模型 logits 集成评估
├── merge_eval.py              # 第 5 步的同物异名"药材组"合并评估
├── summarize_ablations.py     # 消融结果汇总
├── alias_analysis.py          # 同物异名/产地变体标签分析（配合 aliases.json）
├── ssh_helper.py              # AutoDL 训练进程管理辅助
├── notify_email.py            # 训练完成邮件通知
├── run_all_ablations.sh       # 一键后台跑 7 组消融（nohup）
├── run_v2l.sh / watch_v2l.sh  # V2-L 训练与监控
├── watch_and_shutdown.sh / status.sh  # 训练进度监控 / 结束后自动关停云主机
└── README_autodl.md           # AutoDL 云端运行说明
```

### 快速开始

```bash
# 1. 环境
conda create -n herb python=3.10
pip install -r requirements.txt        # 先装与本地 CUDA 匹配的 torch/torchvision

# 2. 数据（约 16.7 万张）
python kaggle/download_kaggle.py       # 下载原始数据集
python kaggle/clean_dataset.py         # 清洗 → dataset_kaggle_clean（9.1GB）

# 3. 训练（建议 GPU ≥ 24GB 显存，如 RTX 3090/4090）
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

## 七、技术细节汇总

需要复现或深挖细节时，这里把口径和数据都摊开。

### 7.1 模型选型对照（879 类，7 模型）

| 模型 | 参数量 | 训练轮次 | 最高 Top-1 | 最优轮次 | 共同预算(≤25轮) |
|:---|:---:|:---:|:---:|:---:|:---:|
| YOLOv8n-cls | 2.72M | 40 | 61.17% | 39 | 59.30% |
| YOLOv8s-cls | 6.36M | 40 | 64.68% | 35 | 63.57% |
| YOLOv8m-cls | 17.05M | 40 | 66.57% | 36 | 65.34% |
| YOLOv8l-cls \* | 37.48M | 25（中止）| 66.28% | 25 | 66.28% |
| EfficientNetV2-M | 52.98M | 40 | 50.78% | 21 | 50.78% |
| ViT-Base | 86.77M | 40 | 74.00% | 40 | 73.54% |
| **ConvNeXt-Base** | 88.47M | 25 | **76.34%** | 20 | **76.34%** |

\* YOLOv8l-cls 按 40 轮配置训练但在第 25 轮中止，末轮学习率仍为 3.5e-4、精度仍在上升，66.28% 是它的**未收敛下界**。

> **解读**：只看 40 轮口径，YOLO 放大 13.8 倍参数（2.72M→37.48M）只买到 +5.40，且增量递减；而同样预算下 ConvNeXt-B 比整个 YOLO 家族高出一大截（+10.06）。**分类头做大不如骨干换强**。

### 7.2 训练配方消融明细（固定 ConvNeXt-B，干净切分，25 轮）

| 配置 | 关键改动 | 最高 Top-1 | 最优轮次 |
|:---|:---|:---:|:---:|
| `base` | 基线（加权采样 + RandAugment m7 + MixUp 1.0，384px）| 76.34% | 20 |
| `no_weighted` | 取消加权采样 → 实例采样 | **76.49%** ✔ | 22 |
| `base_448` | 输入尺寸 448px（batch16×accum4）| 76.42% | 23 |
| `weak_aug` | RandAugment m7→m4，RandomErase 0.2→0.05 | 76.27% | 23 |
| `weak_mixup` | MixUp/CutMix prob 1.0→0.1 | 76.25% | 21 |
| `logit_adjust` | 实例采样 + 关 MixUp + Logit Adjustment Loss | 76.08% | 17 |
| `two_stage` | 前 10 轮冻结 backbone 只训分类头 | 75.64% | 25 |

> **解读**：最优（`no_weighted`，去加权采样）比基线只高 0.15，极差 0.85。可以理解为：**一旦数据干净、默认配方够好，配方细节就不是决定精度的主战场**。

### 7.3 集成 · 容量验证 · 标签合并（汇总）

| 方案 | Top-1 | 说明 |
|:---|:---:|:---|
| 7 模型 logits 集成 | 77.46% | +0.97 over 最优单模型，错误互补 |
| **ConvNeXt-V2-L（197M，15 轮）** | **77.49%** | 单模型即超 7 模型集成 |
| DINOv2 冻结特征线性探针 | 71.24% | 冻结特征不足以支撑 879 类 |
| 按"药材组"合并评估 | **78.31%** | 30 类 15 组，涉及类 58.35% → 84.68% |

同物异名/产地变体合并（部分示例；完整见 `results/merge_eval_result.json`）：

| 药材组 | 样本数 | 原始正确 → 合并正确 |
|:---|:---:|:---:|
| 淫羊藿 / 巫山淫羊藿 | 33 | 15 → 30 |
| 丹参 / 甘肃丹参 | 32 | 14 → 28 |
| 续断 / 川续断 | 30 | 14 → 26 |
| 丝瓜络 / 粤丝瓜络 | 34 | 23 → 32 |
| 七叶一枝花 / 重楼 | 39 | 22 → 35 |
| 葛根 / 粉葛 | 39 | 24 → 34 |

### 7.4 训练环境

| 项 | 配置 |
|:---|:---|
| GPU | AutoDL RTX 3090 / 4090（24GB）；本地 RTX 5070 推理验证 |
| 框架 | PyTorch 2.9 + timm 1.0 + Ultralytics 8.4 |
| 优化器 | AdamW + warmup(5) + 余弦退火 + EMA |
| 正则化 | DropPath 0.2 / weight_decay 0.05 / 标签平滑 0.1 / RandAugment |
| 有效批大小 | 64（batch32 × grad_accum2；448px 时 batch16 × accum4）|

### 7.5 复现口径

- **无泄漏切分**：按 MD5"图组"隔离同图异标签，任何一张图不会同时出现在 train 与 valid。
- **同配方可比**：7 模型对照均采用默认配方单次训练；跨模型结论一律用共同预算（≤25 轮）口径。
- **不去重评估**：验证指标都在原始类标签上计算，不做标签去重。
- 训练曲线、逐轮 history 等原始证据见 [`results/runs_ablation/*/train_results.json`](results/runs_ablation)。

---

## 八、数据许可

[Kaggle chinese-medicine-image 竞赛数据集](https://www.kaggle.com/competitions/chinese-medicine-image)，使用须遵守 Kaggle 竞赛条款；47 类 herbal-eoeld V5 数据集来自 Roboflow 公开数据集。权重发布在 ModelScope，仅供学习研究使用。