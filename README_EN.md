# chinese-herb-879-training

**English · [中文](README.md)**

**Kaggle 879-class fine-grained Chinese herb image classification — training code · training process · training results**

> A research-oriented training project for very-large-scale (879-class) Chinese herbal medicine image recognition. It systematically investigates the **accuracy ceiling of fine-grained classification and its limiting factors**, under a strict clean-evaluation protocol (leakage-free split, no test-set contamination).
>
> Tech stack: PyTorch · timm · Ultralytics · cloud training on AutoDL (RTX 3090/4090)

![Improvement ladder](figs/improvement_ladder.png)

---

## 📊 Key results

| Stage | Metric | Value |
|:---|:---|:---|
| Data cleaning | Dirty-split baseline → clean baseline | 74.87% → **76.34%** (+1.47) |
| Training-recipe ablation | Spread across 7 recipes | only **0.85pt** (best = no-weighted sampling **76.49%**) |
| Model ensembling | Logits average of 7 models | **77.46%** (+0.97 over best single model) |
| Capacity check | ConvNeXt-V2-L single model (197M, 15 epochs) | **77.49%** (single model already beats the 7-model ensemble) |
| Label-merging eval | Merge synonyms by "herb group" | **78.31%** (15 groups / 30 classes, 58.35% → 84.68%) |
| Kaggle public LB ref | V2-L clean-split validation | **0.77490** (top segment, 2025-05) |

> Raw evidence lives in [`results/`](results/): 8 runs of `train_results.json` (with per-epoch history), `ensemble_result.json`, `merge_eval_result.json`.

---

## ⬇️ Model weights (ModelScope)

Every checkpoint exceeds GitHub's 100MB per-file limit, so **weights are not stored in this repo**. They are published on ModelScope:

**Page: https://www.modelscope.cn/models/Lihuahua2022/chinese-herb-879-training**

879-class weights:

| File | Size | Description |
|:---|:---|:---|
| `ConvNeXt-B_879类_no_weighted_best.pth` | 354 MB | **Best ablation recipe** (no-weighted sampling, 76.49%) |
| `ConvNeXt-V2-L_879类_best.pth` | 791 MB | **Capacity check** (197M, single model 77.49%) |
| `training_artifacts/ConvNeXt-B_879类_脏切分.pth` | 354 MB | Dirty-split baseline (74.87%) |
| `training_artifacts/ViT-Base_879类_patch16_384.pth` | 347 MB | Comparison model (74.00%) |
| `training_artifacts/EfficientNetV2-M_879类.pth` | 214 MB | Comparison model (50.78%) |
| `training_artifacts/YOLOv8l-cls_879类_best.pt` | 299 MB | Largest YOLO scale (unconverged lower bound 66.28%) |
| `training_artifacts/YOLOv8m-cls_879类_best.pt` | 34 MB | 66.57% |
| `training_artifacts/YOLOv8s-cls_879类_best.pt` | 12.5 MB | 64.68% |
| `training_artifacts/YOLOv8n-cls_879类_best.pt` | 5.2 MB | 61.17% |

Datasets:

- 879-class (Kaggle chinese-medicine-image): [ModelScope dataset](https://www.modelscope.cn/datasets/Lihuahua2022/Chinese_medicine_image)
- 47-class (herbal-eoeld V5): [ModelScope dataset](https://www.modelscope.cn/datasets/Lihuahua2022/zhongcaoyao)

---

## 🧩 Training code

All training & evaluation scripts live in [`kaggle/`](kaggle/), ready to run on AutoDL or a local GPU.

```
kaggle/
├── train_convnext.py          # ConvNeXt-B training (879-class baseline / V2-L capacity check)
├── train_vit.py               # ViT-Base training
├── train_efficientnet.py      # EfficientNetV2-M training
├── train_yolov8{n,s,m,l}.py   # YOLOv8-cls four scales (Ultralytics)
├── train_ablation.py          # Phase C main script: 7 ablation recipes
├── clean_dataset.py           # Phase A data cleaning (MD5 conflict isolation / leak-free split)
├── check_duplicates.py        # same-image-different-label conflict detection
├── split_dataset.py           # split utilities
├── download_kaggle.py         # kagglehub dataset download
├── diagnose_results.py        # per-class diagnostics (per-class / confusion pairs / buckets)
├── analysis_plots.py          # reject Risk-Coverage / ECE calibration / t-SNE
├── dinov2_probe.py            # DINOv2 frozen-feature linear probing baseline
├── ensemble_eval.py           # 7-model logits ensemble evaluation
├── merge_eval.py              # synonym "herb group" merged evaluation
├── summarize_ablations.py     # ablation summary
├── alias_analysis.py          # synonym/geo-variant label analysis (↔ aliases.json)
├── ssh_helper.py              # AutoDL training process management helper
├── notify_email.py            # training-complete email notification
├── run_all_ablations.sh       # run all 7 ablations (nohup background)
├── run_v2l.sh / watch_v2l.sh  # V2-L training & monitoring
├── watch_and_shutdown.sh / status.sh  # progress monitoring / auto shutdown
└── README_autodl.md           # AutoDL cloud run instructions
```

**Pipeline: data cleaning → recipe ablation → ensembling → capacity check → label-merging eval**

1. **Data cleaning (Phase A)**: full MD5 scan of 167k images → isolate 5,236 same-image-different-label conflict images (2,522 groups) → remove 1,017 cross-split leak images → rebuild a leak-free 90/10 split by "image group" (train 145,537 / valid 16,244 / 879 classes).
2. **Recipe ablation (Phase C)**: fix ConvNeXt-B, unified protocol (25 epochs, batch32×grad_accum2, warmup5, EMA, cosine decay), one-variable comparison across 7 recipes.
3. **Ensembling (Phase D)**: average the logits of 7 ablation checkpoints → 77.46%, showing recipe-specific errors are complementary.
4. **Capacity check**: ImageNet-22K-pretrained ConvNeXt-V2-L (197M) fine-tuned 15 epochs → 77.49%; DINOv2 frozen-feature linear probe reaches only 71.24%, proving the task is highly dependent on domain fine-tuning.
5. **Label-merging eval**: distinguish mergeable synonyms/geo-variants from unmergeable similar-but-different herbs; evaluated by herb group → 78.31%.

---

## 📈 Training process

**7-model convergence comparison** (full training histories, same split & recipe protocol):

![7-model convergence](figs/model_7models_accuracy_comparison_879.png)

**879-class training curves** (validation Top-1 per epoch on the clean split):

![Training curves](figs/training_curves_879.png)

> Key finding: scaling the YOLO family parameters 13.8× (2.72M→37.48M) buys only +5.40 accuracy (with diminishing n→s→m gains), while within the same common budget (≤25 epochs) ConvNeXt-B beats the whole YOLO family by **+10.06 points** — an architecture gap. Fine-grained recognition is sensitive to the backbone's **representational capacity and pretraining distribution**; the size of the explicit classification head is not the bottleneck.

---

## 🏆 Training results

### 3.1 Model selection (879 classes, 7-model comparison)

| Model | Params | Epochs | Best Top-1 | Best epoch | Common budget (≤25 ep) |
|:---|:---:|:---:|:---:|:---:|:---:|
| YOLOv8n-cls | 2.72M | 40 | 61.17% | 39 | 59.30% |
| YOLOv8s-cls | 6.36M | 40 | 64.68% | 35 | 63.57% |
| YOLOv8m-cls | 17.05M | 40 | 66.57% | 36 | 65.34% |
| YOLOv8l-cls \* | 37.48M | 25 (stopped) | 66.28% | 25 | 66.28% |
| EfficientNetV2-M | 52.98M | 40 | 50.78% | 21 | 50.78% |
| ViT-Base | 86.77M | 40 | 74.00% | 40 | 73.54% |
| **ConvNeXt-Base** | 88.47M | 25 | **76.34%** | 20 | **76.34%** |

\* YOLOv8l-cls was configured for 40 epochs but stopped at epoch 25 with the learning rate still at 3.5e-4 and accuracy still rising; 66.28% is its **unconverged lower bound**.

### 3.2 Training-recipe ablation (fixed ConvNeXt-B, clean split, 25 epochs)

| Config | Key change | Best Top-1 | Best epoch |
|:---|:---|:---:|:---:|
| `base` | Baseline (weighted sampling + RandAugment m7 + MixUp 1.0, 384px) | 76.34% | 20 |
| `no_weighted` | Drop weighted sampling → instance sampling | **76.49%** ✔ | 22 |
| `base_448` | Input size 448px (batch16×accum4) | 76.42% | 23 |
| `weak_aug` | RandAugment m7→m4, RandomErase 0.2→0.05 | 76.27% | 23 |
| `weak_mixup` | MixUp/CutMix prob 1.0→0.1 | 76.25% | 21 |
| `logit_adjust` | Instance sampling + MixUp off + Logit Adjustment Loss | 76.08% | 17 |
| `two_stage` | Freeze backbone first 10 epochs, head-only then full FT | 75.64% | 25 |

![Recipe ablation](figs/ablation_879.png)

> The spread across 7 recipes is only 0.85pt: once the split is clean and the default recipe is well-tuned, recipe-level adjustments affect final accuracy very little — **the bottleneck is not the training recipe**.

### 3.3 Ensembling · capacity · label merging

| Method | Top-1 | Note |
|:---|:---:|:---|
| 7-model logits ensemble | 77.46% | +0.97 over best single model, complementary errors |
| **ConvNeXt-V2-L (197M, 15 epochs)** | **77.49%** | single model already beats the 7-model ensemble |
| DINOv2 frozen-feature linear probe | 71.24% | frozen features are insufficient for 879 classes |
| Merged eval by "herb group" | **78.31%** | 15 groups / 30 classes, involved classes 58.35% → 84.68% |

Synonym/geo-variant merging (sample; full list in `results/merge_eval_result.json`):

| Herb group | Samples | Raw correct → merged correct |
|:---|:---:|:---:|
| Epimedii folium / wushan-pinellia variant | 33 | 15 → 30 |
| Salvia miltiorrhiza / Gansu salvia | 32 | 14 → 28 |
| Dipsaci radix / Sichuan dipsacus | 30 | 14 → 26 |
| Luffa sponge / cantonese luffa | 34 | 23 → 32 |
| Paris polyphylla / Chonglou | 39 | 22 → 35 |
| Pueraria / Fen-ge | 39 | 24 → 34 |

---

## 🖥 Training environment

| Item | Setting |
|:---|:---|
| GPU | AutoDL RTX 3090 / 4090 (24GB); local RTX 5070 for inference validation |
| Framework | PyTorch 2.9 + timm 1.0 + Ultralytics 8.4 |
| Optimizer | AdamW + warmup(5) + cosine decay + EMA |
| Regularization | DropPath 0.2 / weight_decay 0.05 / label smoothing 0.1 / RandAugment |
| Effective batch | 64 (batch32 × grad_accum2; 448px → batch16 × accum4) |

---

## 🚀 Quick start

```bash
# 1. Environment
conda create -n herb python=3.10
pip install -r requirements.txt        # install CUDA-matching torch/torchvision first

# 2. Data (~167k images)
python kaggle/download_kaggle.py       # kagglehub download of raw dataset
python kaggle/clean_dataset.py         # clean → dataset_kaggle_clean (9.1GB)

# 3. Training (GPU ≥ 24GB VRAM)
python kaggle/train_convnext.py --epochs 25                      # 879-class baseline ConvNeXt-B
bash kaggle/run_all_ablations.sh                                 # 7 ablation recipes
python kaggle/train_convnext.py --model convnextv2_large --epochs 15 --img_size 384   # V2-L

# 4. Evaluation & analysis
python kaggle/diagnose_results.py --dir results_ablation/runs_ablation/no_weighted
python kaggle/ensemble_eval.py
python kaggle/merge_eval.py
python kaggle/analysis_plots.py
```

Cloud batch training (AutoDL) and monitoring scripts: see [`kaggle/README_autodl.md`](kaggle/README_autodl.md).

---

## 📄 Reproducibility notes

- **Leakage-free split**: MD5 "image-group" isolation — no identical image appears in both train and valid.
- **Comparable protocol**: the 7-model comparison uses the same default recipe for single runs; the common-budget view (≤25 epochs) avoids mixing 40-epoch models with 25-epoch models.
- **Non-deduplicated eval**: all validation metrics are computed on raw class labels, no label dedup.
- Training curves and per-epoch history: see [`results/runs_ablation/*/train_results.json`](results/runs_ablation).

## 📜 Data license

Kaggle chinese-medicine-image competition dataset — usage must follow Kaggle competition terms. The 47-class herbal-eoeld V5 dataset comes from Roboflow public datasets. Weights are published on ModelScope for learning/research purposes only.