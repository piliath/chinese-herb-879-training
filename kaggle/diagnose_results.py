# -*- coding: utf-8 -*-
"""
诊断脚本: 针对 ConvNeXt-B / ViT-B 在 Kaggle 879 类中草药数据集上的分类表现做细粒度诊断。

输入:
  - 训练集计数: dataset_kaggle/train/<类名>/   (用于统计每类样本量)
  - 验证集:      dataset_kaggle/valid/<类名>/   (用于评估)
  - 模型权重:    runs_<model>/best_*.pth

输出(写入 <out_dir>/):
  - report.json                全部指标 + 分桶统计 + 混淆对
  - per_class_accuracy.csv     每类准确率(含训练样本数 / 验证样本数 / 最多被误判为哪类)
  - bucket_accuracy.png        准确率 vs 训练样本量 分组柱状图
  - cumulative_accuracy.png    只保留训练样本量>=阈值的类时, 整体准确率变化曲线
  - top_confusions.png         Top-混淆对 横向条形图
  - summary.txt                文本摘要

用法:
  /d/anaconda/envs/herbal/python.exe diagnose_results.py convnext
  /d/anaconda/envs/herbal/python.exe diagnose_results.py vit
"""
import os
import sys
import json
import csv
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import timm
    from timm.data import create_transform
except ImportError:
    print("缺少 timm, 请先安装: pip install timm")
    sys.exit(1)

# ============ 可调参数 ============
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'dataset_kaggle')
BATCH_SIZE = 48
NUM_WORKERS = 4
BUCKETS = [
    (1, 10,   '1-10'),
    (10, 30,  '10-30'),
    (30, 60,  '30-60'),
    (60, 100, '60-100'),
    (100, 150,'100-150'),
    (150, 200,'150+'),
]
TAIL_THRESHOLD = 30      # 少于该训练样本数视为"尾类"
HEAD_THRESHOLD = 100     # 大于等于该训练样本数视为"头类"
# ====================================

# 模型名 -> (timm 名, 输入尺寸)
MODELS = {
    'convnext': ('convnext_base', 384, 'runs_convnext/best_convnext_base.pth'),
    'vit':      ('vit_base_patch16_384', 384, 'runs_vit/best_vit_base_patch16_384.pth'),
}


def load_checkpoint_path(path):
    """加载 best_*.pth, 返回 state_dict + class_names"""
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        class_names = ckpt.get('class_names', None)
        reported_acc = ckpt.get('best_val_top1_acc', ckpt.get('best_val_top1', None))
    else:
        state = ckpt
        class_names = None
        reported_acc = None
    return state, class_names, reported_acc


def count_train_samples(class_names, data_dir=DATA_DIR):
    """统计每类训练样本数, 返回 {类名: 数量}"""
    counts = {}
    train_dir = os.path.join(data_dir, 'train')
    for name in class_names:
        d = os.path.join(train_dir, name)
        counts[name] = len([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return counts


def run_inference(model, loader, device):
    """返回 preds, preds5, targets"""
    model.eval()
    all_preds, all_preds5, all_targets = [], [], []
    with torch.inference_mode():
        for inputs, targets in tqdm(loader, desc='inference'):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            top1 = outputs.argmax(dim=1).cpu().tolist()
            top5 = outputs.topk(5, dim=1).indices.cpu().tolist()
            all_preds.extend(top1)
            all_preds5.extend(top5)
            all_targets.extend(targets.tolist())
    return np.array(all_preds), np.array(all_preds5), np.array(all_targets)


def bucket_metrics(preds, preds5, targets, class_names, train_counts):
    """按训练样本量分桶统计 top1 / top5"""
    n = len(class_names)
    rows = []
    for lo, hi, label in BUCKETS:
        cls_idx = [i for i in range(n) if lo <= train_counts[class_names[i]] < hi]
        if not cls_idx:
            continue
        mask = np.isin(targets, cls_idx)
        t, p, p5 = targets[mask], preds[mask], preds5[mask]
        top1 = 100.0 * (p == t).mean() if len(t) else 0.0
        top5 = 100.0 * (np.array([t[i] in p5[i] for i in range(len(t))]).mean()) if len(t) else 0.0
        rows.append({
            'bucket': label,
            'sample_range': f'{lo}-{hi-1}',
            'num_classes': len(cls_idx),
            'num_valid_samples': int(len(t)),
            'top1_acc': round(top1, 2),
            'top5_acc': round(top5, 2),
        })
    return rows


def cumulative_metrics(preds, preds5, targets, class_names, train_counts):
    """只保留训练样本数>=阈值的类时, 整体 top1/top5 的变化"""
    out = []
    for thr in [1, 5, 10, 30, 60, 100, 150]:
        cls_idx = [i for i in range(len(class_names)) if train_counts[class_names[i]] >= thr]
        mask = np.isin(targets, cls_idx)
        t, p, p5 = targets[mask], preds[mask], preds5[mask]
        top1 = 100.0 * (p == t).mean() if len(t) else 0.0
        top5 = 100.0 * (np.array([t[i] in p5[i] for i in range(len(t))]).mean()) if len(t) else 0.0
        out.append({'min_train_samples': thr, 'num_classes': len(cls_idx),
                    'num_valid_samples': int(len(t)), 'top1_acc': round(top1, 2), 'top5_acc': round(top5, 2)})
    return out


def confusion_analysis(preds, targets, class_names, valid_per_class, train_counts,
                       correct_per_class, top_n=15):
    """错误样本里的混淆对分析 + 最差类别分析"""
    n = len(class_names)
    # 误判对 (true, pred) 计数
    pair_counter = Counter()
    for t, p in zip(targets, preds):
        if t != p:
            pair_counter[(int(t), int(p))] += 1
    top_pairs = []
    for (t, p), cnt in pair_counter.most_common(top_n * 2):
        top_pairs.append({'true': class_names[t], 'pred': class_names[p], 'count': cnt})
    top_pairs = top_pairs[:top_n]

    # 互相混淆对 (a->b 和 b->a 都频繁)
    mutual = defaultdict(int)
    for (t, p), cnt in pair_counter.items():
        mutual[(min(t, p), max(t, p))] += cnt
    mutual_pairs = sorted(mutual.items(), key=lambda kv: -kv[1])[:top_n]
    mutual_pairs = [{'class_a': class_names[a], 'class_b': class_names[b], 'total_count': cnt}
                    for (a, b), cnt in mutual_pairs]

    # 最差类别: 至少5个验证样本的类里按准确率排序
    worst = []
    for i in range(n):
        if valid_per_class[i] >= 5:
            acc = 100.0 * correct_per_class[i] / valid_per_class[i]
            worst.append((acc, class_names[i], valid_per_class[i], train_counts[class_names[i]]))
    worst.sort()
    worst_classes = [{'class': name, 'acc': round(acc, 2), 'num_valid': nv, 'num_train': nt}
                     for acc, name, nv, nt in worst[:top_n]]

    return top_pairs, mutual_pairs, worst_classes


def save_per_class_csv(path, class_names, train_counts, valid_per_class, correct_per_class,
                       preds, targets):
    """每类准确率 + 最常被误判为哪类"""
    n = len(class_names)
    top_mispred = {}
    pair_counter = Counter()
    for t, p in zip(targets, preds):
        if t != p:
            pair_counter[(int(t), int(p))] += 1
    for (t, p), cnt in pair_counter.items():
        key = top_mispred.get(t)
        if key is None or cnt > key[1]:
            top_mispred[t] = (class_names[p], cnt)

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['类名', '训练样本数', '验证样本数', '正确数', '准确率(%)', '最常被误判为', '误判次数'])
        for i in range(n):
            acc = 100.0 * correct_per_class[i] / valid_per_class[i] if valid_per_class[i] else -1
            mp = top_mispred.get(i, ('', 0))
            w.writerow([class_names[i], train_counts[class_names[i]], valid_per_class[i],
                        correct_per_class[i], round(acc, 2), mp[0], mp[1]])


def plot_bucket_accuracy(rows, out_path, title):
    buckets = [r['bucket'] for r in rows]
    top1 = [r['top1_acc'] for r in rows]
    top5 = [r['top5_acc'] for r in rows]
    num_classes = [r['num_classes'] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(buckets))
    w = 0.38
    ax1.bar(x - w/2, top1, w, label='Top-1 Acc %', color='#1f77b4')
    ax1.bar(x + w/2, top5, w, label='Top-5 Acc %', color='#7fb3d5')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 105)
    ax1.set_title(title, fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n({c}类)" for b, c in zip(buckets, num_classes)], fontsize=9)
    ax1.legend(loc='upper left')
    for xi, (t1, t5, c) in enumerate(zip(top1, top5, num_classes)):
        ax1.text(xi - w/2, t1 + 1, f'{t1:.1f}', ha='center', fontsize=9)
        ax1.text(xi + w/2, t5 + 1, f'{t5:.1f}', ha='center', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cumulative_accuracy(cum, out_path, title):
    thrs = [r['min_train_samples'] for r in cum]
    top1 = [r['top1_acc'] for r in cum]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thrs, top1, marker='o', color='#2ca02c', linewidth=2)
    for xi, (th, t1) in enumerate(zip(thrs, top1)):
        ax.annotate(f'{t1:.1f}%', (th, t1), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=10)
    ax.set_xlabel('仅保留训练样本数 ≥ 该值的类别')
    ax.set_ylabel('整体 Top-1 Acc (%)')
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_confusions(pairs, out_path, title):
    labels = [f"{p['true']}  →  {p['pred']}" for p in pairs]
    counts = [p['count'] for p in pairs]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(labels) + 2)))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color='#d62728', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('误判样本数')
    ax.set_title(title, fontsize=13)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(MODELS.keys()), help='要诊断的模型')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='推理批次大小')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='覆盖权重路径(可用于诊断任意 checkpoint)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='覆盖数据集目录(默认 dataset_kaggle)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='覆盖输出目录(默认 diagnosis_<model>)')
    args = parser.parse_args()

    timm_name, img_size, _ = MODELS[args.model]
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(ROOT, MODELS[args.model][2])
    if args.data_dir:
        DATA_DIR_ACTIVE = os.path.abspath(args.data_dir)
    else:
        DATA_DIR_ACTIVE = DATA_DIR
    out_dir = os.path.join(ROOT, args.out_dir if args.out_dir else f'diagnosis_{args.model}')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[info] device={device}, model={timm_name}')

    # 1. 加载模型
    state, ckpt_classes, reported_acc = load_checkpoint_path(ckpt_path)
    num_classes = len(ckpt_classes) if ckpt_classes else None
    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state)
    model = model.to(device)
    print(f'[info] checkpoint 内记录的 best acc = {reported_acc:.2f}%' if reported_acc else '[info] 未记录 acc')

    # 2. 验证集 & 类名对齐
    val_transform = create_transform(input_size=img_size, is_training=False, interpolation='bicubic')
    valid_dataset = datasets.ImageFolder(os.path.join(DATA_DIR_ACTIVE, 'valid'), transform=val_transform)
    class_names = valid_dataset.classes
    print(f'[info] 验证集类别数={len(class_names)}, 验证样本数={len(valid_dataset)}')
    if ckpt_classes and list(ckpt_classes) != list(class_names):
        print('[warn] checkpoint 的 class_names 与验证集文件夹顺序不一致! 以验证集为准。')
    valid_per_class = np.bincount(valid_dataset.targets, minlength=len(class_names))

    train_counts = count_train_samples(class_names, DATA_DIR_ACTIVE)

    loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # 3. 推理 (带缓存: 已有 npy 则跳过推理)
    cache_dir = out_dir
    cache_paths = [os.path.join(cache_dir, f'{args.model}_preds.npy'),
                   os.path.join(cache_dir, f'{args.model}_preds5.npy'),
                   os.path.join(cache_dir, f'{args.model}_targets.npy')]
    if all(os.path.exists(p) for p in cache_paths):
        print('[info] 发现推理缓存, 跳过推理')
        preds = np.load(cache_paths[0])
        preds5 = np.load(cache_paths[1])
        targets = np.load(cache_paths[2])
    else:
        preds, preds5, targets = run_inference(model, loader, device)
        np.save(cache_paths[0], preds)
        np.save(cache_paths[1], preds5)
        np.save(cache_paths[2], targets)
    correct = preds == targets
    top1_overall = 100.0 * correct.mean()
    top5_overall = 100.0 * np.mean([targets[i] in preds5[i] for i in range(len(targets))])
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    print(f'[info] 复现 Top-1 = {top1_overall:.2f}%  (checkpoint记录 {reported_acc:.2f}%)')

    # 4. 逐类统计 (转成 python int 便于 JSON 序列化)
    correct_per_class = [int(x) for x in np.bincount(targets[correct], minlength=len(class_names))]
    valid_per_class = [int(x) for x in valid_per_class]
    train_counts = {k: int(v) for k, v in train_counts.items()}

    # 5. 分桶 / 累积 / 混淆
    buckets = bucket_metrics(preds, preds5, targets, class_names, train_counts)
    cum = cumulative_metrics(preds, preds5, targets, class_names, train_counts)
    top_pairs, mutual_pairs, worst_classes = confusion_analysis(
        preds, targets, class_names, valid_per_class, train_counts, correct_per_class)

    # 尾类/头类
    tail_idx = [i for i in range(len(class_names)) if train_counts[class_names[i]] < TAIL_THRESHOLD]
    head_idx = [i for i in range(len(class_names)) if train_counts[class_names[i]] >= HEAD_THRESHOLD]
    def _acc(idx):
        mask = np.isin(targets, idx)
        t, p = targets[mask], preds[mask]
        return 100.0 * (p == t).mean() if len(t) else 0.0
    tail_acc = _acc(tail_idx)
    head_acc = _acc(head_idx)

    report = {
        'model': args.model,
        'timm_name': timm_name,
        'img_size': img_size,
        'reported_best_acc': reported_acc,
        'reproduced_top1': round(top1_overall, 3),
        'reproduced_top5': round(top5_overall, 3),
        'macro_f1': round(macro_f1, 4),
        'num_classes': len(class_names),
        'num_valid_samples': len(targets),
        'tail': {'threshold': TAIL_THRESHOLD, 'num_classes': len(tail_idx), 'top1_acc': round(tail_acc, 2)},
        'head': {'threshold': HEAD_THRESHOLD, 'num_classes': len(head_idx), 'top1_acc': round(head_acc, 2)},
        'bucket_metrics': buckets,
        'cumulative_metrics': cum,
        'top_confusion_pairs': top_pairs,
        'mutual_confusion_pairs': mutual_pairs,
        'worst_classes': worst_classes,
    }
    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    save_per_class_csv(os.path.join(out_dir, 'per_class_accuracy.csv'), class_names,
                       train_counts, valid_per_class, correct_per_class, preds, targets)

    # 6. 画图
    plot_bucket_accuracy(buckets, os.path.join(out_dir, 'bucket_accuracy.png'),
                         f'{timm_name} 准确率 vs 训练样本量分组')
    plot_cumulative_accuracy(cum, os.path.join(out_dir, 'cumulative_accuracy.png'),
                             f'{timm_name} 仅保留样本量≥阈值的类时整体Top-1')
    plot_top_confusions(top_pairs, os.path.join(out_dir, 'top_confusions.png'),
                        f'{timm_name} Top-混淆对')

    # 7. 文本摘要
    lines = []
    lines.append(f'===== {timm_name} 诊断摘要 =====')
    lines.append(f'复现 Top-1 = {top1_overall:.2f}%  Top-5 = {top5_overall:.2f}%  Macro-F1 = {macro_f1:.4f}')
    lines.append(f'尾类(训练<{TAIL_THRESHOLD}张, {len(tail_idx)}类) Top-1 = {tail_acc:.2f}%')
    lines.append(f'头类(训练>={HEAD_THRESHOLD}张, {len(head_idx)}类) Top-1 = {head_acc:.2f}%')
    lines.append('')
    lines.append('--- 分桶 ---')
    for r in buckets:
        lines.append(f"  训练{r['sample_range']}张({r['num_classes']}类, {r['num_valid_samples']}个验证样本): "
                     f"Top-1={r['top1_acc']}%  Top-5={r['top5_acc']}%")
    lines.append('')
    lines.append('--- 累积(只保留样本量≥阈值的类) ---')
    for r in cum:
        lines.append(f"  ≥{r['min_train_samples']}张: {r['num_classes']}类/{r['num_valid_samples']}样本 -> Top-1={r['top1_acc']}%")
    lines.append('')
    lines.append('--- Top混淆对 ---')
    for p in top_pairs:
        lines.append(f"  {p['true']} -> {p['pred']} : {p['count']}")
    lines.append('')
    lines.append('--- 互相混淆对 ---')
    for p in mutual_pairs:
        lines.append(f"  {p['class_a']} <-> {p['class_b']} : {p['total_count']}")
    lines.append('')
    lines.append('--- 最差类别(验证样本≥5) ---')
    for w in worst_classes:
        lines.append(f"  {w['class']}: acc={w['acc']}% (训练{w['num_train']}张/验证{w['num_valid']}张)")
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\n[info] 输出已保存到 {out_dir}/')


if __name__ == '__main__':
    main()
