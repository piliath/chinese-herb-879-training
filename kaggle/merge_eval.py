# -*- coding: utf-8 -*-
"""
标签合并评估: 用 V2-L 在干净验证集上预测, 按"近义类分组"重算药材组准确率。
量化"合并同物异名类"能带来的准确率提升。

用法: python merge_eval.py
输出: kaggle/merge_eval_result.json
"""
import os
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
VALID = os.path.join(ROOT, 'dataset_kaggle_clean', 'valid')
CKPT = os.path.join(ROOT, 'results_ablation', 'runs_ablation', 'v2l',
                    'best_convnextv2_large.fcmae_ft_in22k_in1k_384.pth')

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import timm
from timm.data import create_transform


def load_model(device):
    ckpt = torch.load(CKPT, map_location='cpu')
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    names = ckpt.get('class_names') if isinstance(ckpt, dict) else None
    model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=False, num_classes=879)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, names


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[info] device={device}')

    # 加载别名映射
    aliases = json.load(open(os.path.join(ROOT, 'aliases.json'), encoding='utf-8'))
    canon = aliases.get('canonical_map', {})
    core_groups = aliases.get('core_groups', [])

    # 数据集
    ds = datasets.ImageFolder(VALID)
    class_names = ds.classes
    tf = create_transform(input_size=384, is_training=False, interpolation='bicubic')
    ds = datasets.ImageFolder(VALID, transform=tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    targets = np.array(ds.dataset.targets if hasattr(ds, 'dataset') else None) if hasattr(ds, 'dataset') else None

    # 直接重建 targets
    ds_plain = datasets.ImageFolder(VALID)
    tgt_arr = np.array(ds_plain.targets)

    model, names = load_model(device)
    preds = []
    with torch.inference_mode():
        for x, _ in tqdm(loader, desc='V2-L 推理'):
            x = x.to(device)
            with torch.amp.autocast('cuda'):
                out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
    preds = np.array(preds)

    # raw top-1
    raw = 100.0 * (preds == tgt_arr).mean()

    # 合并: canonical map (class_name -> canonical)
    def merged_is_correct(p, t):
        pc, tc = class_names[p], class_names[t]
        cp = canon.get(pc, pc)
        ct = canon.get(tc, tc)
        return cp == ct

    merged = np.array([merged_is_correct(p, t) for p, t in zip(preds, tgt_arr)])
    merged_acc = 100.0 * merged.mean()

    # 每组恢复的错误
    per_group = []
    for grp in core_groups:
        idx = [class_names.index(n) for n in grp if n in class_names]
        if not idx:
            continue
        mask = np.isin(tgt_arr, idx)
        n_total = int(mask.sum())
        n_raw = int(((preds == tgt_arr) & mask).sum())
        n_merged = int((merged & mask).sum())
        per_group.append({
            'group': '/'.join(grp),
            'samples': n_total,
            'raw_correct': n_raw,
            'merged_correct': n_merged,
            'recovered': n_merged - n_raw,
        })

    # 参与合并的类 vs 未参与的
    involved_idx = [class_names.index(n) for n in canon if n in class_names]
    inv_mask = np.isin(tgt_arr, involved_idx)
    inv_raw = 100.0 * ((preds == tgt_arr) & inv_mask).mean() / inv_mask.mean()
    inv_merged = 100.0 * (merged & inv_mask).mean() / inv_mask.mean()

    result = {
        'raw_top1': round(float(raw), 3),
        'merged_top1': round(float(merged_acc), 3),
        'delta': round(float(merged_acc - raw), 3),
        'num_merged_classes': len(canon),
        'num_groups': len(core_groups),
        'involved_classes_raw_top1': round(float(inv_raw), 3),
        'involved_classes_merged_top1': round(float(inv_merged), 3),
        'per_group': per_group,
    }
    with open(os.path.join(ROOT, 'merge_eval_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'\nV2-L 原始 Top-1: {raw:.3f}%')
    print(f'按药材组合并后 Top-1: {merged_acc:.3f}%  (Δ {merged_acc-raw:+.3f})')
    print(f'参与合并的 {len(canon)} 类: 原始 {inv_raw:.2f}% -> 合并后 {inv_merged:.2f}%')
    print('--- 各组恢复的错误 ---')
    for g in per_group:
        if g['recovered'] > 0:
            print(f"  {g['group']}: 恢复 {g['recovered']} 个 (共 {g['samples']} 样本)")
    print(f'\n结果已保存: merge_eval_result.json')


if __name__ == '__main__':
    main()
