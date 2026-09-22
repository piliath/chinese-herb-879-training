# -*- coding: utf-8 -*-
"""
集成评估: 把 7 个消融 checkpoint 的 logits 平均, 在干净验证集上测 ensemble 效果。
用法: python ensemble_eval.py [--valid-dir 验证集目录] [--ckpts-dir 权重目录]
输出: 打印各单模型 + 各集成组合的 Top-1/Top-5
"""
import os
import json
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import timm
from timm.data import create_transform

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VALID = os.path.join(ROOT, 'dataset_kaggle_clean', 'valid')
DEFAULT_CKPTS = os.path.join(ROOT, 'results_ablation', 'runs_ablation')

# 各配置的输入尺寸
IMG_SIZES = {
    'base': 384, 'no_weighted': 384, 'weak_mixup': 384, 'weak_aug': 384,
    'two_stage': 384, 'logit_adjust': 384, 'base_448': 448,
}


def get_logits(model, loader, device, img_size):
    """返回所有验证样本的 logits (N, C)"""
    model.eval()
    all_logits = []
    with torch.inference_mode():
        for x, _ in tqdm(loader, desc=f'infer@{img_size}', leave=False):
            x = x.to(device)
            with torch.amp.autocast('cuda'):
                out = model(x)
            all_logits.append(out.float().cpu())
    return torch.cat(all_logits, dim=0)


def load_model(ckpt_path, num_classes):
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model = timm.create_model('convnext_base', pretrained=False, num_classes=num_classes)
    model.load_state_dict(state)
    return model


def topk_acc(logits, targets, k=1):
    if k == 1:
        return 100.0 * (logits.argmax(1) == targets).float().mean().item()
    pred = logits.topk(k, 1).indices
    ok = pred == targets.unsqueeze(1)
    return 100.0 * ok.any(1).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid-dir', default=DEFAULT_VALID)
    parser.add_argument('--ckpts-dir', default=DEFAULT_CKPTS)
    parser.add_argument('--batch', type=int, default=24)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    torch.backends.cudnn.benchmark = True

    valid_ds = datasets.ImageFolder(args.valid_dir)
    class_names = valid_ds.classes
    num_classes = len(class_names)
    targets = torch.tensor(valid_ds.targets)
    print(f'valid: {len(valid_ds)} images, {num_classes} classes')

    all_logits = {}
    single_top1 = {}
    for cfg, img in IMG_SIZES.items():
        ckpt = os.path.join(args.ckpts_dir, cfg, 'best_convnext_base.pth')
        if not os.path.exists(ckpt):
            print(f'[skip] {cfg} 权重不存在')
            continue
        model = load_model(ckpt, num_classes).to(device)
        tf = create_transform(input_size=img, is_training=False, interpolation='bicubic')
        ds = datasets.ImageFolder(args.valid_dir, transform=tf)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
        logits = get_logits(model, loader, device, img)
        all_logits[cfg] = logits
        t1 = topk_acc(logits, targets, 1)
        t5 = topk_acc(logits, targets, 5)
        single_top1[cfg] = t1
        print(f'  {cfg}: Top-1={t1:.3f}%  Top-5={t5:.3f}%', flush=True)

    print('\n===== 集成组合 =====')
    cfgs = list(all_logits.keys())

    # 全量集成 (7)
    ens_all = sum(all_logits.values()) / len(cfgs)
    print(f'集成全部 {len(cfgs)} 个: Top-1={topk_acc(ens_all, targets, 1):.3f}%  '
          f'Top-5={topk_acc(ens_all, targets, 5):.3f}%', flush=True)

    # 去掉 base_448 (分辨率不同) 的集成
    if 'base_448' in all_logits:
        sub = [all_logits[c] for c in cfgs if c != 'base_448']
        ens_sub = sum(sub) / len(sub)
        print(f'集成除448外 {len(sub)} 个: Top-1={topk_acc(ens_sub, targets, 1):.3f}%  '
              f'Top-5={topk_acc(ens_sub, targets, 5):.3f}%', flush=True)

    # 贪心选择最优子集
    chosen = []
    remaining = list(cfgs)
    cur_best = -1
    cur_logits = None
    while remaining:
        best_c, best_acc, best_logits = None, -1, None
        for c in remaining:
            cand_logits = cur_logits + all_logits[c] if cur_logits is not None else all_logits[c]
            cand_logits = cand_logits / (len(chosen) + 1)
            acc = topk_acc(cand_logits, targets, 1)
            if acc > best_acc:
                best_c, best_acc, best_logits = c, acc, cand_logits
        if best_c is None:
            break
        chosen.append(best_c)
        cur_logits = best_logits  # best_logits 已是当前子集的平均 logits
        cur_best = best_acc
        remaining.remove(best_c)
        print(f'  +{best_c}: 集成[{len(chosen)}] Top-1={best_acc:.3f}%  (chosen: {", ".join(chosen)})', flush=True)

    print(f'\n单模型最佳: {max(single_top1, key=single_top1.get)} = {max(single_top1.values()):.3f}%')
    print(f'贪心最优子集: {", ".join(chosen)} = {cur_best:.3f}%')

    # 保存结果
    result = {
        'single_top1': {k: round(v, 3) for k, v in single_top1.items()},
        'ensemble_all': round(topk_acc(ens_all, targets, 1), 3),
        'greedy_subset': chosen,
        'greedy_top1': round(cur_best, 3),
    }
    with open(os.path.join(ROOT, 'results_ablation', 'ensemble_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('saved: results_ablation/ensemble_result.json')


if __name__ == '__main__':
    main()
