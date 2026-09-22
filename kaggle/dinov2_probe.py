# -*- coding: utf-8 -*-
"""
DINOv2 冻结特征 + 线性探针实验
用 DINOv2 ViT-B/14(冻结)提特征, 训练线性分类头, 评估 879 类 Top-1/Top-5。
目的: 判断"模型特征质量"是否是 ConvNeXt-B 的瓶颈(对比 76.5%)。
特征会缓存到 results_ablation/dinov2_feats/, 重跑秒加载。
用法: python dinov2_probe.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import timm
from timm.data import create_transform

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'dataset_kaggle_clean')
IMG_SIZE = None  # 启动时从模型自动探测
BATCH = 32
CACHE = os.path.join(ROOT, 'results_ablation', 'dinov2_feats')
NUM_CLASSES = 879


def extract(model, split, device):
    os.makedirs(CACHE, exist_ok=True)
    cache_f = os.path.join(CACHE, f'{split}_feats.npy')
    cache_t = os.path.join(CACHE, f'{split}_targets.npy')
    if os.path.exists(cache_f):
        print(f'[cache] {split} 特征已缓存, 加载')
        return np.load(cache_f), np.load(cache_t)
    tf = create_transform(input_size=IMG_SIZE, is_training=False, interpolation='bicubic')
    ds = datasets.ImageFolder(os.path.join(DATA, split), transform=tf)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
    feats, tgts = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in tqdm(loader, desc=f'extract {split}'):
            x = x.to(device)
            with torch.amp.autocast('cuda'):
                f = model(x)
            feats.append(f.float().cpu().numpy())
            tgts.append(y.numpy())
    feats = np.concatenate(feats)
    tgts = np.concatenate(tgts)
    np.save(cache_f, feats)
    np.save(cache_t, tgts)
    print(f'[extract] {split}: {feats.shape}')
    return feats, tgts


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
    model = model.to(device).eval()
    global IMG_SIZE
    IMG_SIZE = model.default_cfg['input_size'][1]
    print(f'[model] DINOv2 ViT-B/14(lvd142m), 特征维度 {model.num_features}, 输入分辨率 {IMG_SIZE}')

    tr_feats, tr_tgts = extract(model, 'train', device)
    va_feats, va_tgts = extract(model, 'valid', device)

    # 线性探针
    Xtr = torch.tensor(tr_feats)
    Ytr = torch.tensor(tr_tgts)
    Xva = torch.tensor(va_feats)
    Yva = torch.tensor(va_tgts)
    head = nn.Linear(Xtr.shape[1], NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3)
    lossfn = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    best_acc1 = 0.0
    for ep in range(20):
        perm = torch.randperm(n)
        head.train()
        tot = 0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            opt.zero_grad()
            loss = lossfn(head(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        head.eval()
        with torch.no_grad():
            logits = torch.cat([head(Xva[i:i + 4096].to(device)).float().cpu()
                                for i in range(0, Xva.shape[0], 4096)])
        acc1 = 100.0 * (logits.argmax(1) == Yva).float().mean().item()
        acc5 = 100.0 * (logits.topk(5, 1).indices == Yva.unsqueeze(1)).any(1).float().mean().item()
        best_acc1 = max(best_acc1, acc1)
        print(f'[probe] ep{ep + 1}: loss={tot / (n // 4096 + 1):.3f}  '
              f'Top-1={acc1:.3f}%  Top-5={acc5:.3f}%', flush=True)

    print(f'\n===== DINOv2 线性探针(20轮) 最佳 Top-1 = {best_acc1:.3f}% (ConvNeXt-B 微调为 76.47%) =====')


if __name__ == '__main__':
    main()
