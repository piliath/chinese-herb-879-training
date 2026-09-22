# -*- coding: utf-8 -*-
"""
Phase C: ConvNeXt-B 训练消融实验 (在清洗后的 dataset_kaggle_clean 上运行)

7 个配置, 每个只改一个变量 (base_448 改输入尺寸):
  base         当前配方 (等价于 train_convnext.py 的配置)
  no_weighted  取消加权采样 -> 普通实例采样
  weak_mixup   MixUp/CutMix prob 1.0 -> 0.1
  weak_aug     RandAugment m7 -> m4, RandomErase 0.2 -> 0.05
  two_stage    前10轮冻结 backbone 只训分类头, 之后全量微调
  logit_adjust 实例采样 + 关闭mixup + Logit Adjustment Loss
  base_448     同 base, 输入尺寸 448 (batch16/accum4)

用法:
  python train_ablation.py --config base
  python train_ablation.py --config base_448 --epochs 30
  python train_ablation.py --config base --smoke          # 本地冒烟测试(几十步)
  python train_ablation.py --config base --no-pretrained  # 本地无权重测试

输出: runs_ablation/<config>/  (config.json, train_results.json, best_*.pth, training_results.png)
"""
import os
import sys
import json
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets as tv_datasets
from tqdm import tqdm

try:
    import timm
    from timm.data import create_transform
    from timm.data.mixup import Mixup
    from timm.loss import SoftTargetCrossEntropy
except ImportError:
    print("缺少 timm: pip install timm")
    sys.exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'dataset_kaggle_clean')
OUT_ROOT = os.path.join(ROOT, 'runs_ablation')

# ============ 7 个消融配置 (只列与 base 不同的项) ============
BASE = dict(
    model='convnext_base', img_size=384, epochs=40, batch_size=32, grad_accum=2,
    lr=5e-4, weight_decay=0.05, drop_path=0.2, warmup_epochs=5, ema_decay=0.9998,
    layer_decay=0.7, aug='rand-m7-mstd0.5-inc1', re_prob=0.2,
    mixup_alpha=0.8, cutmix_alpha=1.0, mixup_prob=1.0, label_smooth=0.1,
    weighted=True, two_stage=False, freeze_epochs=0, logit_adjust=False,
)
CONFIGS = {
    'base': dict(BASE),
    'no_weighted': dict(BASE, weighted=False),
    'weak_mixup': dict(BASE, mixup_prob=0.1),
    'weak_aug': dict(BASE, aug='rand-m4-mstd0.5-inc1', re_prob=0.05),
    'two_stage': dict(BASE, two_stage=True, freeze_epochs=10),
    'logit_adjust': dict(BASE, weighted=False, mixup_prob=0.0, label_smooth=0.0, logit_adjust=True),
    'base_448': dict(BASE, img_size=448, batch_size=16, grad_accum=4),
    # ConvNeXt-V2-L 趋势测试 (no_weighted 配方, 15轮)
    'v2l': dict(BASE, model='convnextv2_large.fcmae_ft_in22k_in1k_384',
                img_size=384, epochs=15, batch_size=16, grad_accum=4,
                warmup_epochs=2, weighted=False),
}
NUM_CLASSES = 879


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)


class LogitAdjustedLoss(nn.Module):
    """Logit Adjustment: 用类别先验对 logits 做偏移, 缓解类别不平衡"""
    def __init__(self, priors, tau=1.0):
        super().__init__()
        self.tau = tau
        self.register_buffer('adj', tau * torch.log(torch.clamp(priors, min=1e-6)))

    def forward(self, logits, targets):
        return nn.functional.cross_entropy(logits + self.adj, targets)


def get_layer_groups(model, lr, weight_decay, layer_decay):
    layer_names = ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3']
    num_layers = len(layer_names) + 1
    groups = []
    for i, prefix in enumerate(layer_names):
        layer_lr = lr * (layer_decay ** (num_layers - 1 - i))
        params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:
            groups.append({'params': params, 'lr': layer_lr, 'weight_decay': weight_decay})
    head_params = [p for n, p in model.named_parameters()
                   if not any(n.startswith(pf) for pf in layer_names) and p.requires_grad]
    if head_params:
        groups.append({'params': head_params, 'lr': lr, 'weight_decay': 0.0})
    return groups


def make_optimizer(model, cfg):
    groups = get_layer_groups(model, cfg['lr'], cfg['weight_decay'], cfg['layer_decay'])
    return optim.AdamW(groups)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def freeze_backbone(model, freeze=True):
    layer_names = ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3']
    for n, p in model.named_parameters():
        if any(n.startswith(pf) for pf in layer_names):
            p.requires_grad = not freeze


def get_dataloaders(cfg):
    train_dir, valid_dir = os.path.join(DATA_DIR, 'train'), os.path.join(DATA_DIR, 'valid')
    train_transform = create_transform(
        input_size=cfg['img_size'], is_training=True, color_jitter=0.3,
        auto_augment=cfg['aug'], interpolation='bicubic',
        re_prob=cfg['re_prob'], re_mode='pixel', re_count=1)
    val_transform = create_transform(input_size=cfg['img_size'], is_training=False, interpolation='bicubic')

    train_ds = tv_datasets.ImageFolder(train_dir, transform=train_transform)
    valid_ds = tv_datasets.ImageFolder(valid_dir, transform=val_transform)
    print(f'[data] train={len(train_ds)} valid={len(valid_ds)} classes={len(train_ds.classes)}')

    if cfg['weighted']:
        counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes))
        counts = np.maximum(counts, 1)
        class_weights = 1.0 / np.sqrt(counts)
        sample_weights = class_weights[train_ds.targets]
        # num_samples 必须是"每样本权重"的长度(=训练集大小), 否则每轮只采样 num_classes 张
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=sampler,
                                  num_workers=cfg['workers'], pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                                  num_workers=cfg['workers'], pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['workers'], pin_memory=True)
    return train_loader, valid_loader, train_ds, valid_ds


def validate(model, valid_loader, device, max_samples=0):
    model.eval()
    correct = correct5 = total = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                out = model(x)
            _, pred = out.topk(1, 1)
            pred5 = out.topk(5, 1).indices
            correct += pred.squeeze(1).eq(y).sum().item()
            correct5 += sum(y[i].item() in pred5[i].tolist() for i in range(len(y)))
            total += y.size(0)
            if max_samples and total >= max_samples:
                break
    model.train()
    return 100.0 * correct / total, 100.0 * correct5 / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=list(CONFIGS.keys()), required=True)
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数')
    parser.add_argument('--smoke', action='store_true', help='冒烟模式: 少量步数快速验证')
    parser.add_argument('--no-pretrained', action='store_true', help='不加载预训练权重(本地测试用)')
    parser.add_argument('--freeze-epochs', type=int, default=None, help='覆盖两阶段冻结轮数')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader worker 数')
    parser.add_argument('--batch', type=int, default=None, help='覆盖 batch_size')
    args = parser.parse_args()

    cfg = dict(CONFIGS[args.config])
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.freeze_epochs is not None:
        cfg['freeze_epochs'] = args.freeze_epochs
    cfg['workers'] = args.workers
    if args.batch is not None:
        cfg['batch_size'] = args.batch
    if args.smoke:
        cfg['epochs'] = 1
        cfg['smoke_steps'] = 25
        cfg['smoke_valid'] = 200
    else:
        cfg['smoke_steps'] = 0
        cfg['smoke_valid'] = 0

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(OUT_ROOT, args.config)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[info] config={args.config} device={device}  {json.dumps(cfg, ensure_ascii=False)}')

    # 有效 batch 与 LR 线性缩放
    eff_batch = cfg['batch_size'] * cfg['grad_accum']
    cfg['lr'] = cfg['lr'] * (eff_batch / 256.0)

    train_loader, valid_loader, train_ds, valid_ds = get_dataloaders(cfg)
    num_classes = len(train_ds.classes)

    # 模型
    model = timm.create_model(cfg['model'], pretrained=not args.no_pretrained,
                              num_classes=num_classes, drop_path_rate=cfg['drop_path'])
    model = model.to(device)

    # Mixup / 损失
    mixup_fn = Mixup(mixup_alpha=cfg['mixup_alpha'], cutmix_alpha=cfg['cutmix_alpha'],
                     prob=cfg['mixup_prob'], switch_prob=0.5, mode='batch',
                     label_smoothing=cfg['label_smooth'], num_classes=num_classes)
    if cfg['logit_adjust']:
        counts = np.bincount(train_ds.targets, minlength=num_classes).astype(np.float64)
        priors = torch.tensor(counts / counts.sum(), device=device)
        train_loss_fn = LogitAdjustedLoss(priors, tau=1.0)
        print('[info] 使用 Logit-Adjusted Loss')
    else:
        train_loss_fn = SoftTargetCrossEntropy()
    valid_loss_fn = nn.CrossEntropyLoss()

    # 优化器与调度器
    if cfg['two_stage']:
        freeze_backbone(model, True)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = make_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg['epochs'] // cfg['grad_accum']
    warmup_steps = len(train_loader) * cfg['warmup_epochs'] // cfg['grad_accum']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda')

    model_ema = ModelEMA(model, cfg['ema_decay'])
    best_acc, best_epoch, history = 0.0, 0, []
    step_count = 0

    for epoch in range(cfg['epochs']):
        # 两阶段: 到达冻结轮次后解冻 backbone 并重建优化器
        if cfg['two_stage'] and epoch == cfg['freeze_epochs']:
            print(f'[two_stage] epoch {epoch+1}: 解冻 backbone, 全量微调')
            freeze_backbone(model, False)
            optimizer = make_optimizer(model, cfg)
            warmup_steps2 = len(train_loader) * cfg['warmup_epochs'] // cfg['grad_accum']
            remaining = total_steps - step_count
            scheduler = get_cosine_schedule_with_warmup(optimizer, min(warmup_steps2, remaining // 2), remaining)

        model.train()
        optimizer.zero_grad()
        run_loss = 0.0
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f'ep{epoch+1}/{cfg["epochs"]} [{args.config}]')):
            if cfg['smoke_steps'] and i >= cfg['smoke_steps']:
                break
            x, y = x.to(device), y.to(device)
            if cfg['mixup_prob'] > 0:
                x, y = mixup_fn(x, y)
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = train_loss_fn(out, y) / cfg['grad_accum']
            scaler.scale(loss).backward()
            if (i + 1) % cfg['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                model_ema.update(model)
                step_count += 1
            run_loss += loss.item() * cfg['grad_accum'] * x.size(0)

        # 验证 (用 EMA)
        val_top1, val_top5 = validate(model_ema.module, valid_loader, device, cfg['smoke_valid'])
        train_avg_loss = run_loss / len(train_ds)
        history.append({'epoch': epoch + 1, 'train_loss': round(train_avg_loss, 4),
                        'val_top1': round(val_top1, 3), 'val_top5': round(val_top5, 3)})
        print(f'[epoch {epoch+1}/{cfg["epochs"]}] train_loss={train_avg_loss:.4f} '
              f'val_top1={val_top1:.3f}% val_top5={val_top5:.3f}%')
        if val_top1 > best_acc:
            best_acc, best_epoch = val_top1, epoch + 1
            torch.save({'model_name': cfg['model'], 'config': cfg,
                        'state_dict': model_ema.module.state_dict(),
                        'epoch': best_epoch, 'best_val_top1': best_acc,
                        'class_names': train_ds.classes},
                       os.path.join(out_dir, f'best_{cfg["model"]}.pth'))
        with open(os.path.join(out_dir, 'train_results.json'), 'w', encoding='utf-8') as f:
            json.dump({'config': args.config, 'cfg': cfg, 'best_epoch': best_epoch,
                       'best_val_top1': best_acc, 'history': history}, f, ensure_ascii=False, indent=2)
        if cfg['smoke_steps']:
            break

    print(f'===== {args.config} 完成: best_val_top1={best_acc:.3f}% @ epoch {best_epoch} =====')
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
