import os
import copy
import json
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    import timm
    from timm.data import create_transform
    from timm.data.mixup import Mixup
    from timm.loss import SoftTargetCrossEntropy
except ImportError:
    print("请先安装必要的库: pip install timm scikit-learn seaborn matplotlib")
    exit(1)

# ==========================================
# 训练参数配置 (全局变量，可直接修改)
# ==========================================
DATA_DIR = 'dataset_kaggle'           # 数据集根目录 (须包含 train 和 valid)
MODEL_NAME = 'efficientnetv2_rw_m'    # EfficientNetV2-M
EPOCHS = 40                           # 训练轮数
BATCH_SIZE = 32                       # 批次大小 (12GB显存建议16, 24GB建议32)
IMG_SIZE = 384                        # 输入图片尺寸 (放大以保留细节)
LR = 5e-4                            # [核心优化] 迁移学习用较小LR保护预训练权重
WEIGHT_DECAY = 1e-4                   # [核心优化] EfficientNet含BN层，用较小的权重衰减
DROP_PATH_RATE = 0.2                  # Stochastic Depth 防止过拟合
MIXUP_ALPHA = 0.4                     # [核心优化] 适度的 Mixup
CUTMIX_ALPHA = 0.5                    # [核心优化] 适度的 Cutmix
WARMUP_EPOCHS = 5                     # [核心优化] 前5轮学习率线性预热
EMA_DECAY = 0.9998                    # [核心优化] 模型EMA指数移动平均衰减率
GRAD_ACCUM_STEPS = 2                  # [核心优化] 梯度累积步数，等效batch=64
LAYER_DECAY = 0.75                    # [核心优化] 层级学习率衰减率
TTA_ENABLED = True                    # [核心优化] 验证时开启 Test-Time Augmentation
PATIENCE = 20                         # [核心优化] Early Stopping 耐心值
OUT_DIR = 'runs_efficientnet'         # 输出结果和权重的保存目录
# ==========================================


def get_layer_groups(model):
    """
    [核心优化] 为 EfficientNetV2 实现层级学习率衰减 (Layer-wise LR Decay / LLRD)
    底层 (靠近输入) 使用更小学习率，顶层 (靠近分类头) 使用更大学习率。
    EfficientNetV2 结构: stem -> blocks.0~6 -> head (conv_head + classifier)
    """
    param_groups = []
    
    # EfficientNetV2 有 stem + 7 个 blocks 阶段 + head
    layer_prefixes = ['stem', 'blocks.0', 'blocks.1', 'blocks.2', 
                      'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6']
    num_layers = len(layer_prefixes) + 1  # +1 for head
    
    assigned_params = set()
    
    for i, prefix in enumerate(layer_prefixes):
        layer_lr = LR * (LAYER_DECAY ** (num_layers - 1 - i))
        params = []
        for name, param in model.named_parameters():
            if name.startswith(prefix) and param.requires_grad:
                params.append(param)
                assigned_params.add(name)
        if params:
            param_groups.append({
                'params': params,
                'lr': layer_lr,
                'weight_decay': WEIGHT_DECAY
            })
    
    # head 层 (conv_head, bn2, classifier 等) 使用完整学习率，不加权重衰减
    head_params = []
    for name, param in model.named_parameters():
        if name not in assigned_params and param.requires_grad:
            head_params.append(param)
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': LR,
            'weight_decay': 0.0
        })
    
    return param_groups


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    [核心优化] Warmup + Cosine Annealing 学习率调度器
    前几个epoch线性预热，之后余弦退火到接近0。
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """
    [核心优化] 模型指数移动平均 (EMA)
    维护一份模型权重的滑动平均，验证时使用EMA模型，通常能提升 0.3~1% 的精度。
    """
    def __init__(self, model, decay=0.9998):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def set(self, model):
        self.module.load_state_dict(model.state_dict())


def get_dataloaders():
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'valid')

    # [核心优化] RandAugment 强度适中 (m7)，过强会导致小数据集欠拟合
    train_transform = create_transform(
        input_size=IMG_SIZE,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m7-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.2,
        re_mode='pixel',
        re_count=1,
    )
    
    val_transform = create_transform(
        input_size=IMG_SIZE,
        is_training=False,
        interpolation='bicubic'
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_transform)

    # [核心优化] 类别不平衡加权采样 (Weighted Random Sampler)
    print("正在统计各类别样本权重以应对长尾效应...")
    class_counts = np.bincount(train_dataset.targets, minlength=len(train_dataset.classes))
    class_counts = np.maximum(class_counts, 1)  # 防止极小众类别数量为0导致除零(inf)进而引发模型崩坏
    class_weights = 1.0 / np.sqrt(class_counts)  # 使用开平方减缓极端不平衡权重
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=4, pin_memory=True)
    
    return train_loader, valid_loader, train_dataset.classes


def get_tta_transforms():
    """
    [核心优化] Test-Time Augmentation (TTA) 变换列表
    验证时对同一张图片做多种变换，取平均预测，稳定并提升精度。
    """
    return [
        # 原始图
        create_transform(input_size=IMG_SIZE, is_training=False, interpolation='bicubic'),
        # 水平翻转
        transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.143), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # 稍微不同的裁剪比例
        transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.2), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]


def validate_with_tta(model, valid_dir, class_names, device):
    """用 TTA 进行验证"""
    tta_transforms = get_tta_transforms()
    num_classes = len(class_names)
    
    base_dataset = datasets.ImageFolder(valid_dir)
    
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(base_dataset)), desc="TTA Validation"):
            img, target = base_dataset[idx]  # PIL Image
            all_targets.append(target)
            
            avg_logits = torch.zeros(num_classes).to(device)
            for t in tta_transforms:
                input_tensor = t(img).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast():
                    logits = model(input_tensor)
                avg_logits += logits.squeeze(0)
            
            avg_logits /= len(tta_transforms)
            pred = avg_logits.argmax().item()
            all_preds.append(pred)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    correct = (all_preds == all_targets).sum()
    total = len(all_targets)
    acc = 100.0 * correct / total
    
    return acc, all_preds, all_targets


def plot_confusion_matrix(cm, classes, out_dir, filename='confusion_matrix.png'):
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()
    print(f"混淆矩阵已保存至 {out_dir}/{filename}")


def count_parameters(model):
    """统计模型总参数量与可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_config(out_dir, config_dict):
    """保存训练配置记录 JSON"""
    config_path = os.path.join(out_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f"训练配置记录已保存至 {config_path}")


def save_results(out_dir, history, summary):
    """保存训练结果记录 JSON 和 CSV"""
    json_path = os.path.join(out_dir, 'train_results.json')
    results_data = {
        'summary': summary,
        'history': history
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
        
    csv_path = os.path.join(out_dir, 'train_results.csv')
    if history:
        keys = history[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
    print(f"训练结果记录已保存至 {json_path} 及 {csv_path}")


def plot_training_results(history, summary, out_dir, filename='training_results.png'):
    """
    绘制并保存训练结果展示图像：
    包含 Loss 曲线、Top-1 准确率曲线、学习率曲线及模型参数/指标汇总面板
    """
    if not history:
        return
        
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    val_accs = [h['val_top1_acc'] for h in history]
    lrs = [h['lr'] for h in history]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 损失曲线 (Loss Curves)
    axs[0, 0].plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    axs[0, 0].plot(epochs, val_losses, label='Valid Loss', color='#ff7f0e', linewidth=2)
    
    # 修复 BUG: 处理模型崩溃导致 loss 为 NaN 的情况
    valid_val_losses = [v for v in val_losses if not np.isnan(v) and not np.isinf(v)]
    if valid_val_losses:
        min_loss_val = min(valid_val_losses)
        min_loss_idx = val_losses.index(min_loss_val)
        min_loss_epoch = epochs[min_loss_idx]
    else:
        min_loss_val, min_loss_epoch = 0.0, epochs[0]
    axs[0, 0].scatter([min_loss_epoch], [min_loss_val], color='red', s=80, zorder=5)
    axs[0, 0].annotate(f'Min Val Loss: {min_loss_val:.4f}\n(Epoch {min_loss_epoch})', 
                       xy=(min_loss_epoch, min_loss_val), 
                       xytext=(min_loss_epoch + 0.3, min_loss_val + 0.05),
                       arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))
    axs[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axs[0, 0].set_xlabel('Epoch', fontsize=12)
    axs[0, 0].set_ylabel('Loss', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend(fontsize=11)

    # 2. 验证集 Top-1 准确率曲线 (Val Top-1 Acc Curve)
    axs[0, 1].plot(epochs, val_accs, label='Valid Top-1 Acc', color='#2ca02c', linewidth=2)
    
    valid_accs = [v for v in val_accs if not np.isnan(v) and not np.isinf(v)]
    if valid_accs:
        best_acc_val = max(valid_accs)
        best_acc_idx = val_accs.index(best_acc_val)
        best_acc_epoch = epochs[best_acc_idx]
    else:
        best_acc_val, best_acc_epoch = 0.0, epochs[0]
    axs[0, 1].scatter([best_acc_epoch], [best_acc_val], color='gold', s=100, edgecolors='black', zorder=5)
    
    text_x_offset = best_acc_epoch - max(1, len(epochs)//4) if best_acc_epoch > len(epochs)//2 else best_acc_epoch + 0.5
    text_y_offset = best_acc_val - 5 if best_acc_val > 20 else best_acc_val + 5
    axs[0, 1].annotate(f'Best Top-1 Acc: {best_acc_val:.2f}%\n(Epoch {best_acc_epoch})', 
                       xy=(best_acc_epoch, best_acc_val), 
                       xytext=(text_x_offset, text_y_offset),
                       arrowprops=dict(facecolor='gold', shrink=0.05, width=1, headwidth=6))
    axs[0, 1].set_title('Validation Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    axs[0, 1].set_xlabel('Epoch', fontsize=12)
    axs[0, 1].set_ylabel('Top-1 Acc (%)', fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend(fontsize=11)

    # 3. 学习率曲线 (Learning Rate)
    axs[1, 0].plot(epochs, lrs, label='Learning Rate', color='#9467bd', linewidth=2, linestyle='-')
    axs[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axs[1, 0].set_xlabel('Epoch', fontsize=12)
    axs[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axs[1, 0].set_yscale('log')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend(fontsize=11)

    # 4. 模型参数与训练汇总仪表盘 (Summary Dashboard)
    axs[1, 1].axis('off')
    total_m = summary.get('total_params', 0) / 1e6
    trainable_m = summary.get('trainable_params', 0) / 1e6
    
    summary_text = (
        "======== Training & Parameter Summary ========\n\n"
        f"  • Model Name         : {summary.get('model_name', 'N/A')}\n"
        f"  • Total Params       : {total_m:.2f} M ({summary.get('total_params', 0):,} Params)\n"
        f"  • Trainable Params   : {trainable_m:.2f} M\n"
        f"  • Total Epochs       : {summary.get('total_epochs', len(epochs))}\n"
        f"  • Best Epoch         : Epoch {summary.get('best_epoch', 'N/A')}\n"
        f"  • Best Val Top-1 Acc : {summary.get('best_val_top1_acc', 0.0):.2f}%\n"
        f"  • Best Val Loss      : {summary.get('best_val_loss', 0.0):.4f}\n"
    )
    if summary.get('tta_val_top1_acc') is not None:
        summary_text += f"  • TTA Val Top-1 Acc  : {summary.get('tta_val_top1_acc'):.2f}%\n"
    
    summary_text += "\n=============================================="

    axs[1, 1].text(0.05, 0.92, summary_text, transform=axs[1, 1].transAxes,
                   fontsize=10.5, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=300)  # 移除 bbox_inches='tight' 防止异常画布扩增
    plt.close()
    print(f"训练结果展示图像已保存至 {save_path}")


def main():
    global LR
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # [核心优化] 学习率线性缩放规则 (Linear Scaling Rule)
    effective_batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS
    LR = LR * (effective_batch_size / 256.0)
    print(f"自动调整学习率 -> {LR:.2e} (有效Batch: {effective_batch_size})")
    
    # [核心优化] 加速卷积运算
    torch.backends.cudnn.benchmark = True

    # 1. 数据集准备
    train_loader, valid_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"加载了 {num_classes} 个类别。")

    # 配置 MixUp 和 CutMix
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, prob=1.0, switch_prob=0.5,
        mode='batch', label_smoothing=0.1, num_classes=num_classes
    )

    # 2. 构建模型
    print(f"正在加载 {MODEL_NAME} 预训练模型...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes, drop_path_rate=DROP_PATH_RATE)
    model = model.to(device)

    # 统计模型参数量
    total_params, trainable_params = count_parameters(model)
    print(f"模型总参数量: {total_params / 1e6:.2f} M ({total_params:,} Params), 可训练参数量: {trainable_params / 1e6:.2f} M")

    # 保存训练配置记录
    config = {
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'img_size': IMG_SIZE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'drop_path_rate': DROP_PATH_RATE,
        'mixup_alpha': MIXUP_ALPHA,
        'cutmix_alpha': CUTMIX_ALPHA,
        'warmup_epochs': WARMUP_EPOCHS,
        'ema_decay': EMA_DECAY,
        'grad_accum_steps': GRAD_ACCUM_STEPS,
        'layer_decay': LAYER_DECAY,
        'tta_enabled': TTA_ENABLED,
        'patience': PATIENCE,
        'data_dir': DATA_DIR,
        'out_dir': OUT_DIR,
        'num_classes': num_classes,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    save_config(OUT_DIR, config)

    # [核心优化] 引入 EMA
    print("初始化 Model EMA...")
    model_ema = ModelEMA(model, decay=EMA_DECAY)

    # 3. 损失函数与优化器
    train_loss_fn = SoftTargetCrossEntropy()
    valid_loss_fn = nn.CrossEntropyLoss()
    
    # [核心优化] 层级学习率衰减 (LLRD)
    print("配置层级学习率衰减 (LLRD)...")
    param_groups = get_layer_groups(model)
    optimizer = optim.AdamW(param_groups)
    
    # [核心优化] Warmup + Cosine Annealing
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    # 混合精度加速
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    best_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0  # Early Stopping 计数器
    history = []

    # 4. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = mixup_fn(inputs, targets)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = train_loss_fn(outputs, targets)
                loss = loss / GRAD_ACCUM_STEPS  # [核心优化] 梯度累积

            scaler.scale(loss).backward()

            # [核心优化] 每 GRAD_ACCUM_STEPS 步更新一次权重
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                # 更新 EMA 模型
                model_ema.update(model)

            train_loss += loss.item() * GRAD_ACCUM_STEPS * inputs.size(0)
            pbar.set_postfix({'loss': f"{loss.item() * GRAD_ACCUM_STEPS:.4f}", 
                              'lr': f"{optimizer.param_groups[-1]['lr']:.2e}"})
            
        train_loss = train_loss / len(train_loader.dataset)

        # 5. 验证循环 (使用 EMA 模型)
        model_ema.module.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets_list = []

        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model_ema.module(inputs)
                    loss = valid_loss_fn(outputs, targets)

                valid_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets_list.extend(targets.cpu().numpy())

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = 100. * correct / total
        current_lr = optimizer.param_groups[-1]['lr']

        # 记录本轮训练结果
        history.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'val_loss': round(valid_loss, 4),
            'val_top1_acc': round(valid_acc, 2),
            'lr': current_lr
        })

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        # 6. 保存最佳模型和混淆矩阵
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            best_val_loss = valid_loss
            epochs_no_improve = 0
            print(f"⭐ 发现最佳模型 (Acc: {best_acc:.2f}%)，正在保存...")
            
            # 保存结合权重、模型参数量、训练配置和结果记录的 Checkpoint
            checkpoint = {
                'model_name': MODEL_NAME,
                'state_dict': model_ema.module.state_dict(),
                'epoch': best_epoch,
                'best_val_top1_acc': best_acc,
                'best_val_loss': best_val_loss,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'config': config,
                'class_names': class_names
            }
            torch.save(checkpoint, os.path.join(OUT_DIR, f'best_{MODEL_NAME}.pth'))
            
            cm = confusion_matrix(all_targets_list, all_preds)
            plot_confusion_matrix(cm, class_names, OUT_DIR)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"⚠️ 连续 {PATIENCE} 轮无提升，触发 Early Stopping。")
                break

        # 实时更新结果记录与训练图表
        summary = {
            'model_name': MODEL_NAME,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_epochs': EPOCHS,
            'best_epoch': best_epoch,
            'best_val_top1_acc': best_acc,
            'best_val_loss': best_val_loss,
            'tta_val_top1_acc': None
        }
        save_results(OUT_DIR, history, summary)
        plot_training_results(history, summary, OUT_DIR)

    print(f"\n常规验证最高准确率: {best_acc:.2f}% (发生在 Epoch {best_epoch})")
    
    # 7. [核心优化] TTA 最终评估
    if TTA_ENABLED:
        print("\n正在使用 TTA (Test-Time Augmentation) 对最佳模型进行终极评估...")
        best_checkpoint = torch.load(os.path.join(OUT_DIR, f'best_{MODEL_NAME}.pth'), map_location=device)
        if isinstance(best_checkpoint, dict) and 'state_dict' in best_checkpoint:
            best_state = best_checkpoint['state_dict']
        else:
            best_state = best_checkpoint

        tta_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
        tta_model.load_state_dict(best_state)
        tta_model = tta_model.to(device)
        
        valid_dir = os.path.join(DATA_DIR, 'valid')
        tta_acc, tta_preds, tta_targets = validate_with_tta(tta_model, valid_dir, class_names, device)
        print(f"🏆 TTA 验证准确率: {tta_acc:.2f}% (常规验证: {best_acc:.2f}%)")
        
        cm_tta = confusion_matrix(tta_targets, tta_preds)
        plot_confusion_matrix(cm_tta, class_names, OUT_DIR, filename='confusion_matrix_tta.png')

        summary['tta_val_top1_acc'] = tta_acc
        save_results(OUT_DIR, history, summary)
        plot_training_results(history, summary, OUT_DIR)
    
    print(f"\n全部训练结束！最佳模型权重、配置文件、结果记录及训练展示图表已保存至目录: {OUT_DIR}/")

if __name__ == '__main__':
    main()

