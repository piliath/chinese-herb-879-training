import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# 训练参数配置 (全局变量，可直接修改)
# ==========================================
MODEL_NAME = 'yolov8n-cls.pt'         # YOLOv8 Nano 分类模型
OUT_DIR = 'runs_yolov8n'               # 输出目录
EPOCHS = 40                            # 训练轮数
BATCH_SIZE = 64                        # 批次大小
IMG_SIZE = 384                         # 图像尺寸
LR0 = 0.001                            # 初始学习率
WEIGHT_DECAY = 0.05                    # 权重衰减
DROPOUT = 0.1                          # Dropout 概率
MIXUP = 0.2                            # MixUp 增强概率
DATA_DIR = os.path.abspath('dataset_kaggle')  # 数据集路径
# ==========================================

epoch_history = []

def on_fit_epoch_end(trainer):
    """Ultralytics 回调: 每轮训练+验证结束后自动采集指标"""
    try:
        epoch = trainer.epoch + 1
        
        # 提取训练 loss
        train_loss = 0.0
        if hasattr(trainer, 'loss') and trainer.loss is not None:
            tl = trainer.loss
            train_loss = tl.item() if hasattr(tl, 'item') else float(tl)
        elif hasattr(trainer, 'tloss') and trainer.tloss is not None:
            tl = trainer.tloss
            train_loss = tl.item() if hasattr(tl, 'item') else float(tl)
        
        # 提取验证 loss
        val_loss = 0.0
        if hasattr(trainer, 'validator') and trainer.validator is not None:
            v = trainer.validator
            if hasattr(v, 'loss') and v.loss is not None:
                vl = v.loss
                val_loss = vl.item() if hasattr(vl, 'item') else float(vl)
        
        # 提取 Top-1 准确率
        val_top1_acc = 0.0
        metrics = trainer.metrics if hasattr(trainer, 'metrics') and trainer.metrics else {}
        for key in ['metrics/accuracy_top1', 'accuracy_top1', 'top1', 'fitness']:
            if key in metrics:
                acc = float(metrics[key])
                val_top1_acc = acc * 100.0 if acc <= 1.0 else acc
                break
        
        if val_top1_acc == 0.0 and hasattr(trainer, 'validator') and trainer.validator is not None:
            vm = trainer.validator.metrics if hasattr(trainer.validator, 'metrics') else None
            if vm is not None and hasattr(vm, 'top1') and vm.top1 is not None:
                acc = float(vm.top1)
                val_top1_acc = acc * 100.0 if acc <= 1.0 else acc
        
        # 提取学习率
        lr_val = 0.0
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            lr_val = trainer.optimizer.param_groups[0].get('lr', 0.0)
        
        record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_top1_acc': round(val_top1_acc, 2),
            'lr': lr_val
        }
        epoch_history.append(record)
        print(f"  [Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1 Acc: {val_top1_acc:.2f}%")
    except Exception as e:
        print(f"  [Epoch 回调采集异常]: {e}")

def count_parameters(model):
    """统计模型参数量"""
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        return total_params, trainable_params
    except Exception:
        return 0, 0

def save_config(out_dir, config_dict):
    """保存训练配置 config.json"""
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f"训练配置已保存至: {config_path}")

def save_results(out_dir, history, summary):
    """保存训练结果记录 train_results.json 和 train_results.csv"""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'train_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'history': history}, f, indent=4, ensure_ascii=False)
        
    csv_path = os.path.join(out_dir, 'train_results.csv')
    if history:
        keys = history[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
    print(f"训练结果记录已保存至: {json_path} 及 {csv_path}")

def plot_training_results(history, summary, out_dir, filename='training_results.png'):
    """绘制单模型训练结果展示仪表盘图表"""
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

    # 1. Loss
    axs[0, 0].plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    axs[0, 0].plot(epochs, val_losses, label='Valid Loss', color='#ff7f0e', linewidth=2)
    axs[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axs[0, 0].set_xlabel('Epoch', fontsize=12)
    axs[0, 0].set_ylabel('Loss', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend(fontsize=11)

    # 2. Accuracy
    axs[0, 1].plot(epochs, val_accs, label='Valid Top-1 Acc', color='#2ca02c', linewidth=2)
    best_acc_idx = np.argmax(val_accs)
    axs[0, 1].scatter([epochs[best_acc_idx]], [val_accs[best_acc_idx]], color='gold', s=100, edgecolors='black', zorder=5)
    axs[0, 1].set_title('Validation Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    axs[0, 1].set_xlabel('Epoch', fontsize=12)
    axs[0, 1].set_ylabel('Top-1 Acc (%)', fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend(fontsize=11)

    # 3. Learning Rate
    axs[1, 0].plot(epochs, lrs, label='Learning Rate', color='#9467bd', linewidth=2)
    axs[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axs[1, 0].set_xlabel('Epoch', fontsize=12)
    axs[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend(fontsize=11)

    # 4. Summary Dashboard
    axs[1, 1].axis('off')
    total_m = summary.get('total_params', 0) / 1e6
    summary_text = (
        "======== Training & Parameter Summary ========\n\n"
        f"  • Model Name         : {summary.get('model_name', 'N/A')}\n"
        f"  • Total Params       : {total_m:.2f} M ({summary.get('total_params', 0):,} Params)\n"
        f"  • Total Epochs       : {summary.get('total_epochs', len(epochs))}\n"
        f"  • Best Epoch         : Epoch {summary.get('best_epoch', 'N/A')}\n"
        f"  • Best Val Top-1 Acc : {summary.get('best_val_top1_acc', 0.0):.2f}%\n"
        f"  • Best Val Loss      : {summary.get('best_val_loss', 0.0):.4f}\n"
        "=============================================="
    )
    axs[1, 1].text(0.05, 0.92, summary_text, transform=axs[1, 1].transAxes,
                   fontsize=10.5, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"单模型展示图表已保存至: {save_path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL_NAME)
    
    total_params, trainable_params = count_parameters(model)
    print(f"正在准备训练 {MODEL_NAME} | 总参数量: {total_params / 1e6:.2f} M")

    config = {
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'imgsz': IMG_SIZE,
        'batch': BATCH_SIZE,
        'optimizer': 'AdamW',
        'lr0': LR0,
        'weight_decay': WEIGHT_DECAY,
        'dropout': DROPOUT,
        'mixup': MIXUP,
        'auto_augment': 'randaugment',
        'out_dir': OUT_DIR,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    save_config(OUT_DIR, config)

    global epoch_history
    epoch_history = []
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

    print(f"开始训练 {MODEL_NAME}...")
    results = model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        optimizer='AdamW',
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        mixup=MIXUP,
        auto_augment='randaugment',
        cos_lr=True,
        label_smoothing=0.1,
        save=True,
        val=True,
        project='runs_yolov8n_project',
        name='yolov8n_run',
        exist_ok=True
    )

    history = list(epoch_history)
    if history:
        best_acc_idx = np.argmax([h['val_top1_acc'] for h in history])
        best_item = history[best_acc_idx]
        best_epoch = best_item['epoch']
        best_acc = best_item['val_top1_acc']
        best_loss = best_item['val_loss']
    else:
        best_epoch, best_acc, best_loss = 0, 0.0, 0.0

    summary = {
        'model_name': 'yolov8n-cls',
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_epochs': len(history) if history else EPOCHS,
        'best_epoch': best_epoch,
        'best_val_top1_acc': best_acc,
        'best_val_loss': best_loss
    }

    save_results(OUT_DIR, history, summary)
    plot_training_results(history, summary, OUT_DIR)
    print(f"\n✅ {MODEL_NAME} 训练结束！模型权重、结果 JSON/CSV 及展示图表已全部保存至 {OUT_DIR}/")

if __name__ == '__main__':
    main()
