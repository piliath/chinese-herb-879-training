import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    src_dir = r"Y:\中草药识别\kagglehub\competitions\chinese-medicine-image\train\train"
    dst_dir = r"Y:\中草药识别\dataset_kaggle"
    
    train_dst = os.path.join(dst_dir, 'train')
    valid_dst = os.path.join(dst_dir, 'valid')
    
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(valid_dst, exist_ok=True)
    
    if not os.path.exists(src_dir):
        print(f"错误: 找不到源数据集目录 {src_dir}")
        return

    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    print(f"找到 {len(classes)} 个类别，开始划分数据集...")
    
    total_train = 0
    total_valid = 0
    
    for cls in tqdm(classes, desc="处理类别"):
        cls_src_path = os.path.join(src_dir, cls)
        cls_train_path = os.path.join(train_dst, cls)
        cls_valid_path = os.path.join(valid_dst, cls)
        
        os.makedirs(cls_train_path, exist_ok=True)
        os.makedirs(cls_valid_path, exist_ok=True)
        
        files = [f for f in os.listdir(cls_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # 使用固定随机种子以保证多次运行划分结果一致
        random.seed(42)
        random.shuffle(files)
        
        split_idx = int(len(files) * 0.9)
        train_files = files[:split_idx]
        valid_files = files[split_idx:]
        
        for f in train_files:
            src_f = os.path.join(cls_src_path, f)
            dst_f = os.path.join(cls_train_path, f)
            if not os.path.exists(dst_f):
                try:
                    os.link(src_f, dst_f)
                except OSError:
                    shutil.copy2(src_f, dst_f)
                    
        for f in valid_files:
            src_f = os.path.join(cls_src_path, f)
            dst_f = os.path.join(cls_valid_path, f)
            if not os.path.exists(dst_f):
                try:
                    os.link(src_f, dst_f)
                except OSError:
                    shutil.copy2(src_f, dst_f)
                    
        total_train += len(train_files)
        total_valid += len(valid_files)
        
    print(f"数据集划分完成！")
    print(f"训练集图片数: {total_train}")
    print(f"验证集图片数: {total_valid}")
    print(f"新数据集已保存至: {dst_dir}")

if __name__ == '__main__':
    main()
