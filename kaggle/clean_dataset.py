# -*- coding: utf-8 -*-
"""
Phase A: 数据清洗与无泄漏切分重建 (Kaggle 中草药 879 类)

输入:  dataset_kaggle/train + dataset_kaggle/valid   (当前脏切分, 全量 167017 张)
输出:  dataset_kaggle_clean/
         train/<类>/*         干净训练集 (无重复图, 无同图异标签, 无跨切分泄漏)
         valid/<类>/*         干净验证集 (同上)
         conflict_holdout/*   同图异标签冲突图 (隔离, 不参与训练/验证)
         clean_report.json    清洗统计
         aliases.json         同物异名 / 产地·炮制变体 alias 映射表 (交付物)

切分原则:
  - 按 MD5 分组, 同一内容的图为一个"图组";
  - 图组出现在 >=2 个不同类名 -> 判定为冲突, 整体隔离;
  - 其余图组按类划分, 类内以"图组"为单位做 90/10 切分 (保证同内容不跨切分);
  - 固定随机种子, 可复现。

用法:
  /d/anaconda/envs/herbal/python.exe clean_dataset.py
"""
import os
import json
import hashlib
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'dataset_kaggle')
DST = os.path.join(ROOT, 'dataset_kaggle_clean')
SEED = 42
TRAIN_RATIO = 0.9

# 同物异名 / 产地·炮制变体组 (与 alias_analysis.py 保持一致, 作为交付物)
CORE_ALIAS_GROUPS = [
    ['丹参', '甘肃丹参'], ['续断', '川续断'], ['丝瓜络', '粤丝瓜络'],
    ['淫羊藿', '巫山淫羊藿'], ['蔓荆子', '单叶蔓荆子'], ['络石藤', '广东络石藤'],
    ['海桐皮', '广东海桐皮'], ['刘寄奴', '北刘寄奴'], ['合欢花', '广东合欢花'],
    ['王不留行', '广东王不留行'], ['升麻', '广升麻'], ['大青叶', '蓼大青叶'],
    ['粉萆薢', '绵萆薢'], ['葛根', '粉葛'], ['肉桂', '桂皮'],
    ['土牛膝', '倒扣草'], ['石菖蒲', '九节菖蒲'], ['五加皮', '香加皮'],
]


def md5_of_file(path, chunk=1024 * 1024):
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def scan():
    """返回 {md5: [(split, class, filepath), ...]}"""
    groups = defaultdict(list)
    for split in ('train', 'valid'):
        split_dir = os.path.join(SRC, split)
        for cls in tqdm(sorted(os.listdir(split_dir)), desc=f'scan {split}'):
            d = os.path.join(split_dir, cls)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                fp = os.path.join(d, fn)
                h = md5_of_file(fp)
                if h:
                    groups[h].append((split, cls, fp))
    return groups


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    random.seed(SEED)
    for sub in ('train', 'valid'):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)
    conflict_dir = os.path.join(DST, 'conflict_holdout')
    os.makedirs(conflict_dir, exist_ok=True)

    groups = scan()
    print(f'[info] 扫描完成: {len(groups)} 个唯一内容, {sum(len(v) for v in groups.values())} 张图')

    conflict_manifest = []   # 冲突图: {file, classes, original_splits}
    clean_items = defaultdict(list)  # class -> [(md5, files)]
    conflict_count = 0

    for h, items in groups.items():
        classes = set(cl for _, cl, _ in items)
        if len(classes) >= 2:
            # 同图异标签 -> 隔离
            conflict_count += len(items)
            for i, (split, cls, fp) in enumerate(items):
                fn = os.path.basename(fp)
                dst = os.path.join(conflict_dir, f'c{conflict_count:06d}_{i}_{fn}')
                link(fp, dst)
                conflict_manifest.append({'md5': h, 'file': dst, 'classes': sorted(classes),
                                          'split': split, 'src_class': cls})
        else:
            cls = classes.pop()
            clean_items[cls].append((h, items))

    print(f'[info] 冲突图(同图异标签): {conflict_count} 张, 已隔离')

    # 类内按图组 90/10 切分
    train_total = valid_total = 0
    stats = {}
    dropped_classes = []
    singleton_classes = []
    for cls, item_list in tqdm(clean_items.items(), desc='split by group'):
        random.shuffle(item_list)
        n = len(item_list)
        if n == 1:
            # 唯一 1 张: 同时进入 train 与 valid, 保证 879 类在两集中齐全
            # (1 张图无法切分, 属被迫重复, 在报告中标注)
            singleton_classes.append(cls)
            train_groups = [item_list[0]]
            valid_groups = [item_list[0]]
        else:
            # 保证 valid 至少 1 个图组
            k = min(n - 1, max(1, int(round(n * TRAIN_RATIO))))
            train_groups, valid_groups = item_list[:k], item_list[k:]

        cls_train_dir = os.path.join(DST, 'train', cls)
        cls_valid_dir = os.path.join(DST, 'valid', cls)
        os.makedirs(cls_train_dir, exist_ok=True)
        os.makedirs(cls_valid_dir, exist_ok=True)

        for h, files in train_groups:
            for split, _, fp in files:
                link(fp, os.path.join(cls_train_dir, os.path.basename(fp)))
        for h, files in valid_groups:
            for split, _, fp in files:
                link(fp, os.path.join(cls_valid_dir, os.path.basename(fp)))

        n_train = sum(len(f) for _, f in train_groups)
        n_valid = sum(len(f) for _, f in valid_groups)
        train_total += n_train
        valid_total += n_valid
        stats[cls] = {'unique_groups': n, 'train_images': n_train, 'valid_images': n_valid}

        if n_valid == 0:
            dropped_classes.append(cls)

    # 写报告
    report = {
        'source': SRC,
        'seed': SEED,
        'train_ratio': TRAIN_RATIO,
        'unique_md5_total': len(groups),
        'conflict_images': conflict_count,
        'conflict_groups': len(conflict_manifest),
        'clean_train_images': train_total,
        'clean_valid_images': valid_total,
        'num_classes': len(clean_items),
        'classes_with_no_valid': dropped_classes,
        'singleton_duplicated_classes': singleton_classes,
        'per_class': stats,
    }
    with open(os.path.join(ROOT, 'clean_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, 'conflict_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(conflict_manifest, f, ensure_ascii=False, indent=1)
    # alias 映射表交付物: {"丹参": "丹参", "甘肃丹参": "丹参", ...}
    alias_map = {}
    for grp in CORE_ALIAS_GROUPS:
        canonical = grp[0]
        for name in grp:
            alias_map[name] = canonical
    with open(os.path.join(ROOT, 'aliases.json'), 'w', encoding='utf-8') as f:
        json.dump({'core_groups': CORE_ALIAS_GROUPS, 'canonical_map': alias_map},
                  f, ensure_ascii=False, indent=2)

    print('===== 清洗完成 =====')
    print(f'唯一内容: {len(groups)}  冲突图: {conflict_count} 张')
    print(f'干净训练集: {train_total} 张  干净验证集: {valid_total} 张')
    print(f'类别数: {len(clean_items)}  无验证样本的类: {dropped_classes}')
    print(f'输出目录: {DST}')
    print(f'报告: clean_report.json / conflict_manifest.json / aliases.json')


if __name__ == '__main__':
    main()
