# -*- coding: utf-8 -*-
"""
数据质量检查: 找出 Kaggle 中草药数据集中 "同图异标签" 的情况。

扫描 train/valid 全部图片, 按文件内容 MD5 分组:
  1. 同一张图出现在 不同类名 下 (同图异标签)
  2. 同一张图既在 train 又在 valid (跨切分泄漏), 进一步区分:
       - 同类别  (同图同标签 跨切分: 轻微泄漏/重复)
       - 不同类别 (同图异标签 跨切分: 严重标签冲突 + 评估失真)

输出:
  - duplicate_report.json  全部统计 + 典型例子
  - summary.txt            文本摘要
用法:
  /d/anaconda/envs/herbal/python.exe check_duplicates.py
"""
import os
import json
import hashlib
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'dataset_kaggle')


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


def collect():
    """返回 {md5: [(split, class_name, filepath), ...]}"""
    groups = defaultdict(list)
    for split in ('train', 'valid'):
        split_dir = os.path.join(DATA_DIR, split)
        class_dirs = sorted(os.listdir(split_dir))
        for cls in tqdm(class_dirs, desc=f'scanning {split}'):
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


def main():
    groups = collect()
    multi_class = []   # 同一 md5 出现在 >=2 个不同类名
    train_valid = []   # 同一 md5 同时出现在 train 与 valid
    tv_same_class = []  # 跨切分且类相同
    tv_diff_class = []  # 跨切分且类不同 (严重)

    for h, items in groups.items():
        if len(items) < 2:
            continue
        class_splits = set((sp, cl) for sp, cl, _ in items)
        classes = set(cl for _, cl, _ in items)
        splits = set(sp for sp, _, _ in items)

        if len(classes) >= 2:
            multi_class.append({'md5': h, 'num_files': len(items),
                                'locations': sorted(class_splits)})
        if 'train' in splits and 'valid' in splits:
            train_cls = [cl for sp, cl, _ in items if sp == 'train']
            valid_cls = [cl for sp, cl, _ in items if sp == 'valid']
            rec = {'md5': h, 'num_train': len(train_cls), 'num_valid': len(valid_cls),
                   'train_classes': sorted(set(train_cls)), 'valid_classes': sorted(set(valid_cls))}
            train_valid.append(rec)
            if set(train_cls) == set(valid_cls):
                tv_same_class.append(rec)
            else:
                tv_diff_class.append(rec)

    def _cls_of(items):
        return sorted(set(cl for _, cl, _ in items))

    # 统计受影响图片数
    affected_multi = sum(g['num_files'] for g in multi_class)
    affected_tv = sum(r['num_train'] + r['num_valid'] for r in train_valid)

    # 具体验证 ChatGPT 提到的几对类
    targets = ['丹参', '甘肃丹参', '丝瓜络', '粤丝瓜络', '七叶一枝花', '七叶莲']
    examples = []
    for g in multi_class:
        cls_here = _cls_of(groups[g['md5']])
        if any(t in cls_here for t in targets):
            examples.append({'md5': g['md5'], 'classes': cls_here,
                             'num_files': len(groups[g['md5']])})
    for r in tv_diff_class:
        if any(t in r['train_classes'] + r['valid_classes'] for t in targets):
            examples.append({'md5': r['md5'], 'train_classes': r['train_classes'],
                             'valid_classes': r['valid_classes'],
                             'num_train': r['num_train'], 'num_valid': r['num_valid']})

    report = {
        'total_unique_md5': len(groups),
        'total_images': sum(len(v) for v in groups.values()),
        'multi_class_groups': len(multi_class),
        'affected_images_multi_class': affected_multi,
        'train_valid_dup_groups': len(train_valid),
        'affected_images_train_valid': affected_tv,
        'tv_same_class_groups': len(tv_same_class),
        'tv_diff_class_groups': len(tv_diff_class),
        'examples_for_claimed_pairs': examples,
        'top_multi_class_groups': multi_class[:50],
        'top_tv_diff_class': tv_diff_class[:30],
    }
    with open(os.path.join(ROOT, 'duplicate_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append('===== 数据去重检查报告 =====')
    lines.append(f'图片总数: {report["total_images"]}  唯一MD5: {report["total_unique_md5"]}')
    lines.append(f'同图出现在>=2个不同类名: {report["multi_class_groups"]}组 / 影响 {affected_multi} 张图')
    lines.append(f'同图跨 train/valid: {report["train_valid_dup_groups"]}组 / 影响 {affected_tv} 张图')
    lines.append(f'  其中 同类别跨切分: {len(tv_same_class)}组')
    lines.append(f'  其中 不同类别跨切分(严重): {len(tv_diff_class)}组')
    lines.append('')
    lines.append('--- 典型例子 (ChatGPT 声称的几对) ---')
    for e in examples[:20]:
        lines.append(f"  {e}")
    lines.append('')
    lines.append('--- 同图异标签 Top 组 (跨类名) ---')
    for g in multi_class[:20]:
        lines.append(f"  {g['num_files']}张: {g['locations']}")
    lines.append('')
    lines.append('--- 跨切分且标签冲突 Top 组 ---')
    for r in tv_diff_class[:20]:
        lines.append(f"  train={r['train_classes']} valid={r['valid_classes']} (train {r['num_train']}张/valid {r['num_valid']}张)")
    with open(os.path.join(ROOT, 'duplicate_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
