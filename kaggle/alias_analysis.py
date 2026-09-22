# -*- coding: utf-8 -*-
"""
同物异名/近义标签合并分析:
量化 "标签体系本身" 对准确率的损失。

读取诊断缓存的 preds/targets (由 diagnose_results.py 生成), 若把一组"同物异名/
产地/炮制变体"类合并视为同一类, 计算合并后的 Top-1 准确率提升。

用法:
  /d/anaconda/envs/herbal/python.exe alias_analysis.py
"""
import os
import json
import numpy as np
from torchvision import datasets

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'dataset_kaggle')

# 同物异名 / 近义 / 产地·炮制变体组
#  核心组: 语义上确属同物异名或产地/炮制变体
CORE_GROUPS = [
    ['丹参', '甘肃丹参'],
    ['续断', '川续断'],
    ['丝瓜络', '粤丝瓜络'],
    ['淫羊藿', '巫山淫羊藿'],
    ['蔓荆子', '单叶蔓荆子'],
    ['络石藤', '广东络石藤'],
    ['海桐皮', '广东海桐皮'],
    ['刘寄奴', '北刘寄奴'],
    ['合欢花', '广东合欢花'],
    ['王不留行', '广东王不留行'],
    ['升麻', '广升麻'],
    ['大青叶', '蓼大青叶'],
    ['粉萆薢', '绵萆薢'],
    ['葛根', '粉葛'],
    ['肉桂', '桂皮'],
    ['土牛膝', '倒扣草'],
    ['石菖蒲', '九节菖蒲'],
    ['五加皮', '香加皮'],
]
#  扩展组: 视觉极近但语义上算不同品种/来源 (合并仅作"可视化近邻"参考)
EXTRA_GROUPS = [
    ['玄参', '苦玄参'],
    ['川贝母', '平贝母'],
    ['大驳骨', '小驳骨'],
    ['金果榄', '青牛胆'],
    ['贯众', '绵马贯众'],
    ['莪术', '片姜黄'],
    ['牛膝', '川牛膝'],
]


def load(model):
    class_names = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid')).classes
    idx = {n: i for i, n in enumerate(class_names)}
    preds = np.load(os.path.join(ROOT, f'diagnosis_{model}', f'{model}_preds.npy'))
    targets = np.load(os.path.join(ROOT, f'diagnosis_{model}', f'{model}_targets.npy'))
    return class_names, idx, preds, targets


def merge_acc(class_names, idx, preds, targets, groups, group_name):
    """把每组类合并, 统计合并后的 top-1"""
    # 建分组 id
    group_of = {}
    for gid, grp in enumerate(groups):
        for name in grp:
            group_of[idx[name]] = gid
    mask = np.array([t in group_of for t in targets])
    # 合并后判断: 真实类属于某组时, 预测类也必须属于同一组才算对
    merged_correct = np.array([
        group_of.get(int(t), -1) == group_of.get(int(p), -2)
        if int(t) in group_of else (t == p)
        for t, p in zip(targets, preds)
    ])
    n = len(targets)
    raw_top1 = 100.0 * (preds == targets).mean()
    merged_top1 = 100.0 * merged_correct.mean()

    # 每组恢复的错误数 (误判为组内另一成员的数量)
    per_group = []
    for gid, grp in enumerate(groups):
        g_idx = set(idx[n] for n in grp)
        wrong_inside = sum(
            1 for t, p in zip(targets, preds)
            if int(t) in g_idx and int(p) in g_idx and t != p
        )
        per_group.append({'group': ' / '.join(grp), 'recovered_errors': wrong_inside})

    return {
        'scenario': group_name,
        'num_groups': len(groups),
        'num_classes_involved': len(group_of),
        'raw_top1': round(raw_top1, 2),
        'merged_top1': round(merged_top1, 2),
        'delta': round(merged_top1 - raw_top1, 2),
        'per_group': per_group,
    }


def main():
    all_results = {}
    for model in ('convnext', 'vit'):
        class_names, idx, preds, targets = load(model)
        core = merge_acc(class_names, idx, preds, targets, CORE_GROUPS, 'core')
        extra = merge_acc(class_names, idx, preds, targets, CORE_GROUPS + EXTRA_GROUPS, 'core+extra')
        all_results[model] = {'core': core, 'core_plus_extra': extra}
        # 打印摘要
        print(f'===== {model} =====')
        for sc in (core, extra):
            print(f"{sc['scenario']}: {sc['num_groups']}组/{sc['num_classes_involved']}类参与合并 -> "
                  f"Top-1 {sc['raw_top1']}% → {sc['merged_top1']}%  (Δ{sc['delta']:+.2f})")
        print('  核心组逐组可恢复错误数:')
        for g in core['per_group']:
            if g['recovered_errors']:
                print(f"    {g['group']}: {g['recovered_errors']}")
    with open(os.path.join(ROOT, 'alias_report.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
