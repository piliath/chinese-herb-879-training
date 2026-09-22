# -*- coding: utf-8 -*-
"""
汇总 Phase C 消融结果: 读取 runs_ablation/<config>/train_results.json
输出对比表 (纯文本, 无 matplotlib, 避免中文/字体问题)
用法: python summarize_ablations.py [--json out.json]
"""
import os
import json
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs_ablation')

CONFIG_DESC = {
    'base': 'baseline (weighted+strong aug+mixup1.0)',
    'no_weighted': 'no weighted sampler',
    'weak_mixup': 'mixup prob 1.0 -> 0.1',
    'weak_aug': 'RandAug m7->m4, Erase 0.2->0.05',
    'two_stage': 'two-stage (freeze 10 ep then full)',
    'logit_adjust': 'instance sampling + Logit Adjustment',
    'base_448': 'img 448',
}


def main():
    rows = []
    for name in sorted(os.listdir(OUT)):
        jp = os.path.join(OUT, name, 'train_results.json')
        if not os.path.exists(jp):
            continue
        with open(jp, encoding='utf-8') as f:
            r = json.load(f)
        hist = r.get('history', [])
        last = hist[-1] if hist else {}
        rows.append({
            'config': name,
            'desc': CONFIG_DESC.get(name, ''),
            'img_size': r.get('cfg', {}).get('img_size'),
            'best_val_top1': r.get('best_val_top1'),
            'best_epoch': r.get('best_epoch'),
            'final_val_top1': last.get('val_top1'),
            'final_val_top5': last.get('val_top5'),
            'num_epochs': len(hist),
        })
    rows.sort(key=lambda x: -(x['best_val_top1'] or 0))

    print('=' * 92)
    print(f"{'config':<14}{'desc':<42}{'img':<5}{'best@ep':<10}{'best%':<10}{'final%':<10}{'top5%':<8}")
    print('-' * 92)
    for r in rows:
        print(f"{r['config']:<14}{r['desc']:<42}{r['img_size']:<5}"
              f"{str(r['best_epoch']):<10}{str(r['best_val_top1']):<10}"
              f"{str(r['final_val_top1']):<10}{str(r['final_val_top5']):<8}")
    print('=' * 92)
    best = max(rows, key=lambda x: x['best_val_top1'] or 0)
    print(f"BEST: {best['config']} = {best['best_val_top1']}% @ epoch {best['best_epoch']}")

    if '--json' in sys.argv:
        i = sys.argv.index('--json')
        with open(sys.argv[i + 1], 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f'saved: {sys.argv[i+1]}')


if __name__ == '__main__':
    main()
