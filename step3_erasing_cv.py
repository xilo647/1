# -*- coding: utf-8 -*-
"""思考题 2：把数据增强再加强（`RandomErasing`），观察置信度分布变化。

指导书第三步的 A/B/C 三组只跑了一次 80/20 划分，验证集仅 20 张——
1 张图 = 5%，任何小于 10% 的准确率差都落在抽样噪声里（报告 4.3 节已经点明这一点）。
本脚本把这个坑补上，做法是**多种子重复实验**：

    * 4 种增强配置：A 无增强 / B 基础 / C 强 / D = C + RandomErasing
    * 5 个数据划分种子：42, 43, 44, 45, 46
    * 每个 (配置, 种子) 组合训练 5 epoch，超参与 step3_augment.py 完全一致
    * 报告两套指标：
        1. 5 折准确率的 均值 ± 标准差（看**配置之间**的差异是否超过**折之间**的波动）
        2. 把 5×20 = 100 个验证样本**池化**后统计置信度分布（看增强对"自信程度"的影响）

RandomErasing 的定义复用 ``step3_augment.aug_transform("erase")``，
保证与主实验第三步的 D 组是同一套变换。

运行：``python step3_erasing_cv.py``

预计耗时：CPU 上约 5~6 分钟（4 配置 × 5 折 × 5 epoch）。
"""
from __future__ import annotations

import numpy as np

import common
from common import (DATA_DIR, get_loaders, print_header, save_json, set_seed,
                    tee_stdout, train_and_eval)
from step3_augment import aug_transform, build_resnet18_pretrained

CONFIGS = [
    ("none", "A. 无增强"),
    ("basic", "B. 基础增强(裁剪+翻转)"),
    ("strong", "C. 强增强(+色彩抖动+旋转)"),
    ("erase", "D. 强增强+RandomErasing"),
]
SEEDS = [42, 43, 44, 45, 46]
EPOCHS, LR, BATCH = 5, 1e-3, 16
BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def pooled_stats(probs, preds, labels):
    """把若干折的验证集预测池化后统计置信度分布。

    注意：argmax 出来的置信度天然 >= 0.5，所以分箱从 0.5 起。
    """
    conf = np.array([max(p) for p in probs], dtype=float)
    preds = np.array(preds)
    labels = np.array(labels)
    correct = preds == labels

    hist = {}
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == len(BINS) - 2 else (conf >= lo) & (conf < hi)
        hist[f"{lo:.1f}~{hi:.1f}"] = int(mask.sum())

    return {
        "n": int(conf.size),
        "acc": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "std_conf": float(conf.std(ddof=1)) if conf.size > 1 else 0.0,
        "median_conf": float(np.median(conf)),
        "mean_conf_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_conf_wrong": float(conf[~correct].mean()) if (~correct).any() else None,
        "conf_gt_0.9": int((conf > 0.9).sum()),
        "conf_lt_0.6": int((conf < 0.6).sum()),
        "hist": hist,
    }


def main():
    per_fold_acc = {k: [] for k, _ in CONFIGS}
    per_fold_time = {k: [] for k, _ in CONFIGS}
    pool = {k: {"probs": [], "preds": [], "labels": []} for k, _ in CONFIGS}

    print_header("思考题 2：RandomErasing 对置信度分布的影响（4 配置 × 5 种子 × 5 epoch）")
    for seed in SEEDS:
        print(f"\n---------- 数据划分种子 seed={seed} ----------")
        for key, title in CONFIGS:
            set_seed(seed)
            train_loader, val_loader = get_loaders(
                DATA_DIR, batch_size=BATCH, train_transform=aug_transform(key), seed=seed)
            net = build_resnet18_pretrained(num_classes=2, freeze=True)
            acc = train_and_eval(net, train_loader, val_loader, epochs=EPOCHS, lr=LR,
                                 device="cpu", verbose=False)
            run = common.LAST_RUN
            per_fold_acc[key].append(float(acc))
            per_fold_time[key].append(float(run["mean_epoch_time"]))
            pool[key]["probs"] += run["probs"]
            pool[key]["preds"] += run["preds"]
            pool[key]["labels"] += run["labels"]
            print(f"  {title:<26} 准确率 {acc:.4f} | 每轮 {run['mean_epoch_time']:.1f}s")

    summary = {}
    for key, title in CONFIGS:
        accs = np.array(per_fold_acc[key], dtype=float)
        stats = pooled_stats(pool[key]["probs"], pool[key]["preds"], pool[key]["labels"])
        summary[key] = {
            "title": title,
            "folds": SEEDS,
            "fold_acc": [float(a) for a in accs],
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std(ddof=1)),
            "acc_min": float(accs.min()),
            "acc_max": float(accs.max()),
            "epoch_time_mean": float(np.mean(per_fold_time[key])),
            "pooled": stats,
        }

    # ---------------- 表 1：5 折准确率 ----------------
    print_header("表 1　4 种增强配置的 5 折准确率（同超参，只改增强强度）")
    head = f"{'配置':<26}{'均值':>8}{'标准差':>9}{'最低':>8}{'最高':>8}{'每轮(s)':>10}   逐折准确率"
    print(head)
    print("-" * len(head))
    for key, title in CONFIGS:
        s = summary[key]
        folds = " ".join(f"{a:.2f}" for a in s["fold_acc"])
        print(f"{title:<26}{s['acc_mean']:>8.4f}{s['acc_std']:>9.4f}"
              f"{s['acc_min']:>8.2f}{s['acc_max']:>8.2f}{s['epoch_time_mean']:>10.1f}   {folds}")

    # ---------------- 表 2：池化置信度统计 ----------------
    print_header("表 2　池化 100 个验证样本的置信度统计（5 折 × 20 张）")
    head = (f"{'配置':<26}{'准确率':>8}{'平均置信度':>12}{'中位数':>9}"
            f"{'判对样本':>10}{'判错样本':>10}{'>0.9':>7}{'<0.6':>7}")
    print(head)
    print("-" * len(head))
    for key, title in CONFIGS:
        p = summary[key]["pooled"]
        mc = f"{p['mean_conf_correct']:.4f}" if p["mean_conf_correct"] is not None else "—"
        mw = f"{p['mean_conf_wrong']:.4f}" if p["mean_conf_wrong"] is not None else "—"
        print(f"{title:<26}{p['acc']:>8.4f}{p['mean_conf']:>12.4f}{p['median_conf']:>9.4f}"
              f"{mc:>10}{mw:>10}{p['conf_gt_0.9']:>7}{p['conf_lt_0.6']:>7}")

    # ---------------- 表 3：置信度分箱直方图 ----------------
    print_header("表 3　置信度分箱分布（样本数，n=100）")
    labels = [f"{BINS[i]:.1f}~{BINS[i+1]:.1f}" for i in range(len(BINS) - 1)]
    print(f"{'配置':<26}" + "".join(f"{lb:>10}" for lb in labels))
    print("-" * (26 + 10 * len(labels)))
    for key, title in CONFIGS:
        h = summary[key]["pooled"]["hist"]
        print(f"{title:<26}" + "".join(f"{h[lb]:>10d}" for lb in labels))

    # ---------------- 结论提示 ----------------
    print_header("自动对照（供报告引用）")
    none_acc = summary["none"]["acc_mean"]
    erase_acc = summary["erase"]["acc_mean"]
    fold_std = np.mean([summary[k]["acc_std"] for k, _ in CONFIGS])
    gap = erase_acc - none_acc
    print(f"A 无增强 5 折均值          : {none_acc:.4f}")
    print(f"D 加 RandomErasing 5 折均值 : {erase_acc:.4f}")
    print(f"两者差值                    : {gap:+.4f}")
    print(f"配置内折间标准差的平均值     : {fold_std:.4f}")
    print(f"差值是否超过折间波动         : {'是' if abs(gap) > fold_std else '否（差异不显著）'}")
    print(f"平均置信度 A -> D           : "
          f"{summary['none']['pooled']['mean_conf']:.4f} -> "
          f"{summary['erase']['pooled']['mean_conf']:.4f}")
    print(f"判对/判错置信度差 A         : "
          f"{summary['none']['pooled']['mean_conf_correct'] - summary['none']['pooled']['mean_conf_wrong']:+.4f}")
    print(f"判对/判错置信度差 D         : "
          f"{summary['erase']['pooled']['mean_conf_correct'] - summary['erase']['pooled']['mean_conf_wrong']:+.4f}")

    save_json(summary, "step3_erasing_cv.json")


if __name__ == "__main__":
    with tee_stdout("step3_erasing_cv"):
        main()
