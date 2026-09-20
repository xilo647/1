# -*- coding: utf-8 -*-
"""第三步：加入数据增强（三个实验横向对比）。

对应指导书第三步。指导书的参考代码给出了"预训练 + 数据增强"的一种配置，
标题要求的是**三个实验横向对比**，因此这里固定其他所有变量（同一份 80/20
划分 seed=42、同一个手写 ResNet18 + 同一份 ImageNet 预训练权重、同样的
5 epoch / lr=1e-3 / 冻结卷积只训 fc），只改"训练集增强强度"这一个变量：

    A. 无增强      —— 只 Resize(224,224)
    B. 基础增强    —— RandomResizedCrop + RandomHorizontalFlip
    C. 强增强      —— B + ColorJitter + RandomRotation(10)
    D. 强增强+擦除 —— C + RandomErasing（思考题 2 追加的第四组）

验证集始终只用干净的 Resize + Normalize。

注意：20 张验证集上的差异落在统计噪声里，单次划分不足以判断增强的好坏，
多种子重复实验见 ``step3_erasing_cv.py``。

运行：``python step3_augment.py``
"""
from __future__ import annotations

import os

import common  # noqa: F401  先导入以设定 TORCH_HOME
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from common import (DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, RESULT_DIR, count_params,
                    count_trainable, get_loaders, print_header, save_json, set_seed,
                    tee_stdout, train_and_eval)
from step1_scratch import ResNet18Manual

WS = os.path.dirname(os.path.abspath(__file__))


def aug_transform(strength: str):
    """按增强强度构造训练集 transform。"""
    if strength == "none":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if strength == "basic":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机裁剪缩放
            transforms.RandomHorizontalFlip(),                    # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if strength == "strong":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),                # 随机亮度/对比度/饱和度
            transforms.RandomRotation(10),                        # 随机小角度旋转
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if strength == "erase":
        # 思考题 2：在 C 组基础上再加 RandomErasing（随机遮挡一块矩形区域）。
        # 两个必须注意的点：
        #   1) RandomErasing 只能作用在 **Tensor** 上，所以必须排在 ToTensor 之后；
        #   2) 这里排在 Normalize 之后，因此 value=0 填充的是"数据集均值色"而不是黑色，
        #      相当于挖掉一块中性的灰斑，不会额外引入"黑色矩形"这种新伪影。
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])
    raise ValueError(strength)


def build_resnet18_pretrained(num_classes=2, freeze=True):
    """手写 ResNet18 + 官方预训练权重，只训练 fc。

    注意：指导书参考代码写的是
        net = ResNet18Manual(num_classes=2)
        net.load_state_dict(official.state_dict(), strict=False)
    这条语句实际会报错——``strict=False`` 只容忍"多键/少键"，不容忍形状不匹配，
    而官方权重的 ``fc.weight`` 是 [1000,512]。所以这里按指导书第二步处理 AlexNet
    的同一套路改成本仓库的写法：先建 1000 类结构灌权重，再换分类头。
    """
    net = ResNet18Manual(num_classes=1000)
    official = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.load_state_dict(official.state_dict(), strict=True)   # 此时形状完全一致
    net.fc = nn.Linear(net.fc.in_features, num_classes)       # 换成 2 类分类头
    if freeze:
        for name, param in net.named_parameters():
            param.requires_grad = 'fc' in name
    return net


def confidence_stats(probs, preds, labels):
    """统计置信度分布，用于观察增强对"过度自信"的影响。"""
    conf = [max(p) for p in probs]
    correct = [c for c, y, p in zip(conf, labels, preds) if y == p]
    wrong = [c for c, y, p in zip(conf, labels, preds) if y != p]
    return {
        "mean_confidence": float(np.mean(conf)),
        "mean_confidence_correct": float(np.mean(correct)) if correct else None,
        "mean_confidence_wrong": float(np.mean(wrong)) if wrong else None,
        "count_conf_gt_0.9": int(sum(1 for c in conf if c > 0.9)),
        "count_total": len(conf),
    }


def main():
    device = "cpu"
    epochs, lr, batch_size = 5, 1e-3, 16
    configs = [("none", "A. 无增强"), ("basic", "B. 基础增强(裁剪+翻转)"),
               ("strong", "C. 强增强(+色彩抖动+旋转)"),
               ("erase", "D. 强增强+RandomErasing(思考题2)")]

    results = {}
    for key, title in configs:
        set_seed(42)
        train_loader, val_loader = get_loaders(DATA_DIR, batch_size=batch_size,
                                               train_transform=aug_transform(key))
        print_header(f"第三步：{title}（epochs={epochs}, lr={lr}, 预训练 ResNet18 冻结卷积）")
        net = build_resnet18_pretrained(num_classes=2, freeze=True)
        print(f"参数量 {count_params(net):,} / 可训练 {count_trainable(net):,}\n")
        acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr,
                             device=device, tag=f"step3_{key}")
        run = dict(common.LAST_RUN)
        run["confidence"] = confidence_stats(run["probs"], run["preds"], run["labels"])
        run["augmentation"] = key
        run["title"] = title
        results[key] = run
        torch.save(net.state_dict(), os.path.join(RESULT_DIR, f"model_step3_{key}.pth"))
        print()

    print_header("第三步 横向对比（同一个 seed=42 的 80/20 划分，只改增强）")
    print(f"{'配置':<28}{'准确率':>8}{'精确率':>10}{'召回率':>10}{'F1':>8}"
          f"{'每轮(s)':>10}{'平均置信度':>12}{'>0.9 的样本':>12}")
    for key, title in configs:
        r = results[key]
        f, c = r["final"], r["confidence"]
        print(f"{title:<28}{f['acc']:>8.4f}{f['precision']:>10.4f}{f['recall']:>10.4f}"
              f"{f['f1']:>8.4f}{r['mean_epoch_time']:>10.1f}{c['mean_confidence']:>12.4f}"
              f"{c['count_conf_gt_0.9']:>8d}/{c['count_total']:<3d}")

    save_json({k: {kk: vv for kk, vv in v.items() if kk not in ("labels", "preds", "probs")}
               for k, v in results.items()}, "step3_summary.json")


if __name__ == '__main__':
    with tee_stdout("step3_augment"):
        main()
