# -*- coding: utf-8 -*-
"""第四步：调用 torchvision.models 封装好的模型横向对比。

对应指导书 4.1 / 4.2 节。三个模型都是"ImageNet 预训练 + 冻结特征层 +
只训练分类头"，同样的数据划分（seed=42）、同样 5 epoch / lr=1e-3 / batch=16，
记录准确率、每轮训练用时、模型参数量与磁盘体积，为第六步部署选型做铺垫。

运行：``python step4_compare.py``
"""
from __future__ import annotations

import os
import time

import common  # noqa: F401  先导入以设定 TORCH_HOME
import torch
import torch.nn as nn
from torchvision import models

from common import (DATA_DIR, RESULT_DIR, count_params, count_trainable, get_loaders,
                    print_header, save_json, set_seed, tee_stdout, train_and_eval)

WS = os.path.dirname(os.path.abspath(__file__))
MODELS = ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]


def build_model(name, num_classes=2, freeze=True):
    """统一接口：加载预训练模型并替换分类头"""
    constructors = {
        "resnet18":            lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "mobilenet_v3_small":  lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "shufflenet_v2_x1_0":  lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
        "efficientnet_b0":     lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    }
    net = constructors[name]()

    # 找到并替换分类头（不同模型的头名字不一样）
    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "mobilenet_v3_small":
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head_params = net.classifier[3].parameters()
    elif name == "shufflenet_v2_x1_0":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "efficientnet_b0":
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        head_params = net.classifier[1].parameters()

    if freeze:  # 冻结特征提取层，只训练分类头（CPU 友好）
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return net


def main():
    epochs, lr, batch_size, device = 5, 1e-3, 16, "cpu"
    train_loader, val_loader = get_loaders(DATA_DIR, batch_size=batch_size)
    results = {}
    for name in MODELS:
        set_seed(42)
        print_header(f"第四步：模型 {name}（预训练+冻结卷积, epochs={epochs}, lr={lr}）")
        net = build_model(name)
        print(f"参数量 {count_params(net):,} / 可训练 {count_trainable(net):,}")
        acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr,
                             device=device, tag=f"step4_{name}")
        run = dict(common.LAST_RUN)
        save_path = os.path.join(RESULT_DIR, f"model_{name}.pth")   # 顺手保存，第五步/第六步要用
        torch.save(net.state_dict(), save_path)
        run["model"] = name
        run["params"] = count_params(net)
        run["file_size_mb"] = round(os.path.getsize(save_path) / 1024 / 1024, 2)
        results[name] = run
        print(f"模型已保存: {save_path} ({run['file_size_mb']} MB)\n")

    print_header("第四步 横向对比结果（同一个 seed=42 的 80/20 划分）")
    print(f"{'模型':<22}{'参数量':>12}{'准确率':>9}{'精确率':>9}{'召回率':>9}"
          f"{'F1':>8}{'每轮(s)':>9}{'总用时(s)':>10}{'体积(MB)':>10}")
    for name, _ in sorted(results.items(), key=lambda x: -x[1]["final"]["acc"]):
        r = results[name]
        f = r["final"]
        print(f"{name:<22}{r['params']:>12,}{f['acc']:>9.4f}{f['precision']:>9.4f}"
              f"{f['recall']:>9.4f}{f['f1']:>8.4f}{r['mean_epoch_time']:>9.1f}"
              f"{r['total_time']:>10.1f}{r['file_size_mb']:>10.2f}")

    save_json({k: {kk: vv for kk, vv in v.items() if kk not in ("labels", "preds", "probs")}
               for k, v in results.items()}, "step4_summary.json")


if __name__ == '__main__':
    with tee_stdout("step4_compare"):
        main()
