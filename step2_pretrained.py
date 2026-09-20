# -*- coding: utf-8 -*-
"""第二步：在手动模型基础上加载 ImageNet 预训练参数（迁移学习）。

对应指导书 2.1 / 2.2 节：

* 做法一（默认）：冻结卷积层，只训练分类头（classifier）；
* 做法二（进阶）：解冻全部层，用小学习率 1e-5 微调，对比准确率与训练时间。

数据划分沿用第一步生成的同一份 txt 索引（seed=42），保证与第一步可比。

运行：``python step2_pretrained.py``
"""
from __future__ import annotations

import os
import time

import common  # noqa: F401  —— 先导入 common，它会设定 TORCH_HOME（预训练权重缓存目录）
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import models

from common import RESULT_DIR, count_params, count_trainable, print_header, save_json, set_seed, tee_stdout
from step1_scratch import dataloader

WS = os.path.dirname(os.path.abspath(__file__))


class AlexNetManual(nn.Module):
    """结构、层名与 torchvision.models.alexnet 完全一致，方便直接灌入官方权重。"""

    def __init__(self, num_classes=2):
        super().__init__()
        # 卷积部分必须命名为 features，且层顺序与 torchvision 一致
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 分类部分必须命名为 classifier，索引 6 是最后一层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),   # 索引 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # 索引 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # 索引 6 <- 换这里
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_pretrained_alexnet(num_classes=2, freeze_conv=True):
    """加载官方 AlexNet 权重，把分类头换成 num_classes 类。"""
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)          # 1. 先建 1000 类的完整结构
    net.load_state_dict(official.state_dict())          # 2. 此时形状完全一致，strict=True 也能加载
    net.classifier[6] = nn.Linear(4096, num_classes)   # 3. 再把分类头换成 2 类（新头自动随机初始化）

    if freeze_conv:
        # 冻结：只训练 classifier
        for name, param in net.named_parameters():
            param.requires_grad = name.startswith("classifier")
    return net


def train_loop(net, train_iter, test_iter, epochs, lr, device, title):
    """指导书第一步的训练循环，抽成函数供两步（冻结 / 微调）复用。"""
    loss_fn = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    print_header(title)
    print(f"参数量 {count_params(net):,} / 可训练 {count_trainable(net):,} | "
          f"epochs={epochs}, lr={lr}, device={device}\n")

    history = []
    t_start = time.time()
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        t0 = time.time()
        for X, Y in train_iter:
            X, Y = X.to(device), Y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, Y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            running_loss += l.item() * X.size(0)
        epoch_time = time.time() - t0
        train_loss = running_loss / len(train_iter.dataset)

        net.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in test_iter:
                X, Y = X.to(device), Y.to(device)
                y_hat = net(X)
                val_loss += loss_fn(y_hat, Y).item() * X.size(0)
                all_preds.extend(y_hat.argmax(1).cpu().numpy())
                all_labels.extend(Y.cpu().numpy())
        test_loss = val_loss / len(test_iter.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0)

        print(f"EPOCH: {epoch+1}/{epochs}")
        print(f"  -> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"  -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | F1-Score: {f1:.4f} | 用时: {epoch_time:.1f}s\n")

        history.append({
            "epoch": epoch + 1, "train_loss": float(train_loss), "test_loss": float(test_loss),
            "acc": float(accuracy), "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "epoch_time": round(epoch_time, 2),
        })

    total_time = time.time() - t_start
    last = history[-1]
    print(f"[{title}] 最终准确率 {last['acc']:.4f} | 总用时 {total_time:.1f}s "
          f"(平均 {total_time/epochs:.1f}s/轮)")
    return {
        "history": history,
        "final": last,
        "best": max(history, key=lambda h: h["acc"]),
        "total_time": round(total_time, 2),
        "mean_epoch_time": round(total_time / epochs, 2),
        "params": count_params(net),
        "trainable_params": count_trainable(net),
        "config": {"epochs": epochs, "lr": lr, "device": device},
        "labels": [int(v) for v in all_labels],
        "preds": [int(v) for v in all_preds],
    }


def main():
    set_seed(42)
    device = "cpu"
    base_path = os.path.join(WS, "datasets")

    # 与第一步完全相同的划分
    train_iter, test_iter = dataloader(base_path, batch_size=16, seed=42)

    # ---------------- 做法一：冻结卷积层，只训练分类头 ----------------
    net = build_pretrained_alexnet(num_classes=2, freeze_conv=True).to(device)
    frozen = train_loop(net, train_iter, test_iter, epochs=30, lr=1e-4, device=device,
                        title="第二步-做法一：ImageNet 预训练 + 冻结卷积层，只训练分类头（30 epoch, lr=1e-4）")
    torch.save(net.state_dict(), os.path.join(RESULT_DIR, "model_step2_alexnet_frozen.pth"))
    save_json(frozen, "step2_frozen_history.json")

    # ---------------- 做法二（进阶）：解冻全部层，小学习率微调 ----------------
    set_seed(42)
    net2 = build_pretrained_alexnet(num_classes=2, freeze_conv=False).to(device)
    finetuned = train_loop(net2, train_iter, test_iter, epochs=5, lr=1e-5, device=device,
                           title="第二步-做法二（进阶）：解冻全部层微调（5 epoch, lr=1e-5）")
    torch.save(net2.state_dict(), os.path.join(RESULT_DIR, "model_step2_alexnet_finetune.pth"))
    save_json(finetuned, "step2_finetune_history.json")

    print_header("冻结 vs 微调 对比")
    print(f"{'方案':<24}{'准确率':>8}{'精确率':>10}{'召回率':>10}{'F1':>8}{'总用时(s)':>12}{'每轮(s)':>10}")
    for name, r in [("冻结卷积(30轮)", frozen), ("全部微调(5轮)", finetuned)]:
        f = r["final"]
        print(f"{name:<24}{f['acc']:>8.4f}{f['precision']:>10.4f}{f['recall']:>10.4f}"
              f"{f['f1']:>8.4f}{r['total_time']:>12.1f}{r['mean_epoch_time']:>10.1f}")
    save_json({"frozen": {k: v for k, v in frozen.items() if k != "history"},
               "finetuned": {k: v for k, v in finetuned.items() if k != "history"}},
              "step2_summary.json")


if __name__ == '__main__':
    with tee_stdout("step2_pretrained"):
        main()
