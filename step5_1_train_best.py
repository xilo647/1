# -*- coding: utf-8 -*-
"""第五步（上）：训练时保存验证集最佳模型。

对应指导书 5.1 / 5.2 节。核心思想是"早停 + 只留历史最佳"：
逐 epoch 在验证集上评估，只有当准确率超过历史最佳时才写盘，
这样最终磁盘上的 ``best_model.pth`` 一定是泛化最好的那一版。

在指导书代码基础上补充：
* 记录每个 epoch 的准确率，输出"最佳出现在第几轮"；
* 同时保存最后一轮的权重 ``last_model.pth``，用来对比"最佳"与"最后"的差别。

运行：``python step5_1_train_best.py``
"""
from __future__ import annotations

import os
import time

import common  # noqa: F401  先导入以设定 TORCH_HOME
import torch
import torch.nn as nn
from torchvision import models

from common import (DATA_DIR, RESULT_DIR, count_params, count_trainable, get_loaders,
                    print_header, save_json, set_seed, tee_stdout)

WS = os.path.dirname(os.path.abspath(__file__))


def train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3,
                         save_path=None, device="cpu"):
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    best_acc = 0.0
    best_epoch = 0
    history = []
    t_start = time.time()
    for epoch in range(epochs):
        net.train()
        t0 = time.time()
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        epoch_time = time.time() - t0

        # 验证
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                correct += (net(X).argmax(1) == Y).sum().item()
                total += Y.size(0)
        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Val Acc: {acc:.4f} | "
              f"Train Loss: {running_loss/len(train_loader.dataset):.4f} | 用时: {epoch_time:.1f}s")
        history.append({"epoch": epoch + 1, "val_acc": float(acc),
                        "train_loss": float(running_loss / len(train_loader.dataset)),
                        "epoch_time": round(epoch_time, 2)})

        if acc > best_acc:                      # 只保留历史最佳
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(net.state_dict(), save_path)
            print(f"  ^ 最佳模型已保存到 {save_path}")
    total_time = time.time() - t_start
    print(f"最佳准确率: {best_acc:.4f}（出现在第 {best_epoch} 轮）| 总用时 {total_time:.1f}s")
    return {"best_acc": float(best_acc), "best_epoch": best_epoch, "history": history,
            "total_time": round(total_time, 2), "params": count_params(net)}


def main():
    set_seed(42)
    device = "cpu"
    epochs, lr, batch_size = 5, 1e-3, 16
    save_path = os.path.join(RESULT_DIR, "best_model.pth")
    last_path = os.path.join(RESULT_DIR, "last_model.pth")

    print_header("第五步：训练并在验证集最佳时保存模型（resnet18 预训练+冻结卷积）")
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True
    print(f"参数量 {count_params(net):,} / 可训练 {count_trainable(net):,}\n")

    train_loader, val_loader = get_loaders(DATA_DIR, batch_size=batch_size)
    info = train_with_best_save(net, train_loader, val_loader, epochs=epochs, lr=lr,
                                save_path=save_path, device=device)
    torch.save(net.state_dict(), last_path)     # 最后一轮（用于对比"最佳 vs 最后"）
    info["save_path"] = save_path
    info["last_path"] = last_path
    info["last_acc"] = info["history"][-1]["val_acc"]
    info["best_size_mb"] = round(os.path.getsize(save_path) / 1024 / 1024, 2)
    print(f"\n最佳模型: {save_path} ({info['best_size_mb']} MB, 验证准确率 {info['best_acc']:.4f})")
    print(f"最后一轮模型: {last_path} (验证准确率 {info['last_acc']:.4f})")
    save_json(info, "step5_best_model_info.json")


if __name__ == '__main__':
    with tee_stdout("step5_1_train_best"):
        main()
