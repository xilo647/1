# -*- coding: utf-8 -*-
"""第一步：手动搭建 AlexNet，从零训练。

对应指导书 1.2 / 1.3 节。代码结构、超参（batch_size=16, lr=1e-4, epochs=30,
device='cpu'）与指导书一致，只做了三处必要改动：

* ``base_path`` 换成本仓库的 ``datasets/``（指导书里是老师电脑上的绝对路径）；
* 生成 train/test 索引文件前固定 ``random.seed(42)``，否则每次运行的 80/20
  划分都不同，无法与其他步骤对比（指导书"控制变量提醒"的要求）；
* 索引文件写到 ``results/`` 下，逐 epoch 指标存成 JSON，方便写报告。

另外，本模块顺带提供 ``ResNet18Manual``（手写 ResNet18，层名与 torchvision
完全一致），第三步的参考代码 ``from step1_scratch import ResNet18Manual``
可以直接运行。

运行：``python step1_scratch.py``
"""
from __future__ import annotations

import os
import random
import sys
import time

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import RESULT_DIR, count_params, count_trainable, print_header, save_json, set_seed, tee_stdout

WS = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# 1.2 模型定义（指导书原版）
# --------------------------------------------------------------------------- #
class AlexNet(nn.Module):
    def __init__(self):
        """初始化"""
        super().__init__()
        self.net = nn.Sequential(
            # 这里，我们使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 2))

    def forward(self, X):
        """前向传播"""
        return self.net(X)


# --------------------------------------------------------------------------- #
# 手写 ResNet18（供第三步使用；层名与 torchvision.models.resnet18 完全一致，
# 因此可以用 load_state_dict(official.state_dict(), strict=False) 灌入预训练权重）
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity          # 残差连接：梯度可以"抄近道"
        return self.relu(out)


class ResNet18Manual(nn.Module):
    """手写 ResNet18，结构与参数名对齐 torchvision.models.resnet18。"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# --------------------------------------------------------------------------- #
# 1.2 自定义数据集：从 txt 索引文件读 (路径, 标签)
# --------------------------------------------------------------------------- #
class DatasetLoader(Dataset):
    def __init__(self, dataset_path):
        self.images_path = []
        self.labels = []
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    self.images_path.append(parts[0])
                    self.labels.append(int(parts[1]))

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform(image)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        label = self.labels[index]
        image = self.preprocess_image(image_path)
        return image, label

    def __len__(self):
        return len(self.labels)


def dataloader(base_path, batch_size, seed=42):
    """划分数据集并生成 txt 索引文件（80% 训练 / 20% 测试）。

    注意：指导书原版没有固定 random.sample 的种子，这里固定为 seed=42，
    以满足"全程同一份数据划分"的控制变量要求。
    """
    imgs_abs_path = sorted(
        os.path.join(base_path, i) for i in os.listdir(base_path)
        if i.lower().endswith(('.jpg', '.jpeg', '.png')))
    random.seed(seed)
    test_size = int(0.2 * len(imgs_abs_path))
    test_files = set(random.sample(imgs_abs_path, test_size))

    os.makedirs(RESULT_DIR, exist_ok=True)
    train_txt = os.path.join(RESULT_DIR, "trainDatasets.txt")
    test_txt = os.path.join(RESULT_DIR, "testDatasets.txt")

    with open(test_txt, 'w', encoding='utf-8') as f1, \
            open(train_txt, 'w', encoding='utf-8') as f2:
        for i in imgs_abs_path:
            label = 0 if 'cat' in i.lower() else 1
            if i in test_files:
                f1.write(f"{i} {label}\n")
            else:
                f2.write(f"{i} {label}\n")

    TrainDataset = DatasetLoader(train_txt)
    TestDataset = DatasetLoader(test_txt)
    print(f"[数据] 训练 {len(TrainDataset)} 张, 测试 {len(TestDataset)} 张"
          f" (索引文件: {train_txt} / {test_txt})")
    return (DataLoader(TrainDataset, batch_size=batch_size, shuffle=True),
            DataLoader(TestDataset, batch_size=batch_size, shuffle=False))


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


def main():
    set_seed(42)
    batch_size = 16
    lr = 0.0001  # 从头训练自定义网络，建议使用较小的学习率
    epochs = 30
    device = 'cpu'

    print_header("第一步：手动搭建 AlexNet，从零训练（30 epoch, lr=1e-4）")

    # 4. 实例化自定义 AlexNet 模型
    net = AlexNet()
    net.apply(init_weight)
    net.to(device)
    print(f"参数量: {count_params(net):,}（全部可训练: {count_trainable(net):,}）")
    print(f"设备: {device}\n")

    loss_fn = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    # 5. 加载自定义数据集
    base_path = os.path.join(WS, "datasets")
    train_iter, test_iter = dataloader(base_path, batch_size=batch_size)

    history = []
    t_start = time.time()
    # 6. 模型训练循环
    print("开始训练...")
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

        # 7. 验证与评价指标计算
        net.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for X, Y in test_iter:
                X, Y = X.to(device), Y.to(device)
                y_hat = net(X)
                loss = loss_fn(y_hat, Y)
                val_loss += loss.item() * X.size(0)

                _, preds = torch.max(y_hat, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(Y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0)
        test_loss = val_loss / len(test_iter.dataset)

        print(f"EPOCH: {epoch+1}/{epochs}")
        print(f"  -> Test Loss: {test_loss:.4f}")
        print(f"  -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        print(f"  -> 用时: {epoch_time:.1f}s\n")

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "acc": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "epoch_time": round(epoch_time, 2),
        })

    total_time = time.time() - t_start
    last = history[-1]
    print("-" * 78)
    print(f"最终（第 {epochs} 轮）: 准确率 {last['acc']:.4f} | 精确率 {last['precision']:.4f} | "
          f"召回率 {last['recall']:.4f} | F1 {last['f1']:.4f}")
    print(f"30 轮总用时 {total_time:.1f}s（平均 {total_time/epochs:.1f}s/轮）")
    best = max(history, key=lambda h: h["acc"])
    print(f"过程中最好的一轮: 第 {best['epoch']} 轮, 准确率 {best['acc']:.4f}")

    torch.save(net.state_dict(), os.path.join(RESULT_DIR, "model_step1_alexnet_scratch.pth"))
    save_json({
        "history": history,
        "final": last,
        "best": best,
        "total_time": round(total_time, 2),
        "mean_epoch_time": round(total_time / epochs, 2),
        "params": count_params(net),
        "config": {"batch_size": batch_size, "lr": lr, "epochs": epochs, "device": device},
        "labels": [int(v) for v in all_labels],
        "preds": [int(v) for v in all_preds],
    }, "step1_scratch_history.json")
    return last


if __name__ == '__main__':
    with tee_stdout("step1_scratch"):
        main()
