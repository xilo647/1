# -*- coding: utf-8 -*-
"""第五步（下）：加载保存好的权重，做单张图片推理测试。

对应指导书 5.3 节。三种用法：

    python step5_2_test.py                                  # 默认 best_model.pth + datasets/cat.1.jpg
    python step5_2_test.py results/best_model.pth datasets/dog.5.jpg
    python step5_2_test.py results/best_model.pth --all     # 跑完整个 datasets/，输出全量结果

推理部分与训练完全解耦：这里只有 ``torchvision.models.resnet18(weights=None)``
（不下载权重）+ ``load_state_dict``，这也是工程上"训练与推理分离"的标准做法。
"""
from __future__ import annotations

import json
import os
import sys

import common  # noqa: F401  先导入以设定 TORCH_HOME
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common import DATA_DIR, RESULT_DIR, print_header, tee_stdout

WS = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["cat 猫", "dog 狗"]

DEFAULT_WEIGHT = os.path.join(RESULT_DIR, "best_model.pth")
DEFAULT_IMAGE = os.path.join(DATA_DIR, "cat.1.jpg")


def load_model(weight_path=DEFAULT_WEIGHT):
    net = models.resnet18(weights=None)   # 结构一致即可，不用下载权重
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict(image_path, net, transform=None):
    transform = transform or build_transform()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)              # 增加 batch 维 -> [1,3,224,224]
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]  # 得分转概率
    pred = probs.argmax().item()
    return CLASSES[pred], probs[pred].item(), probs


def test_all(net, transform):
    """在整个 datasets/ 上跑一遍，相当于用"训练时见过的验证集 + 训练集"全量回归测试。"""
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    rows, n_correct = [], 0
    for f in files:
        path = os.path.join(DATA_DIR, f)
        label, conf, probs = predict(path, net, transform)
        truth = "cat 猫" if f.lower().startswith("cat") else "dog 狗"
        ok = (label == truth)
        n_correct += ok
        rows.append({"file": f, "truth": truth, "pred": label,
                     "conf": round(float(conf), 4),
                     "p_cat": round(float(probs[0]), 4), "p_dog": round(float(probs[1]), 4),
                     "correct": bool(ok)})
    print_header("全量推理测试（100 张，含训练集与验证集）")
    for r in rows:
        flag = "OK " if r["correct"] else "ERR"
        print(f"  [{flag}] {r['file']:<14} 真值 {r['truth']} | 预测 {r['pred']} | "
              f"置信度 {r['conf']:.2%} (猫 {r['p_cat']:.2%} / 狗 {r['p_dog']:.2%})")
    acc = n_correct / len(rows)
    print(f"\n全量准确率: {acc:.4f} ({n_correct}/{len(rows)})")
    out = os.path.join(RESULT_DIR, "step5_all_predictions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "n": len(rows), "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"[已保存] {out}")
    return acc


def main(argv):
    weight_path = DEFAULT_WEIGHT
    args = [a for a in argv if a != "--all"]
    if args:
        weight_path = args[0]
    if not os.path.exists(weight_path):
        print(f"找不到权重文件 {weight_path}，请先运行 step5_1_train_best.py")
        return 1

    print_header("第五步：模型保存与单张图片推理测试")
    print(f"权重: {weight_path} ({os.path.getsize(weight_path)/1024/1024:.2f} MB)")
    net = load_model(weight_path)
    transform = build_transform()

    if "--all" in argv:
        test_all(net, transform)
        return 0

    path = args[1] if len(args) > 1 else DEFAULT_IMAGE
    label, conf, probs = predict(path, net, transform)
    print(f"图片: {path}")
    print(f"预测: {label}，置信度 {conf:.2%}")
    print(f"各类概率: 猫 {probs[0]:.2%} | 狗 {probs[1]:.2%}")
    return 0


if __name__ == '__main__':
    # 单张与全量分别留一份日志，互不覆盖
    _tag = "step5_2_test_all" if "--all" in sys.argv else "step5_2_test_single"
    with tee_stdout(_tag):
        sys.exit(main(sys.argv[1:]))
