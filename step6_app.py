# -*- coding: utf-8 -*-
"""第六步：封装成可拍照分类的小程序（Gradio）。

对应指导书第六步。部署三要素：加载权重 -> 接收用户输入（拍照/上传）-> 返回预测。

启动：``python step6_app.py``
  * 本机浏览器访问      http://127.0.0.1:7860
  * 手机连同一 Wi-Fi 访问 http://<本机IP>:7860 ，可直接调用手机摄像头拍照识别

可用环境变量覆盖默认值：
  CATDOG_WEIGHT   权重路径，默认 results/best_model.pth
  CATDOG_PORT     端口，默认 7860
"""
from __future__ import annotations

import os
import socket

import common  # noqa: F401  先导入以设定 TORCH_HOME
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

WS = os.path.dirname(os.path.abspath(__file__))
CLASSES = {0: "🐱 猫", 1: "🐶 狗"}
WEIGHT = os.environ.get("CATDOG_WEIGHT", os.path.join(WS, "results", "best_model.pth"))
PORT = int(os.environ.get("CATDOG_PORT", "7860"))

# ---------- 1. 加载模型（启动时只做一次） ----------
net = models.resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, 2)
net.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True))
net.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------- 2. 定义预测函数（gradio 每次拍照/上传都会调用） ----------
def predict(img: Image.Image):
    if img is None:
        return "请先拍照或上传一张图片", {}
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    conf, pred = probs.max(0)
    detail = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
    return f"识别结果：{CLASSES[int(pred)]}（置信度 {conf:.2%}）", detail


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def build_demo():
    import gradio as gr
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", sources=["webcam", "upload"], label="拍照或上传"),
        outputs=[gr.Textbox(label="识别结果"), gr.Label(label="各类别概率")],
        title="🐱🐶 猫狗分类器",
        description="基于 ResNet18 迁移学习训练。对准猫或狗拍照，或上传图片即可识别。",
        # 关掉 gradio 默认的"标记(flag)"按钮，否则它会在工作目录下建一个 flagged/
        allow_flagging="never",
    )


if __name__ == "__main__":
    print(f"权重: {WEIGHT}")
    print(f"本机访问: http://127.0.0.1:{PORT}")
    print(f"手机(同一局域网)访问: http://{local_ip()}:{PORT}")
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=PORT)
