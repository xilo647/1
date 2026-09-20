# -*- coding: utf-8 -*-
"""思考题 3（加分项）：把 predict 封装成 HTTP 接口，供微信小程序等前端调用。

FastAPI 版推理服务，和第六步的 Gradio 界面共用同一份 ``predict`` 逻辑：

启动：``python step6_api.py``          默认 http://127.0.0.1:8000
接口：
    GET  /health          -> {"status": "ok"}
    POST /predict         -> multipart/form-data，字段名 file；返回 JSON
    GET  /                -> 内置的极简测试页（可用手机浏览器上传/拍照）

微信小程序端只要
    wx.uploadFile({url: 'http://<电脑IP>:8000/predict', filePath, name: 'file', ...})
即可，与这里完全一致。
"""
from __future__ import annotations

import io
import os

import common  # noqa: F401  先导入以设定 TORCH_HOME
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import models, transforms

WS = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.environ.get("CATDOG_WEIGHT", os.path.join(WS, "results", "best_model.pth"))
PORT = int(os.environ.get("CATDOG_API_PORT", "8000"))
CLASSES = ["cat 猫", "dog 狗"]

net = models.resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, 2)
net.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True))
net.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="猫狗分类 API")

PAGE = """<!doctype html><meta charset="utf-8"><title>猫狗分类 API 测试</title>
<h3>🐱🐶 猫狗分类 API 测试</h3>
<input type="file" id="f" accept="image/*" capture="environment">
<button onclick="go()">识别</button><pre id="out"></pre>
<script>
async function go(){
  const f=document.getElementById('f').files[0]; if(!f){alert('先选一张图片');return;}
  const fd=new FormData(); fd.append('file', f);
  const r=await fetch('/predict',{method:'POST',body:fd}); const j=await r.json();
  document.getElementById('out').textContent=JSON.stringify(j,null,2);
}
</script>"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    pred = int(probs.argmax().item())
    return {
        "label": CLASSES[pred],
        "label_id": pred,
        "confidence": round(float(probs[pred]), 6),
        "probs": {"cat 猫": round(float(probs[0]), 6), "dog 狗": round(float(probs[1]), 6)},
    }


if __name__ == "__main__":
    import uvicorn
    print(f"权重: {WEIGHT}")
    print(f"API 文档: http://127.0.0.1:{PORT}/docs")
    print(f"测试页  : http://127.0.0.1:{PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
