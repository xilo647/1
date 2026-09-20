# 猫狗图像分类实验：从零搭建到部署上线

本仓库是《猫狗分类实验指导书》六个步骤的**完整可运行实现 + 实验记录**。
数据是聊天里发来的 100 张图片（`cat.1~50.jpg` / `dog.51~100.jpg`），
标签藏在文件名里，与指导书 0.2 节的 `FileNameDataset` 约定一致。

> 环境：Windows 10 / Python 3.8.6（CPU 版 PyTorch 2.4.1）。
> 全部代码在 CPU 上运行，与指导书里 `device = 'cpu'` 的设定一致，
> 单卡 RTX 4060 本可用，但为保持与指导书、与同学结果可比，本实验未启用 GPU。

---

## 1. 目录结构

```
.
├── datasets/                    # 100 张原始图片（cat.1~50 / dog.51~100）
├── image.png                    # 指导书里的 LeNet→AlexNet 结构图
├── 猫狗分类实验指导书.md          # 原始指导书（只读留档）
├── common.py                    # 0.2/0.3 公共代码：数据加载 + 训练/评估
├── step1_scratch.py             # 第一步：手写 AlexNet 从零训练（含手写 ResNet18）
├── step2_pretrained.py          # 第二步：ImageNet 预训练迁移（冻结 / 微调）
├── step3_augment.py             # 第三步：数据增强四组横向对比（含 RandomErasing）
├── step3_erasing_cv.py          # 思考题2：RandomErasing 多种子重复实验（4 配置 × 5 种子）
├── step4_compare.py             # 第四步：resnet18 / mobilenet / shufflenet 对比
├── step5_1_train_best.py        # 第五步：训练并保存验证集最佳模型
├── step5_2_test.py              # 第五步：单张推理 + 全量回归测试
├── step6_app.py                 # 第六步：Gradio 拍照分类小程序
├── step6_api.py                 # 思考题3：FastAPI 版 HTTP 接口
├── results/                     # 指标 JSON、日志、模型权重（运行后生成）
├── tools/                       # 数据集构建、体检、冒烟测试、一键运行
└── requirements.txt
```

## 2. 环境准备

系统 Python 3.8.6 里没有 torch，且沙箱只允许写仓库目录，
因此依赖装在仓库内的 `.venv2`（不污染系统环境）：

```powershell
# 1) 在仓库内建一个不带 pip 的虚拟环境（Windows 下带 pip 的 venv 会被沙箱的
#    ensurepip 步骤挡住，所以改成"建空壳 + 用系统 pip 装到它的 site-packages"）
python -m venv --without-pip .venv2

# 2) 用系统 pip 把依赖装进该虚拟环境的 site-packages
python -m pip install --target .venv2\Lib\site-packages `
    -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 3) 验证
.venv2\Scripts\python.exe -c "import torch, torchvision, sklearn, gradio; print(torch.__version__)"
```

预训练权重缓存固定在仓库内 `.torch_cache`（由 `common.py` 设置 `TORCH_HOME`），不写系统盘。

## 3. 运行

一键跑完 1~5 步：

```powershell
pwsh -File tools\run_all.ps1
```

或者单步运行（每一步都会把标准输出同时存到 `results\logs\`）：

```powershell
.venv2\Scripts\python.exe step1_scratch.py
.venv2\Scripts\python.exe step2_pretrained.py
.venv2\Scripts\python.exe step3_augment.py
.venv2\Scripts\python.exe step3_erasing_cv.py   # 思考题2，CPU 上约 5~6 分钟
.venv2\Scripts\python.exe step4_compare.py
.venv2\Scripts\python.exe step5_1_train_best.py
.venv2\Scripts\python.exe step5_2_test.py results\best_model.pth datasets\cat.1.jpg
.venv2\Scripts\python.exe step5_2_test.py results\best_model.pth --all
```

第六步（拍照小程序，单独常驻运行）：

```powershell
.venv2\Scripts\python.exe step6_app.py
# 本机：http://127.0.0.1:7860
# 手机（同一 Wi-Fi）：http://<电脑IP>:7860
```

思考题 3 的 HTTP 接口：

```powershell
.venv2\Scripts\python.exe step6_api.py
# 文档：http://127.0.0.1:8000/docs
# 测试页：http://127.0.0.1:8000/
```

## 4. 结果一览

详细数字、混淆矩阵与讨论见 [`实验报告.md`](实验报告.md)；原始指标在 `results/*.json`，
训练输出在 `results/logs/*.log`，汇总表格在 `results/summary_tables.md`。

| 步骤 | 核心改动 | 本实验准确率 | 每轮用时 |
|---|---|---|---|
| 1 手写 AlexNet 从零训练 | 随机权重，30 epoch | **0.85**（峰值 0.90） | 3.1 s |
| 2 +ImageNet 预训练（冻结卷积） | 只训分类头，30 epoch | **1.00**（第 3 轮起） | 2.4 s |
| 2b 进阶：全部微调 | lr=1e-5, 5 epoch | 0.90（仍在上升） | 3.2 s |
| 3 +数据增强 | 无 / 基础 / 强 / 强+RandomErasing 四组 | 单次 0.90/0.80/0.90/0.95；**5 折均值 0.90/0.88/0.91/0.89（差异不显著）** | 1.9~3.4 s |
| 4 多模型对比 | resnet18 / mobilenet_v3_small / shufflenet_v2_x1_0 | 0.90 / 0.90 / 0.75 | 1.9 / 2.2 / 2.4 s |
| 5 保存最佳模型 + 推理 | state_dict + 单图/全量测试 | 全量 100 张 **0.91**（91/100） | — |
| 6 Gradio 小程序 | 摄像头拍照识别 | 实测 `http://127.0.0.1:7860` HTTP 200 | — |

部署（已实测）：

```powershell
.venv2\Scripts\python.exe step6_app.py
# 本机：http://127.0.0.1:7860
# 手机（同一 Wi-Fi）：http://192.168.2.116:7860
```

## 5. 与指导书的差异说明

代码严格沿用了指导书的结构、超参与函数命名，只在"能跑起来 / 可复现"上做了必要调整，
每一处都在对应脚本的 docstring 里写明：

1. `base_path` 换成仓库内的 `datasets/`（指导书里是老师电脑上的绝对路径）。
2. 第一步 `dataloader()` 里补了 `random.seed(42)`。指导书原版用 `random.sample`
   随机划分且没固定种子，会导致每次运行的 80/20 划分不同，违反其"全程同一份数据划分"
   的控制变量要求。
3. `FileNameDataset` 内部对 `os.listdir` 结果排序，理由同上。
4. 第三步参考代码里 `from step1_scratch import ResNet18Manual`：指导书第一步讲的是
   AlexNet，没有这个类，属于指导书的笔误；本实现在 `step1_scratch.py` 中补上了
   一个与 `torchvision.models.resnet18` 层名/形状完全一致的手写 ResNet18，
   使该行 import 与后续 `load_state_dict(official.state_dict(), strict=False)`
   能按指导书原样运行。
5. 训练日志与指标统一落盘到 `results/`，便于写报告；不影响任何计算结果。
6. 第三步在指导书的 A/B/C 三组之外补了 **D 组 = C + `RandomErasing`**（思考题 2），
   并用 `step3_erasing_cv.py` 做 4 配置 × 5 数据划分种子的重复实验。
   原因：20 张验证集上 1 张图 = 5%，单次划分看到的增强差异（0.80 与 0.95）
   完全落在抽样噪声里，不足以支撑任何结论。实测结论也确实**推翻了**
   "RandomErasing 会提高平均置信度"的直觉预期，详见 `实验报告.md` 第九节。
