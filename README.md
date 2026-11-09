# 👷 工业安全合规智能巡检 Agent (Vision-Guarded Agent)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.4.204-green)
![LangChain](https://img.shields.io/badge/LangChain-1.7.6-yellowgreen)
![Gradio](https://img.shields.io/badge/Gradio-Web_UI-red)

本项目是一个融合了**计算机视觉（CNN）**与**大语言模型（LLM）**的多模态智能应用。针对传统工业安全巡检高度依赖人工、效率低、易漏检的痛点，系统以 YOLOv8 为视觉感知核心，结合 OpenCV 与 Gradio，并利用 LangChain 构建本地大模型 Agent，实现了**“自然语言指令 -> 意图解析 -> 视觉检测 -> 智能报告生成”**的全链路闭环。

## 🌟 核心特性

*   **🎯 自定义目标检测**：基于 VOC2028 安全帽数据集（7000+ 图像）微调 YOLOv8n 模型，mAP@0.5 达到 **82.5%**，实现高精度的人员与安全帽识别。
*   **🧠 多模态 Agent 调度**：基于 LangChain 构建 ReAct 模式 Agent。将视觉模型封装为 `@tool`，大模型根据用户自然语言自动判断是否需要调用视觉工具。
*   **🛡️ 结构化输出与防幻觉**：视觉工具不返回原始像素，而是提取检测数据生成高度结构化的 JSON（包含安全帽数量、违规状态），大幅降低大模型的推理成本与幻觉。
*   **⚙️ 异常兜底机制**：内置文件路径校验与 `try-except` 异常捕获。在未检测到目标或路径错误时，Agent 能优雅回复用户提示，避免系统崩溃。
*   **💻 零门槛 Web 交互**：基于 Gradio 搭建前端，用户仅需上传图片并输入口语化指令，即可同时获得“带检测框的渲染图”与“自然语言整改报告”。

## 🏗️ 系统架构

本项目采用**“感知-决策-交互”三层解耦架构**：

1.  **视觉感知层 (Vision Perception)**：OpenCV 预处理图像 -> YOLOv8n 模型推理 -> 业务规则判断 -> 输出结构化 JSON。
2.  **Agent 决策层 (Agent Orchestration)**：LM Studio 驱动本地 Qwen 模型 -> 解析用户意图 -> 调度视觉工具 -> 整合结果并生成自然语言。
3.  **交互层 (UI Layer)**：Gradio 提供 Web 界面，实现数据流转与可视化展示。

## 📊 模型训练与性能指标

*   **数据集**：VOC2028 (PASCAL VOC 格式转 YOLO 格式)。
*   **数据划分**：训练集 6056 张，验证集 1515 张。
*   **训练环境**：PyTorch 2.15.0 (CPU 版本，受限于本地 CUDA 环境)。
*   **训练参数**：图像尺寸 `416x416`，`batch_size=4`，训练轮数 `30 Epochs`，优化器 `AdamW`。
*   **评估结果**：
    *   总体 mAP@0.5：**82.5%**
    *   `hat`（安全帽）精度：mAP50 = 82.2%
    *   `person`（未戴安全帽人员）精度：mAP50 = 82.7%
    *   单张图片 CPU 推理速度：**20.0 ms**

## 📂 项目代码结构

```text
VisionAgent_Project/
├── app.py                # 【交互层】Gradio Web 界面与 OpenCV 渲染逻辑
├── agent_app.py          # 【决策与感知层】LangChain Agent 构建、YOLO 工具封装
├── best.pt               # 基于 VOC2028 训练出的 YOLOv8 权重
├── data.yaml             # 数据集类别配置文件
├── train.py              # 模型训练脚本
├── split_dataset.py      # 数据集划分脚本
├── voc2yolo.py           # PASCAL VOC 转 YOLO 格式转换脚本
└── requirements.txt      # 项目依赖库列表
