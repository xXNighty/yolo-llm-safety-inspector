# 👷 工业安全合规智能巡检 Agent (Vision-Guarded Agent)

本项目是一个融合了**计算机视觉（CNN）**与**大语言模型（LLM）**的多模态智能系统。以 YOLOv8 为视觉感知核心，结合 OpenCV 与 Gradio，并利用 LangChain 构建本地大模型 Agent，实现“自然语言指令 -> 意图解析 -> 视觉检测 -> 智能报告”的闭环。

## 🌟 核心特性
- **自定义目标检测**：基于 VOC2028 数据集训练 YOLOv8 安全帽检测模型（mAP@0.5 达 82.5%）。
- **工具调用编排**：将视觉模型封装为 LangChain Tool，Agent 自动识别意图并调用工具。
- **异常兜底机制**：完善的 `try-except` 与路径校验，防止大模型幻觉与程序崩溃。
- **Web 交互界面**：基于 Gradio，支持图片上传、检测框渲染与自然语言报告生成。

## ⚙️ 环境与运行
1. 安装依赖：`pip install -r requirements.txt`
2. 启动本地大模型（LM Studio 加载 Qwen 并开启 Local Server）。
3. 运行主程序：`python app.py`
4. 访问网页：`http://127.0.0.1:7860`

## 🏗️ 架构设计
用户输入 -> Agent意图解析 -> 调用 YOLOv8 工具 -> 返回结构化 JSON -> LLM 生成最终自然语言巡检报告。