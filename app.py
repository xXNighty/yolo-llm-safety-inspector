# 导入所需库
import gradio as gr  # 用于快速构建 Web 界面
from agent_app import agent  # 直接从我们写好的后端代码里导入 Agent 实例
import cv2  # 用于读取和保存图片
import os
from ultralytics import YOLO

# 1. 加载我们训练好的 YOLO 模型（用于在界面上画出检测框）
yolo_model = YOLO('./best.pt')


# 2. 定义处理图片的核心函数（前端点击“提交”后触发的逻辑）
def process_image(image, query):
    """
    image: Gradio 传入的图片数据（Numpy 数组，RGB 格式）
    query: 用户在文本框输入的自然语言指令
    """
    # --- 步骤 A：处理图片并保存到本地 ---
    temp_path = "uploaded_image.jpg"
    # OpenCV 默认使用 BGR 格式，而 Gradio 传入的是 RGB，所以需要转换颜色
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # --- 步骤 B：调用 YOLO 画框，生成可视化结果 ---
    results = yolo_model(temp_path)
    # results[0].plot() 是 YOLO 自带的神器，直接在原图上画好框、写好类别和置信度
    res_plotted = results[0].plot()
    result_img_path = "result_image.jpg"
    # 将画好框的图片保存到本地，准备展示到前端
    cv2.imwrite(result_img_path, res_plotted)

    # --- 步骤 C：调用 Agent 生成智能文字报告 ---
    # 拼接一个完整的提示词，把图片的绝对路径告诉大模型
    full_query = f"图片路径是 {os.path.abspath(temp_path)}。{query}"
    # 触发 Agent 运行（大模型会自己决定调用刚才的 vision_inspection 工具）
    response = agent.invoke({"messages": [("human", full_query)]})
    # 提取大模型返回的最终文本报告
    final_report = response["messages"][-1].content

    # --- 步骤 D：将结果返回给前端展示 ---
    return result_img_path, final_report


# 3. 构建 Gradio 交互界面
interface = gr.Interface(
    fn=process_image,  # 绑定上面写的处理函数
    inputs=[  # 定义输入组件
        gr.Image(label="上传现场照片"),  # 图片上传框
        gr.Textbox(
            label="输入巡检指令",
            value="请帮我检查一下这张图里有没有人没戴安全帽？"  # 默认填入一个示例问题
        )
    ],
    outputs=[  # 定义输出组件
        gr.Image(label="YOLO 视觉检测结果"),  # 显示画好框的图片
        gr.Textbox(label="Agent 智能巡检报告", lines=6)  # 显示大模型生成的文字报告
    ],
    title="智能视觉安全巡检 Agent",
    description="基于 YOLOv8 与本地大模型 (LM Studio) 的多模态巡检系统"
)

# 4. 启动服务
if __name__ == "__main__":
    # launch() 运行后，会自动在浏览器打开 http://127.0.0.1:7860
    interface.launch()