# ==========================================
# 1. 导入所需的依赖库
# ==========================================
import json  # 用于将 Python 字典转换成 JSON 格式的字符串，方便大模型读取
import os  # 用于进行操作系统相关的操作，比如检查文件路径是否存在
from langchain.agents import create_agent  # LangChain 新版构建 Agent 的核心函数
from langchain_openai import ChatOpenAI  # 用于连接 LM Studio 提供的兼容 OpenAI 格式的 API
from langchain_core.tools import tool  # 装饰器：用于将普通的 Python 函数转换为 Agent 可调用的工具
from ultralytics import YOLO  # 导入 YOLOv8 的核心库
from langchain_core.messages import AIMessage  # 用于类型标注，告诉 PyCharm 返回的是 AI 消息对象，消除波浪线警告

# ==========================================
# 2. 初始化视觉模型
# ==========================================
# 加载本地 YOLOv8 预训练模型。这里 './' 表示当前项目目录。确保 yolov8n.pt 文件已经放在同级目录下。
yolo_model = YOLO('./yolov8n.pt')


# ==========================================
# 3. 定义视觉检测工具函数 (Agent 的“眼睛”)
# ==========================================
@tool  # 【核心装饰器】加上这个装饰器，LangChain 就会自动解析这个函数的名称、注释和参数，封装成一个 Agent 能够使用的 Tool
def vision_inspection(image_path: str) -> str:
    """
    传入图像路径，返回检测结果JSON字符串。
    (注意：这里的注释非常重要，Agent 是通过读取这段注释来理解这个工具是干嘛的！)
    """
    # 异常兜底 1：检查文件是否存在（防止用户传错路径导致程序崩溃）
    if not os.path.exists(image_path):
        # 如果文件不存在，返回一个规范的错误 JSON 字符串给大模型
        return json.dumps({
            "status": "error",
            "message": f"图片路径不存在: {image_path}，请检查路径是否正确。"
        }, ensure_ascii=False)  # ensure_ascii=False 保证中文不乱码

    print(f"\n[系统日志] 正在调用 YOLO 检测图片: {image_path}")

    try:
        # 核心推理：将图片路径传给 YOLO 模型进行检测
        results = yolo_model(image_path)

        objects = []  # 创建一个空列表，用于存放检测到的物体类别名称
        # 遍历检测结果（可能包含多张图片，虽然这里只传了一张）
        for r in results:
            # 遍历这张图片中检测到的所有边界框（bounding boxes）
            for box in r.boxes:
                # 获取类别索引，并通过 yolo_model.names 字典映射回类别名称（如 'person', 'car'）
                cls_name = yolo_model.names[int(box.cls[0])]
                objects.append(cls_name)  # 将识别到的类别加入列表

        # 将提取到的信息整理成字典
        detection_data = {
            "image_path": image_path,
            "detected_classes": objects,
            "total_objects": len(objects)
        }

        # 将字典转化为 JSON 字符串并返回。Agent 拿到这个字符串后，会用它的大模型大脑去理解。
        return json.dumps(detection_data, ensure_ascii=False)

    except Exception as e:
        # 异常兜底 2：如果 YOLO 在推理过程中发生任何意外（如图片损坏、内存溢出等），捕获错误并返回给 Agent
        return json.dumps({
            "status": "error",
            "message": f"YOLO 检测失败: {str(e)}"
        }, ensure_ascii=False)


# 将定义好的工具放入一个列表中，因为 Agent 可能拥有多个工具（未来你可以再加一个“语音识别工具”或“数据库查询工具”）
tools = [vision_inspection]

# ==========================================
# 4. 初始化大语言模型 (Agent 的“大脑”)
# ==========================================
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",  # LM Studio 本地服务的默认地址，告诉 LangChain 往哪里发请求
    api_key="lm-studio",  # 本地服务不需要真实的 key，随便填一个占位即可
    model="qwen3-8b",  # 【注意】这里填你 LM Studio 中实际加载的模型名称（如 qwen2.5-7b-instruct）
    temperature=0.1  # 温度值。越低（0.1）代表模型回答越严谨稳定，适合做逻辑和工具调用；越高越有创造力
)

# ==========================================
# 5. 构建 Agent 调度器
# ==========================================
agent = create_agent(
    model=llm,  # 告诉 Agent 使用哪个大脑
    tools=tools,  # 告诉 Agent 它拥有哪些工具
    # 系统提示词：设定 Agent 的角色。这里让它作为一个有用的助手，并且告知它可以使用工具。
    system_prompt="You are a helpful assistant. You have access to the following tools."
)

# ==========================================
# 6. 运行测试
# ==========================================
if __name__ == "__main__":
    # 定义你要测试的图片路径。使用 r"" 表示原始字符串，防止 Windows 路径里的反斜杠 \ 被识别为转义字符
    test_image_path = r"A:\pycharm\Lab\VisionAgent\test.jpg"

    # 拼接用户的提问，把图片路径包含进去，引导 Agent 去调用工具
    user_query = f"帮我检查一下 {test_image_path} 这张图，里面有什么？"
    print(f"用户提问: {user_query}")

    # 核心运行步骤：
    # 使用 invoke 方法触发 Agent 运行。
    # 输入必须是一个字典，包含 "messages" 键，里面是一个元组 ("human", 提问内容) 代表用户输入。
    response: dict = agent.invoke({"messages": [("human", user_query)]})

    # 类型标注：告诉编辑器 response["messages"][-1] 是一个 AIMessage 对象，消除 PyCharm 的警告
    final_message: AIMessage = response["messages"][-1]

    # 提取最终的消息内容并打印。此时打印出来的就是大模型结合了 YOLO 检测结果后，总结出来的自然语言报告
    print("\n【Agent最终报告】:\n", final_message.content)