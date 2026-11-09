import os
import xml.etree.ElementTree as ET
from PIL import Image

# ================= 1. 配置实际路径 =================
# 请把这里的路径改成电脑上真实存放 VOC2028 的路径
xml_dir = r"A:\\pycharm\\Lab\\VisionAgent\\data\\Annotations"  # XML 文件夹路径
img_dir = r"A:\\pycharm\\Lab\\VisionAgent\\data\\JPEGImages"  # 图片文件夹路径
yolo_label_dir = r"A:\\pycharm\\Lab\\VisionAgent\\result\\labels"  # 转换后 TXT 的输出路径

# 如果 labels 文件夹不存在，自动创建
os.makedirs(yolo_label_dir, exist_ok=True)

# ================= 2. 定义类别映射 =================
# 根据 VOC2028 数据集的官方说明：
# hat 代表佩戴了安全帽，person 代表未佩戴（或人头）
# YOLO 需要数字类别，所以我们将 hat 映射为 0，person 映射为 1
class_map = {"hat": 0, "person": 1}

# ================= 3. 开始循环转换 =================
# 遍历 Annotations 文件夹下的所有文件
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue  # 跳过非 XML 文件

    xml_path = os.path.join(xml_dir, xml_file)

    # 解析 XML 文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取文件名（不带后缀），例如 "000001"
    base_name = os.path.splitext(xml_file)[0]

    # 找到对应的图片，获取图片的宽高（用于归一化计算）
    img_path = os.path.join(img_dir, base_name + ".jpg")
    if not os.path.exists(img_path):
        print(f"找不到图片 {img_path}，跳过该 XML")
        continue

    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size  # 获取图片的宽度和高度
    except Exception as e:
        print(f"读取图片 {img_path} 失败: {e}")
        continue

    txt_lines = []  # 用于存放这个图片所有目标的 YOLO 格式数据

    # 遍历 XML 里的每一个 object（目标）
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None:
            continue

        label = name_node.text.strip()
        if label not in class_map:
            continue  # 如果出现了不认识的类别，跳过

        class_id = class_map[label]  # 获取类别数字 id

        # 获取边界框坐标 (左上角 xmin, ymin 和 右下角 xmax, ymax)
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # ================= 4. 核心公式：转换为 YOLO 格式 =================
        # YOLO 格式需要归一化到 0~1 之间，格式为：类别 x中心点 y中心点 宽度 高度
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        box_w = (xmax - xmin) / img_w
        box_h = (ymax - ymin) / img_h

        # 保留 6 位小数，用空格隔开，追加到列表
        txt_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    # ================= 5. 保存 TXT 文件 =================
    # 如果没有目标，生成空文件（YOLO 认为这是背景图）
    txt_file_path = os.path.join(yolo_label_dir, base_name + ".txt")
    with open(txt_file_path, "w") as f:
        f.write("\n".join(txt_lines))

print("格式转换完成！请去 A:/pycharm/Lab/VisionAgent/result/labels 检查是否生成了大量 txt 文件。")