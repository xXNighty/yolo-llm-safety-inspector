import os
import random
import shutil

# ================= 1. 配置路径 =================
# 原数据路径
img_dir = r"A:\\pycharm\\Lab\\VisionAgent\\data\\JPEGImages"
label_dir = r"A:\\pycharm\\Lab\\VisionAgent\\result\\labels"

# YOLOv8 目标存放路径（最终会把这个路径写到 data.yaml 里）
base_out_dir = r"A:\\pycharm\\Lab\\VisionAgent\\result\\VOC2028_YOLO"

# 定义子文件夹路径
train_img_dir = os.path.join(base_out_dir, "images/train")
val_img_dir = os.path.join(base_out_dir, "images/val")
train_label_dir = os.path.join(base_out_dir, "labels/train")
val_label_dir = os.path.join(base_out_dir, "labels/val")

# ================= 2. 自动创建文件夹 =================
for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(d, exist_ok=True)

# ================= 3. 获取所有图片文件名并打乱 =================
# 获取所有图片名字（只取文件名，如 000001.jpg）
all_imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
random.seed(42)  # 设置随机种子，保证每次运行划分的结果一样
random.shuffle(all_imgs)  # 随机打乱

# 计算 80% 的划分点
split_idx = int(len(all_imgs) * 0.8)
#split_idx = 1000  #为了节约时间平替上行代码
train_imgs = all_imgs[:split_idx]
val_imgs = all_imgs[split_idx:]

print(f"总图片数: {len(all_imgs)}, 训练集: {len(train_imgs)}, 验证集: {len(val_imgs)}")


# ================= 4. 复制文件 =================
def copy_files(img_list, target_img_dir, target_label_dir):
    for img_name in img_list:
        # 复制图片
        src_img = os.path.join(img_dir, img_name)
        dst_img = os.path.join(target_img_dir, img_name)
        shutil.copy(src_img, dst_img)

        # 复制对应的 txt 标签
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + ".txt"
        src_label = os.path.join(label_dir, label_name)
        dst_label = os.path.join(target_label_dir, label_name)

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)


# 执行复制
print("正在复制训练集...")
copy_files(train_imgs, train_img_dir, train_label_dir)
print("正在复制验证集...")
copy_files(val_imgs, val_img_dir, val_label_dir)

print(f"数据集整理完成！请查看 {base_out_dir} 文件夹。")