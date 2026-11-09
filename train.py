from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',
        epochs=30,  # CPU 训练慢，30 轮足够看到效果了，不用跑 50 轮
        imgsz=416,  # 【关键】把图片尺寸从 640 降到 416，速度能快 2-3 倍
        batch=4,  # 【关键】CPU 内存有限，batch 设小一点，防止电脑卡死
        device='cpu',  # 强制指定用 CPU
        workers=2,  # Windows 下数据加载线程不要太多
        project='safety_helmet',
        name='cpu_run'
    )