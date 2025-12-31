from ultralytics import YOLO
import cv2
import numpy as np

if __name__ == '__main__':
    # 加载模型
    model = YOLO(model=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp121\weights\best.pt')

    # 使用OpenCV直接读取图像，确保一致性
    image_path = r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp121\weights\test1.jpg'
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("无法读取图像文件")
        exit()

    # 进行预测
    results = model.predict(
        source=original_image,  # 直接传入OpenCV图像
        save=False,
        show=False
    )

    # 获取第一个结果
    result = results[0]

    # 创建BGR颜色映射
    colors = [
        (0, 0, 255),  # 红色 - pedestrian
        (0, 255, 0),  # 绿色 - people
        (255, 0, 0),  # 蓝色 - bicycle
        (0, 255, 255),  # 黄色 - car
        (255, 0, 255),  # 紫色 - van
        (255, 255, 0),  # 青色 - truck
        (128, 0, 128),  # 深紫色 - tricycle
        (0, 128, 128),  # 橄榄色 - awning-tricycle
        (128, 128, 0),  # 深青色 - bus
        (0, 0, 128)  # 深红色 - motor
    ]

    # 复制原始图像
    image_with_boxes = original_image.copy()

    # 绘制检测框
    for box in result.boxes:
        # 获取框坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # 获取类别ID
        class_id = int(box.cls[0])
        # 获取对应颜色
        color = colors[class_id % len(colors)]
        # 绘制矩形框
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 3)

    # 直接保存，不进行任何颜色转换
    output_path = r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp121\test1_detect.jpg'
    cv2.imwrite(output_path, image_with_boxes)
    print(f"保存成功: {output_path}")