import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json


def cross_dataset_evaluation():
    # 加载模型
    model = YOLO(r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp72\weights\best.pt')

    # AI-TOD类别列表
    ai_tod_classes = ['airplane', 'bridge', 'storage-tank', 'ship',
                      'swimming-pool', 'vehicle', 'person', 'wind-mill']

    # VisDrone类别列表 (根据你的训练确定)
    visdrone_classes = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

    print("注意: 这是跨数据集零样本测试，类别不完全匹配!")
    print(f"模型训练类别: {visdrone_classes}")
    print(f"测试数据集类别: {ai_tod_classes}")

    # 验证
    results = model.val(
        data=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\ultralytics\cfg\datasets\AI-TOD.yaml',
        imgsz=640,
        batch=4,
        save_json=True,
        workers=0,
        device='0',
        save_txt=True,  # 保存文本结果


    )

    return results


if __name__ == '__main__':
    results = cross_dataset_evaluation()