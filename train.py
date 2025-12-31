
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(r'E:\BaiduNetdiskDownload\change-ASFF-Rep\ultralytics\cfg\models\11\v11(1).yaml')
    model.train(data=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\ultralytics\cfg\datasets\VisDrone.yaml',
                imgsz=640,
                epochs=200,
                batch=4,
                workers=0,
                device='0',
                patience=30,
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,

                # optimizer='AdamW',  # 改用更稳定的AdamW优化器
                # lr0=0.0008,  # 降低基础学习率 (原SGD通常0.01，AdamW用更小值)
                # momentum=0.9,  # 添加动量
                # weight_decay=0.05,  # 增加权重衰减
                # #clip_grad=5.0,  # 梯度裁剪阈值 (防止梯度爆炸)
                # warmup_epochs=10,  # 延长预热期
                # warmup_momentum=0.8,  # 预热期动量
                # warmup_bias_lr=0.1,  # 预热期偏置学习率
                # lrf=0.01,  # 最终学习率=lr0*lrf=0.000008
                # amp=False,  # 保持关闭AMP


                )

# if __name__ == '__main__':
#     # 直接加载中断的检查点（关键修改点）
#     model = YOLO(r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp121\weights\last.pt')
#
#     model.train(
#         data=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\ultralytics\cfg\datasets\VisDrone.yaml',
#         imgsz=640,
#         epochs=200,
#         batch=4,
#         workers=0,
#         device='0',
#         optimizer='SGD',
#         close_mosaic=10,
#         resume=True,  # 必须保持True
#         project='runs/train',
#         name='exp121',  # 必须与中断的实验名称一致
#         single_cls=False,
#         cache=False
#     )


#下面这个是跑了之后，重新看下实验结果（精度）
# if __name__ == '__main__':
#     # 加载模型权重
#     model = YOLO(r'best.pt')
#
#     # 验证模型
#     results = model.val(
#         data=r'VisDrone.yaml',
#         imgsz=640,
#         batch=8,
#         workers=0,
#         device='0',
#         save_txt=True,  # 保存文本结果
#         save_json=True  # 保存 JSON 结果
#     )

