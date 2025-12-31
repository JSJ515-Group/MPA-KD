from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(model=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp121\weights\best.pt')

    # 进行预测，不显示置信度分数
    model.predict(
        source=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\无人机航拍3.jpg',
        save=True,
        show=True,
        show_conf=False,  # 关键参数：不显示置信度分数
        show_labels=True  # 仍然显示类别标签
    )