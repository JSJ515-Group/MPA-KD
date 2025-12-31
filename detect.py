# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp72\weights\best.pt')
    model.predict(source=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp72\weights\test2.jpg',
                  save=True,
                  show=True,
                  )