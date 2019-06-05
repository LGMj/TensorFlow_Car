# coding:utf-8
# 加载模块
import numpy as np
import cv2 as cv
import os 
import sys 
import pickle
from progressbar import *

def loadData(imageDir):
    # 读取图像保存路径中的所有图像和转向角信息
    files = os.listdir(imageDir)
    path = os.path.join(imageDir, files[0])
    img = cv.imread(path, 0)
    ty = img.dtype 
    data = []
    label = []
    progress = ProgressBar()
    for i in progress(range(1, len(files))):
        path = os.path.join(imageDir, files[i])
        if os.path.isfile(path):
            img = cv.imread(path, 0)  # 读取一帧图像
            img = cv.resize(img, (240, 60))  # 转为240*60大小
            img = img.tolist()
            data.append(img)  # 保存至数组
            angle = int(path.split('_')[2].split('.')[0])  # 读取转向角信息
            angle = 620 - int(angle*240 / 1079) - 381
            angle = angle / 240
            label.append(angle)
    data = np.array(data, dtype=ty)
    label = np.array(label, dtype=np.float32)
    return data, label     # 返回图像和转向角数据   


x, y = loadData('./train_image/')  # 训练图像路径
with open('train_data.kpl', 'wb') as fp:
    pickle.dump([x,y], fp)  # 保存为kpl文件

x, y = loadData('./test_image/')  # 测试图像路径
with open('test_data.kpl', 'wb') as fp:
    pickle.dump([x,y], fp)  # 保存为kpl文件

