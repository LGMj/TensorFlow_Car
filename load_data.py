import numpy as np
import cv2 as cv
import os 
import sys 
from progressbar import *

def loadData():
    imageDir = './image/'
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
            img = cv.imread(path, 0)
            img = img.tolist()
            data.append(img)
            angle = int(path.split('_')[1].split('.')[0])
            angle = 620 - int(angle*240 / 1079) - 381
            label.append(angle)
    data = np.array(data, dtype=ty)
    label = np.array(label, dtype=int)
    return data, label        
