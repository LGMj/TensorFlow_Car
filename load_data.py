import numpy as np
import cv2 as cv
import os 
import sys 
import pickle
from progressbar import *

def loadData(imageDir):
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
            img = cv.resize(img, (200, 66))
            img = img.tolist()
            data.append(img)
            angle = int(path.split('_')[2].split('.')[0])
            angle = 620 - int(angle*240 / 1079) - 381
            angle = angle / 240
            label.append(angle)
    data = np.array(data, dtype=ty)
    label = np.array(label, dtype=np.float32)
    return data, label        


x, y = loadData('./train_image/')
with open('train_data.kpl', 'wb') as fp:
    pickle.dump([x,y], fp)

x, y = loadData('./test_image/')
with open('test_data.kpl', 'wb') as fp:
    pickle.dump([x,y], fp)

