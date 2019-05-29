import cv2 as cv
import keras
import numpy as np 
import time 
import pickle

with open('./test_data_240.kpl', 'rb') as f:
     x_test, y_test = pickle.load(f)

x_test4D = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')
x_text4D_normalize = x_test4D / 255

model = keras.models.load_model('model.h5')

loss = model.evaluate(x_text4D_normalize, y_test, verbose=1)
print(loss)
print(type(loss))

