import cv2 as cv
import keras
import numpy as np 
import time 

img_path = 'image/20676_501.jpeg'

model = keras.models.load_model('model1.h5')

start = time.clock()
img_data = []
img = cv.imread(img_path, 0)
img = cv.resize(img, (240, 60))
img = img.tolist()
img_data.append(img)
img_data = np.array(img_data, dtype=np.uint8)
img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], img_data.shape[2], 1).astype('float32')
img_data = img_data / 255

prediction = model.predict(img_data)
pred = np.argmax(prediction, 1)
end = time.clock()
print(str(end-start))
angle = int(img_path.split('_')[1].split('.')[0])
angle = 620 - int(angle*240/1079) - 381
print(pred, angle)

