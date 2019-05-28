import cv2 as cv
import keras
import numpy as np 
import time 

images = []
angles = []
for i in range(4):
    name = input("input image name:")
    img_path = 'test_image/' + name + '.jpeg'
    angle = int(img_path.split('_')[2].split('.')[0])
    angle = 620 - int(angle*240/1079) - 381
    print(angle)
    print('path:' + img_path)
    images.append(img_path)
    angles.append(angle)

model = keras.models.load_model('model.h5')

for i in range(4):
    start = time.clock()
    img_data = []
    img = cv.imread(images[i], 0)
    img_show = img 
    img = cv.resize(img, (200, 66))
    img = img.tolist()
    img_data.append(img)
    img_data = np.array(img_data, dtype=np.uint8)
    img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], img_data.shape[2], 1).astype('float32')
    img_data = img_data / 255
    
    prediction = model.predict(img_data)
    pred = 240 * prediction
    end = time.clock()
    print('time:' + str(end-start))
    print(pred, angles[i])
    loss = abs(angles[i] - pred[0]) / 240 * 100
    print(loss)
    loss = ('%.2f'%loss)
    string = 'act:' + str(angles[i]) + ', ' + 'pre:' + str(pred[0])
    cv.putText(img_show, string, (10,20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img_show, 'loss:' + str(loss) + '%', (10,40), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    cv.imshow("img", img_show)
    cv.waitKey(0)
    cv.destroyAllWindows()
