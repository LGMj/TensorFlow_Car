# coding: utf-8

import load_data as ld
from keras.utils import np_utils
import numpy as np
import keras
from progressbar import *
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

x_train, y_train= ld.loadData()

x_train4D = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
 
x_train4D_normalize = x_train4D / 255
y_trainOneHot = np_utils.to_categorical(y_train)

model = Sequential()
model.add(Conv2D(filters=16,
                kernel_size=(5,5),
                padding='same',#补零
                input_shape=(64,128,1),
                activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(filters=36,
                kernel_size=(5,5),
                padding='same',#补零
                activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(240,activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train4D_normalize,
                       y=y_trainOneHot,validation_split=0.2,
                       epochs=200,batch_size=300,verbose=2)

model.save('model.h5')
