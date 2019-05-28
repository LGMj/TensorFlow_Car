# coding: utf-8

import load_data as ld
from keras.utils import np_utils
import numpy as np
import keras
from progressbar import *
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import callbacks

print('Loading data...')
x_train, y_train= ld.loadData()

x_train4D = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
 
x_train4D_normalize = x_train4D / 255
# y_trainOneHot = np_utils.to_categorical(y_train)

print('建立模型')
model = Sequential()
model.add(Conv2D(filters=8,
                kernel_size=(3,3),
                padding='same',#补零
                input_shape=(60,240,1),
                activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(filters=16,
                kernel_size=(3,3),
                padding='same',#补零
                activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,
                kernel_size=(3,3),
                padding='same',#补零
                activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(1))
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_logarithmic_error',
             optimizer=adam, metrics=['accuracy'])
print(model.summary())

model_path = os.path.expanduser('./model.h5')
save_best = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
callbacks_list = [save_best, early_stop]
model.fit(x=x_train4D_normalize,y=y_train,validation_split=0.2,epochs=10,batch_size=30,callbacks=callbacks_list)

