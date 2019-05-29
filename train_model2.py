# coding: utf-8

from keras.utils import np_utils
import numpy as np
import pickle
import keras
from progressbar import *
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ELU, Lambda
from keras import callbacks
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K

print('Loading data...')
with open('./train_data_240.kpl', 'rb') as f:
    x_train, y_train = pickle.load(f)

x_train4D = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
 
x_train4D_normalize = x_train4D / 255
# y_trainOneHot = np_utils.to_categorical(y_train)

print('建立模型')
init = 'glorot_uniform'
if K.backend() == 'theano':
    input_frame = Input(shape=(1, 60, 240))
else:
    input_frame = Input(shape=(60, 240, 1))
x = Conv2D(24, (5, 5), padding='valid', strides=(2,2), kernel_initializer=init)(input_frame)
x = ELU()(x)
x = Dropout(0.2)(x)
x = Conv2D(36, (5, 5), padding='valid', strides=(2,2), kernel_initializer=init)(x)
x = ELU()(x)
x = Dropout(0.2)(x)
x = Conv2D(48, (5, 5), padding='valid', strides=(2,2), kernel_initializer=init)(x)
x = ELU()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), padding='valid', strides=(2,2), kernel_initializer=init)(x)
x = ELU()(x)
x = Dropout(0.2)(x)

x = Flatten()(x)

x = Dense(100, kernel_initializer=init)(x)
x = ELU()(x)
x = Dropout(0.5)(x)
x = Dense(50, kernel_initializer=init)(x)
x = ELU()(x)
x = Dropout(0.5)(x)
x = Dense(10, kernel_initializer=init)(x)
x = ELU()(x)
out = Dense(1, kernel_initializer=init)(x)

model = Model(input=input_frame, output=out)
model.compile(optimizer='adam',loss='mean_squared_error')
print(model.summary())

model_path = os.path.expanduser('./model.h5')
save_best = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
callbacks_list = [save_best, early_stop]
model.fit(x=x_train4D_normalize,y=y_train,validation_split=0.2,epochs=50,batch_size=256,callbacks=callbacks_list)

