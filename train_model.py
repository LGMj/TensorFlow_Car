# coding: utf-8
# 加载模块
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

# 加载训练数据
print('Loading data...')
with open('./train_data_240.kpl', 'rb') as f:
    x_train, y_train = pickle.load(f)
# 将数据转成n*240*60*1
x_train4D = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
# 归一化
x_train4D_normalize = x_train4D / 255

print('建立模型')
init = 'glorot_uniform'
# 定义模型输入形状
if K.backend() == 'theano':
    input_frame = Input(shape=(1, 60, 240))
else:
    input_frame = Input(shape=(60, 240, 1))
x = Conv2D(24, (5, 5), padding='valid', strides=(2,2), kernel_initializer=init)(input_frame)  # 第一卷积层，24个5*5卷积核，卷积步长为2
x = ELU()(x)  # 激活函数为ELU函数
x = Dropout(0.2)(x)  # 训练时随机屏蔽20%的神经元      
x = Conv2D(36, (5, 5), padding='valid', strides=(2,2), kernel_initializer=init)(x)  # 第二卷积层，36个5*5卷积核，卷积步长为2
x = ELU()(x)  # 激活函数为ELU函数
x = Dropout(0.2)(x)  # 训练时随机屏蔽20%的神经元      
x = Conv2D(48, (5, 5), padding='valid', strides=(2,2), kernel_initializer=init)(x)  # 第三卷积层，48个5*5卷积核，卷积步长为2
x = ELU()(x)  # 激活函数为ELU函数
x = Dropout(0.2)(x)  # 训练时随机屏蔽20%的神经元
x = Conv2D(64, (3, 3), padding='valid', strides=(2,2), kernel_initializer=init)(x)  # 第四卷积层，64个5*5卷积核，卷积步长为2
x = ELU()(x)  # 激活函数为ELU函数
x = Dropout(0.2)(x)  # 训练时随机屏蔽20%的神经元

x = Flatten()(x)  # 将卷积后的图像展开成一维数据

x = Dense(100, kernel_initializer=init)(x)  # 全连接网络隐藏层，100个神经元
x = ELU()(x)  # 激活函数为ELU函数
x = Dropout(0.5)(x)  # 训练时随机屏蔽50%的神经元
x = Dense(50, kernel_initializer=init)(x)  # 全连接网络隐藏层，50个神经元
x = ELU()(x)  # 激活函数为ELU函数
x = Dropout(0.5)(x)  # 训练时随机屏蔽50%的神经元
x = Dense(10, kernel_initializer=init)(x)  # 全连接网络隐藏层，10个神经元
x = ELU()(x)  # 激活函数为ELU函数
out = Dense(1, kernel_initializer=init)(x)  # 网络输出

model = Model(input=input_frame, output=out)  #生成网络模型
# 设置优化器为adam优化器，损失函数为均方差
model.compile(optimizer='adam',loss='mean_squared_error') 
print(model.summary())  # 打印模型参数

model_path = os.path.expanduser('./model.h5')  # 模型保存位置
# 保存训练参数最优的模型
save_best = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
callbacks_list = [save_best, early_stop]
# 训练模型，共训练50轮，每次喂入256组数据
model.fit(x=x_train4D_normalize,y=y_train,validation_split=0.2,epochs=50,batch_size=256,callbacks=callbacks_list)

