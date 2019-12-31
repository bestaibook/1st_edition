## train model
#from keras.models import Sequential
#from keras.layers import Dense, Activation
# 
#model = Sequential()
#model.add(Dense(input_dim=x_data.shape[1], units=100))
#model.add(Activation("relu"))
#model.add(Dense(input_dim=100, units=100))
#model.add(Activation("relu"))
#model.add(Dense(input_dim=100, units=100))
#model.add(Activation("relu"))
#model.add(Dense(input_dim=100, units=y_data.shape[1]))
#model.add(Activation("softmax"))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model.fit(x_train, y_train, epochs=1000, batch_size=20, verbose=2)
#scores = model.evaluate(x_test, y_test, batch_size=20, verbose=1)
#print(scores)
#
#

# -*- coding: utf-8 -*-

'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 训练数据 60000张手写图片，28*28*1
# 测试数据 10000张手写图片，28*28*1

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化到0-1
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# to_categorical(y, nb_classes=None)
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
# y: 类别向量; nb_classes:总共类别数
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Dense层:即全连接层
# keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)


model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递activation参数实现。
# 以下两行等价于：model.add(Dense(512,activation='relu'))
model.add(Dense(512))
model.add(Activation('relu'))

# Dropout  需要断开的连接的比例
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# 打印出模型概况
print('model.summary:')
model.summary()

# 在训练模型之前，通过compile来对学习过程进行配置
# 编译模型以供训练
# 包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']
# 如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 训练模型
# Keras以Numpy数组作为输入数据和标签的数据类型
# fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
# nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。

# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))


# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
# 按batch计算在某些输入数据上模型的误差
print('-------evaluate--------')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])



