'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function  #预设一个输出函数

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop #导入优化算法，还有SGD,Adam

batch_size = 128
num_classes = 10
epochs = 20  #训练20轮

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#这里的x_train是60000*28*28的三维数组

#将x_train变成两维60000*（28*28），因为全连接层不能处理图像（如28*28）这样二维的结构，只能变成一列的数据
x_train = x_train.reshape(60000, 784)  #60000表示数据条数
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')  #784个像素点对应的黑白值是0到255，变成浮点值
x_test = x_test.astype('float32')
x_train /= 255  #归一化
x_test /= 255
print(x_train.shape[0], 'train samples')  #x_train是60000*784，x_train.shape[0]返回的是60000
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes) #y_train为60000*1的数据，表示y属于哪个类型，keras.utils.to_categorical将数据转为类型矩阵，返回的是60000*10
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential() #定义一个序列模型
model.add(Dense(512, activation='relu', input_shape=(784,))) #增加一个全连接层，有512个隐藏神经元，因为是第一层，要指定输入形状为784
model.add(Dropout(0.2))  #防止过拟合
model.add(Dense(512, activation='relu'))  #制定激活函数
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()  #看一下模型的参数介绍
 
 #编译一下模型，确定损失函数，优化算法，度量标准
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
#拟合模型
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
