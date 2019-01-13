

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
手写数字识别
'''

from __future__ import print_function  #预设了一个输出函数
import keras
from keras.datasets import mnist   #keras自带了mnist数据集
from keras.models import Sequential   #导入序列模型
from keras.layers import Dense, Dropout, Flatten  
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#由于TensorFlow和Theano里的参数位置是不一样的，比如Theano中参数表示x_train.shape[0]数据条数，1卷积层数量，图像的行和列
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32') #x_train由0到255的像素点表示
x_test = x_test.astype('float32') 
x_train /= 255  #归一化
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples') #输出样本数
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)  #将类别表示为矩阵,类似[0,0,0,0,1,0,0,0,0]
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#指定卷积层数，卷积核大小3*3，激活函数，输入大小
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #定义池化大小
model.add(Dropout(0.25))
model.add(Flatten())  #把前面一层中的多个二维的面，拉直连接成一维的数据，才能与后面的全连接层相接
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  #输出层的激活函数

#编译模型参数
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
