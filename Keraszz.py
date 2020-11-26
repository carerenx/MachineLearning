from __future__ import print_function

from keras.models import Model
from keras.utils import np_utils
import numpy as np
import pandas as pd
import keras
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import dataprocess
# 训练次数
nb_epochs = 1000
data=dataprocess.Dataprocess
# 读取训练集和测试集、标签
# 第0列为标签
# 第1列往后为数据
x_train =  data.x_train# tsv文件中使用\t作为分隔符，如果是csv文件使用‘，’作为分隔符就把这里的关键字改一下
y_train = data.y_train
x_test = data.x_test
y_test = data.y_test


# 有几个类别，np.nique函数是去除数组中的重复数字
nb_classes = len(np.unique(y_test))
# 设定批的大小，//表示整除
batch_size = min(x_train.shape[0] // 10, 16)

# 归一化
y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

# 转换为独热编码
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

x_train_mean = x_train.mean()
# std代表标准差
x_train_std = x_train.std()
# z-score标准化
x_train = (x_train - x_train_mean) / (x_train_std)

x_test = (x_test - x_train_mean) / (x_train_std)

x_train = x_train.reshape(x_train.shape + (1, 1,))
x_test = x_test.reshape(x_test.shape + (1, 1,))

x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)
conv1 = keras.layers.Conv2D(128, 8, 1, border_mode='same')(x)
conv1 = keras.layers.normalization.BatchNormalization()(conv1)
conv1 = keras.layers.Activation('relu')(conv1)

#    drop_out = Dropout(0.2)(conv1)
conv2 = keras.layers.Conv2D(256, 5, 1, border_mode='same')(conv1)
conv2 = keras.layers.normalization.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

#    drop_out = Dropout(0.2)(conv2)
conv3 = keras.layers.Conv2D(128, 3, 1, border_mode='same')(conv2)
conv3 = keras.layers.normalization.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)
out = keras.layers.Dense(nb_classes, activation='softmax')(full)

model = Model(input=x, output=out)

optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=50, min_lr=0.0001)
history = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                    verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr])
model.save('FCN_CBF_1500.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()