#coding: utf-8
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD

#data
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
y_train = np_utils.to_categorical(y_train,num_classes = 10)
y_test = np_utils.to_categorical(y_test,num_classes = 10)

#build CNN
model = Sequential()
model.add(Convolution2D(
    filters=32,
    kernel_size=(3,3),
    strides=(1,1),
    padding='same',
    input_shape=(28,28,1),

))
model.add(Activation('relu'))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size =(2, 2),
    strides=(2, 2),
    padding='same',

))
model.add(Dropout(0.25))

model.add(Convolution2D(filters=64, kernel_size = (3,3),strides=(1,1), padding='same',))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same',))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr = 0.01, decay = 1e-6,momentum = 0.9, nesterov = True)
model.compile(optimizer=sgd,loss = 'categorical_crossentropy',metrics = ['accuracy'])


print('Training-------------')
model.fit(X_train, y_train, batch_size = 32,epochs = 1,)


print('\nTesting-------------')
loss,accuracy = model.evaluate(X_test,y_test)


print('\ntest loss: ',loss)
print('\ntest accuracy: ',accuracy)

