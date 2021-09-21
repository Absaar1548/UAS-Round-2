import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.utils.np_utils import to_categorical
import numpy as np

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
y_test = to_categorical(y_test)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 784

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(28,28,1),activation='relu')))
    model.add((Conv2D(noOfFilters,sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10,activation = 'softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer=opt,loss = 'categorical_crossentropy',metrics=['accuracy'])

    return model

model = myModel()

model.fit(x_train,y_train,epochs=4)

model.evaluate(x_test,y_test)
model.save(r'Task 2\my_model_rand')
