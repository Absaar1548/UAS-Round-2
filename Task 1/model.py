import keras
from keras.backend import learning_phase
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizer_v1 import Adam
import numpy as np
from sklearn.model_selection import train_test_split


x_train = np.load(r'Task 1\x_train.npy')
y_train = np.load(r'Task 1\y_train.npy')


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

    model.add(Dense(36,activation = 'softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer=opt,loss = 'categorical_crossentropy',metrics=['accuracy'])

    return model

model = myModel()
model.fit(x_train,y_train,epochs=25,steps_per_epoch=100,batch_size=14)

weight_list = []

for layer in model.layers:
    weight_list.append(layer)

np.save(r'Task 1\weight_list.npy',weight_list)
model.save(r'Task 1\my_model')