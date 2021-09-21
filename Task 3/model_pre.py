from keras.utils.np_utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizer_v1 import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras


x_train = np.load(r'Task 3\x_train.npy')
y_train = np.load(r'Task 3\y_train.npy')

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.1)

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

weight_list = np.load(r'Task 1\weight_list.npy',allow_pickle=True)

for i in range(len(weight_list)-1):
    model.layers[i] = weight_list[i]

model.fit(x_train,y_train,epochs=10)
model.evaluate(x_test,y_test)

model.save(r'Task 3\my_model_pre')
