import cv2 as cv
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import keras

def create_dataset():

    dir = r'Task 3\mnistTask'
    images = []

    for names in os.listdir(dir):
        path = os.path.join(dir,names)
        print(f'Folder {names} complete')

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
            img28 = cv.resize(img_array,(28,28))

            images.append(img28)

    return images

x_train = create_dataset()

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)

y_train = []

model = keras.models.load_model(r'Task 2\my_model_rand')
y = model.predict(x_train)


for prediction in y:
    MaxPosition=np.argmax(prediction)  
    y_train.append(MaxPosition)


y_train = to_categorical(y_train)


np.save(r'Task 3\x_train',x_train)
np.save(r'Task 3\y_train',y_train)