import cv2 as cv
import os
import numpy as np
from keras.utils.np_utils import to_categorical

def create_dataset():
    
    dir = r'Task 1\trainPart1\train'

    labels = []
    images = []

    folders = []
    for i in os.listdir(dir):
        folders.append(i)

    for names in folders:
        path = os.path.join(dir,names)
        label = folders.index(names)
        print(f'Folder {label} processing...')

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
            img_array = cv.bitwise_not(img_array)
            img28 = cv.resize(img_array,(28,28))

            labels.append(label)
            images.append(img28)

    return images,labels

x_train,y_train = create_dataset()

print('Process Completed')
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
y_train = to_categorical(y_train)

np.save(r'Task 1\x_train',x_train)
np.save(r'Task 1\y_train',y_train)