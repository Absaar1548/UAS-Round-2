import cv2 as cv
import os
import numpy as np
import keras

dir = r'C:\Users\ABSAAR ALI\Desktop\test_set'

X_train = []


for img in os.listdir(dir):
    img_path = os.path.join(dir,img)
    img_array = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
    img_array = cv.bitwise_not(img_array)
    img28 = cv.resize(img_array,(28,28))

    X_train.append(img28)

X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)

model = keras.models.load_model(r'Task 2\my_model_rand')

prediction = model.predict(X_train)
y = []

for i in prediction:
    MaxPosition=np.argmax(i)
    y.append(MaxPosition)  

print(y)

np.savetxt(r'C:\Users\ABSAAR ALI\Desktop\prediction.txt',y,fmt='%i')
