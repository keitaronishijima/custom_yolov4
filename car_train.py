import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation,BatchNormalization,Dropout

import warnings
warnings.filterwarnings('ignore')

train_path = "real_car/data/train"
validation_path = "real_car/data/val"
test_path  = "real_car/data/test"

categories = os.listdir(train_path)
categories.sort()
"Categories Count :" ,len(categories)

train = []

IMG_SIZE=100
print(categories)
for category in categories:
    if category == '.DS_Store':
        continue
    folder = os.path.join(train_path,category)
    label = categories.index(category)
    
    for file in os.listdir(folder):
        file = os.path.join(folder,file)
        img = cv2.imread(file)
        #if type(img) == "NoneType":
        #print(file)
        
        try:
            img_arr = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            train.append([img_arr,label])
            
        except:
            pass
        
            
print("Training Image Count",len(train))

validation = []

count = int(0)
for category in categories:
    folder = os.path.join(validation_path,category)
    label = categories.index(category)
    if category == '.DS_Store':
            continue
    for file in os.listdir(folder):
        file = os.path.join(folder,file)
        #print(file)
        img = cv2.imread(file)
        try:
            img_arr = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            validation.append([img_arr,label])
        except:
            count+=1
            pass

test = []
for category in categories:
    if category == '.DS_Store':
        continue
    folder = os.path.join(test_path,category)
    label = categories.index(category)
    if category == '.DS_Store':
            continue
    for file in os.listdir(folder):
        file = os.path.join(folder,file)
        img = cv2.imread(file)
        try:
            img_arr = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            test.append([img_arr,label])
        except:
            pass

x_train = []
y_train = []
x_validation = []
y_validation = []
x_test = []
y_test = []

for features , label  in train:
    x_train.append(features)
    y_train.append(label)
y_train = pd.get_dummies(y_train)
for features , label  in validation:
    x_validation.append(features)
    y_validation.append(label)
y_validation = pd.get_dummies(y_validation)
for features , label  in test:
    x_test.append(features)
    y_test.append(label)
y_test = pd.get_dummies(y_test)


x_train = np.array(x_train)/255
y_train = np.array(y_train)

x_validation = np.array(x_validation)/255
y_validation = np.array(y_validation)

x_test = np.array(x_test)/255
y_test = np.array(y_test)

model = Sequential()

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(3,3))

model.add(Conv2D(64,(5,5),activation="relu"))
model.add(MaxPooling2D(5,5)) 


model.add(Flatten())

#-------------------

model.add(Dense(64,input_shape = x_train.shape[1:],activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics=["accuracy"])

checkpoint_path = "checkpoints"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(x_train,y_train,epochs=15,validation_data=(x_test,y_test), callbacks=[cp_callback])

model.summary()

#os.listdir('checkpoint_dir')

model.save("real_car_model.h5")

# def predict_fruit():
#     model = Sequential
    