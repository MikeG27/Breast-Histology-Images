#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:07:47 2018

@author: michalgorski
"""

import time 
import os
import zipfile
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
from modules import myplots



directories = os.listdir() #Get list of directories
dataset = zipfile.ZipFile('/home/michal/Pulpit/Breast-Histology-Images/dataset/dataset.zip')
dataset.extractall()


X = np.load("X.npy") # features
y = np.load("Y.npy") # Outputs
y = y.reshape(y.shape[0],1)

# Plot some images
#myplots.plot_img(X_train,y_train)

# =============================================================================
#                           Data Overview
# =============================================================================


print("\nImage data : \n")
print("Number of images : ", len(X) ) 
print("Image dims : " , X[0].shape)
print("Width : " , X.shape[1])
print("Height : " , X.shape[1])
print("Depth : ", X.shape[3])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255


# =============================================================================
                            # Define model
# =============================================================================

model = Sequential()
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(50, 50, 3))

model.add(conv_base)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
          loss='binary_crossentropy',
          metrics=['accuracy'])

batch_size = 32
epochs = 40

datagen = ImageDataGenerator(zoom_range=0.2,      
                             rotation_range=0.2,
                             fill_mode="nearest",
                             horizontal_flip = True,
                             vertical_flip=True)

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,y_train,batch_size = batch_size),
                               steps_per_epoch=(len(X_train)/batch_size),epochs=epochs,
                               validation_data=(X_test,y_test),validation_steps=(len(X_test)/batch_size))

loss1,accuracy1 = model.evaluate(X_test,y_test)
#plot training
myplots.plot_training(history,save_fig=True)
#plot confusion_matrix
y_pred = model.predict(X_test)
class_names = ["Health","Cancer"]
myplots.plot_confusion_matrix(y_pred,y_test,class_names)

model.save("VGG_81%")
myplots.predict_cancer(randint(0, len(X_test)),X_test,y_test,y_pred)




