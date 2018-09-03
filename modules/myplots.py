#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:06:08 2018

@author: michal
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

sns.set()




def plot_img(X, y, n_rows = 2, n_cols = 5, n_img = 10, save = False):
    
    """
    Plot images from numpy data
    
    inputs:
        * X -- array of images (n_images,width,height,depth)
        * y -- output labels 
        * n_rows -- number of rows
        * n_cols -- number of columns
        * n_img -- number of images
        
    outputs:
        
        * plot images
        * save images
    """
    
    plt.figure()
    
    for i in range(0,n_img):
        plt.suptitle("Breast_histology",fontsize = 30)
        plt.subplot(n_rows,n_cols,i+1)
        a = y[i+2754]
        if a == 0:
            plt.title("Negative")
        else:
            plt.title("Positive")
        plt.imshow(X[i+2754])

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
        return smoothed_points     

def plot_training(history,val = True,save_fig = False):
    
    acc = history.history["acc"]
    loss = history.history["loss"]
    val_acc = history.history["val_acc"]
    val_loss = history.history["val_loss"]
    
    epochs = range(len(acc))
    
    plt.figure((2),figsize=(20,10))
    
    plt.subplot(2,1,1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    
    if save_fig == True:
        plt.savefig("learning")
        
  
        
def plot_confusion_matrix(y_pred,y_test,class_names):
    
    for i in range(0,len(y_pred)):
        if y_pred[i] > 0.5 :
            y_pred[i] = 1
        else :
            y_pred[i] = 0
        
    cnf_matrix = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plt.title("Confusion Matrix")
    sns.heatmap(cnf_matrix,annot = True, xticklabels=class_names,
                      yticklabels=class_names,fmt=".1f",square=True,robust=True,cmap="Blues",
                      linewidths=4,linecolor='white')

    plt.subplot(1,2,2)
    cnf_matrix_normalized = cnf_matrix/cnf_matrix.sum(axis=0)
    plt.title("Confusion Matrix normalized")
    sns.heatmap(cnf_matrix_normalized,annot = True, xticklabels=class_names,
                      yticklabels=class_names,fmt="0f",square=True,robust=True,cmap="Blues",
                      linewidths=4,linecolor='white')
    
    
def predict_cancer(pic_number,X_test,y_test,y_pred):

    plt.figure()
    plt.suptitle("Cancer detection system",fontsize = 20)
    plt.subplot(1,2,1)
    plt.imshow(X_test[pic_number])
    print(int(y_test[pic_number]))
    
    if y_test[pic_number] == 1:
        plt.title("Diagnosis : " + "positive")
    else :
        plt.title("Diagnosis : " + "negative")
    
    plt.subplot(1,2,2)
    plt.imshow(X_test[pic_number])
    print(int(y_pred[pic_number]))
    
    if y_pred[pic_number] == 1:
        plt.title("Prediction : " + "positive")
    else :
        plt.title("Prediction : " + "negative")