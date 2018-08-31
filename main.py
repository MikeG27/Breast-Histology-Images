#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:07:47 2018

@author: michalgorski
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.load("X.npy") # features
y = np.load("y.npy") # Outputs

# Plot some images
plt.figure()
for i in range(0,10):
    plt.subplot(2,5,i+1)
    a = y[i+2754]
    if a == 0:
        plt.title("Negative")
    else:
        plt.title("Positive")
    plt.imshow(X[i+2754])
    plt.suptitle("Cancer images",fontsize = 30 )
    
# =============================================================================
#                           Data Overview
# =============================================================================



def split_data(X,y,test_size):
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for i in len(y):
        if i < float(len(y)*test_size):
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
        
    return X_train, y_train, X_test , y_test

X_train, y_train, X_test , y_test = split_data(X,y,test_size=0.33)
