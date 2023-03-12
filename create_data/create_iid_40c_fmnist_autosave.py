# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:47:23 2021

@author: liush
"""

#Generation of Local Data Sets.
#import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
import random
import numpy as np
import math



img_rows, img_cols = 28, 28# input image dimensions
num_classes = 10
num_client = 40
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]

num_per = x_train.shape[0]//num_client

for j in range(1,num_client+1):###############################################################
   
    if j<=num_client:
        x_train_local = x_train[int((j-1)*num_per):int(j*num_per)] #local train data
        y_train_local = y_train[int((j-1)*num_per):int(j*num_per)]
    else:
        x_train_local = x_train[int((j-1)*num_per):] #local train data
        y_train_local = y_train[int((j-1)*num_per):]

    if j <10:
        name_x_train = 'data_fmnist_iid/' + 'client_0' + str(j) +'_x_train.npy'
        name_y_train = 'data_fmnist_iid/' + 'client_0' + str(j) +'_y_train.npy'
    else:
        name_x_train = 'data_fmnist_iid/' + 'client_' + str(j) +'_x_train.npy'
        name_y_train = 'data_fmnist_iid/' + 'client_' + str(j) +'_y_train.npy'
   
    np.save(name_x_train, x_train_local)################################################
    np.save(name_y_train, y_train_local)
