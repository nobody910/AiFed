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
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

for j in range(1,41):###############################################################
    Nc = [2,3,4,5,6] # the number of classes in each local data set.
    Nc = Nc[random.randint(0,4)]
    labels = [0,1,2,3,4,5,6,7,8,9] #the names of classes
    classes = random.sample(labels, Nc) #name of the classes in this client 
    classes_weight = np.zeros(shape=10)
    sum_weight = 0
    
    for i in range(0,Nc):
        classes_weight[classes[i]] = random.random() #random of (0,1)
        sum_weight = sum_weight + classes_weight[classes[i]]
        
    Smin, Smax =1000,2000 #the range of the local train data size
    num = random.randint(Smin,Smax) #the number of local train data size
    Pclasses = classes_weight/sum_weight*num #the train number of each classes    
    
    for i in range(0,Nc):
        Pclasses[classes[i]] = math.floor(Pclasses[classes[i]]) #round down
        
    Pclasses = Pclasses.astype('int')
    idx = np.argsort(y_train) #sort,Index values from small to large
    x_train_sorted = x_train[idx]
    y_train_sorted = y_train[idx]
    idx_total = []
    idx_test_total = []
    
    for i in range(0,60000):
        idx_total.append(i)        
        
    idx_local = []
    
    for i in range(0,Nc):
        index_range =  np.argwhere(y_train_sorted == classes[i])
        index_max = max(index_range)
        index_min = min(index_range)
        idx_local = idx_local+random.sample(
            idx_total[int(index_min):int(index_max)],Pclasses[classes[i]] )

    x_train_local = x_train_sorted[idx_local] #local train data
    y_train_local = y_train_sorted[idx_local]

    if j <10:
        name_x_train = 'data_fmnist_noniid/' + 'client_0' + str(j) +'_x_train.npy'
        name_y_train = 'data_fmnist_noniid/' + 'client_0' + str(j) +'_y_train.npy'
    else:
        name_x_train = 'data_fmnist_noniid/' + 'client_' + str(j) +'_x_train.npy'
        name_y_train = 'data_fmnist_noniid/' + 'client_' + str(j) +'_y_train.npy'
   
    np.save(name_x_train, x_train_local)################################################
    np.save(name_y_train, y_train_local)
