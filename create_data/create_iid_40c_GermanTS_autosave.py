# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:47:23 2021

@author: liush
"""

#Generation of Local Data Sets.
#import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import random
import numpy as np
import pandas as pd
import math
import pickle

num_client = 40

with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)

signnames = pd.read_csv('german-traffic-signs/signnames.csv')
#print(len(signnames))
#print(type(train_data))
#print(train_data.keys())

x_train, y_train = train_data['features'], train_data['labels']

img_rows, img_cols = 32, 32# input image dimensions
num_classes = 43
input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_train /= 255

idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]

num_per = 850#x_train.shape[0]//num_client

for j in range(1,num_client+1):###############################################################
   
    if j<=num_client:
        x_train_local = x_train[int((j-1)*num_per):int(j*num_per)] #local train data
        y_train_local = y_train[int((j-1)*num_per):int(j*num_per)]
    else:
        x_train_local = x_train[int((j-1)*num_per):] #local train data
        y_train_local = y_train[int((j-1)*num_per):]

    if j <10:
        name_x_train = 'data_GermanTS_iid/' + 'client_0' + str(j) +'_x_train.npy'
        name_y_train = 'data_GermanTS_iid/' + 'client_0' + str(j) +'_y_train.npy'
    else:
        name_x_train = 'data_GermanTS_iid/' + 'client_' + str(j) +'_x_train.npy'
        name_y_train = 'data_GermanTS_iid/' + 'client_' + str(j) +'_y_train.npy'
   
    np.save(name_x_train, x_train_local)################################################
    np.save(name_y_train, y_train_local)
