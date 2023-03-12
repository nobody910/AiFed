# -*- coding: utf-8 -*-
"""
@author: liush
"""

#Generation of Local Data Sets.
#import keras
import pickle
import pandas as pd
import random
import numpy as np
import math

with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

signnames = pd.read_csv('german-traffic-signs/signnames.csv')
#print(len(signnames))
#print(type(train_data))
#print(train_data.keys())

x_train, y_train = train_data['features'], train_data['labels']
x_val, y_val = val_data['features'], val_data['labels']
x_test, y_test = test_data['features'], test_data['labels']

#print('Training data:', x_train.shape)
#print('Validation data:', x_val.shape)
#print('Test data:', x_test.shape)

img_rows, img_cols = 32, 32# input image dimensions
num_classes = 43
input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255

'''
#save val data
name_x_val = 'data_trafficsigns_noniid/' + 'x_val.npy'
name_y_val = 'data_trafficsigns_noniid/' + 'y_val.npy'
np.save(name_x_val, x_val)################################################
np.save(name_y_val, y_val)
'''

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

for j in range(1,41):###############################################################
    #Nc = [5,10,15,20,6] # the number of classes in each local data set.
    Nc = [i for i in range(4,13)]
    Nc = Nc[random.randint(0,8)]
    #labels = [0,1,2,3,4,5,6,7,8,9] #the names of classes
    labels = [i for i in range(0,num_classes)]
    classes = random.sample(labels, Nc) #name of the classes in this client 
    classes_weight = np.zeros(shape=num_classes)
    sum_weight = 0
    
    for i in range(0,Nc):
        classes_weight[classes[i]] = random.random() #random of (0,1)
        sum_weight = sum_weight + classes_weight[classes[i]]
        
    Smin, Smax =650,1050 #the range of the local train data size
    num = random.randint(Smin,Smax) #the number of local train data size
    Pclasses = classes_weight/sum_weight*num #the train number of each classes
    
    for i in range(0,Nc):
        Pclasses[classes[i]] = math.floor(Pclasses[classes[i]]) #round down
        
    Pclasses = Pclasses.astype('int')
    idx = np.argsort(y_train) #sort,Index values from small to large
    x_train_sorted = x_train[idx]
    y_train_sorted = y_train[idx]
    idx_total = []
    
    for i in range(0,x_train.shape[0]):
        idx_total.append(i)
        
    idx_local = []
    
    for i in range(0,Nc):
        index_range =  np.argwhere(y_train_sorted == classes[i])
        index_max = max(index_range)
        index_min = min(index_range)
        if (index_max - index_min) >= Pclasses[classes[i]]:
            idx_local = idx_local+random.sample(
                idx_total[int(index_min):int(index_max)],Pclasses[classes[i]] )
        else:
            idx_local = idx_local + idx_total[int(index_min):int(index_max)]

    x_train_local = x_train_sorted[idx_local] #local train data
    y_train_local = y_train_sorted[idx_local]

    if j <10:
        name_x_train = 'data_GermanTS_noniid/' + 'client_0' + str(j) +'_x_train.npy'
        name_y_train = 'data_GermanTS_noniid/' + 'client_0' + str(j) +'_y_train.npy'
    else:
        name_x_train = 'data_GermanTS_noniid/' + 'client_' + str(j) +'_x_train.npy'
        name_y_train = 'data_GermanTS_noniid/' + 'client_' + str(j) +'_y_train.npy'

   
    np.save(name_x_train, x_train_local)################################################
    np.save(name_y_train, y_train_local)
# client_01-50 :train:1000-2000; test:300-700
# client_51-60 :train:100-200; test:30-60