# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:47:23 2021

@author: liush
"""

#Generation of Local Data Sets.
import pickle
import random
import numpy as np

img_rows, img_cols = 32, 32# input image dimensions
num_classes = 43
num_per_class_stimulus = 5
# the data, shuffled and split between train and test sets
with open('GermanTS/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('GermanTS/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('GermanTS/test.p', 'rb') as f:
    test_data = pickle.load(f)

#signnames = pd.read_csv('german-traffic-signs/signnames.csv')
#print(len(signnames))
#print(type(train_data))
#print(train_data.keys())

x_train, y_train = train_data['features'], train_data['labels']
x_val, y_val = val_data['features'], val_data['labels']
x_test, y_test = test_data['features'], test_data['labels']

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
idx_test_total = []
for i in range(0,len(y_test)):
    idx_test_total.append(i)
idx_test_local = []
idx_test = np.argsort(y_test) #sort,Index values from small to large
y_test_sorted = y_test[idx_test]
x_test_sorted = x_test[idx_test]
for i in range(num_classes):
    index_test_range =  np.argwhere(y_test_sorted == i)
    index_test_max = max(index_test_range)
    index_test_min = min(index_test_range)
    idx_test_local = idx_test_local+random.sample(
        idx_test_total[int(index_test_min):int(index_test_max)],num_per_class_stimulus )

x_test_local = x_test_sorted[idx_test_local] #local test data
#y_test_local = y_test_sorted[idx_test_local]

name_x_test = 'data_stimulus/' + 'GermanTS' +'_x_stimulus.npy'
    #name_y_test = 'data_stimulus/' + 'client_' + str(j) +'_y_test.npy'

np.save(name_x_test, x_test_local)





    #np.save(name_y_test, y_test_local)

# client_01-50 :train:1000-2000; test:300-700
# client_51-60 :train:100-200; test:30-60