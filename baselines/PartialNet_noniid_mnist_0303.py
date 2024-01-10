# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:16:36 2021

@author: liush
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import random
import os
import pandas as pd
from function import plot_result
import time

start=time.time() 

img_rows, img_cols = 28, 28
batch_size = 48 #dynamic
num_classes = 10
epochs = 2
rounds = 400 #communication round
time_range_1, time_range_2 = [3,30]
round_in_phase = 3 
set_es = [2] #MLU

k, c = 40, 0.2 #total number of clients, fraction
m = int(k*c)
(X_train, Y_train), (x_test, y_test) = mnist.load_data() #only need test set
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

type_dateset = 'noniid'
name_dataset = 'mnist'
save_path = 'results_'+type_dateset+'_'+name_dataset+'_save/PartialNet_0303_result1.xlsx'
path = 'data_'+name_dataset+'_'+type_dateset+'/'
model = load_model('w0_'+name_dataset+'.h5')#initialized model w0
file_list = []
file_list = os.listdir(path)
for s in file_list:
    #print(s)
    b = s[10] #client_01_x_train.npy, client_01_x_test.npy, etc
    path_s = path + s
    if ((b == 'x') & (len(s)==21)):
        x_train_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==21)):
        y_train_all.append(np.load(path_s))
    elif ((b=='x') & (len(s)==20)):
        x_test_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==20)):
        y_test_all.append(np.load(path_s))    

clients_index = []
for i in range(0,k):
    clients_index.append(i)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train =X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_test = keras.utils.to_categorical(y_test, num_classes)


cost_unit_model = 13.81#--------------------------------------MB
cost_dense = 12.5#--------------------------------------------MB
time_cost_all=np.zeros(shape=k)

weights = []#weights of each client
length_all = 0 #total size of data
length = np.zeros(shape = k) #size of each client
accum_time = []#accumulated time of each round
accum_cost = []#---------------------------------------------------------------accum_cost
sum_wait_time = 0
sum_cost = 0

for i in range(0,k):
    x_train = x_train_all[i]
    length[i] = len(x_train)
    weights.append(np.array(model.get_weights())) #initialize local model
    time_cost_all[i] = random.uniform(time_range_1, time_range_2)#--------------------------------

length_norm = length/sum(length)

global_model_weights = []
global_model_weights.append(model.get_weights())

global_model_test_loss = [] 
global_model_test_acc = []
length_save_all = []
length_save_test = [] 
s0_all = []

for r in range(0,rounds):
    if (r % round_in_phase) in set_es:
        flag = True
    else:
        flag = False
    s0 = random.sample(clients_index, m) #clients of rounds r
    s0_all.append(s0)
    time_thisround = []
    length_all = 0
    for i in range(0,m):
        time_thisround.append(time_cost_all[s0[i]])
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(global_model_weights[r]) #current local model
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_split=0.1)
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
    
    wait_time = max(time_thisround)
    sum_wait_time += wait_time
    accum_time.append(sum_wait_time)
    if flag == False: #only update dense layer
        sum_cost += m*cost_dense
    else:
        sum_cost += m*cost_unit_model
    accum_cost.append(sum_cost)
    weights_new = length_norm[0]*weights[0]
    for i in range(1,k):
        #weights_new = weights_new + length[s0[i]]*weights[s0[i]] # aggregate selected m
        weights_new += length_norm[i]*weights[i] # aggregate all k    
    
    if (flag == False): #only update dense layer
        for i in [0,1,2,3,6,7,8,9]:
            weights_new[i] = global_model_weights[r][i]
    
    
    model.set_weights(weights_new)     # global model update
    global_model_weights.append(model.get_weights())
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    if (r+1)<=40 or (r+1)%50==0:
        print ("round %d:"%(r+1),end = '\n')
        print('Global Model Test loss:', score[0])
        print('Global Model Test accuracy:', score[1])
        print('\n')

plot_result(global_model_test_acc,'PartialNet','accuracy')

save_name = list(zip(global_model_test_acc, global_model_test_loss,accum_time,accum_cost))
dataframe = pd.DataFrame(save_name, columns=['accuracy', 'loss','accum_time','accum_cost'])
dataframe.to_excel(save_path, index=False)

end=time.time()
run_time = end-start
print('Running time: %d Seconds'%(run_time))

import yagmail
yag = yagmail.SMTP(user = '1481209521@qq.com', password = 'lvirgyuqrrldbaga', host = 'smtp.qq.com')
yag.send(to = ['1481209521@qq.com'], 
             subject = '代码已运行完成', 
             contents = ['noniid-mnist代码已运行完成'])