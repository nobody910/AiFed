# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:16:36 2021

@author: liush
"""
import pickle
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import random
import os
import pandas as pd
from function import plot_result
import time
import math

os.environ['CUDA_VISIBLE_DEVICES']='0'
start=time.time() 

#img_rows, img_cols = 32, 32
batch_size = 48 #dynamic
num_classes = 10
epochs = 2
rounds = 3200 #communication round
mu = 1
a_async = 0.5
k, c = 40, 0.2 #total number of clients, fraction
#m = int(k*c)
time_range_1, time_range_2 = [3,40]
m = 1
(X_train, Y_train), (x_test, y_test) = mnist.load_data() #only need test set
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

type_dateset = 'noniid'
name_dataset = 'mnist'
save_path = 'results_'+type_dateset+'_'+name_dataset+'_save/FedConD_0226_result1.xlsx'
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

img_rows, img_cols = x_train_all[0].shape[1], x_train_all[0].shape[2]
num_channel = x_train_all[0].shape[3]
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (num_channel, img_rows, img_cols)
else:
    X_train =X_train.reshape(X_train.shape[0], img_rows, img_cols, num_channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channel)
    input_shape = (img_rows, img_cols, num_channel)

#input_shape = (img_rows, img_cols, num_channel)
y_test = keras.utils.to_categorical(y_test, num_classes)
index_test = np.arange(0,x_test.shape[0])

weights = []#weights of each client
weights_glo = []
global_model_weights = []
global_model_weights.append(model.get_weights())
length_all = 0 #total size of data
length = np.zeros(shape = k) #size of each client
update_fre_all = np.zeros(shape = k)

cost_unit_model = 13.81#--------------------------------------MB

time_cost_all = np.zeros(shape=k)
twf_all = np.zeros(shape = k)
timestamp_all = np.zeros(shape = k)

accum_time = []#accumulated time of each round
accum_cost = []#---------------------------------------------------------------accum_cost
sum_wait_time = 0
sum_cost = 0

for i in range(0,k):
    x_train = x_train_all[i]
    length[i] = len(x_train)
    weights.append(np.array(model.get_weights())) #initialize local model
    weights_glo.append(np.array(model.get_weights()))
    time_cost_all[i] = random.uniform(time_range_1, time_range_2)

length_norm = length/sum(length)
global_model_test_loss = [] 
global_model_test_acc = []
s0_all = []

def my_loss(w_global,w_local,mu):
    def loss(y_true,y_pred):
        proximal_term = 0
        for w_g, w_l in zip(w_global, w_local):
            proximal_term += np.linalg.norm([w_g,w_l])
        return tensorflow.keras.losses.categorical_crossentropy(y_true,y_pred) +mu/2*proximal_term
    return loss

for r in range(0,1):
    act_flag_all = np.zeros(shape = k)
    index_fre = np.argsort(update_fre_all)
    n_select1 = int(0.2*k)#select top 0.5*k data size
    client_index_fre = index_fre[:n_select1] #select top 0.5*k client
    s0_unact = index_fre[n_select1:]#--------------------------------------------unactivated
    for act_idx in client_index_fre:
        act_flag_all[act_idx] = 1
    for i in range(k):
        if act_flag_all[i] == 0:
            time_cost_all[i] = time_range_2 + 1
    
    tmp_index = time_cost_all.argsort() #ascending sorted index by time#-------
    s0 = tmp_index[:m]#clients of current round
    #s0 = random.sample(list(client_index_acw1), m) #selected clients of rounds r
    s0_wait_time = []#---------------------------------------------------------
    for i in range(0,m):
        s0_wait_time.append(time_cost_all[s0[i]])
        time_cost_all[s0[i]] = random.uniform(time_range_1, time_range_2)#-----
    wait_time = max(s0_wait_time)
    sum_wait_time += wait_time
    accum_time.append(sum_wait_time)
    for i in range(m,k):
        time_cost_all[tmp_index[i]] -= wait_time#update cost time for unfinished clients
    sum_cost += m*cost_unit_model
    accum_cost.append(sum_cost)
    
    for i in range(0,m):
        timestamp_all[s0[i]] = r+1
        #length_all += length_all_new[s0[i]]
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(global_model_weights[r]) #current local model
        model.compile(\
                      loss = my_loss(w_global=global_model_weights[r],w_local=model.get_weights(),mu=mu),
                      #loss=keras.losses.categorical_crossentropy +0.01/2*\
                      #np.linalg.norm([np.array(global_model_weights[r]),np.array(model.get_weights())], ord=2),
                      #optimizer=tensorflow.keras.optimizers.gradient_descent_v2.SGD(lr = 0.003),
                      optimizer=keras.optimizers.SGD(lr = 0.003),
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
         validation_split = 0.1)#--------------------------------------
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        update_fre_all[s0[i]] += 1#--------------------------------------------
    

    weights_new = np.array(global_model_weights[-1]) - length_norm[s0[0]]*\
        (np.array(global_model_weights[-1])-np.array(model.get_weights()))
    model.set_weights(weights_new)     # global model update
    global_model_weights.append(model.get_weights())

    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])

    print ("round %d:"%(r+1),end = '\n')
    print('Global Model Test loss:', global_model_test_loss[r])
    print('Global Model Test accuracy:', global_model_test_acc[r])
    #print('train_all',length_all)
    #print('test_all:',length_test)
    print('\n')
    
#------------------------------------------------------------------------------
for r in range(1,rounds):
    #act_flag_all = np.zeros(shape = k)
    s0_unact = list(s0_unact)+list(s0)

    fre_un = []
    for i in s0_unact:
        fre_un.append(update_fre_all[i])
    fre_un = np.array(fre_un)
    s0_unact = np.array(s0_unact)
    index_fre = np.argsort(fre_un)
    s0_un_sort = s0_unact[index_fre]
    
    n_select1 = m#select top 0.8*k data size
    s0_act = s0_un_sort[:n_select1]#--------------------------------activated   
    s0_unact = s0_un_sort[n_select1:]#--------------------------------------------unactivated
    for i in s0_act:
        time_cost_all[i] = random.uniform(time_range_1, time_range_2)#---------
    for i in s0_unact:
        time_cost_all[i] = time_range_2 + 1
    
    tmp_index = time_cost_all.argsort() #ascending sorted index by time#-------
    s0 = tmp_index[:m]#clients of current round
    #s0 = random.sample(list(client_index_acw1), m) #selected clients of rounds r
    s0_wait_time =[]
    for i in range(0,m):
        s0_wait_time.append(time_cost_all[s0[i]])
        time_cost_all[s0[i]] = random.uniform(time_range_1, time_range_2)#-----
    wait_time = max(s0_wait_time)
    sum_wait_time += wait_time
    accum_time.append(sum_wait_time)
    for i in range(m,k):
        time_cost_all[tmp_index[i]] -= wait_time#update cost time for unfinished clients

    sum_cost += m*cost_unit_model
    accum_cost.append(sum_cost)
    
    num_train = 0  #-----------------------------------------------------------  
    length_all = 0 #sum size of seclected clients
    for i in range(0,m):
        #length_all += length_all_new[s0[i]]
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(global_model_weights[int(timestamp_all[s0[i]])]) #current local model
        old_model_weight = np.array(model.get_weights())
        timestamp_all[s0[i]] = r+1
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
         validation_split = 0.1)#--------------------------------------
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        update_fre_all[s0[i]] += 1#--------------------------------------------   
  
    weights_new = np.array(global_model_weights[-1]) - length_norm[s0[0]]*\
        (old_model_weight-np.array(model.get_weights()))
    
    model.set_weights(weights_new)     # global model update
    global_model_weights.append(model.get_weights())
    
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    
    if ((r+1)%100)==0:
        print ("round %d:"%(r+1),end = '\n')
        print('Global Model Test loss:', global_model_test_loss[r])
        print('Global Model Test accuracy:', global_model_test_acc[r])
    #print('train_all',length_all)
    #print('test_all:',length_test)
        print('\n')    

plot_result(global_model_test_acc,'FedConD','accuracy')

save_name = list(zip(global_model_test_acc, global_model_test_loss,accum_time,accum_cost))
dataframe = pd.DataFrame(save_name, columns=['accuracy', 'loss','accum_time','accum_cost'])
dataframe.to_excel(save_path, index=False)

end=time.time()
run_time = end-start
print('Running time: %d Seconds'%(run_time))

import yagmail
yag = yagmail.SMTP(user = '1481209521@qq.com', password = 'lvirgyuqrrldbaga', host = 'smtp.qq.com')
yag.send(to = ['1481209521@qq.com'], subject = '72服务器代码', contents = ['FedConD-noniid-minst代码已运行完成'])