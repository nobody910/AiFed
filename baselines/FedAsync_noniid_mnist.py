# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:16:36 2021

@author: liush
"""
import pickle
import numpy as np
import os
import tensorflow
tensorflow.device('/device:GPU:2')
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DECIVES']='3'
#gpus = tensorflow.config.list_physical_devices("GPU")
 
#if gpus:
   # gpu0 = gpus[2] #如果有多个GPU，仅使用第0个GPU
   # tensorflow.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import random
import pandas as pd
from function import plot_result
import time
import math


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
time_range_1, time_range_2 = [3,30]
m = 1
(X_train, Y_train), (x_test, y_test) = mnist.load_data() #only need test set
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

type_dateset = 'noniid'
name_dataset = 'mnist'
save_path = 'results_'+type_dateset+'_'+name_dataset+'_save/FedAsync_0525_result1.xlsx'
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
    X_train =X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, num_channel)

#input_shape = (img_rows, img_cols, num_channel)
y_test = keras.utils.to_categorical(y_test, num_classes)
index_test = np.arange(0,x_test.shape[0])

weights = []#weights of each client
weights_glo = []
length_all = 0 #total size of data
length = np.zeros(shape = k) #size of each client

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

global_model_weights = np.array(model.get_weights())
length_norm = length/sum(length)

global_model_test_loss = [] 
global_model_test_acc = []
length_save_all = []
length_save_test = [] 
s0_all = []

def my_loss(w_global,w_local,mu):
    def loss(y_true,y_pred):
        proximal_term = 0
        for w_g, w_l in zip(w_global, w_local):
            proximal_term += np.linalg.norm([w_g,w_l])
        return tensorflow.keras.losses.categorical_crossentropy(y_true,y_pred) +mu/2*proximal_term
    return loss

for r in range(0,rounds):
    tmp_index = time_cost_all.argsort() #ascending sorted index by time
    s0 = tmp_index[:m]#clients of current round
    #s0 = random.sample(clients_index, m) #clients of rounds r
    s0_all.append(s0)
    wait_time = time_cost_all[s0[m-1]]# m th client cost time as wait time
    sum_wait_time += wait_time
    accum_time.append(sum_wait_time)
    sum_cost += m*cost_unit_model
    accum_cost.append(sum_cost)
    weights_new = length_norm[0]*weights[0]
    for i in range(m,k):
        time_cost_all[tmp_index[i]] -= wait_time#update cost time for unfinished clients
        
    for i in range(0,m): #new cost time for finished clients
        time_cost_all[s0[i]] = random.uniform(time_range_1, time_range_2)
        twf_all[s0[i]] = math.pow(r-timestamp_all[s0[i]]+1, (-a_async))
    
    #length_save_all.append(length_all)
    #length_save_test.append(length_test)
    length_all = 0
    for i in range(0,m):
        #length_all += length[s0[i]]
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(weights_glo[s0[i]]) #current local model
        model.compile(\
                      loss = my_loss(w_global=weights_glo[s0[i]],w_local=model.get_weights(),mu=mu),
                      #loss=keras.losses.categorical_crossentropy +0.01/2*\
                      #np.linalg.norm([np.array(global_model_weights[r]),np.array(model.get_weights())], ord=2),
                      #optimizer=tensorflow.keras.optimizers.gradient_descent_v2.SGD(lr = 0.003),
                      optimizer=tensorflow.keras.optimizers.SGD(lr = 0.003),
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_split=0.1)
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        timestamp_all[s0[i]] = r+1#update timestamp
    
    alpha_all = list(twf_all)
    weights_new = alpha_all[s0[0]]*weights[s0[0]] + (1-alpha_all[s0[0]])*np.array(global_model_weights)
    for i in range(1,m):
        #weights_new = weights_new + length[s0[i]]*weights[s0[i]] # aggregate selected m
        weights_new=alpha_all[s0[i]]*weights[s0[i]]+(1-alpha_all[s0[0]])*weights_new    
    
    
    model.set_weights(weights_new)     # global model update
    global_model_weights = model.get_weights()
    for i in range(0,m):
        weights_glo[s0[i]] = model.get_weights()
    
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    if (r+1)<=40 or (r+1)%50==0:
        print ("round %d:"%(r+1),end = '\n')
        print('Global Model Test loss:', score[0])
        print('Global Model Test accuracy:', score[1])
    

plot_result(global_model_test_acc,'FedAsync','accuracy')

save_name = list(zip(global_model_test_acc, global_model_test_loss,accum_time,accum_cost))
dataframe = pd.DataFrame(save_name, columns=['accuracy', 'loss','accum_time','accum_cost'])
dataframe.to_excel(save_path, index=False)

end=time.time()
run_time = end-start
print('Running time: %d Seconds'%(run_time))