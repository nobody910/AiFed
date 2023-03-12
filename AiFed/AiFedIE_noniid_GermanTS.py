# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:15:35 2021

@author: liush
"""

import numpy as np
import tensorflow.compat.v1.keras
from tensorflow import keras
import pickle
from tensorflow.compat.v1.keras import backend as K
from information_function import compute_information_entropy
from tensorflow.keras.models import load_model
import random
import os
#import matplotlib.pyplot as plt
from function import plot_result 
import tensorflow
import pandas as pd
import time
import math
#from scipy.spatial.distance import pdist

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用第一, 三块GPU
#tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.99)
#gpu_options = tensorflow.compat.v1.GPUOptions(allow_growth=True)
#sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(gpu_options=gpu_options))  
#tensorflow.compat.v1.disable_eager_execution()
img_rows, img_cols = 32, 32
batch_size = 48
num_classes = 43
epochs = 2
rounds = 400 #communication round
time_range_1, time_range_2 = [3,30]
a = 1.36

k, c = 40, 0.2 #total number of clients, fraction
m = int(k*c)
with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)
x_val, y_val = val_data['features'], val_data['labels']
x_test, y_test = test_data['features'], test_data['labels']
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

type_dateset = 'noniid'
name_dataset = 'GermanTS'
save_path = 'results_'+type_dateset+'_'+name_dataset+'_save/AiFedIE_0601_result2.xlsx'
path = 'data_'+name_dataset+'_'+type_dateset+'/'
model = load_model('w0_'+name_dataset+'.h5')#initialized model w0
#save_path = 'results_fmnist_save/ALU_0308_result1_direct.xlsx'
#model = load_model('w0_fmnist.h5')#initialized model w0
#path = "data_fmnist_noniid/"
#------------------------------------------------------------------------------local test
#name_x_test_local = 'data_test_local/'+name_dataset+'_x_test_local.npy'
#name_y_test_local = 'data_test_local/'+name_dataset+'_y_test_local.npy'
#x_test_local = np.load(name_x_test_local)
#y_test_local = np.load(name_y_test_local)
#y_test_local = tensorflow.keras.utils.to_categorical(y_test_local, num_classes)
#local_model_acc_all = np.zeros(shape = (k,rounds))
#delta_local_model_acc_all = np.zeros(shape = (k,rounds))
#local_model_acc_m = np.zeros(shape = (m,rounds))
#delta_local_model_acc_m = np.zeros(shape = (m,rounds))
#------------------------------------------------------------------------------consistency
from compute_RDM import compute_rda
from compute_RDM import compute_rc_simp
name_x_stimulus = 'data_stimulus/' + name_dataset +'_x_stimulus.npy'
stimulus = np.load(name_x_stimulus)
#distance_measure = 'correlation', 'cosine', 'euclidean'
distance_measure = 'cosine'
layer_idx_all = [6]
#[1,2,6,7]#related to model structure, model.layers in compute_rdm function
#1,2-->cnn layers; 3-->max pooling layer; 4--> dropout layer; 5--> flatten; 6,7--> FC layers 
global_rdm_allayer =[]
for layer_idx in layer_idx_all:
    global_rdm_onelayer = compute_rda(model, stimulus, distance_measure, layer_idx-1)#initialize global_rdm
    global_rdm_allayer.append(global_rdm_onelayer)
#global_rdm_allayer = np.array(global_rdm_allayer)
flag_upload_deeplayer = np.zeros(shape = (k,rounds))
update_prob = np.ones(shape = (m,1))/m
for i in range(m):
    update_prob[i] *= (i+1) 
    
#representational_consistency_all = np.zeros(shape = (k, len(layer_idx_all), rounds))
#representational_consistency_all = np.zeros(shape = (k, rounds))
#delta_representational_consistency_all = np.zeros(shape = (k, rounds))
representational_consistency_m = np.zeros(shape = (m, rounds))
#delta_representational_consistency_m = np.zeros(shape = (m, rounds))
#save_path_rc_all = 'results_'+type_dateset+'_'+name_dataset+'_save/ALMU_0601_result1_rc_all'#-----------
save_path_rc_m = 'results_'+type_dateset+'_'+name_dataset+'_save/AiFedIE_0601_result2_rc_m'#-----------
#####################################################################

file_list = []
file_list = os.listdir(path)
for s in file_list:
    #print(s)
    b = s[10] #client_001_x_train.npy, client_001_x_test.npy, etc
    path_s = path + s
    if ((b == 'x') & (len(s)==21)):
        x_train_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==21)):
        y_train_all.append(np.load(path_s))
    elif ((b=='x') & (len(s)==20)):
        x_test_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==20)):
        y_test_all.append(np.load(path_s))  
    
start=time.time()    

s0_all = []
iw_all = np.zeros(shape = k)#informative weights
twf_all = np.zeros(shape = k)
timestamp_all = np.zeros(shape = k)
clients_index = []
for i in range(0,k):
    clients_index.append(i)

img_rows, img_cols = x_train_all[0].shape[1], x_train_all[0].shape[2]
num_channel = x_train_all[0].shape[3]

input_shape = (img_rows, img_cols, num_channel)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

global_model_weights = model.get_weights()
weights = []#weights of each client
weights_glo = []
length_all = np.zeros(shape = k)#total size of data
time_cost_all=np.zeros(shape=k)
for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    iw_all[i] = compute_information_entropy(y_train)#informative weights
    length_all[i] = len(x_train)
    weights.append(np.array(model.get_weights())) #initialize
    weights_glo.append(np.array(model.get_weights()))
    time_cost_all[i] = random.uniform(time_range_1, time_range_2)#--------------------------------

size_all = length_all/sum(length_all)
info_all = iw_all/sum(iw_all)

set_es = [0]
global_model_weights = np.array(model.get_weights())
global_model_test_loss = [] 
global_model_test_acc = [] 
#layer_dist_euclidean_all = np.zeros(shape = (8,rounds))
#layer_dist_cosine_all = np.zeros(shape = (8,rounds))
cost_unit_model = 9.97#-------------------------------------------------------MB
cost_unit_shallow = 0.97


accum_time = []#accumulated time of each round
accum_cost = []#---------------------------------------------------------------accum_cost
sum_wait_time = 0
sum_cost = 0
for r in range(0,rounds):
    s0 = random.sample(clients_index, int(m)) #clients of rounds r
    s0_all.append(s0)
    time_thisround = []   
    #length_sum = 0    
    for i in range(0,int(m)):
        time_thisround.append(time_cost_all[s0[i]])
        time_cost_all[s0[i]] = random.uniform(time_range_1, time_range_2)#new cost time for finished clients
        timestamp_all[s0[i]] = r+1#update timestamp
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(global_model_weights) #current local model
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_split=0.1)
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        #score_local = model.evaluate(x_test_local,y_test_local, verbose=0)
        #local_model_acc_all[s0[i]][r] = score_local[1]
        #local_model_acc_m[i][r] = score_local[1]
        #if r>=1:
            #delta_local_model_acc_all[s0[i]][r] = local_model_acc_all[s0[i]][r]-global_model_test_acc[r-1]
            #delta_local_model_acc_m[i][r] = local_model_acc_m[s0[i]][r]-global_model_test_acc[r-1]
    for i in range(k):
        twf_all[i] = math.pow(a,-r+timestamp_all[i]-1)    
        
    alpha_all = list(twf_all)/sum(twf_all)
    coe_all = alpha_all*size_all*info_all
    coe_norm = coe_all/sum(coe_all)
    
    wait_time = max(time_thisround)
    sum_wait_time += wait_time
    accum_time.append(sum_wait_time)
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        
        #layer_weights_new, layer_weights_old = [], []
        #layer_dist_euclidean = np.zeros(shape = 8)
        #layer_dist_cosine = np.zeros(shape = 8)
        #----------------------------------------------------------------------compute rdms
    rc_this_round = []
    for i in range(m): 
        num_layer = 0
        model.set_weights(weights[s0[i]])
        for layer_idx in layer_idx_all:
            local_rdm_onelayer = compute_rda(model, stimulus, distance_measure, layer_idx-1) 
            Pearson_coefficient_onelayer = compute_rc_simp(
                local_rdm_onelayer, global_rdm_allayer[num_layer])
            representational_consistency_onelayer = Pearson_coefficient_onelayer[0]*\
                Pearson_coefficient_onelayer[0]
            #representational_consistency_all[i][num_layer][r]= representational_consistency_onelayer
            representational_consistency_m[i][r]= representational_consistency_onelayer
            rc_this_round.append(representational_consistency_onelayer)
            #if r>=1:
                #delta_representational_consistency_all[i][r]=representational_consistency_all[i][r]\
                    #-representational_consistency_all[i][r-1]
            #num_layer+=1
        #Pearson_coefficient_layer = compute_representational_consistency(local_rdm[j],global_rdm[j])
        #representational_consistency_layer = Pearson_coefficient_layer[0]*Pearson_coefficient_layer[0]
        #representational_consistency_all[i][j][r]= representational_consistency_layer
            #layer_weights_new.append(model.get_weights()[j])#layers of the trained model
            #layer_weights_old.append(global_model_weights[r][j])#layers of the current global model
        #layer_weights_new, layer_weights_old = np.array(layer_weights_new), np.array(layer_weights_old)
    rc_this_round = np.array(rc_this_round)
    idx_rc_m = rc_this_round.argsort()#sorted idx, min first
    #idx_rc_m = rc_this_round.argsort()[::-1]#sorted idx, max first
    #sum_upload_deeplayer = 0
    for idx in idx_rc_m:
        AT = random.random()#[0,1]
        if AT >= update_prob[idx]:#upload
            flag_upload_deeplayer[s0[idx]][r] = 1
            #sum_upload_deeplayer += 1
            sum_cost += cost_unit_model
        else:#not upload
            weights[s0[idx]][4] = global_model_weights[4]
            sum_cost += cost_unit_shallow

    accum_cost.append(sum_cost)    
    weights_new = coe_norm[0]*weights[0]
    for i in range(1,k):
        weights_new += coe_norm[i]*weights[i] # aggregate

    model.set_weights(weights_new)     # global model update
    global_model_weights = model.get_weights()
    for i in range(0,m):
        weights_glo[s0[i]] = model.get_weights()
    num_layer = 0
    for layer_idx in layer_idx_all:
        global_rdm_onelayer = compute_rda(model, stimulus, distance_measure, layer_idx-1)#initialize global_rdm
        global_rdm_allayer[num_layer] = (global_rdm_onelayer)
        num_layer+=1
    #global_rdm_allayer = np.array(global_rdm_allayer)            
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    if (r+1)<=40 or (r+1)%50 ==0:
        print ("round %d:"%(r+1),end = '\n')
        print('Global Model Test loss:', score[0])
        print('Global Model Test accuracy:', score[1])
        print('\n')

plot_result(global_model_test_acc,'AiFedIE','accuracy')
save_name = list(zip(global_model_test_acc, global_model_test_loss,accum_time,accum_cost))
dataframe = pd.DataFrame(save_name, columns=['accuracy', 'loss','accum_time','accum_cost'])
dataframe.to_excel(save_path, index=False)
#------------------------------------------------------------------------------aver rc
#for i in range()

#------------------------------------------------------------------------------save rc
#np.save(save_path_rc_all, representational_consistency_all)
np.save(save_path_rc_m, representational_consistency_m)
'''
save_name_flag = list(flag_update_deeplayer)
dataframe_flag = pd.DataFrame(save_name_flag)
dataframe_flag.to_excel(save_path_flag, index=False)
'''
##############################################################################
end=time.time()
run_time = end-start
print('Running time: %d Seconds'%(run_time))


