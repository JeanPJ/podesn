#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:54:43 2020

@author: jean-jordanou
"""

import scipy.io as io
import scipy.linalg as sla
import RNN as RNN
import pickle
import numpy.linalg as nla
import numpy as np
import matplotlib.pyplot as plt
from GalerkinESN import *


def change_or_not(x,min_val,max_val):
    y = 0
    p_change = np.random.rand()
    if p_change < 0.5:
        y = x
    else:
        y = min_val + (max_val - min_val)*np.random.rand()
    return y

def RFRAS(min,max,num_steps,minimum_step):
    RFRAS_sig = np.empty(num_steps)
    val = min + (max - min)*np.random.rand()
    for i in range(num_steps):

        if i % minimum_step  == 0:
            val = change_or_not(val,min,max)


        RFRAS_sig[i] = val

    return RFRAS_sig

def feature_scaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = (x - xmin)/(xmax - xmin)
    else:
        y = x
    return y

def feature_descaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = x*(xmax-xmin) + xmin
    else:
        y = x

    return y








data = io.loadmat('data_platform.mat')

y_data = data['y']
y_max = np.array([220.0,220.0])
y_min = np.array([170.0,170.0])
y_data_normal = feature_scaling(y_data,y_max,y_min)

u_data = data['u']
u_max = np.array([1.0,1.0])
u_min = np.array([0.01,0.01])
u_data_normal = feature_scaling(u_data,u_max,u_min)

training_set_size = 3000

y_data_training = y_data_normal[:training_set_size,:]
u_data_training = u_data_normal[:training_set_size,:]

y_data_val = y_data_normal[training_set_size:,:]
u_data_val = u_data_normal[training_set_size:,:]

simtime = u_data.shape[0]

regularization = 1e-5
drop = 200

y_pred = np.empty_like(y_data)
y_pred_gar1 = np.empty_like(y_data)
y_pred_gar2 = np.empty_like(y_data)

y_pred_gar_original = np.empty_like(y_data)
y_pred_gar_original2 = np.empty_like(y_data)

y_pred_deim = np.empty_like(y_data)


for k in range(simtime):
    
    y_pred[k] = esn_platform.update(u_data_normal[k]).flatten()
    y_pred_gar1[k] = reduced_esn.update(u_data_normal[k],case = 'case1').flatten()
    y_pred_gar2[k] = reduced_esn.update(u_data_normal[k],case = 'case2').flatten()
    y_pred_gar_original[k] = reduced_esn.update(u_data_normal[k],case = 'vanilla').flatten()
    y_pred_deim[k] = reduced_esn.update(u_data_normal[k],case = 'deim').flatten()
    y_pred_gar_original2[k] = reduced_esn2.update(u_data_normal[k],case = 'vanilla').flatten()
    

error = np.abs(y_pred - y_data_normal)
error_gar1 = np.abs(y_pred_gar1 - y_data_normal)
error_gar2 = np.abs(y_pred_gar2 - y_data_normal)
error_gar3 = np.abs(y_pred_gar_original - y_data_normal)
error_deim = np.abs(y_pred_deim - y_data_normal)

training_error = np.mean(error[200:40000],axis = 0)
val_error = np.mean(error[40000:], axis = 0)

training_error_gar1 = np.mean(error_gar1[200:40000],axis = 0)
val_error_gar1 = np.mean(error_gar1[40000:], axis = 0)

training_error_gar2 = np.mean(error_gar2[200:40000],axis = 0)
val_error_gar2 = np.mean(error_gar2[40000:], axis = 0)

training_error_gar3 = np.mean(error_gar3[200:40000],axis = 0)
val_error_gar3 = np.mean(error_gar3[40000:], axis = 0)

training_error_deim = np.mean(error_deim[200:40000],axis = 0)
val_error_deim = np.mean(error_deim[40000:], axis = 0)

print("para a rede original, o erro foi", training_error, "erro de validação:", val_error)

print("para a rede reduzida 1, o erro foi", training_error_gar1, "erro de validação:", val_error_gar1)
print("para a rede reduzida 2, o erro foi", training_error_gar2, "erro de validação:", val_error_gar2)
print("para a rede reduzida sem simplificação, o erro foi", training_error_gar3, "erro de validação:", val_error_gar3)
print("para a rede reduzida com DEIM, o erro foi", training_error_deim, "erro de validação:", val_error_deim)


plt.plot(y_pred[48000:,0])
#plt.plot(y_pred_gar1)
plt.plot(y_pred_gar1[48000:,0])
plt.show()

n = 400

T = reduced_esn.V

# x = Tz T -> 400 x 36
e_bound = (1 + np.sqrt(2*n))*(1/np.linalg.norm(T[:,0],ord=np.inf))*np.linalg.norm(np.eye(n) - T@T.T)

print(e_bound)



plt.plot(y_pred_deim[40000:42000,0],label="DEIM")
plt.plot(y_pred_gar_original[40000:42000,0],label="pure POD")
plt.plot(y_pred[40000:42000,0],label="original ESN")
plt.plot(y_data_normal[40000:42000,0],label = "data")
plt.legend()
plt.grid(True)
plt.show()


simtime_test = 1000

u_test = np.empty([simtime_test,2])
u_test[:,0] = RFRAS(0.1,1.0,simtime_test,200)
u_test[:,1] = RFRAS(0.1,1.0,simtime_test,200)

y_pred_test = np.empty([simtime_test,2])
y_gar_test = np.empty([simtime_test,2])
y_deim_test = np.empty([simtime_test,2])
y_gar2_test = np.empty([simtime_test,2])


for k in range(simtime_test):
    
    
    y_pred_test[k] = esn_platform.update(u_test[k]).flatten()
    y_gar_test[k] = reduced_esn.update(u_test[k],case = 'vanilla').flatten()
    y_gar2_test[k] = reduced_esn2.update(u_test[k],case = 'vanilla').flatten()
    y_deim_test[k] = reduced_esn.update(u_test[k],case = 'deim').flatten()


plt.plot(y_deim_test[:,0],label="DEIM (m = 230, 36 N)")
plt.plot(y_gar_test[:,0],label="pure POD (36 N)")
plt.plot(y_gar2_test[:,0],label="pure POD (11 N)")
plt.plot(y_pred_test[:,0],label="original ESN")
plt.legend()
plt.grid(True)
ax = plt.gca()

plt.ylabel("output")
plt.xlabel("simulation time")
plt.show()