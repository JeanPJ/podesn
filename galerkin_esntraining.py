#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:58:49 2020

@author: jean-jordanou
"""

import pickle
import numpy as np
import RNN as RNN
import scipy.io as io

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



pickle_file = open("esn_platform_dummy.pickle","rb")
esn_platform = pickle.load(pickle_file)


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


initial_state = esn_platform.get_current_state()

states = np.empty([simtime,initial_state.shape[0]])
for k in range(simtime):
    
    y_pred[k] = esn_platform.update(u_data_normal[k]).flatten()
    states[k] = esn_platform.get_current_state().flatten()
    

error = np.abs(y_pred - y_data_normal)

training_error = np.mean(error[200:40000],axis = 0)
val_error = np.mean(error[40000:], axis = 0)

print("para esse treinamento, o erro foi", training_error, "erro de validação:", val_error)

training_error_full = training_error
val_error_full = val_error


file_states = open("states.mat","wb")
io.savemat(file_states,{'states':states})






