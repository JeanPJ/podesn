#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as io
import RNN
import pickle
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
"""
Created on Mon Apr 25 13:05:26 2022

@author: jeanpj
"""

neu_list= [200,400,600,800,1000,1200,1400,1600,1800,2000]
energy_cutoff = [0.01,0.05,0.1]
deim_cutoff = [0.01,0.05,0.1]

reduced_esn_list = []

data = io.loadmat('data_platform.mat')


y_data = data['y']
y_max = np.array([220.0,220.0])
y_min = np.array([170.0,170.0])
y_data_normal = feature_scaling(y_data,y_max,y_min)
n_out = y_data.shape[1]
u_data = data['u']
u_max = np.array([1.0,1.0])
u_min = np.array([0.01,0.01])
u_data_normal = feature_scaling(u_data,u_max,u_min)

simtime = u_data_normal.shape[0]

pseudo_var = np.sum((y_data_normal[20000:30000] - np.mean(y_data_normal[20000:30000],axis=0)**2))

red_size = 10000

r_list = np.empty(len(neu_list))
for i,neu in enumerate(neu_list):
    
    cur_esn_string = "esn_platform_" + str(neu) +".pickle"
    cur_esn = pickle.load(open(cur_esn_string,'rb'))
    #states = galerkin_esntraining(cur_esn,u_data_normal[:red_size])
    y_esn = np.empty([simtime,n_out])
    for t0 in range(simtime):
        y_esn[t0] = cur_esn.update(u_data_normal[t0]).flatten()
        
    cur_error = np.sum((y_data_normal[20000:30000] - y_esn[20000:30000])**2,axis=0)
    r = 1 - cur_error/pseudo_var
    r_list[i] = np.mean(r)
    
print(r_list)
    
        
    