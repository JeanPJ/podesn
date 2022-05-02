#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:45:11 2022

@author: jeanpj
"""

import pickle
import RNN
import numpy as np 
from GalerkinESN import *
from galerkin_esntraining import *

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

list_of_yesn = []
list_of_y_gar_lists = []
list_of_y_deim_lists_lists = []

list_of_reduced_states = []

pseudo_var = np.sum((y_data_normal[20000:30000] - np.mean(y_data_normal[20000:30000],axis=0)**2))


red_size = 10000
for i,neu in enumerate(neu_list):
    
    cur_esn_string = "esn_platform_" + str(neu) +".pickle"
    cur_esn = pickle.load(open(cur_esn_string,'rb'))
    states = galerkin_esntraining(cur_esn,u_data_normal[:red_size])
    y_esn = np.empty([simtime,n_out])
    for t0 in range(simtime):
        y_esn[t0] = cur_esn.update(u_data_normal[t0]).flatten()
        
    list_of_yesn = list_of_yesn + [y_esn]
    list_of_y_gar = []
    reduced_states = np.empty_like(energy_cutoff)
    for j,ec in enumerate(energy_cutoff):
        print("oi")
        cur_reduction = GalerkinESN(cur_esn,states,u_data_normal[:red_size],ec)
        reduced_states[j] = cur_reduction.reduced_size
        y_gar = np.empty_like(y_esn)
        for t1 in range(simtime):
            y_gar[t1] = cur_reduction.update(u_data_normal[t1],"vanilla").flatten()
        list_of_y_gar = list_of_y_gar + [y_gar]
        list_of_y_deim_lists = []
        for k,dc in enumerate(deim_cutoff):
            list_of_y_deim = []
            y_deim = np.empty_like(y_esn)
            cur_reduction.calculate_deim(dc)
            for t2 in range(simtime):
                y_deim[t2] = cur_reduction.update(u_data_normal[t2],"deim").flatten()
            list_of_y_deim = list_of_y_deim + [y_deim]
        list_of_y_deim_lists = list_of_y_deim_lists + [list_of_y_deim]
    list_of_y_gar_lists = list_of_y_gar_lists + [list_of_y_gar]
    list_of_y_deim_lists_lists = list_of_y_deim_lists_lists + [list_of_y_deim_lists]
    list_of_reduced_states = list_of_reduced_states + [reduced_states]
                
            
    
save_dict = {'y_esn_list': list_of_yesn,
             'list_of_y_gar_lists':list_of_y_gar_lists,
             'list_of_y_deim_lists':list_of_y_deim_lists_lists,
             'list_of_reduced_states':list_of_reduced_states}


io.savemat('two_wells_galerkin_data_try1.mat',save_dict)

    
    
    