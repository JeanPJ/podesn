#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:25:56 2022

@author: jeanpj
"""

import numpy as np
import scipy.io as io
import RNN
import pickle

from GalerkinESN import *
from galerkin_esntraining import *


neu = 1400
cur_esn_string = "esn_platform_" + str(neu) +".pickle"
cur_esn = pickle.load(open(cur_esn_string,'rb'))


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
pseudo_var = np.sum((y_data_normal[30000:] - np.mean(y_data_normal[30000:],axis=0)**2))
red_size = 10000
states = galerkin_esntraining(cur_esn,u_data_normal[:red_size])

simtime = u_data_normal.shape[0]

ec = [1e-1,5e-2,1e-2,5e-3,1e-3,1e-4,1e-5]
dc = 1e-5


size_list = np.empty(len(ec))
r_list = np.empty([len(ec),n_out])
for k,ec_num in enumerate(ec):
    red_esn = GalerkinESN(cur_esn,states,u_data_normal[:red_size],ec_num,dc)
    size_list[k] = red_esn.reduced_size
    y_esn = np.empty([simtime,n_out])
    y_gar = np.empty_like(y_esn)
    y_deim = np.empty_like(y_esn)
    for t1 in range(simtime):
        y_gar[t1] = red_esn.update(u_data_normal[t1],"vanilla").flatten()
    


    cur_error_pod = np.sum((y_data_normal[30000:] - y_gar[30000:])**2,axis=0)





    r_list[k] = 1 - cur_error_pod/pseudo_var



save_dict = {'r_list':r_list,'size_list':size_list}
io.savemat('pod_project_result.mat',save_dict)

#print(r_esn,"R for ESN",r_gar,"R for POD",r_deim,"R for POD-DEIM")