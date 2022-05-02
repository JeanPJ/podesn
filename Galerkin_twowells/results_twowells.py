#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:38:27 2022

@author: jeanpj
"""
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
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

# save_dict = {'y_esn_list': list_of_yesn,
#              'list_of_y_gar_lists':list_of_y_gar_lists,
#              'list_of_y_deim_lists':list_of_y_deim_lists_lists,
#              'list_of_reduced_states':list_of_reduced_states}


#io.savemat('two_wells_galerkin_data_try1.mat',save_dict)

load_dict = io.loadmat('two_wells_galerkin_data_try1.mat')


y_esn_list = load_dict['y_esn_list']
list_of_y_gar_lists = load_dict['list_of_y_gar_lists']
list_of_y_deim_lists_lists = load_dict['list_of_y_deim_lists']
list_of_reduced_states = load_dict['list_of_reduced_states']

print(y_esn_list)

neu_list= [200,400,600,800,1000,1200,1400,1600,1800,2000]
energy_cutoff = [0.01,0.05,0.1]
deim_cutoff = [0.01,0.05,0.1]

#explanation: y_esn_list, list of y_esn_ responses over time.

real_data = io.loadmat('data_platform.mat')


y_data = real_data['y']
y_max = np.array([220.0,220.0])
y_min = np.array([170.0,170.0])
y_data_normal = feature_scaling(y_data,y_max,y_min)


full_esn_error_plot = np.empty([len(neu_list),2])
full_esn_r_plot = np.empty([len(neu_list),2])
gar_r_list = []
deim_r_list_of_lists = []
for i,neu in enumerate(neu_list):
    full_esn_error_plot[i] = np.mean(np.abs(y_data_normal - y_esn_list[i]),axis=0)
    residual_sum = np.sum((y_data_normal - y_esn_list[i])**2,axis=0)
    variance_sum = np.sum((y_data_normal - np.mean(y_data_normal,axis=0))**2,axis=0)
    full_esn_r_plot[i] = 1 - residual_sum/variance_sum
    gar_r_plot = np.empty([len(energy_cutoff),2])
    for j,ec in enumerate(energy_cutoff):
        residual_sum = np.sum((y_data_normal - list_of_y_gar_lists[i][j])**2,axis=0)
        variance_sum = np.sum(np.abs(y_data_normal - np.mean(y_data_normal,axis=0)**2),axis=0)
        gar_r_plot[j] = 1 - residual_sum/variance_sum
        
    gar_r_list += [gar_r_plot]
        


#reorganize gar list in terms of neurons



gar_r_list_neurons = []

for j,ec in enumerate(energy_cutoff):
    gar_r_plot_neurons = np.empty([len(neu_list),2])
    for n,neu in enumerate(neu_list):
        gar_r_plot_neurons[n] = gar_r_list[n][j]
        
    gar_r_list_neurons += [gar_r_plot_neurons]
    
print(gar_r_list_neurons)
    
plt.plot(neu_list,full_esn_r_plot[:,0],'o')
for j,ec in enumerate(energy_cutoff):
    plt.plot(neu_list,gar_r_list_neurons[j][:,0],'o')
    
plt.grid(True)
plt.show()
plt.plot(neu_list,full_esn_r_plot[:,1],'o')
for j,ec in enumerate(energy_cutoff):
    plt.plot(neu_list,gar_r_list_neurons[j][:,1],'o')
    
plt.grid(True)
plt.show()