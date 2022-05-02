#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:43:36 2022

@author: jeanpj
"""

import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt



# save_dict = {'mc_coef_mean_grid':mc_coef_mean_grid,
#              'mc_coef_std_grid':mc_coef_std_grid,
#              'mc_coef_min_grid':mc_coef_min_grid,
#              'mc_coef_max_grid':mc_coef_max_grid,
#              'mc_red_coef_mean_grid':mc_red_coef_mean_grid,
#              'mc_red_coef_std_grid':mc_red_coef_std_grid,
#              'mc_red_coef_min_grid':mc_red_coef_min_grid,
#              'mc_red_coef_max_grid':mc_red_coef_max_grid,
#              'mc_deim_coef_mean_grid':mc_deim_coef_mean_grid,
#              'mc_deim_coef_std_grid':mc_deim_coef_std_grid,
#              'mc_deim_coef_min_grid':mc_deim_coef_min_grid,
#            'mc_deim_coef_max_grid':mc_deim_coef_max_grid,
#            'redneu_mean_grid':redneu_mean_grid,
#            'redneu_std_grid':redneu_std_grid,
#            'redneu_min_grid':redneu_min_grid,
#            'redneu_max_grid':redneu_max_grid,
#            'deimsize_mean_grid':deimsize_mean_grid,
#            'deimsize_min_grid':deimsize_min_grid,
#            'deimsize_std_grid':deimsize_std_grid,
#            'deimsize_max_grid':deimsize_max_grid}


#Task 1, get all the MC_data

mc_data_string = "MC_data"


data_dict = io.loadmat("MC_data0")

mc_coef_mean_grid = data_dict['mc_coef_mean_grid'].flatten()
mc_coef_std_grid = data_dict['mc_coef_std_grid'].flatten()
mc_coef_min_grid = data_dict['mc_coef_min_grid'].flatten()
mc_coef_max_grid = data_dict['mc_coef_max_grid'].flatten()

mc_red_coef_mean_grid = np.array(data_dict['mc_red_coef_mean_grid'])
mc_red_coef_std_grid = np.array(data_dict['mc_red_coef_std_grid'])
mc_red_coef_max_grid = np.array(data_dict['mc_red_coef_max_grid'])
mc_red_coef_min_grid = np.array(data_dict['mc_red_coef_min_grid'])

redneu_mean_grid = data_dict['redneu_mean_grid']
redneu_min_grid = data_dict['redneu_min_grid']
redneu_max_grid = data_dict['redneu_max_grid']


mc_deim_coef_mean_grid = np.array(data_dict['mc_deim_coef_mean_grid'])
mc_deim_coef_std_grid = np.array(data_dict['mc_deim_coef_std_grid'])


deimsize_mean_grid = data_dict['deimsize_mean_grid']
deimsize_std_grid = data_dict['deimsize_std_grid']


for i in range(1,6):
    filename = mc_data_string + str(i)
    
    data_dict = io.loadmat(filename)
    
    mc_coef_mean_grid += data_dict['mc_coef_mean_grid'].flatten()
    mc_coef_std_grid += data_dict['mc_coef_std_grid'].flatten()
    mc_coef_min_grid += data_dict['mc_coef_min_grid'].flatten()
    mc_coef_max_grid += data_dict['mc_coef_max_grid'].flatten()

    mc_red_coef_mean_grid += np.array(data_dict['mc_red_coef_mean_grid'])
    mc_red_coef_std_grid += np.array(data_dict['mc_red_coef_std_grid'])
    mc_red_coef_max_grid += np.array(data_dict['mc_red_coef_max_grid'])
    mc_red_coef_min_grid += np.array(data_dict['mc_red_coef_min_grid'])

    redneu_mean_grid += data_dict['redneu_mean_grid']
    redneu_min_grid += data_dict['redneu_min_grid']
    redneu_max_grid += data_dict['redneu_max_grid']


    mc_deim_coef_mean_grid += np.array(data_dict['mc_deim_coef_mean_grid'])
    mc_deim_coef_std_grid += np.array(data_dict['mc_deim_coef_std_grid'])
    
    deimsize_mean_grid += data_dict['deimsize_mean_grid']
    deimsize_std_grid += data_dict['deimsize_std_grid']
    

mc_coef_mean_grid = mc_coef_mean_grid/6
mc_coef_std_grid = mc_coef_std_grid/6
mc_coef_min_grid = mc_coef_min_grid/6
mc_coef_max_grid = mc_coef_max_grid/6

mc_red_coef_mean_grid = mc_red_coef_mean_grid/6
mc_red_coef_std_grid = mc_red_coef_std_grid/6
mc_red_coef_max_grid = mc_red_coef_max_grid/6
mc_red_coef_min_grid = mc_red_coef_min_grid/6
redneu_mean_grid = redneu_mean_grid/6
redneu_min_grid = redneu_min_grid/6
redneu_max_grid = redneu_max_grid/6

mc_deim_coef_mean_grid = mc_deim_coef_mean_grid/6
mc_deim_coef_std_grid = mc_deim_coef_std_grid/6


deimsize_mean_grid /= 6 
deimsize_std_grid /= 6

neuron_num = [100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200]
energy_cutoff = [0.01,0.05,0.1]
deim_cutoff = [0.01,0.05,0.1,0.2]

#PLOT 1: Full neurons vs POD

plt.errorbar(neuron_num,mc_coef_mean_grid,mc_coef_std_grid,linestyle='None', marker='^',label = "full ESN")
plt.errorbar(neuron_num,mc_red_coef_mean_grid[:,0],mc_red_coef_std_grid[:,0],linestyle='None', marker='^',label = "EC= 0.01")
plt.errorbar(neuron_num,mc_red_coef_mean_grid[:,1],mc_red_coef_std_grid[:,1],linestyle='None', marker='^',label = "EC= 0.05")
plt.errorbar(neuron_num,mc_red_coef_mean_grid[:,2],mc_red_coef_std_grid[:,2],linestyle='None', marker='^',label = "EC= 0.1")
plt.grid(True)
plt.ylim([19.5,20.0])
plt.legend()
plt.ylabel("MC")
plt.xlabel("neurons of original ESN")
plt.show()


#plot 2: X axis is the number of states.
plt.errorbar(neuron_num,mc_coef_mean_grid,mc_coef_std_grid,linestyle='None', marker='^',label = "full")
plt.errorbar(redneu_mean_grid[:,0],mc_red_coef_mean_grid[:,0],mc_red_coef_std_grid[:,0],linestyle='None', marker='^',label = "EC 0.01")

plt.errorbar(redneu_mean_grid[:,1],mc_red_coef_mean_grid[:,1],mc_red_coef_std_grid[:,1],linestyle='None', marker='^',label = "EC 0.05")

plt.errorbar(redneu_mean_grid[:,2],mc_red_coef_mean_grid[:,2],mc_red_coef_std_grid[:,2],linestyle='None', marker='^',label = "EC 0.1")

plt.plot(neuron_num,np.ones(len(neuron_num))*mc_red_coef_mean_grid[-1,2],"--",label="best MC (EC0.1)")
plt.plot(neuron_num,np.ones(len(neuron_num))*mc_red_coef_mean_grid[-1,1],"--",label="best MC (EC0.05)")
plt.plot(neuron_num,np.ones(len(neuron_num))*mc_red_coef_mean_grid[-1,0],"--",label="best MC (EC0.01)")

plt.ylabel("MC")
plt.xlabel("number of model states")
plt.ylim([19.5,20.0])
plt.grid(True)
plt.legend()
plt.show()
    
