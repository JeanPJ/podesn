#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:59:07 2022

@author: jeanpj
"""

import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np

neuron_num = [100,200,300,400,500,600,700,800,900,1000,1200,1400]
energy_cutoff = [0.01,0.05,0.1]

data_dict = io.loadmat("MC_data_from_server.mat")

# structure of data file: 
    #{'mc_coef_mean_grid':mc_coef_mean_grid,
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
#            'redneu_min_grid':redneu_min_grid,
#            'redneu_max_grid':redneu_max_grid,
#            'deimsize_mean_grid':deimsize_mean_grid,
#            'deimsize_min_grid':deimsize_min_grid,
#            'deimsize_max_grid':deimsize_max_grid}

mc_coef_mean_grid = data_dict['mc_coef_mean_grid'].flatten()
mc_coef_std_grid = data_dict['mc_coef_std_grid'].flatten()
mc_coef_min_grid = data_dict['mc_coef_min_grid'].flatten()
mc_coef_max_grid = data_dict['mc_coef_max_grid'].flatten()



plt.errorbar(neuron_num,mc_coef_mean_grid,mc_coef_std_grid,linestyle='None', marker='^',label = "full")

mc_red_coef_mean_grid = np.array(data_dict['mc_red_coef_mean_grid'])
mc_red_coef_std_grid = np.array(data_dict['mc_red_coef_std_grid'])
mc_red_coef_max_grid = np.array(data_dict['mc_red_coef_max_grid'])
mc_red_coef_min_grid = np.array(data_dict['mc_red_coef_min_grid'])
print(mc_red_coef_mean_grid)

plt.errorbar(neuron_num,mc_red_coef_mean_grid[:,0],mc_red_coef_std_grid[:,0],linestyle='None', marker='^',label = "EC 0.01")

plt.errorbar(neuron_num,mc_red_coef_mean_grid[:,1],mc_red_coef_std_grid[:,1],linestyle='None', marker='^',label = "EC 0.05")

plt.errorbar(neuron_num,mc_red_coef_mean_grid[:,2],mc_red_coef_std_grid[:,2],linestyle='None', marker='^',label = "EC 0.1")
plt.ylabel("MC")
plt.xlabel("neurons")
plt.grid(True)
plt.legend()
plt.show()



redneu_mean_grid = data_dict['redneu_mean_grid']
redneu_min_grid = data_dict['redneu_min_grid']
redneu_max_grid = data_dict['redneu_max_grid']


plt.plot(neuron_num,redneu_mean_grid[:,0],'o')
plt.plot(neuron_num,redneu_mean_grid[:,1],'o')
plt.plot(neuron_num,redneu_mean_grid[:,2],'o')
plt.grid(True)
plt.show()


# deimsize_mean_grid = data_dict['deimsize_mean_grid']
# deimsize_min_grid = data_dict['deimsize_min_grid']
# deimsize_max_grid = data_dict['deimsize_max_grid']


# plt.plot(neuron_num,deimsize_mean_grid[:,0],'o')
# plt.plot(neuron_num,deimsize_mean_grid[:,1],'o')
# plt.plot(neuron_num,deimsize_mean_grid[:,2],'o')
# plt.grid(True)
# plt.show()


# mc_deim_coef_mean_grid = np.array(data_dict['mc_deim_coef_mean_grid'])
# mc_deim_coef_std_grid = np.array(data_dict['mc_deim_coef_std_grid'])

# print(mc_red_coef_mean_grid)
# plt.errorbar(neuron_num,mc_coef_mean_grid,mc_coef_std_grid,linestyle='None', marker='^',label = "full")

# plt.errorbar(neuron_num,mc_deim_coef_mean_grid[:,0],mc_deim_coef_std_grid[:,0],linestyle='None', marker='^',label = "EC 0.01")

# plt.errorbar(neuron_num,mc_deim_coef_mean_grid[:,1],mc_deim_coef_std_grid[:,1],linestyle='None', marker='^',label = "EC 0.05")

# plt.errorbar(neuron_num,mc_deim_coef_mean_grid[:,2],mc_deim_coef_std_grid[:,2],linestyle='None', marker='^',label = "EC 0.1")
# plt.ylabel("MC")
# plt.xlabel("neurons")
# plt.grid(True)
# plt.legend()
# plt.show()


plt.errorbar(neuron_num,mc_coef_mean_grid,mc_coef_std_grid,linestyle='None', marker='^',label = "full")
plt.errorbar(redneu_mean_grid[:,0],mc_red_coef_mean_grid[:,0],mc_red_coef_std_grid[:,0],linestyle='None', marker='^',label = "EC 0.01")

plt.errorbar(redneu_mean_grid[:,1],mc_red_coef_mean_grid[:,1],mc_red_coef_std_grid[:,1],linestyle='None', marker='^',label = "EC 0.05")

plt.errorbar(redneu_mean_grid[:,2],mc_red_coef_mean_grid[:,2],mc_red_coef_std_grid[:,2],linestyle='None', marker='^',label = "EC 0.1")

plt.plot(neuron_num,np.ones(len(neuron_num))*mc_red_coef_mean_grid[-1,2],"--",label="best MC (EC0.1)")
plt.plot(neuron_num,np.ones(len(neuron_num))*mc_red_coef_mean_grid[-1,1],"--",label="best MC (EC0.05)")
plt.plot(neuron_num,np.ones(len(neuron_num))*mc_red_coef_mean_grid[-1,0],"--",label="best MC (EC0.01)")

plt.ylabel("MC")
plt.xlabel("neurons/reduced size")
plt.grid(True)
plt.legend()
plt.show()