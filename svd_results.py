#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:45:36 2022

@author: jeanpj
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

load_dict = io.loadmat('svd_data.mat')

# save_dict = {'white_noise':white_noise,'brown_noise_list':brown_noise_list,
#              'ec_for_wn_mean':ec_for_wn_mean,'ec_for_wn_std':ec_for_wn_std,
#              'ec_for_brown_noise_mean_list':ec_for_brown_noise_mean_list,
#              'ec_for_brown_noise_std_list':ec_for_brown_noise_std_list}


white_noise = load_dict['white_noise'].flatten()
brown_noise_list = load_dict['brown_noise_list']
ec_for_wn_mean = load_dict['ec_for_wn_mean'].flatten()
ec_for_wn_std = load_dict['ec_for_wn_std'].flatten()
ec_for_brown_noise_mean_list = load_dict['ec_for_brown_noise_mean_list']
ec_for_brown_noise_std_list = load_dict['ec_for_brown_noise_std_list']


omega_list = [0.2,0.4,0.6,0.8]
neu = 2000
print(ec_for_wn_std,ec_for_wn_mean)
plt.errorbar(np.arange(1,neu+1),ec_for_wn_mean,ec_for_wn_std,linestyle='None', marker='^',label = "white noise")
for brown_noise_mean,brown_noise_std,omega in zip(ec_for_brown_noise_mean_list,ec_for_brown_noise_std_list,omega_list):
    label_string = "brown noise" + str(omega)
    plt.errorbar(np.arange(neu),brown_noise_mean,brown_noise_std,linestyle='None', marker='^',label = label_string)
plt.show()