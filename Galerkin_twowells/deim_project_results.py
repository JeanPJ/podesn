#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:04:46 2022

@author: jeanpj
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

load_dict = io.loadmat('deim_project_result.mat')

size_list = load_dict['size_list'].flatten()

r_list = load_dict['r_list']

plt.plot(size_list,r_list,'o')
plt.xlabel("Number of states")
plt.ylabel("R^2")
plt.grid(True)
plt.ylim([0.8,1])
plt.show()

N = 1400
ec = [1e-1,5e-2,1e-2,5e-3,1e-3,1e-4,1e-5]


plt.semilogx(ec,size_list*100/N)
plt.xlabel("Energy Cutoff")
plt.ylabel("States proportion (%)")
plt.grid(True)
plt.show()

plt.semilogx(ec,r_list,'o')
plt.xlabel("Energy Cutoff")
plt.ylabel("R^2")
plt.ylim([0.8,1])
plt.grid(True)


plt.show()