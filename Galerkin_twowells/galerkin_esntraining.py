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


def galerkin_esntraining(esn,u_data):

    esn_platform = esn
    u_data_normal = u_data

    simtime = u_data.shape[0]

    initial_state = esn_platform.get_current_state()

    states = np.empty([simtime,initial_state.shape[0]])
    for k in range(simtime):
    
        y = esn_platform.update(u_data_normal[k]).flatten()
        states[k] = esn_platform.get_current_state().flatten()
    


    return states






