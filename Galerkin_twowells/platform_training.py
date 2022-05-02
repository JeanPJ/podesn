#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:25:11 2019

@author: jean
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
#import Project_esnmpc.cyclic_ESN as RNN
#import Project_esnmpc.RNN as RNN
import RNN
import pickle
from galerkin_esntraining import *

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

neurons = [200,400,600,800,1000,1200,1400,1600,1800,2000]

for i,neu in enumerate(neurons):

    esn_4tanks = RNN.EchoStateNetwork(
        neu = neu,
        n_in = 2,
        n_out = 2,
        gama = 0.7,
        ro = 0.99,
        psi = 0.0,
        in_scale = 0.1,
        bias_scale = 0.1,
        initial_filename = "4tanks1",
        load_initial = False,
        save_initial = False,
        output_feedback = False)


    data = io.loadmat('data_platform.mat')

    y_data = data['y']
    y_max = np.array([220.0,220.0])
    y_min = np.array([170.0,170.0])
    y_data_normal = feature_scaling(y_data,y_max,y_min)

    u_data = data['u']
    u_max = np.array([1.0,1.0])
    u_min = np.array([0.01,0.01])
    u_data_normal = feature_scaling(u_data,u_max,u_min)

    training_set_size = 10000

    y_data_training = y_data_normal[:training_set_size,:]
    u_data_training = u_data_normal[:training_set_size,:]

    y_data_val = y_data_normal[training_set_size:,:]
    u_data_val = u_data_normal[training_set_size:,:]

    simtime = u_data.shape[0]

    regularization = 1e-2
    drop = 200
    esn_4tanks.offline_training(u_data_training,y_data_training,regularization,drop)
    esn_4tanks.reset()

    y_pred = np.empty_like(y_data)
    for k in range(simtime):
    
        y_pred[k] = esn_4tanks.update(u_data_normal[k]).flatten()
    

    error = (y_pred - y_data_normal)**2

    training_error = np.sqrt(np.mean(error[200:10000],axis = 0))
    val_error = np.sqrt(np.mean(error[20000:30000], axis = 0))

    print("para esse treinamento, o erro foi", training_error, "erro de validação:", val_error)

    esn_name = "esn_platform_" + str(neu) +".pickle"
    pickle_file = open(esn_name,"wb")
    pickle.dump(esn_4tanks,pickle_file)

    pickle_file.close()

    plt.plot(y_pred[45000:,0])
    plt.plot(y_data_normal[45000:,0])
    plt.show()
