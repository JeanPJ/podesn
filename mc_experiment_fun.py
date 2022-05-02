#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:45:01 2021
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Wed Nov 17 16:45:01 2021

@author: jeanpj
"""

import numpy as np
import RNN
# import matplotlib.pyplot as plt
import scipy.linalg as sla
import numpy.linalg as nla
import scipy.io as io
from GalerkinESN import GalerkinESN
import os.path

def k_delayed_signal(signal,k):
    
    
    
    input_signal = signal[k:,:]
    delayed_signal = signal[:-k,:]
    return input_signal,delayed_signal

def col_2d(x):
    
    y = x.reshape([x.size,1])
    return y

def memory_capacity_k(delayed_signal,network_signal):
    
    inp = np.hstack([delayed_signal,network_signal])
    mc = np.corrcoef(inp,rowvar = False)
    
    return mc


#EXPERIMENT SPECIFICATIONS
total_mc = 100
ran_num = 2
input_size = 20000
neuron_num = [100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200]
energy_cutoff = [0.01,0.05,0.1]
deim_cutoff = [0.01,0.05,0.1,0.2]

def run_experiment(neu,energy_cutoff,deim_cutoff,input_size):

    simtime = input_size
    mc_input = np.random.randn(simtime,1)
    
    esn_mc = RNN.EchoStateNetwork(
        neu = neu,
        n_in = 1,
        n_out = 1,
        gama = 1.0,
        ro = 0.99,
        psi = 0.0,
        in_scale = 0.1,
        bias_scale = 0.1,
        initial_filename = "mc",
        load_initial = False,
        save_initial = False,
        output_feedback = False)
    
    
    total_mc = 20
    drop = 100
    regularization = 1e-9

    mc_plot = np.empty(total_mc)
    mc_plot_red = np.empty([total_mc,len(energy_cutoff)])
    mc_plot_deim = np.empty([total_mc,len(energy_cutoff),len(deim_cutoff)])
    states = np.empty([simtime,esn_mc.neu])
    for t in range(simtime):
        esn_mc.update(mc_input[t])
        states[t] = esn_mc.get_current_state().flatten()
        
    reduced_esn_list = []
    reduced_neurons_list = []
    for ec in energy_cutoff:
        
        reduced_esn = GalerkinESN(esn_mc,states,mc_input,energy_cutoff = ec)
        
        reduced_esn_list= reduced_esn_list  + [reduced_esn]
    
        reduced_neurons_list = reduced_neurons_list  +  [reduced_esn.reduced_size]
        
    deim_size_list = []
    esn_weight_list = []
    for j,dc in enumerate(deim_cutoff):
        #calculate every new deim
        for resn in reduced_esn_list:
            resn.calculate_deim(dc)
        deim_size_list = deim_size_list + [reduced_esn_list[0].m]
        for k in range(1,total_mc+1):
            (input_signal,delayed_signal) = k_delayed_signal(mc_input,k)
            if j == 0:
                network_signal = np.empty([simtime-k,1])                
                esn_mc.offline_training(input_signal,delayed_signal,regularization,drop)
                esn_weight_list = esn_weight_list + [esn_mc.Wro]
                esn_mc.reset()
            reduced_signal_deim = np.empty([simtime-k,len(energy_cutoff)])
            for resn in reduced_esn_list:
                resn.new_output_weights_from_matrix(esn_weight_list[k-1])
                reduced_signal = np.empty([simtime-k,len(energy_cutoff)])
                resn.reset()     
            for t in range(simtime-k):
                if j == 0:
                    network_signal[t,0] = esn_mc.update(input_signal[t])               
                for i,resn in enumerate(reduced_esn_list):
                    reduced_signal[t,i] = resn.update(input_signal[t],'vanilla')
                    reduced_signal_deim[t,i] = resn.update(input_signal[t],'deim')
            
            if j == 0:
                mc_k = memory_capacity_k(delayed_signal[drop:,:],network_signal[drop:,:])
                mc_plot[k-1] = mc_k[1,0]
        
            for i in range(len(reduced_esn_list)):
                mc_red_k = memory_capacity_k(delayed_signal[drop:,:],col_2d(reduced_signal[drop:,i]))
                mc_plot_red[k-1,i] = mc_red_k[1,0]
                mc_deim_k = memory_capacity_k(delayed_signal[drop:,:],col_2d(reduced_signal_deim[drop:,i]))    
                mc_plot_deim[k-1,i,j] = mc_deim_k[1,0]
        
            
    

        
        
        
    mc_coef = np.nansum(mc_plot)
    print(mc_coef,"Coef for",neu,"neurons,full")
    mc_red_coef = np.nansum(mc_plot_red,axis = 0)
    print(mc_red_coef,"Coef for",neu,"neurons,POD")
    mc_deim_coef = np.nansum(mc_plot_deim,axis = 0)
    print(mc_deim_coef,"Coef for",neu,"neurons,DEIM")
    print(deim_size_list,"list of deim size")
    # plt.plot(mc_plot,label = "full ESN")
    # for i in range(len(reduced_esn_list)):
    #     vanilla_string = "POD-ESN" + str(energy_cutoff[i])
    #     deim_string = "DEIM-ESN" + str(energy_cutoff[i])
    #     plt.plot(mc_plot_red[:,i],label = vanilla_string)
    # ax = plt.gca()
    # ax.set_xlabel("k")
    # ax.set_ylabel("$MC_k$")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # for i in range(len(reduced_esn_list)):
    #     for j in range(len(deim_cutoff)):
    #         vanilla_string = "POD-ESN" + str(energy_cutoff[i])
    #         deim_string = "DEIM-ESN" + str(deim_cutoff[j])
    #         plt.plot(mc_plot_red[:,i],label = vanilla_string)
    #         plt.plot(mc_plot_deim[:,i,j],label = deim_string)
    #     ax = plt.gca()
    #     ax.set_xlabel("k")
    #     ax.set_ylabel("$MC_k$")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    
    return mc_coef,mc_red_coef,mc_deim_coef,reduced_neurons_list,deim_size_list

mc_coef_mean_grid = np.empty([len(neuron_num)])
mc_coef_std_grid = np.empty([len(neuron_num)])
mc_coef_min_grid = np.empty([len(neuron_num)])
mc_coef_max_grid = np.empty([len(neuron_num)])

mc_red_coef_mean_grid = np.empty([len(neuron_num),len(energy_cutoff)])
mc_red_coef_std_grid = np.empty([len(neuron_num),len(energy_cutoff)])
mc_red_coef_min_grid = np.empty([len(neuron_num),len(energy_cutoff)])
mc_red_coef_max_grid = np.empty([len(neuron_num),len(energy_cutoff)])

mc_deim_coef_mean_grid = np.empty([len(neuron_num),len(energy_cutoff),len(deim_cutoff)])
mc_deim_coef_std_grid = np.empty([len(neuron_num),len(energy_cutoff),len(deim_cutoff)])
mc_deim_coef_min_grid = np.empty([len(neuron_num),len(energy_cutoff),len(deim_cutoff)])
mc_deim_coef_max_grid = np.empty([len(neuron_num),len(energy_cutoff),len(deim_cutoff)])

redneu_mean_grid = np.empty([len(neuron_num),len(energy_cutoff)])
redneu_std_grid = np.empty([len(neuron_num),len(energy_cutoff)])
redneu_min_grid = np.empty([len(neuron_num),len(energy_cutoff)])
redneu_max_grid = np.empty([len(neuron_num),len(energy_cutoff)])

deimsize_mean_grid = np.empty([len(neuron_num),len(deim_cutoff)])
deimsize_std_grid = np.empty([len(neuron_num),len(deim_cutoff)])
deimsize_min_grid = np.empty([len(neuron_num),len(deim_cutoff)])
deimsize_max_grid = np.empty([len(neuron_num),len(deim_cutoff)])

load_runtime_data = False

key = 0

filename = 'MC_data' + str(key) + '.mat'
while(os.path.isfile(filename)):
    key +=1
    filename = 'MC_data' + str(key) + '.mat'

save_dict = {'mc_coef_mean_grid':mc_coef_mean_grid,
             'mc_coef_std_grid':mc_coef_std_grid,
             'mc_coef_min_grid':mc_coef_min_grid,
             'mc_coef_max_grid':mc_coef_max_grid,
             'mc_red_coef_mean_grid':mc_red_coef_mean_grid,
             'mc_red_coef_std_grid':mc_red_coef_std_grid,
             'mc_red_coef_min_grid':mc_red_coef_min_grid,
             'mc_red_coef_max_grid':mc_red_coef_max_grid,
             'mc_deim_coef_mean_grid':mc_deim_coef_mean_grid,
             'mc_deim_coef_std_grid':mc_deim_coef_std_grid,
             'mc_deim_coef_min_grid':mc_deim_coef_min_grid,
           'mc_deim_coef_max_grid':mc_deim_coef_max_grid,
           'redneu_mean_grid':redneu_mean_grid,
           'redneu_std_grid':redneu_std_grid,
           'redneu_min_grid':redneu_min_grid,
           'redneu_max_grid':redneu_max_grid,
           'deimsize_mean_grid':deimsize_mean_grid,
           'deimsize_min_grid':deimsize_min_grid,
           'deimsize_std_grid':deimsize_std_grid,
           'deimsize_max_grid':deimsize_max_grid}

runtimename = 'rundime_data' + str(key) + '.mat'
i0 = 0
io.savemat(filename,save_dict)
if load_runtime_data:
    key = input("enter the run's key:")
    filename = 'MC_data' + key + '.mat'
    runtimename = 'rundime_data' + key + '.mat'
    where_runtime = io.loadmat(runtimename)
    unfinished = 'unfinished_' + filename
    load_dict = io.loadmat(unfinished)
    
    mc_coef_mean_grid = load_dict['mc_coef_mean_grid']
    mc_coef_std_grid = load_dict['mc_coef_std_grid']
    mc_coef_min_grid = load_dict['mc_coef_min_grid']
    mc_coef_max_grid = load_dict['mc_coef_max_grid']

    mc_red_coef_mean_grid = load_dict['mc_red_coef_mean_grid']
    mc_red_coef_std_grid = load_dict['mc_red_coef_std_grid']
    mc_red_coef_min_grid = load_dict['mc_red_coef_min_grid']
    mc_red_coef_max_grid = load_dict['mc_red_coef_max_grid']

    mc_deim_coef_mean_grid = load_dict['mc_deim_coef_mean_grid']
    mc_deim_coef_std_grid =load_dict['mc_deim_coef_std_grid']
    mc_deim_coef_min_grid = load_dict['mc_deim_coef_min_grid']
    mc_deim_coef_max_grid = load_dict['mc_deim_coef_max_grid']

    redneu_mean_grid = load_dict['redneu_mean_grid']
    redneu_std_grid = load_dict['redneu_std_grid']
    redneu_min_grid = load_dict['redneu_min_grid']
    redneu_max_grid = load_dict['redneu_max_grid']

    deimsize_mean_grid = load_dict['deimsize_mean_grid']
    deimsize_std_grid = load_dict['deimsize_std_grid']
    deimsize_min_grid = load_dict['deimsize_min_grid']
    deimsize_max_grid = load_dict['deimsize_max_grid']
    
    i0 = where_runtime['i']
    print(i0)
    neu0 = where_runtime['neu']
    
    

for i,neu in enumerate(neuron_num):
    
    if i < i0:
        continue
    runtime_data = {'i':i,'neu':neu,'key':key}
    
    io.savemat(runtimename,runtime_data)
    print(key,"this is this guys key")
    print(i)
    mc_coef_plot = np.empty(ran_num)
    mc_red_plot = np.empty([ran_num,len(energy_cutoff)])
    mc_deim_plot = np.empty([ran_num,len(energy_cutoff),len(deim_cutoff)])
    redneu_plot = np.empty([ran_num,len(energy_cutoff)])
    deimsize_plot = np.empty([ran_num,len(deim_cutoff)])

    for k in range(ran_num):
        print("run number>",k,"number of neurons:",neu)
        (mc_coef_plot[k],mc_red_plot[k],mc_deim_plot[k],redneu_plot[k],deimsize_plot[k]) = run_experiment(neu,energy_cutoff,deim_cutoff,input_size)
        
    print(np.nanmean(mc_coef_plot),mc_coef_mean_grid)
    mc_coef_mean_grid[i] = np.nanmean(mc_coef_plot)
    mc_coef_std_grid[i] = np.nanstd(mc_coef_plot)
    mc_coef_min_grid[i] = np.nanmin(mc_coef_plot)
    mc_coef_max_grid[i] = np.nanmax(mc_coef_plot)

    mc_red_coef_mean_grid[i] = np.nanmean(mc_red_plot,axis=0)
    mc_red_coef_std_grid[i] = np.nanstd(mc_red_plot,axis=0)
    mc_red_coef_min_grid[i] =  np.nanmin(mc_red_plot,axis=0)
    mc_red_coef_max_grid[i] = np.nanmax(mc_red_plot,axis=0)

    mc_deim_coef_mean_grid[i] = np.nanmean(mc_deim_plot,axis=0)
    mc_deim_coef_std_grid[i] = np.nanstd(mc_deim_plot,axis=0)
    mc_deim_coef_min_grid[i] = np.nanmin(mc_deim_plot,axis=0)
    mc_deim_coef_max_grid[i] = np.nanmax(mc_deim_plot,axis=0)

    redneu_mean_grid[i] = np.nanmean(redneu_plot,axis=0)
    redneu_std_grid[i] = np.nanmean(redneu_plot,axis=0)
    redneu_min_grid[i] = np.nanmin(redneu_plot,axis=0)
    redneu_max_grid[i] = np.nanmax(redneu_plot,axis=0)
    
    deimsize_mean_grid[i] = np.nanmean(deimsize_plot,axis=0)
    deimsize_std_grid[i] = np.nanstd(deimsize_plot,axis=0)
    deimsize_min_grid[i] = np.nanmin(deimsize_plot,axis=0)
    deimsize_max_grid[i] = np.nanmax(deimsize_plot,axis=0)
    
    save_dict = {'mc_coef_mean_grid':mc_coef_mean_grid,
                 'mc_coef_std_grid':mc_coef_std_grid,
                 'mc_coef_min_grid':mc_coef_min_grid,
                 'mc_coef_max_grid':mc_coef_max_grid,
                 'mc_red_coef_mean_grid':mc_red_coef_mean_grid,
                 'mc_red_coef_std_grid':mc_red_coef_std_grid,
                 'mc_red_coef_min_grid':mc_red_coef_min_grid,
                 'mc_red_coef_max_grid':mc_red_coef_max_grid,
                 'mc_deim_coef_mean_grid':mc_deim_coef_mean_grid,
                 'mc_deim_coef_std_grid':mc_deim_coef_std_grid,
                 'mc_deim_coef_min_grid':mc_deim_coef_min_grid,
               'mc_deim_coef_max_grid':mc_deim_coef_max_grid,
               'redneu_mean_grid':redneu_mean_grid,
               'redneu_std_grid':redneu_std_grid,
               'redneu_min_grid':redneu_min_grid,
               'redneu_max_grid':redneu_max_grid,
               'deimsize_mean_grid':deimsize_mean_grid,
               'deimsize_min_grid':deimsize_min_grid,
               'deimsize_std_grid':deimsize_std_grid,
               'deimsize_max_grid':deimsize_max_grid}

    unfinished = 'unfinished_' + filename
    io.savemat(unfinished,save_dict)
    



print(mc_coef_plot,"full",mc_red_plot,"reduced",mc_deim_plot,"DEIM")


save_dict = {'mc_coef_mean_grid':mc_coef_mean_grid,
             'mc_coef_std_grid':mc_coef_std_grid,
             'mc_coef_min_grid':mc_coef_min_grid,
             'mc_coef_max_grid':mc_coef_max_grid,
             'mc_red_coef_mean_grid':mc_red_coef_mean_grid,
             'mc_red_coef_std_grid':mc_red_coef_std_grid,
             'mc_red_coef_min_grid':mc_red_coef_min_grid,
             'mc_red_coef_max_grid':mc_red_coef_max_grid,
             'mc_deim_coef_mean_grid':mc_deim_coef_mean_grid,
             'mc_deim_coef_std_grid':mc_deim_coef_std_grid,
             'mc_deim_coef_min_grid':mc_deim_coef_min_grid,
           'mc_deim_coef_max_grid':mc_deim_coef_max_grid,
           'redneu_mean_grid':redneu_mean_grid,
           'redneu_std_grid':redneu_std_grid,
           'redneu_min_grid':redneu_min_grid,
           'redneu_max_grid':redneu_max_grid,
           'deimsize_mean_grid':deimsize_mean_grid,
           'deimsize_min_grid':deimsize_min_grid,
           'deimsize_std_grid':deimsize_std_grid,
           'deimsize_max_grid':deimsize_max_grid}

io.savemat(filename,save_dict)