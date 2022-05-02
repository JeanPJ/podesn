#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:45:01 2021

@author: jeanpj
"""

import numpy as np
import RNN
import matplotlib.pyplot as plt
import scipy.linalg as sla
import numpy.linalg as nla
from GalerkinESN import GalerkinESN

def k_delayed_signal(signal,k):
    
    
    
    input_signal = signal[k:,:]
    delayed_signal = signal[:-k,:]
    return input_signal,delayed_signal



def memory_capacity_k(delayed_signal,network_signal):
    
    inp = np.hstack([delayed_signal,network_signal])
    mc = np.corrcoef(inp,rowvar = False)
    
    return mc


simtime = 10000
mc_input = np.random.randn(simtime,1)


esn_mc = RNN.EchoStateNetwork(
        neu = 1200,
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



total_mc = 100
drop = 100
regularization = 1e-9

mc_plot = np.empty(total_mc)
mc_plot_red = np.empty(total_mc)
mc_plot_deim = np.empty(total_mc)

#obtaining reduced esn:
    
states = np.empty([simtime,esn_mc.neu])
for t in range(simtime):
    esn_mc.update(mc_input[t])
    states[t] = esn_mc.get_current_state().flatten()
    
    
reduced_esn = GalerkinESN(esn_mc,states,mc_input,energy_cutoff = 0.1,deim_cutoff = 0.01)




for k in range(1,total_mc+1):
    network_signal = np.empty([simtime-k,1])                
    (input_signal,delayed_signal) = k_delayed_signal(mc_input,k)
    esn_mc.offline_training(input_signal,delayed_signal,regularization,drop)
    esn_mc.reset()
    reduced_esn.new_output_weights(esn_mc)
    reduced_esn.reset()
    reduced_signal = np.empty([simtime-k,1])
    reduced_signal_deim = np.empty([simtime-k,1])          
    for t in range(simtime-k):
        network_signal[t,0] = esn_mc.update(input_signal[t])
        reduced_signal[t,0] = reduced_esn.update(input_signal[t],'vanilla')
        reduced_signal_deim[t,0] = reduced_esn.update(input_signal[t])
    
    #reduced_esn = GalerkinESN(esn_mc,states)
    
    
   # for t in range(simtime-k):
   #     reduced_signal[t] = reduced_esn.update(input_signal[t],"vanilla")
        
    mc_k = memory_capacity_k(delayed_signal[drop:,:],network_signal[drop:,:])
    
    mc_red_k = memory_capacity_k(delayed_signal[drop:,:],reduced_signal[drop:,:])
    
    mc_deim_k = memory_capacity_k(delayed_signal[drop:,:],reduced_signal_deim[drop:,:])
    
    mc_plot[k-1] = mc_k[1,0]
    mc_plot_red[k-1] = mc_red_k[1,0]
    mc_plot_deim[k-1] = mc_deim_k[1,0]
        


mc_coef = np.nansum(mc_plot)
mc_red_coef = np.sum(mc_plot_red)
plt.plot(mc_plot,label = "full ESN")
plt.plot(mc_plot_red,label = "POD-ESN")
plt.plot(mc_plot_deim,label = "DEIM-ESN")
ax = plt.gca()
ax.set_xlabel("k")
ax.set_ylabel("$MC_k$")
plt.grid(True)
plt.legend()
plt.show()
print("memory capacity:",mc_coef)

print("memory capacity reduced ESN:",mc_red_coef)
    