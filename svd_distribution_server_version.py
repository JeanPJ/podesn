import numpy as np
import RNN
#import matplotlib.pyplot as plt
import scipy.linalg as sla
import numpy.linalg as nla
import scipy.io as io

esn_list = []



esn_number = 20

neu = 2000

for i in range(esn_number):

    esn = RNN.EchoStateNetwork(
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
    
    esn_list += [esn]
    
    

#Define input signals:
    
simtime = 10000
    
white_noise = np.random.randn(simtime)


brown_noise_list = []

omega_list = [0.2,0.4,0.6,0.8]

for omega in omega_list:
    brown_noise = np.empty(simtime)
    brown_noise[0] = 0
    for t in range(1,simtime):
        brown_noise[t] = (1-omega)*brown_noise[t-1] + omega*white_noise[t-1]
        
    brown_noise_list += [brown_noise]
    
    
#obtain state profiles:
    
ec_for_white_noise_list = np.empty([esn_number,neu])

for n,esn in enumerate(esn_list):
    states = np.empty([simtime,neu])
    for t in range(simtime):
        states[t] = esn.update(white_noise[t])
    
    U, s, V  = sla.svd(states,full_matrices = False)
    
    total_energy = np.sum(s)
    ec = s/total_energy
        
    ec_for_white_noise_list[n] = ec
    

ec_for_wn_mean = np.mean(ec_for_white_noise_list,axis = 0)
ec_for_wn_std = np.std(ec_for_white_noise_list,axis = 0)


ec_for_brown_noise_mean_list = []
ec_for_brown_noise_std_list = []

for brown_noise in brown_noise_list:
    
    ec_for_brown_noise_list = np.empty([esn_number,neu])
    for n,esn in enumerate(esn_list):
        states = np.empty([simtime,neu])
        for t in range(simtime):
            states[t] = esn.update(brown_noise[t])
        
        U, s, V  = sla.svd(states,full_matrices = False)
        
        total_energy = np.sum(s)
        ec = s/total_energy
            
        ec_for_brown_noise_list[n] = ec
        

    ec_for_bn_mean = np.mean(ec_for_white_noise_list,axis = 0)
    ec_for_bn_std = np.std(ec_for_white_noise_list,axis = 0)
    
    ec_for_brown_noise_mean_list += [ec_for_bn_mean]
    ec_for_brown_noise_std_list += [ec_for_bn_std]



save_dict = {'white_noise':white_noise,'brown_noise_list':brown_noise_list,
             'ec_for_wn_mean':ec_for_wn_mean,'ec_for_wn_std':ec_for_wn_std,
             'ec_for_brown_noise_mean_list':ec_for_brown_noise_mean_list,
             'ec_for_brown_noise_std_list':ec_for_brown_noise_std_list}


io.savemat('svd_data.mat',save_dict)