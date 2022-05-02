import numpy as np
import RNN
#import matplotlib.pyplot as plt
import scipy.linalg as sla
import numpy.linalg as nla
import scipy.io as io

import matplotlib.pyplot as plt

esn_list = []

def change_or_not(x,min_val,max_val):
    y = 0
    p_change = np.random.rand()
    if p_change < 0.5:
        y = x
    else:
        y = min_val + (max_val - min_val)*np.random.rand()
    return y

def RFRAS(min,max,num_steps,minimum_step):
    RFRAS_sig = np.empty(num_steps)
    val = min + (max - min)*np.random.rand()
    for i in range(num_steps):

        if i % minimum_step  == 0:
            val = change_or_not(val,min,max)


        RFRAS_sig[i] = val

    return RFRAS_sig

esn_number = 20

neu = 500

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

# omega_list = [0.2,0.4,0.6,0.8]

# for omega in omega_list:
#     brown_noise = np.empty(simtime)
#     brown_noise[0] = 0
#     for t in range(1,simtime):
#         brown_noise[t] = (1-omega)*brown_noise[t-1] + omega*white_noise[t-1]
        
#     brown_noise_list += [brown_noise]
    
    
# brown_noise_list = []

omega_list = [10,100,500,1000]

for omega in omega_list:
    aprbs_signal = RFRAS(-1,1,simtime,omega)
        
    brown_noise_list += [aprbs_signal]
    
    
#plot input signals
f, sub = plt.subplots(5, sharey=True)
sub[0].plot(white_noise,label = "white noise")
sub[0].grid(True)
plotnum = 0
for brown_noise, omega in zip(brown_noise_list,omega_list):
    plotnum += 1
    label_string = "brown noise" + str(omega)
    sub[plotnum].plot(brown_noise,label = label_string)
    sub[plotnum].grid(True)
    
plt.show()

#obtain state profiles:
    
ec_for_white_noise_list = np.empty([esn_number,neu])

for n,esn in enumerate(esn_list):
    states = np.empty([simtime,neu])
    for t in range(simtime):
        esn.update(white_noise[t])
        states[t] = esn.get_current_state().flatten()
    
    U, s, V  = sla.svd(states,full_matrices = False)
    
    total_energy = np.sum(s)
    print(total_energy,s[0]/total_energy)
    ec = s/total_energy
    print(ec[0],n)
        
    ec_for_white_noise_list[n] = ec
    esn.reset()
    

ec_for_wn_mean = np.mean(ec_for_white_noise_list,axis = 0)
ec_for_wn_std = np.std(ec_for_white_noise_list,axis = 0)


ec_for_brown_noise_mean_list = []
ec_for_brown_noise_std_list = []

for brown_noise in brown_noise_list:
    
    ec_for_brown_noise_list = np.empty([esn_number,neu])
    for n,esn in enumerate(esn_list):
        states = np.empty([simtime,neu])
        for t in range(simtime):
            esn.update(brown_noise[t])
            states[t] = esn.get_current_state().flatten()
        
        #plt.plot(states)
        #plt.show()
        U, s, V  = sla.svd(states,full_matrices = False)
        total_energy = np.sum(s)
        print(total_energy,s[0]/total_energy,n)
        ec = s/total_energy
        print(ec[0],n)
            
        ec_for_brown_noise_list[n] = ec
        
        esn.reset()
        

    ec_for_bn_mean = np.mean(ec_for_brown_noise_list,axis = 0)
    ec_for_bn_std = np.std(ec_for_brown_noise_list,axis = 0)
    
    ec_for_brown_noise_mean_list += [ec_for_bn_mean]
    ec_for_brown_noise_std_list += [ec_for_bn_std]


cut = 10
plt.errorbar(np.arange(cut),ec_for_wn_mean[:cut],ec_for_wn_std[:cut],linestyle='None', marker='^',label = "white noise")
for brown_noise_mean,brown_noise_std,omega in zip(ec_for_brown_noise_mean_list,ec_for_brown_noise_std_list,omega_list):
    label_string = "APRBS, min period:" + str(omega)
    plt.errorbar(np.arange(cut),brown_noise_mean[:cut],brown_noise_std[:cut],linestyle='None', marker='^',label = label_string)
    
plt.grid(True)
plt.legend()
plt.xlabel("Singular Value")
plt.ylabel("Energy Contribution")
plt.show()
        
    


    
