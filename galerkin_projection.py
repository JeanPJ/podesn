#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:54:43 2020

@author: jean-jordanou
"""

import scipy.io as io
import scipy.linalg as sla
import RNN as RNN
import pickle
import numpy.linalg as nla
import numpy as np
import matplotlib.pyplot as plt


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



class GalerkinESN:
    
    def __init__(self,esn,states,energy_cutoff = 0.001):
        
        
        self.esn = esn
        self.U, self.s, self.V  = sla.svd(states,full_matrices = False)
        
        
        reduced_size = self.s[self.s> np.max(self.s)*energy_cutoff].size
        self.V = self.V.T
        m = 230
        
        self.T = np.copy(self.V[:,:reduced_size])
        
        self.Tm = np.copy(self.T)
        self.Ts = np.array([])
        
        self.Wrr_v = np.dot(self.esn.Wrr,self.T)
        self.Wrr_case1 = self.Wrr_v
        self.Wir_v = self.esn.Wir
        self.Wir_gar = self.Wir_v
        self.Wbr_v = self.esn.Wbr
        self.Wbr_gar = self.Wbr_v
        
        
        self.Wro_case1 = np.hstack([np.atleast_2d(self.esn.Wro[:,0]).T,np.dot(self.esn.Wro[:,1:],self.T)])
        
        self.Wrr_case2 = self.esn.Wrr
        
        independence = 0.0
        e = np.empty_like(self.T.diagonal())
        k = 0
        while k < 10:
            A = nla.inv(np.dot(self.Tm.T,self.Tm))
            E = np.dot(np.dot(self.Tm,A),self.Tm.T)
            cut = np.argmin(E.diagonal())
#            for i in range(e.size):
#                e[i] = np.abs(np.dot(self.T[i,:],self.T[i,:]) - 1.0)
#                
           # cut = np.argmax(e)
            
            if self.Ts.size == 0:
                self.Ts = self.Tm[cut,:]
            else:
                self.Ts = np.vstack([self.Tm[cut,:],self.Ts])
                
            if cut == self.Tm.shape[0] - 1:
                self.Tm = self.Tm[:cut,:]
                self.Wir_gar = self.Wir_gar[:cut,:]
                self.Wbr_gar = self.Wbr_gar[:cut,:]
                self.Wrr_case1 = self.Wrr_case1[:cut,:]
                self.Wrr_case2 = self.Wrr_case2[:cut,:]
                
            elif cut == 0:
                self.Tm = self.Tm[1:,:]
                self.Wir_gar = self.Wir_gar[1:,:]
                self.Wbr_gar = self.Wbr_gar[1:,:]
                self.Wrr_case1 = self.Wrr_case1[1:,:]
                self.Wrr_case2 = self.Wrr_case2[1:,:]
                                
            else:
                self.Tm = np.vstack([self.Tm[:cut,:],self.Tm[cut+1:,:]])
                self.Wir_gar = np.vstack([self.Wir_gar[:cut,:],self.Wir_gar[cut+1:,:]])
                self.Wbr_gar = np.vstack([self.Wbr_gar[:cut,:],self.Wbr_gar[cut+1:,:]])
                self.Wrr_case1 = np.vstack([self.Wrr_case1[:cut,:],self.Wrr_case1[cut+1:,:]])
                self.Wrr_case2 = np.vstack([self.Wrr_case2[:cut,:],self.Wrr_case2[cut+1:,:]])
                
            
            independence = np.min(E.diagonal())
            k+=1
           # print np.max(e)
            #independence = np.max(e)
            
        
        self.z = np.zeros([reduced_size,1])
        self.am = np.zeros([self.Tm.shape[0],1])
        self.z_v = np.zeros([reduced_size,1])
        self.z_deim = np.zeros([reduced_size,1])
        
        map_matrix = np.vstack([np.eye(self.Tm.shape[0]),np.dot(self.Ts,self.Tm.T)])
        
        self.Wro_case2 = np.hstack([np.atleast_2d(self.esn.Wro[:,0]).T,np.dot(self.esn.Wro[:,1:],map_matrix)])
        
        self.Wrr_case2 = np.dot(self.Wrr_case2,map_matrix)
        self.reduced_size = reduced_size
        
        
        self.m = m
        self.U_deim = np.atleast_2d(np.copy(self.V[:,0])).T
        
        max_elem = np.max(np.abs(self.U_deim).flatten())
        max_index = np.argmax(np.abs(self.U_deim).flatten())
        
        self.Pivot = np.zeros([self.esn.neu,1])
        
        self.Pivot[max_index,0] = 1.0
        
        max_index_list = []
        
        for l in range(1,m):
            
            current_u =  np.atleast_2d(np.copy(self.V[:,l])).T
            #obtaining residue
            c = np.linalg.solve(self.Pivot.T@self.U_deim,self.Pivot.T@current_u)
            r = current_u - self.U_deim@c
            
            
            #getting max residue
            max_elem = np.max(np.abs(r).flatten())
            max_index = np.argmax(np.abs(r).flatten())
            
            max_index_list = max_index_list + [max_index]
            
            #setting pivoting collumn of respective iteration
            pivot_vector = np.zeros([self.esn.neu,1])
            pivot_vector[max_index,0] = 1.0
            
            #stacking in main matrices:
                
            self.U_deim = np.hstack([self.U_deim,current_u])
            self.Pivot = np.hstack([self.Pivot,pivot_vector])
            print(self.Pivot)
            
        #get new DEIM update function    
        self.T_deim = self.T.T@self.U_deim@np.linalg.inv(self.Pivot.T@self.U_deim)
        self.T_deim_for_linear_term = self.T.T@self.U_deim@np.linalg.inv(self.Pivot.T@self.U_deim)@self.Pivot.T@self.T
        print(self.T_deim.shape, "shape da matriz redutora")
        self.Wrr_deim = self.Pivot.T@self.Wrr_v
        self.Wir_deim = self.Pivot.T@self.Wir_v
        self.Wbr_deim = self.Pivot.T@self.Wbr_v
        
        self.max_index_list = max_index_list
        
        
        c = np.linalg.norm(np.linalg.inv(self.Pivot.T@self.U_deim))
        epsilon_star = np.linalg.norm(np.eye(self.esn.neu) - self.U_deim@self.U_deim.T)
        e_bound = c*epsilon_star
        
        
        print(e_bound, "multiplicador do bound de erro do DEIM")
        print(c,"C")
        print(epsilon_star,"E")
        
        
        
                
            
        
    
    def update(self,inp,case = 'case1'):
        # input has to have same size
        # as n_in. Returns the output as shape (2,1), so if yo
        # u want to plot the data, a buffer is mandatory.
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)
        
        if case == 'case1':
            aux = np.dot(self.Wrr_case1,self.z) + np.dot(self.Wir_gar,Input) + self.Wbr_gar                
            self.z = (1-self.esn.leakrate)*self.z + np.dot(self.Tm.T,self.esn.leakrate*np.tanh(aux))            
            z_wbias = np.vstack((np.atleast_2d(1.0),self.z))
            y = np.dot(self.Wro_case1,z_wbias)
            
            
        if case == 'case2':
            aux = np.dot(self.Wrr_case2,self.am) + np.dot(self.Wir_gar,Input) + self.Wbr_gar                
            self.am = (1-self.esn.leakrate)*self.am + self.esn.leakrate*np.tanh(aux)            
            am_wbias = np.vstack((np.atleast_2d(1.0),self.am))
            y = np.dot(self.Wro_case2,am_wbias)
            
        if case == 'vanilla':
            aux = np.dot(self.Wrr_v,self.z_v) + np.dot(self.Wir_v,Input) + self.Wbr_v                
            self.z_v = (1-self.esn.leakrate)*self.z_v + np.dot(self.T.T,self.esn.leakrate*np.tanh(aux))            
            z_wbias = np.vstack((np.atleast_2d(1.0),self.z_v))
            y = np.dot(self.Wro_case1,z_wbias)
            
        if case == 'deim':
            aux = np.dot(self.Wrr_deim,self.z_deim) + np.dot(self.Wir_deim,Input) + self.Wbr_deim                
            self.z_deim = (1-self.esn.leakrate)*self.T_deim_for_linear_term@self.z_deim + np.dot(self.T_deim,self.esn.leakrate*np.tanh(aux))            
            z_wbias = np.vstack((np.atleast_2d(1.0),self.z_deim))
            y = np.dot(self.Wro_case1,z_wbias)
        
        
        return y
        
        
        
pickle_file = open("esn_platform_dummy.pickle","rb")
esn_platform = pickle.load(pickle_file)
load_dict = io.loadmat("states.mat")
states = load_dict['states']


reduced_esn = GalerkinESN(esn_platform,states)


reduced_esn2 = GalerkinESN(esn_platform,states,energy_cutoff=0.01)








data = io.loadmat('data_platform.mat')

y_data = data['y']
y_max = np.array([220.0,220.0])
y_min = np.array([170.0,170.0])
y_data_normal = feature_scaling(y_data,y_max,y_min)

u_data = data['u']
u_max = np.array([1.0,1.0])
u_min = np.array([0.01,0.01])
u_data_normal = feature_scaling(u_data,u_max,u_min)

training_set_size = 3000

y_data_training = y_data_normal[:training_set_size,:]
u_data_training = u_data_normal[:training_set_size,:]

y_data_val = y_data_normal[training_set_size:,:]
u_data_val = u_data_normal[training_set_size:,:]

simtime = u_data.shape[0]

regularization = 1e-5
drop = 200

y_pred = np.empty_like(y_data)
y_pred_gar1 = np.empty_like(y_data)
y_pred_gar2 = np.empty_like(y_data)

y_pred_gar_original = np.empty_like(y_data)
y_pred_gar_original2 = np.empty_like(y_data)

y_pred_deim = np.empty_like(y_data)


for k in range(simtime):
    
    y_pred[k] = esn_platform.update(u_data_normal[k]).flatten()
    y_pred_gar1[k] = reduced_esn.update(u_data_normal[k],case = 'case1').flatten()
    y_pred_gar2[k] = reduced_esn.update(u_data_normal[k],case = 'case2').flatten()
    y_pred_gar_original[k] = reduced_esn.update(u_data_normal[k],case = 'vanilla').flatten()
    y_pred_deim[k] = reduced_esn.update(u_data_normal[k],case = 'deim').flatten()
    y_pred_gar_original2[k] = reduced_esn2.update(u_data_normal[k],case = 'vanilla').flatten()
    

error = np.abs(y_pred - y_data_normal)
error_gar1 = np.abs(y_pred_gar1 - y_data_normal)
error_gar2 = np.abs(y_pred_gar2 - y_data_normal)
error_gar3 = np.abs(y_pred_gar_original - y_data_normal)
error_deim = np.abs(y_pred_deim - y_data_normal)

training_error = np.mean(error[200:40000],axis = 0)
val_error = np.mean(error[40000:], axis = 0)

training_error_gar1 = np.mean(error_gar1[200:40000],axis = 0)
val_error_gar1 = np.mean(error_gar1[40000:], axis = 0)

training_error_gar2 = np.mean(error_gar2[200:40000],axis = 0)
val_error_gar2 = np.mean(error_gar2[40000:], axis = 0)

training_error_gar3 = np.mean(error_gar3[200:40000],axis = 0)
val_error_gar3 = np.mean(error_gar3[40000:], axis = 0)

training_error_deim = np.mean(error_deim[200:40000],axis = 0)
val_error_deim = np.mean(error_deim[40000:], axis = 0)

print("para a rede original, o erro foi", training_error, "erro de validação:", val_error)

print("para a rede reduzida 1, o erro foi", training_error_gar1, "erro de validação:", val_error_gar1)
print("para a rede reduzida 2, o erro foi", training_error_gar2, "erro de validação:", val_error_gar2)
print("para a rede reduzida sem simplificação, o erro foi", training_error_gar3, "erro de validação:", val_error_gar3)
print("para a rede reduzida com DEIM, o erro foi", training_error_deim, "erro de validação:", val_error_deim)


plt.plot(y_pred[48000:,0])
#plt.plot(y_pred_gar1)
plt.plot(y_pred_gar1[48000:,0])
plt.show()

n = 400

T = reduced_esn.V

# x = Tz T -> 400 x 36
e_bound = (1 + np.sqrt(2*n))*(1/np.linalg.norm(T[:,0],ord=np.inf))*np.linalg.norm(np.eye(n) - T@T.T)

print(e_bound)



plt.plot(y_pred_deim[40000:42000,0],label="DEIM")
plt.plot(y_pred_gar_original[40000:42000,0],label="pure POD")
plt.plot(y_pred[40000:42000,0],label="original ESN")
plt.plot(y_data_normal[40000:42000,0],label = "data")
plt.legend()
plt.grid(True)
plt.show()


simtime_test = 1000

u_test = np.empty([simtime_test,2])
u_test[:,0] = RFRAS(0.1,1.0,simtime_test,200)
u_test[:,1] = RFRAS(0.1,1.0,simtime_test,200)

y_pred_test = np.empty([simtime_test,2])
y_gar_test = np.empty([simtime_test,2])
y_deim_test = np.empty([simtime_test,2])
y_gar2_test = np.empty([simtime_test,2])


for k in range(simtime_test):
    
    
    y_pred_test[k] = esn_platform.update(u_test[k]).flatten()
    y_gar_test[k] = reduced_esn.update(u_test[k],case = 'vanilla').flatten()
    y_gar2_test[k] = reduced_esn2.update(u_test[k],case = 'vanilla').flatten()
    y_deim_test[k] = reduced_esn.update(u_test[k],case = 'deim').flatten()


plt.plot(y_deim_test[:,0],label="DEIM (m = 230, 36 N)")
plt.plot(y_gar_test[:,0],label="pure POD (36 N)")
plt.plot(y_gar2_test[:,0],label="pure POD (11 N)")
plt.plot(y_pred_test[:,0],label="original ESN")
plt.legend()
plt.grid(True)
ax = plt.gca()

plt.ylabel("output")
plt.xlabel("simulation time")
plt.show()