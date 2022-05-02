#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:06:36 2022

@author: jeanpj
"""
import numpy as np
import RNN
#import matplotlib.pyplot as plt
import scipy.linalg as sla
import numpy.linalg as nla
import scipy.io as io
def col_2d(x):
    
    y = x.reshape([x.size,1])
    return y
class GalerkinESN:
    
    def __init__(self,esn,states,input_to_states,energy_cutoff = 0.05,deim_cutoff = 0.01):
        
        
        self.esn = esn
        self.U, self.s, self.V  = sla.svd(states,full_matrices = False)
        self.V = self.V.T
        
        
        #reduced_size = self.s[self.s> np.max(self.s)*energy_cutoff].size
        #print(reduced_size)
        
        self.set_reduced_size_ec(energy_cutoff)
        
        #END POD
        #BEGIN DEIM
        
        #z_snapshots = states@self.T
        z_snapshots = np.copy(states)
        
        fun_snapshots = np.empty_like(states)
        
        for i in range(states.shape[0]):
            fun_snapshots[i] = self.nonlinear_fun_for_deim(z_snapshots[i],input_to_states[i]).flatten()
            
        
        self.U_deim_dump, self.s_deim, self.V_deim  = sla.svd(fun_snapshots,full_matrices = False)
        self.V_deim = self.V_deim.T
        self.calculate_deim(deim_cutoff)
        
        
        
    
    def set_reduced_size_ec(self,energy_cutoff):
        reduced_size = 0
        total_energy = np.sum(self.s)
        contribution = 0
        m = 0
        for i in range(self.s.shape[0]):
            contribution = contribution + self.s[i]/total_energy
            if contribution > (1.0-energy_cutoff):
                break
            reduced_size +=1
            
        
        self.T = np.copy(self.V[:,:reduced_size])
        
        
        self.Wrr_v = np.dot(self.esn.Wrr,self.T)
        self.Wir_v = self.esn.Wir
        self.Wir_gar = self.Wir_v
        self.Wbr_v = self.esn.Wbr
        self.Wbr_gar = self.Wbr_v
        
        self.Wro_case1 = np.hstack([np.atleast_2d(self.esn.Wro[:,0]).T,
                                    np.dot(self.esn.Wro[:,1:],self.T)])
        
        
            #independence = np.max(e)
            
        
        self.z_deim = np.zeros([reduced_size,1])
        self.z_v = np.zeros([reduced_size,1])
        
        self.reduced_size = reduced_size
              
    def calculate_deim(self,deim_cutoff):
        total_energy = np.sum(self.s_deim)
        contribution = 0
        m = 0
        for i in range(self.s_deim.shape[0]):
            contribution = contribution + self.s_deim[i]/total_energy
            if contribution > (1.0- deim_cutoff):
                break
            m +=1 
        self.m = m
        
        self.U_deim = np.atleast_2d(np.copy(self.V_deim[:,0])).T
        
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
            
        #get new DEIM update function    
        self.T_deim = self.T.T@self.U_deim@np.linalg.inv(self.Pivot.T@self.U_deim)
        #self.T_deim_for_linear_term = self.T.T@self.U_deim@np.linalg.inv(self.Pivot.T@self.U_deim)@self.Pivot.T@self.T
        print(self.T_deim.shape, "shape da matriz redutora")
        self.Wrr_deim = self.Pivot.T@self.Wrr_v
        self.Wir_deim = self.Pivot.T@self.Wir_v
        self.Wbr_deim = self.Pivot.T@self.Wbr_v
        
        self.max_index_list = max_index_list
        
        
        c = np.linalg.norm(np.linalg.inv(self.Pivot.T@self.U_deim))
        epsilon_star = np.linalg.norm(np.eye(self.esn.neu) - self.U_deim@self.U_deim.T)
        e_bound = c*epsilon_star
        
        
        print(e_bound, "multiplicador do bound de erro do DEIM")
        
    
    def update(self,inp,case = 'deim'):
        # input has to have same size
        # as n_in. Returns the output as shape (2,1), so if yo
        # u want to plot the data, a buffer is mandatory.
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)
            
        if case == 'vanilla':
            aux = np.dot(self.Wrr_v,self.z_v) + np.dot(self.Wir_v,Input) + self.Wbr_v                
            self.z_v = (1-self.esn.leakrate)*self.z_v + np.dot(self.T.T,self.esn.leakrate*np.tanh(aux))            
            z_wbias = np.vstack((np.atleast_2d(1.0),self.z_v))
            y = np.dot(self.Wro_case1,z_wbias)
            
        if case == 'deim':
            aux = np.dot(self.Wrr_deim,self.z_deim) + np.dot(self.Wir_deim,Input) + self.Wbr_deim                
            self.z_deim = (1-self.esn.leakrate)*self.z_deim + np.dot(self.T_deim,self.esn.leakrate*np.tanh(aux))            
            z_wbias = np.vstack((np.atleast_2d(1.0),self.z_deim))
            y = np.dot(self.Wro_case1,z_wbias)
        
        
        return y
    
    def nonlinear_fun_for_deim(self,z,u):
        z = col_2d(z)
        u = col_2d(u)
        aux = np.dot(self.esn.Wrr,z) + np.dot(self.Wir_v,u) + self.Wbr_v             
        return self.esn.leakrate*np.tanh(aux)  
    
    
    def reset(self):
        
        self.z_v = np.zeros([self.reduced_size,1])
        self.z_deim = np.zeros([self.reduced_size,1])
        
    def new_output_weights(self,esn):
        
        self.Wro_case1 = np.hstack([np.atleast_2d(esn.Wro[:,0]).T,
                                    np.dot(esn.Wro[:,1:],self.T)])
        
    def new_output_weights_from_matrix(self,Wro):
        self.Wro_case1 = np.hstack([np.atleast_2d(Wro[:,0]).T,
                                    np.dot(Wro[:,1:],self.T)])