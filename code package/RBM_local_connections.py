# -*- coding: utf-8 -*-
"""
RBM representation of quantum states for real positive states with limited connection.
Hidden neurons are only connected to adjacent visible neurons
"""

import numpy as np
import itertools
import random
from time import time


import sys
sys.path.insert(1, '../')

import ising_model_

import matplotlib.pyplot as plt


class RBM():
    
    def __init__(self, n_visible, hamiltonian = None, number_of_connections = 3):
        # n_visible (int): number of spins in the system
        # n_hidden: number of hidden units in the RBM
        # hamiltonian: matrix 2^n_visible x 2^n_visible representing the hamiltonina of the system whose ground state is approximated by the RBM
        

        self.nv = n_visible
        self.nh = n_visible

        self.hamil = hamiltonian
                
        # Initialising value for visible and hidden biases to zero
        self.a = np.zeros(self.nv)
        self.b = np.zeros(self.nh)
        
        # Initialising weights value based on random distribution       
        var = 0.25 if self.nv <= 4 else 1.0/self.nv
        
        if number_of_connections != 2 and number_of_connections != 3:
            print("invalid number of connections, choices are 2 or 3")
        
        self.w_1 = np.random.normal(0, var, size = self.nv)
        self.w_2 = np.random.normal(0, var, size = self.nv)
        
        self.connect_3 = True if number_of_connections ==3 else False
        
        if self.connect_3: self.w_3 = np.random.normal(0, var, size = self.nv)
        
        # Array of all possible spin configurations
        self.configurations = np.array(self.permutations(self.nv))
                
        

    def train(self, n_epoch, learning_rate, sample_size):
        # n_epoch (int): number of steps to make during training (e.g. 150)
        # learning_rate (float): size of step for gradient descent (e.g. 0.01)
        # sample_size (int): size of the sample used for calculations (max = 2 ** number of spins)
                        
        # constant defining the step of parameters variation
        cst = 2 * learning_rate / (sample_size)
        
        self.range_nv = range(self.nv)
        self.range_sample = range(sample_size)
        self.range_len_perm = range(2**self.nv)
        
        self.b_shape = (self.nh, sample_size)
        
        self.w_shape = (self.nv, sample_size)
    
        for p in range(n_epoch):
            if p%25 == 0: print("iter = ", p)
            
            ### optimisation propsed in article: How to use Neural Networks to Investigate Quantum Many-Body Physics ###
            
            # parameters variation = 2/Ns * 
            # sum_1toNs (energy gradient with respoect to param) * (local energy - expectation value)
            
            self.b_plus_configs_dot_w = self.b + self.compute_config_dot_w(self.configurations)
            
           
            # RBM normalized state during this iteration
            psi_norm = self.psi_normalized()
               

            ### Samplig the data to optimize code ###
            
            sample_indices = random.choices(self.range_len_perm, np.square(psi_norm), k = sample_size)
            self.sample = self.configurations[sample_indices]


            self.b_plus_configs_dot_w_sample = self.b_plus_configs_dot_w[sample_indices]
              
            # local energy for all configuration
            local_energy = np.dot(psi_norm, self.hamil) / psi_norm
            
            # local energy for configurations in the sample
            local_energy_sample = local_energy[sample_indices]
            
            # Approximation the energy (expectation value) for a sample of the data
            E_approx = np.mean(local_energy_sample)
            
            # Term used to calculate variation in parameters, the negative term is for stabilization
            E_loc_minus_H_expect = local_energy_sample - E_approx
            
            # energy gradients of weigths and biases
            
            grad_a, grad_b, grad_w1, grad_w2, grad_w3 = self.energy_grad()
            
            # updating weights and biases with energy gradient with respect to parameters and energy term
            self.a -= cst/10 * np.dot(E_loc_minus_H_expect, grad_a)
            self.b -= cst/10 * np.dot(E_loc_minus_H_expect, grad_b)

            self.w_1 -= cst  * np.dot(E_loc_minus_H_expect, grad_w1)
            self.w_2 -= cst  * np.dot(E_loc_minus_H_expect, grad_w2)
            
            if self.connect_3: self.w_3 -= cst * np.dot(E_loc_minus_H_expect, grad_w3)
            

    def energy_grad(self):
        tanh_prod_weights_config = np.tanh(self.b_plus_configs_dot_w_sample)
        
        grad_a = self.sample
        grad_b = tanh_prod_weights_config
        grad_w1 = np.multiply(self.sample, tanh_prod_weights_config)
        grad_w2 = np.multiply(self.sample, np.roll(tanh_prod_weights_config, -1, axis = 1))
        grad_w3 = np.multiply(self.sample, np.roll(tanh_prod_weights_config, 1, axis = 1))
    
        return grad_a, grad_b, grad_w1, grad_w2, grad_w3
    
    def compute_config_dot_w(self, configs):
        if self.connect_3:
            return np.multiply(configs, self.w_1) + np.roll(np.multiply(configs, self.w_2), 1, axis = 1) + np.roll(np.multiply(configs, self.w_3), -1, axis = 1)
        else:
            return np.multiply(configs, self.w_1) + np.roll(np.multiply(configs, self.w_2), 1, axis = 1)
    

    def psi(self):
        # Implementation of the formula to determine RBM psi (the vector is not normalized):
        # |Ψ> = exp(sum_i(a_i * v_i)x(prod_j(cosh(b_i + sum_i(b_j * w_ij)))))
        
        b_plus_configs_dot_w = self.b + self.compute_config_dot_w(self.configurations)
        exp_part = np.exp(np.dot(self.configurations, self.a))
        
        cosh_part = np.cosh(b_plus_configs_dot_w)
        
        return exp_part * np.prod(cosh_part, axis = 1)
    

    def psi_normalized(self):
        # Normalizes the RBM state vector
        psi = self.psi()
        return psi / np.linalg.norm(psi)

    def expect_H(self):
        # Returns the expectation value of the hamiltonian (energy of the state)
        psi = self.psi_normalized()
        return np.dot(psi, np.dot(self.hamil, psi))
    
    def permutations(self, n):
        # returns a list of all binary permutations of values 1 or -1 of length n
        string_perms = ["".join(seq) for seq in itertools.product("01", repeat=n)]
        perms = [[(2 * int(n) - 1) for n in string_perms[c]] for c in range( 2**n )]
        return perms
        
       
## Testing the machine ##

n_spins = 10
h = 1
n_epoch, lr, sample_size = 200, 0.025, 500



    
### Calculting hamiltonian, exact ground state and exact energy of ground state ###
hamil = ising_model_.hamiltonian_1d(n_spins, h)
target_function, target_energy = ising_model_.ground_state(hamil)

print('NumSites: %i, h: %.1f' %
      (n_spins, h))


debut = time()

### Set up the RBM ###
rbm = RBM(n_visible = n_spins, hamiltonian = hamil, number_of_connections=3)
 
# print(rbm.w)
### Train the RBM ###
rbm.train(n_epoch, lr, sample_size)


### Print the results ###
print("temps: %.1f s" %(time() - debut))

print("\nExact energy: %f, RBM energy: %f" %(target_energy.real, rbm.expect_H()))


overlap = np.dot(rbm.psi_normalized(),target_function)**2
rel_err = (target_energy.real - rbm.expect_H()) / np.abs(target_energy.real)

print("\nOverlap between states: %.2f %%, Relative error in energy: %.1f %%" %(overlap *100, rel_err *100))