# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:49:07 2023

@author: ericb
"""

# -*- coding: utf-8 -*-
"""
RBM representation of quantum states for real positive states
"""

import numpy as np
import itertools
import random
from time import time


import ising_model_


class RBM():
    
    def __init__(self, n_visible, n_hidden, hamiltonian = None):
        # n_visible (int): number of spins in the system
        # n_hidden: number of hidden units in the RBM
        # hamiltonian: matrix 2^n_visible x 2^n_visible representing the hamiltonian of the system whose ground state is approximated by the RBM
        
        self.nv = n_visible
        self.nh = n_hidden

        self.hamil = hamiltonian
                
        # Initialising value for visible and hidden biases to zero
        self.a = np.zeros(self.nv)
        self.b = np.zeros(self.nh)
        
        # Initialising weights value based on random distribution
        var = 0.25 if self.nv <= 4 else 1.0/self.nv
        
        self.w = np.array(np.random.normal(0, var, size = (self.nv, self.nh)))
       
        # Array of all possible spin configurations
        self.configurations = np.array(self.permutations(self.nv))
        
        
        
    def train(self, n_epoch, learning_rate, sample_size):
        # n_epoch (int): number of steps to make during training (e.g. 150)
        # learning_rate (float): size of step for gradient descent (e.g. 0.01)
        # sample_size (int): size of the sample used for calculations (max = 2 ** number of spins)
                        
        # constant defining the step of parameters variation
        cst = 2 * learning_rate / (sample_size)
        
        self.range_sample = range(sample_size)
        range_len_configs = range(2**self.nv)
       
    
        for p in range(n_epoch):
            if p%25 == 0: print("iter = ", p)
            
            ### optimisation propsed in article: How to use Neural Networks to Investigate Quantum Many-Body Physics ###
            
            # parameters variation = 2/Ns * 
            # sum_1toNs (energy gradient with respoect to param) * (local energy - expectation value)
            
            self.b_plus_w_dot_conf = self.b + np.matmul(self.configurations, self.w)
            
            # RBM normalized state during this iteration
            
            psi_norm = self.psi_normalized()
            
                        
            ### Samplig the data to optimize code ###
            
            self.sample_indices = random.choices(range_len_configs, np.square(psi_norm), k = sample_size)
            
            self.sample = self.configurations[self.sample_indices]

            
            # local energy for all configuration
            local_energy = np.dot(psi_norm, self.hamil) / psi_norm
            
            # local energy for configurations in the sample
            local_energy_sample = local_energy[self.sample_indices]
            
            # Approximation the energy (expectation value) for a sample of the data
            E_approx = np.mean(local_energy_sample)
            
            # Term used to calculate variation in parameters, the negative term is for stabilization
            E_loc_minus_H_expect = local_energy_sample - E_approx
            
            # energy gradients of weigths and biases
            grad_a, grad_b, grad_w = self.energy_grad()
            
            # updating weights and biases with energy gradient with respect to parameters and energy term
            self.a -= cst/10 * np.matmul(E_loc_minus_H_expect, grad_a)
            self.b -= cst/10 * np.matmul(E_loc_minus_H_expect, grad_b)
            self.w -= cst    * np.matmul(grad_w, E_loc_minus_H_expect)

    
    def energy_grad(self):
        b_plus_w_dot_conf_sample = self.b_plus_w_dot_conf[self.sample_indices]
        
        tanh_sample = np.tanh(b_plus_w_dot_conf_sample)
       
        grad_a = self.sample
        grad_b = tanh_sample
        grad_w = [np.outer(self.sample[c], tanh_sample[c]) for c in self.range_sample]
        grad_w = np.transpose(grad_w, axes =[1,2,0])
        
        return grad_a, grad_b, grad_w



    def psi(self): 
        # Implementation of the formula to determine RBM psi (the vector is not normalized):
        # |Ψ> = exp(sum_i(a_i * v_i)x(prod_j(cosh(b_i + sum_i(b_j * w_ij)))))
        
        exp_part = np.exp(np.dot(self.configurations, self.a))
        cosh_part = np.cosh(self.b_plus_w_dot_conf)
        
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
       
    

       
## Example use of the machine ##

# Parameters for the initialisation of the neural network
n_spins = 10                # number of spin in the spin chain
h = 1                       # parameter value of eternal field for Ising
n_hidden = 2*n_spins        # number of hidden parameters in the neural network

# parameters for the training the neural network
n_epoch = 100               # number of steps for training
lr = 0.02                   # size of step for training
sample_size = 500           # sample of the data to be used (max = 2**n_spins)


    
### Calculting hamiltonian, exact ground state and exact energy of ground state for reference ###
hamil = ising_model_.hamiltonian_1d(n_spins, h)
target_function, target_energy = ising_model_.ground_state(hamil)

print('\nNumSites: %i, h: %.1f, n_hidden: %i\n' %
      (n_spins, h, n_hidden))


debut = time()

### Set up the RBM ###
rbm = RBM(n_visible = n_spins, n_hidden = n_hidden, hamiltonian = hamil)
 
### Train the RBM ###
rbm.train(n_epoch, lr, sample_size)


### Print the results ###

print("\nComputation time: %.1f s" %(time() - debut))

print("\nExact energy: %.3f, RBM energy: %.3f" %(target_energy.real, rbm.expect_H()))

overlap = np.dot(rbm.psi_normalized().conjugate(),target_function)**2
rel_err = (target_energy.real - rbm.expect_H()) / np.abs(target_energy.real)

print("\nOverlap between states: %.2f %%, Relative error in energy: %.2f %%" %(overlap * 100, rel_err * 100))



