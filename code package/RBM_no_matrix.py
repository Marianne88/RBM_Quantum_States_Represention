# -*- coding: utf-8 -*-
"""
RBM representation of quantum states for real positive states. Uses linear operators instead of matrix to represent the hamiltonian
"""

import numpy as np
import itertools
import random
from time import time
from scipy.sparse.linalg import LinearOperator, eigsh


class RBM():
    
    def __init__(self, n_visible, n_hidden, hamiltonian = None):
        # n_visible (int): number of spins in the system
        # n_hidden: number of hidden units in the RBM
        # hamiltonian: linear operator representing the hamiltonian of the system whose ground state is approximated by the RBM
        
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
        cst = 4 * learning_rate / sample_size
        
        self.range_sample = range(sample_size)
        range_len_configs = range(2**self.nv)


        for p in range(n_epoch):
            if p%25 == 0: print("iter = ", p)
            
            ### optimisation propsed in article: How to use Neural Networks to Investigate Quantum Many-Body Physics ###
            
            # parameters variation = 2/Ns * 
            # sum_1toNs (energy gradient with respoect to param) * (local energy - expectation value)
            
            
            self.b_plus_w_dot_conf = self.b + np.matmul(self.configurations, self.w)
            
            
            # update sample half the time
            # if p%2 == 0:
            psi_complet = self.psi_normalized()
   
            self.sample_indices = random.choices(range_len_configs, np.square(psi_complet), k = sample_size)
            self.sample = self.configurations[self.sample_indices]
            
            
            self.b_plus_w_dot_conf_sample = self.b_plus_w_dot_conf[self.sample_indices]
            

            # local energy for all configuration
            local_energy = self.hamil.matvec(psi_complet) / psi_complet
            
            # local energy for configurations in the sample
            local_energy_sample = local_energy[self.sample_indices]
            
            # Approximation the energy (expectation value) for a sample of the data
            E_approx = np.mean(local_energy_sample)
            
            # Term used to calculate variation in parameters, the negative term is for stabilization
            E_loc_minus_H_expect = local_energy_sample - E_approx
            
            # energy gradients of weigths and biases
            grad_a, grad_b, grad_w = self.energy_grad()
            

            # updating weights and biases with energy gradient with respect to parameters and energy term          
           
            self.a -= cst/5 * np.matmul(E_loc_minus_H_expect, grad_a)
            self.b -= cst/5 * np.matmul(E_loc_minus_H_expect, grad_b)
            self.w -= cst   * np.matmul(grad_w, E_loc_minus_H_expect)

    
    def energy_grad(self):
        
        tanh_sample = np.tanh(self.b_plus_w_dot_conf_sample)
       
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
        return np.dot(psi, self.hamil.matvec(psi))
    
    
    def permutations(self, n):
        # returns a list of all binary permutations of values 1 or -1 of length n
        string_perms = ["".join(seq) for seq in itertools.product("01", repeat=n)]
        perms = [[(1 if n =="1" else -1) for n in c] for c in string_perms]
        return perms
       
    
    
## Code taken from Tensors.net   
    
def doApplyHam(psiIn: np.ndarray,
               hloc: np.ndarray,
               N: int,
               usePBC: bool):
  """
  Applies local Hamiltonian, given as sum of nearest neighbor terms, to
  an input quantum state.
  Args:
    psiIn: vector of length d**N describing the quantum state.
    hloc: array of ndim=4 describing the nearest neighbor coupling.
    N: the number of lattice sites.
    usePBC: sets whether to include periodic boundary term.
  Returns:
    np.ndarray: state psi after application of the Hamiltonian.
  """
  d = hloc.shape[0]
  
  psiOut = np.zeros(psiIn.size)
  
  for k in range(N - 1):
    # apply local Hamiltonian terms to sites [k,k+1]
    psiOut += np.tensordot(hloc.reshape(d**2, d**2),
                           psiIn.reshape(d**k, d**2, d**(N - 2 - k)),
                           axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)

  if usePBC:
    # apply periodic term
    psiOut += np.tensordot(hloc.reshape(d, d, d, d),
                           psiIn.reshape(d, d**(N - 2), d),
                           axes=[[2, 3], [2, 0]]
                           ).transpose(1, 2, 0).reshape(d**N)

  return psiOut


n_spins = 15
h = 0.6
usePBC = True  # use periodic or open boundaries

# Define Hamiltonian (quantum XX model)
# d = 2  # local dimension
sX = np.array([[0, 1.0], [1.0, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])


# hamloc represents the interaction model, in this case, transverse field Ising model 1D
hamloc = (-h*np.kron(sX, sI) - np.kron(sZ, sZ)).reshape(2, 2, 2, 2)


# Other example of local hamiltonian
# sY = np.array([[0, -1j], [1j, 0]])
# hamloc = (np.real(np.kron(sX,sX) + np.kron(sY,sY))).reshape(2,2,2,2)

def doApplyHamClosed(psiIn):
  return doApplyHam(psiIn, hamloc, n_spins, usePBC)


H = LinearOperator((2**n_spins, 2**n_spins), matvec=doApplyHamClosed)


Energy, psi = eigsh(H, k=1, which='SA')



### Set up the RBM ###
n_hidden = 2 * n_spins
n_epoch = 100
lr = 0.02
sample_size = 500


debut = time()
rbm = RBM(n_visible = n_spins, n_hidden = n_hidden, hamiltonian = H)
 
### Train the RBM ###
rbm.train(n_epoch, lr, sample_size)

### Print the results ###

print("\nComputation time: %.1f s" %(time() - debut))

# Compare results with exact ground state and energy
print("\nExact energy: %.3f, RBM energy: %.3f" %(Energy, rbm.expect_H()))

overlap = np.dot(rbm.psi_normalized().conjugate(),psi)**2
rel_err = (Energy - rbm.expect_H()) / np.abs(Energy)

print("\nOverlap between states: %.2f %%, Relative error in energy: %.2f %%" %(overlap * 100, rel_err * 100))

