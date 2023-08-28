# -*- coding: utf-8 -*-
"""
Calculations for entanglement
"""

import numpy as np
import functools
import scipy

import qiskit.quantum_info as qi
from qutip.qobj import Qobj
from qutip import partial_transpose


class entanglement:
    
    def __init__(self, density_matrix = None, state_vectors = None, state_prob = None):
        
        # density matrix : matrix representing the whole system. Can be list or qi.DensityMatrix
        # states_vectors : list of one or multiple (mix state) states vector(s) representing the system
        # if states_vector contains more than one vector, a probability list must be given
        # subsys : list of subsystems to trace over for the partial trace (to get the density matrix of the rest of the system)
        
        if type(density_matrix) == qi.DensityMatrix: self.density_mat = density_matrix.data
            
        elif type(density_matrix) == np.matrix or type(density_matrix) == list: self.density_mat = qi.DensityMatrix(density_matrix)
       
        elif density_matrix == None:
            self.states_vect = state_vectors
            self.state_prob = state_prob
            self.density_mat = self._density_matrix()
        
        else: print("error data type for density matrix")
        
        self.n_spins_tot = self._get_spins(self.density_mat)
        
        
# -------------------- UTILITARY FUNCTIONS -------------------- #

    def _density_matrix(self):
        # states_vect : list of pure states describing the sys
        # states_prob : list of probabilities associated with the corresponding quantum pure state vector
        # returns list representing density matrix of system
        
        if type(self.states_vect[0]) != list or len(self.states_vect) == 1:
            return qi.DensityMatrix(np.dot(np.matrix(self.states_vect).conj().T, np.matrix(self.states_vect)))

        # sum of values in states_prob must give 1
        if np.abs(self.states_prob != None and sum(self.states_prob) - 1) > 1e-10: print("list of state vectors probabilities non valid"); return
        
        temp = map(lambda p, state : p * (np.dot(state.getH() , state)), self.states_prob, map(lambda arr : np.matrix(np.array(arr)), self.states_vect))
        ini = np.matrix([[0]*len(self.states_vect[0])] * len(self.states_vect[0]))
        
        rho = functools.reduce(lambda accum, elem: accum + elem, temp, ini)
        
        return qi.DensityMatrix(rho)


    def _red_dens_mat(self, subsys):
        # subsys: list of states to trace over ex [1, 2] would trace over the second and third subsystems 
        # partial trace of empty set return density matrix unmodified
        return qi.partial_trace((self.density_mat), subsys) if subsys != None else self.density_mat
    
    
    def _get_spins(self, mat):
        # mat is a density matrix of spins system
        # function returns the number of spins corresponding to the dimension of matrix
        return int(np.log2(len(mat))) if( type(mat) == list or type(mat) == np.matrix) else len(mat.dims())
        
    
    def _trace_norm(self, mat):
        # trace norm = sum of singular values of mat = roots of the eigenvalues of ρρ †
        # for hermitian mat (such as density mat) = sum of the absolute value of the eigenvalues of the density matrix.
        
        eigenvalues, eigenvectors = np.linalg.eig(np.matrix(mat))
        return sum(np.abs(eigenvalues))
        

    def _zero_approx(self, n):
        return n if np.abs(n) >= 5e-15 else 0



# -------------------- ENTANGLEMENT CALCULATIONS -------------------- #


    def _ent_entropy(self, subsys = None):
        # print("calculating entanglement entropy (von Neumann entropy)...")
        dens_mat = self._red_dens_mat(subsys)
             
        if not dens_mat.is_valid():
            print("reduced density matrix invalid"); return
        
        eigenvalues, eigenvectors = np.linalg.eig(np.matrix(dens_mat))
        eigenvalues = list(filter(self._zero_approx, eigenvalues.real))
    
        S = functools.reduce(lambda accum, eigenval : accum - eigenval * np.log(eigenval), eigenvalues, 0)
        return self._zero_approx(S)

    def _mutual_info(self, subsys_AB, subsys_A, subsys_B):
        # subsys_AB, subsys_A, subsys_B : subsystem considered to compute mutual information
        #print("Calculating mutual information...")
        
        set_AB, set_A, set_B = map(set, (subsys_AB, subsys_A, subsys_B))
        
        if not set_A.isdisjoint(set_B) or not set_A.union(set_B) == set_AB:
            print("Wrong choice of subsets"); return
        
        set_tot_sys = set(range(self.n_spins_tot))
        subsys_part_AB, subsys_part_A, subsys_part_B = map(lambda x: list(set_tot_sys - x), (set_AB, set_A, set_B))

        S_AB, S_A, S_B = map(self._ent_entropy, (subsys_part_AB, subsys_part_A, subsys_part_B))

        MI = self._zero_approx(S_A + S_B - S_AB)
        
        if MI < 0 : print("subadditivity not respected, S_AB >= S_A + S_B"); return
            
        return MI
    
    
    def _log_neg(self, subsys = None, mask = None):
        #print("calculating logarithmic negativity...")
        
        dens_mat = self._red_dens_mat(subsys)
        
        if not dens_mat.is_valid(): 
            print("reduced density matrix invalid"); return
        
        dims = [[2] * self._get_spins(dens_mat)]*2
       
        qobj = Qobj(dens_mat.data, dims)
        return self._zero_approx(np.log(self._trace_norm(partial_transpose(qobj, mask))))


    def _renyi_entropy(self, n, subsys = None):
        if n == 1: return self._ent_entropy(subsys)
        return self._zero_approx((1/(1-n) * np.log(np.trace(scipy.linalg.fractional_matrix_power(self._red_dens_mat(subsys), n))).real))
