# -*- coding: utf-8 -*-
"""
Computation for 1D and 2D transverse field Ising model.
The hamiltonian and ground state can be calculated (ground state calculated using exact diagonalisaiton)
"""

import numpy as np
import functools



def hamiltonian_1d(num_spins, h):
    
    iden = [[1, 0], [0, 1]]

    S_x = [[0, 1], [1, 0]]
    S_z = [[1, 0], [0, -1]]
    
    S_x_arr = np.zeros((num_spins, 2**num_spins, 2**num_spins))
    S_z_arr = np.zeros((num_spins, 2**num_spins, 2**num_spins))
    
        
    for i in range(num_spins):
        
        op1, op2 = [iden] * num_spins, [iden] * num_spins
        op1[i], op2[i] = S_x, S_z
               
        S_x_arr[i], S_z_arr[i] = map(lambda array : functools.reduce(lambda acc, elem : np.kron(acc, elem), array), [np.array(op1), np.array(op2)])

    
    hamiltonian = - h * sum(S_x_arr) - sum([np.dot(S_z_arr[i], S_z_arr[(i+1)%num_spins]) for i in range(num_spins)])
    
    return hamiltonian


def hamiltonian_2d(n_x, n_y, h):
    num_spins = n_x*n_y

    iden = [[1, 0], [0, 1]]

    S_x = [[0, 1], [1, 0]]
    S_z = [[1, 0], [0, -1]]
    
    S_x_arr = np.zeros((num_spins, 2**num_spins, 2**num_spins))
    S_z_arr = np.zeros((num_spins, 2**num_spins, 2**num_spins))
    
        
    for i in range(num_spins):
        
        op1, op2 = [iden] * num_spins, [iden] * num_spins
        op1[i], op2[i] = S_x, S_z
               
        S_x_arr[i], S_z_arr[i] = map(lambda array : functools.reduce(lambda acc, elem : np.kron(acc, elem), array), [np.array(op1), np.array(op2)])

    voisins = np.zeros((2**num_spins, 2**num_spins))

    for i in range(num_spins):
        x = i % n_x
        y = i // n_x
        voisin1 = (x + 1)%n_x + y * n_x
        voisin2 = (x - 1)%n_x + y * n_x
        voisin3 = x + ((y + 1)%n_y) * n_x
        voisin4 = x + ((y - 1)%n_y) * n_x
       
        voisins += np.dot(S_z_arr[i], S_z_arr[voisin1]) + np.dot(S_z_arr[i], S_z_arr[voisin2])+np.dot(S_z_arr[i], S_z_arr[voisin3])+np.dot(S_z_arr[i], S_z_arr[voisin4])

    
    hamiltonian = - h * sum(S_x_arr) - voisins
    
    return hamiltonian



def ground_state(hamiltonian):
    # Return the ground state, the associated energy and the hamiltonian of the system 
    # with the chosen parameter using exact diagonalisation
    # Use as reference
    
    
    eigenValues, eigenVectors = np.linalg.eig(hamiltonian)
    eigenVectors = eigenVectors.transpose()
    
    idx = np.argsort(eigenValues)
    
    ground_state = eigenVectors[idx[0]]    
    ground_state_norm = np.array(ground_state) / np.linalg.norm(ground_state)

    ground_energy = eigenValues[idx[0]]
    
    
    return ground_state_norm.real , ground_energy


