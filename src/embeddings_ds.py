"""
This code is adapted from the repository:
https://github.com/jmmartyn/Neural-Network-Quantum-Field-States

Original work by J. M. Martyn et al.
"""

import jax.numpy as jnp
import numpy as np

def Embedding1(model, x, valid_mask):
    '''
    Constructs the embedding of particle positions x for DS1 in a multi-dimensional space.

    Parameters:
        x: a single particle configuration of shape (n_max, phys_dim)
        mask_valid: mask indicating the existing particles in the configuration
        
    Returns:
        embedded_data: tensor of shape (n_particles, 2*dim) containing the embedding.
    '''
    x_norm = x / model.L  

    if model.periodic:
        sines = jnp.where(valid_mask[:, None], jnp.sin(2 * np.pi * x_norm), 0)
        cosines = jnp.where(valid_mask[:, None], jnp.cos(2 * np.pi * x_norm), 0)
        embedded_data = jnp.concatenate((sines, cosines), axis=1)
    else:
        first = jnp.where(valid_mask[:, None], x_norm, 0)
        second = jnp.where(valid_mask[:, None], 1 - x_norm, 0)
        embedded_data = jnp.concatenate((first, second), axis=1)
        
    return embedded_data

def Embedding2(model, x, valid_mask):
    '''
    Constructs the embedding of particle positions x for DS2 in a multi-dimensional space.

    Parameters:
        x: a single particle configuration of shape (n_max, phys_dim)
        mask_valid: mask indicating the existing particles in the configuration
        
    Returns:
        embedded_data: tensor of shape (n_particles * (n_particles - 1) / 2, dim) containing pairwise differences.
    '''    

    n = x.shape[0]
    row_idx, col_idx = jnp.triu_indices(n, k=1)
    diffs = x[row_idx] - x[col_idx]
    upper_mask = valid_mask[row_idx] & valid_mask[col_idx]

    if model.periodic:
        embedded_data = jnp.where(upper_mask[:, None], jnp.cos(2 * np.pi / model.L * diffs), 0)
    else:
        embedded_data = jnp.where(upper_mask[:, None], (diffs / model.L) ** 2, 0)

    return embedded_data


