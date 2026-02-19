"""
This code is adapted from the repository:
https://github.com/jmmartyn/Neural-Network-Quantum-Field-States

Original work by J. M. Martyn et al.
"""

import numpy as np
from flax import linen as nn
import jax.numpy as jnp

def CS_n_1d(L, m, mu, g):
    '''
    Calculates the ground state particle number of the CS 1D model 

    Parameters:
        L: system size
        m: particle mass
        mu: chemical potential
        g: CS interaction strength

    Returns:
        n: ground state particle number of the CS 1D model 
    '''

    lam = 1 / 2 * (1 + (1 + 4 * m * g) ** 0.5)
    a = np.pi ** 2 * lam ** 2 / (6 * m * L ** 2)
    n = (mu / (3 * a) + 1 / 3) ** 0.5
    n = int(np.round(n))
    return n

def CS_energy_1d(L, m, mu, g):
    '''
    Calculates the ground state energy of the CS 1D model  

    Parameters:
        L: system size
        m: particle mass
        mu: chemical potential
        g: CS interaction strength

    Returns:
        E: ground state energy of the CS 1D model 
    '''

    n = CS_n_1d(L, m, mu, g)
    lam = 1 / 2 * (1 + (1 + 4 * m * g) ** 0.5)
    a = np.pi ** 2 * lam ** 2 / (6 * m * L ** 2)
    E = a * (n ** 3 - n) - mu * n
    return E
        
def CS_n_2d(omega, mu, g):
    '''
    Calculates the ground state particle number of the CS 2D model 

    Parameters:
        omega: trap angular frequency
        mu: chemical potential
        g: CS interaction strength

    Returns:
        n: ground state particle number of the CS 1D model 
    '''
    lam = np.sqrt(g/2)
    n = ((mu / omega - 2) / lam + 1 )/2
    return int(np.round(n))

def CS_energy_2d(omega, mu, g):
    '''
    Calculates the energy of the CS ground state

    Parameters:
        omega: trap angular frequency
        mu: chemical potential
        g: CS interaction strength

    Returns:
        E: ground state energy of the CS 2D model 
    '''

    n = CS_n_2d(omega, mu, g)
    lam = np.sqrt(g/2)
    E = omega*(2*n + lam*n*(n-1)) - mu*n
    return E


class cs_nqfs_exact(nn.Module):
    """
    Exact ansatz for the CS 2D model 
    
    Attributes:
      L: System length.
      m: Particle mass.
      g: Interaction strength.
      omega: Trap angular frequency
    """
    L: float = 5.0
    m: float = 1.0
    g: float = 1.0
    phys_dim: int = 2
    omega : float = 1.0
    def __call__(self, x: jnp.ndarray, mask_valid: jnp.ndarray) -> jnp.ndarray:

        if self.phys_dim == 1:
            n = x.shape[0]
            row_idx, col_idx = jnp.triu_indices(n, k=1)
            interparticle_seps = x[row_idx] - x[col_idx]
            interparticle_seps = interparticle_seps - self.L * jnp.round(interparticle_seps / self.L)

            upper_mask = mask_valid[row_idx] & mask_valid[col_idx]
            lam = 0.5 * (1 + jnp.sqrt(1 + 4 * self.m * self.g))
            jastrows = jnp.where(upper_mask[:, None], jnp.abs(jnp.sin(jnp.pi * interparticle_seps / self.L)) ** lam, 1)

            return jnp.sum(jnp.log(jastrows), where=upper_mask[:, None])
        else:
            n = x.shape[0]
            row_idx, col_idx = jnp.triu_indices(n, k=1)
            diffs = x[row_idx] - x[col_idx]
            interparticle_seps = jnp.linalg.norm(diffs, axis=-1)
            upper_mask = mask_valid[row_idx] & mask_valid[col_idx]
            lam_D = jnp.sqrt(self.g/2)
            jastrows = jnp.sum(jnp.where(upper_mask, lam_D * jnp.log(interparticle_seps), 0))
            jastrows -= self.omega*jnp.sum(jnp.linalg.norm(x - self.L/2, axis=-1)**2, where=mask_valid)/2
            return jastrows
