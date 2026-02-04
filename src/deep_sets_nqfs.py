"""
This code is adapted from the repository:
https://github.com/jmmartyn/Neural-Network-Quantum-Field-States

Original work by J. M. Martyn et al.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from src.jastrow_factors import log_jastrow
from src.deep_sets import Deep_Set
from src.embeddings_ds import Embedding1, Embedding2

class NQFS(nn.Module):
    """
    Neural Quantum Field State (NQFS) ansatz using Deep Sets architecture.
    

    This ansatz models the log-wavefunction as a sum of permutation-invariant 
    Deep Set network outputs, physical Jastrow factors, and boundary terms.

    Attributes:
      width: Width of the hidden layers in the Deep Sets.
      depth_phi: Depth of the particle-wise network (phi).
      depth_rho: Depth of the aggregation network (rho).
      L: System length.
      periodic: Whether to enforce periodic boundary conditions.
      phys_dim: Physical dimension.
      m: Particle mass.
      g: Interaction strength.
      jastrow_type: Type of Jastrow correlation factor ('CS1D', 'LL1D', etc.).
      k: Wavevector for the embedding (default 5.0).
    """
    width : int
    depth_phi : int
    depth_rho :int
    L: float = 1.0
    periodic: bool = False
    phys_dim: int = 1
    m: float = 1.0
    g: float = 1.0
    jastrow_type: str = None
    k: float = 5.0
    
    q_n_mean_init: float = 5.0
    q_n_inv_softplus_width_init: float = 3.0
    q_n_inv_softplus_slope_init: float = 1.0

    def setup(self):
        # Initializes deep sets DS1 and DS2
        input_dim1 = 2*self.phys_dim  # The input to DS1 is a 2d embedding
        self.DS1 = Deep_Set(input_dim1, self.width, self.depth_phi, self.depth_rho)
        input_dim2 = 1*self.phys_dim  # The input to DS2 is a 1d embedding
        self.DS2 = Deep_Set(input_dim2, self.width, self.depth_phi, self.depth_rho)

        self.q_n_mean = self.param('q_n_mean', nn.initializers.constant(self.q_n_mean_init), (1,))
        self.q_n_inv_softplus_width = self.param('q_n_inv_softplus_width', nn.initializers.constant(self.q_n_inv_softplus_width_init), (1,))
        self.q_n_inv_softplus_slope = self.param('q_n_inv_softplus_slope', nn.initializers.constant(self.q_n_inv_softplus_slope_init), (1,))

    def __call__(self, x, mask_valid):

        mask1 = jax.lax.stop_gradient(mask_valid[:, None])
        # Contribution from DS1
        x_emb1 = Embedding1(self, x, mask_valid)
        val = jnp.log(self.DS1(jnp.nan_to_num(x_emb1), mask1))
        
        # Contribution from DS2
        n = jnp.sum(mask_valid)
        x_emb2 = Embedding2(self, x, mask_valid)
        mask2 = jax.lax.stop_gradient(x_emb2 != 0)[:, 0][:, None]
        val += (n >= 2) * jnp.log(self.DS2(jnp.nan_to_num(x_emb2), mask2))

        # L^(-n/2) contribution
        val += - n * self.phys_dim / 2 * jnp.log(self.L)

        # Jastrow contribution
        if self.jastrow_type is not None:
            val += (n >= 2) * log_jastrow(self.jastrow_type)(x, mask_valid, self.L, self.m, self.g, self.periodic)

        # q_n contribution
        val += 0.5 * self.log_q_n(n)

        # Cutoff factor for non-periodic systems
        if not self.periodic:
            val += self.log_cutoff_factor(x, n, mask_valid)
        return val.squeeze()


    def log_q_n(self, n):
        """
        Computes the log-probability factor for the particle number distribution.
        
        This implements a smoothed "box" or "window" function centered at `q_n_mean`
        with a learnable width and slope, penalizing particle numbers outside the preferred range.
        """
        q_n_width = jnp.log1p(jnp.exp(self.q_n_inv_softplus_width))
        c_1 = (2*self.q_n_mean - q_n_width)/2
        c_2 = (2*self.q_n_mean + q_n_width)/2
        s = jnp.log1p(jnp.exp(self.q_n_inv_softplus_slope))
        val = -jnp.log1p(jnp.exp(-s*(n-c_1))) - jnp.log1p(jnp.exp(s*(n-c_2)))
        return val

    def log_cutoff_factor(self, x, n, mask_valid):
        """
        Computes the boundary cutoff factor for open boundary conditions (Hard Wall).
        
        Forces the wavefunction to zero at the boundaries (0 and L) by adding
        log(x/L * (1 - x/L)), which goes to -infinity as x->0 or x->L.
        """
        val = (x % self.L)/self.L * (1 - (x % self.L)/self.L)
        val = jnp.sum(jnp.where(mask_valid[:, None], jnp.log(val+1e-16), 0))
        val -= n * jnp.log(self.L/30)
        return val
