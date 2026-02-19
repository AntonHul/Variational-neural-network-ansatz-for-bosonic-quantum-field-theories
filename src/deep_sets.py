"""
This code is adapted from the repository:
https://github.com/jmmartyn/Neural-Network-Quantum-Field-States

Original work by J. M. Martyn et al.
"""

from flax import linen as nn
import jax.numpy as jnp

def log_cosh(x):
    """
    Computes the natural logarithm of cosh(x) in a numerically stable way.
    
    Approximation:
      log(cosh(x)) = |x| - log(2) + log(1 + exp(-2|x|))
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

class Deep_Set(nn.Module):
    """
    A Permutation Invariant Deep Set architecture.

    Attributes:
      width: Width of the hidden layers.
      depth_phi: Number of layers in the 'phi' (particle-wise) network.
      depth_rho: Number of layers in the 'rho' (global) network.
      input_dim: Dimension of input features.
    """
    input_dim : int
    width: int
    depth_phi : int
    depth_rho : int

    @nn.compact
    def __call__ (self, x_emb, mask):
        """
        Forward pass.

        Parameters:
          x_emb: Input embeddings of shape (n_max, features).
          mask_valid: Mask indicating the existing particles in the configuration.

        Returns:
          Scalar output.
        """
        y = log_cosh(nn.Dense(features=self.width)(x_emb))
        for i in range(self.depth_phi-1):
            y = log_cosh(nn.Dense(features=self.width)(y))

        y = jnp.sum(y*mask, axis=0)
        for i in range(self.depth_rho-1):
            y = log_cosh(nn.Dense(features=self.width)(y))
        y = nn.softplus(nn.Dense(features=1)(y))  

        return y
    

