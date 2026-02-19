import jax
import jax.numpy as jnp
import itertools
from typing import Callable

def embeddings(embed_type: str = 'Gaussian') -> Callable:
    """
    Function to select the embedding strategy.
    
    Parameters:
      embed_type: 'Gaussian' for localized RBF grid or 'Periodic' for Fourier features.
      
    Returns:
      The corresponding embedding function.
    """
    if embed_type == 'Gaussian':
        return GS_Embedding
    elif embed_type == 'Periodic':
        return PW_Embedding
    else:
        raise ValueError("Choose between 'Gaussian' and 'Periodic'!")
    
def GS_Embedding(x: jnp.ndarray, L: float, embed_dim: int, phys_dim: int, sigma: float, periodic: bool) -> jnp.ndarray:
    """
    Computes Gaussian embeddings on a regular grid for a given particle configuration.

    This embedding maps particle positions to a high-dimensional feature space defined
    by Gaussian activations centered at grid points uniformly spaced in the simulation box.

    Parameters:
      x: Particle configuration of shape (n_max, phys_dim).
      L: System length.
      embed_dim: Number of grid points per spatial dimension.
      phys_dim: Physical dimension of the local Hilbert space.
      sigma: Width of the Gaussian functions.
      periodic: If True, uses Minimum Image Convention for distance calculations.

    Returns:
      embeddings: Array of shape (n_max, embed_dim**phys_dim).
    """

    grid_1d = jnp.linspace(0, L, embed_dim)
    mesh = jnp.array(list(itertools.product(*([grid_1d] * phys_dim)))) 

    def ff_fn(v: jnp.ndarray) -> jnp.ndarray:
        diffs = mesh - v[None, :]  
        if periodic:
            diffs = jnp.mod(diffs + L / 2, L) - L / 2
        squared_norms = jnp.sum(diffs**2, axis=-1) 
        return jnp.exp(-squared_norms / (2 * sigma**2))

    return jax.vmap(ff_fn)(x)


def PW_Embedding(x: jnp.ndarray, L: float, embed_dim: int, phys_dim: int, sigma=None, periodic=None) -> jnp.ndarray:
    """
    Computes truncated L-periodic Fourier basis embeddings (Plane Waves).

    Maps particle positions to sinusoidal features, preserving periodicity naturally.
    
    Parameters:
      x: Particle configuration of shape (n_max, phys_dim).
      L: System length (periodicity).
      embed_dim: Number of frequency modes.
      phys_dim: Physical dimension of the local Hilbert space.
      sigma: (Unused) Kept for API compatibility.
      periodic: (Unused) Kept for API compatibility.

    Returns:
      embeddings: Array of shape (n_max, 2 * phys_dim * embed_dim).
    """
    grid_k = jnp.arange(1, embed_dim + 1)

    def ff_fn(v: jnp.ndarray) -> jnp.ndarray:

        phases = (2 * jnp.pi / L) * v[:, None] * grid_k[None, :]  
        sin_part = jnp.sin(phases)
        cos_part = jnp.cos(phases)
        return jnp.concatenate([sin_part, cos_part], axis=-1).reshape(-1)

    return jax.vmap(ff_fn)(x)