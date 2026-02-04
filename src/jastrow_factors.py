import jax
from jax import numpy as jnp
import jax.scipy.special as jsp
import numpy as np


def log_jastrow(jastrow_type="Const"):
    """
    Function to select the log-Jastrow factor function.
    """
    if jastrow_type == 'CS1D':
        return log_jastrow_CS1D
    elif jastrow_type == 'CS2D':
        return log_jastrow_CS2D
    elif jastrow_type == 'LL1D':
        return log_jastrow_LL1D
    else:
        raise ValueError("Jastrow factor hasn't been implemented for this mdoel yet!")


def log_jastrow_CS1D(x, mask_valid, L, m, g, periodic):
    """
    Computes a 1D Jastrow factor with cusp conditions suitable for Calogero-Sutherland-like interactions.
    
    Parameters:
      x: a single particle configuration of shape (n_max, phys_dim)
      mask_valid: mask indicating the existing particles in the configuration
      L: System length.
      m: Particle mass.
      g: Interaction coupling constant.
      periodic: Whether to apply Minimum Image Convention.

    Returns:
      Scalar log-Jastrow value.
    """
    n = x.shape[0]
    row_idx, col_idx = jnp.triu_indices(n, k=1)
    interparticle_seps = x[row_idx] - x[col_idx]
    if periodic:
        interparticle_seps = interparticle_seps - L * jnp.round(interparticle_seps / L)
    upper_mask = mask_valid[row_idx] & mask_valid[col_idx]
    lam = 0.5 * (1 + (1 + 4 * m * g) ** 0.5)
    jastrows = jnp.where(upper_mask[:, None], jnp.tanh(17 * jnp.abs(interparticle_seps / L)) ** lam, 1)
    jastrows *= jnp.where(upper_mask[:, None], jnp.tanh(17 * (1 - jnp.abs(interparticle_seps / L))) ** lam, 1)
    return jnp.sum(jnp.log(jastrows), where=upper_mask[:, None])


def log_jastrow_CS2D(x, mask_valid, L, m, g, periodic):
    """
    Computes a 2D Jastrow factor with cusp conditions suitable for Calogero-Sutherland-like interactions.
    
    Parameters:
      x: a single particle configuration of shape (n_max, phys_dim)
      mask_valid: mask indicating the existing particles in the configuration
      L: System length.
      m: Particle mass.
      g: Interaction coupling constant.
      periodic: Whether to apply Minimum Image Convention.

    Returns:
      Scalar log-Jastrow value.
    """
    n = x.shape[0]
    row_idx, col_idx = jnp.triu_indices(n, k=1)
    diffs = x[row_idx] - x[col_idx]
    if periodic:
        diffs = diffs - L * jnp.round(diffs / L)
    interparticle_seps = jnp.linalg.norm(diffs, axis=-1)
    upper_mask = mask_valid[row_idx] & mask_valid[col_idx]
    lam = (m * g) ** 0.5
    jastrows = jnp.where(upper_mask, jnp.tanh(12 * interparticle_seps / L) ** lam, 1)
    return jnp.sum(jnp.log(jastrows), where=upper_mask)

def log_jastrow_LL1D(x, mask_valid, L, m, g, periodic):
    """
    Computes the Log-Jastrow factor for a 1D Lieb-Liniger gas (contact repulsion).

    Parameters:
      x: a single particle configuration of shape (n_max, phys_dim)
      mask_valid: mask indicating the existing particles in the configuration
      L: System length.
      m: Particle mass.
      g: Interaction strength (coupling constant).
      periodic: If True, applies Minimum Image Convention.

    Returns:
      val: Scalar log-amplitude of the Jastrow factor.
    """
    n = x.shape[0]
    row_idx, col_idx = jnp.triu_indices(n, k=1)
    diffs = x[row_idx] - x[col_idx]
    if periodic:
        diffs = diffs - L * jnp.round(diffs / L)
    pairwise_diffs = jnp.linalg.norm(diffs, axis=-1)
    upper_mask = mask_valid[row_idx] & mask_valid[col_idx]

    val = jnp.sum(jnp.where(upper_mask, jnp.log(pairwise_diffs  + 1 / (m * g)), 0))

    # Tonks-Girardeau limit correction for large g
    if g > 1e4:
        log_Selberg_vals = [
            (2 * jsp.gammaln(0 * n + 1 + j) +
             jsp.gammaln(0 * n + 2 + j) -
             jsp.gammaln(1 + n + j)) for j in range(n)
        ]
        log_Selberg_vals = jnp.stack(log_Selberg_vals)
        n_to_sum = jnp.sum(mask_valid)
        indices = jnp.arange(n)
        mask_sum = indices < n_to_sum
        log_Selberg = jnp.sum(log_Selberg_vals, where=mask_sum)
        val += - n_to_sum / 2 * jnp.log(L) -  1/2 * log_Selberg

    return val
