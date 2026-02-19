import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple, Union, Optional, Dict
from src.metropolis_sampling import MetropolisHastings_sampler, MetropolisHastings_proposal
from src.vmap_chunked import vmap


def canonicalize_ansatz(logpsi: Union[Callable, Any]) -> Callable:
    '''
    Ensures the wave function ansatz is a callable function.

    Parameters:
     logpsi: the ansatz, which can be a raw python function or a Flax 
        module (with an .apply method)

    Returns:
     logpsi_apply: the callable ansatz function signature f(params, x, ...)
    '''
    if hasattr(logpsi, 'apply'):
        return logpsi.apply
    elif callable(logpsi):
        return logpsi
    else:
        raise ValueError("logpsi must be a function or a Flax module")

def Marginal_logpsi(logpsi: Callable, params: Dict) -> Callable:
    """
    Creates a function to evaluate the logpsi for a modified configuration.

    Parameters:
      logpsi: the wavefunction ansatz (function or Flax module)
      params: The current variational ansatz parameters.

    Returns:
      marginal_logpsi_fn: A function that returns log(Psi(x_new)).
    """

    logpsi = canonicalize_ansatz(logpsi)

    def marginal_logpsi_fn(x: jnp.ndarray, mask_valid, x_marg: jnp.ndarray, inds: jnp.ndarray):
        x = x.at[inds,:].set(x_marg)
        return logpsi(params, x, mask_valid)
    
    return marginal_logpsi_fn

def Marginal_gradient_square_sum(logpsi: Callable, params: Dict) -> Callable:
    """
    Creates a function to evaluate the sum of squared gradients of the wavefunction (Psi) for a modified configuration.

    Parameters:
      logpsi: the wavefunction ansatz (function or Flax module)
      params: The current variational ansatz parameters.

    Returns:
      marginal_grad_fn: A function that returns sum(|grad_x Psi(x_new)|^2) over selected indices.
    """
    logpsi = canonicalize_ansatz(logpsi)
    psi = lambda p, x, mask_valid: jnp.exp(logpsi(p, x, mask_valid))
    std_grad_fn = jax.grad(psi, argnums=1)

    def marginal_grad_fn(x: jnp.ndarray, mask_valid, x_marg: jnp.ndarray, inds: jnp.ndarray) -> jnp.ndarray:
        x = x.at[inds,:].set(x_marg)
        grad = jnp.sum(jnp.where(mask_valid[inds], std_grad_fn(params, x, mask_valid)[inds]**2, 0))
        return grad
    
    return marginal_grad_fn


def marginal_grid_integration(logpsi: Callable, params: Dict, bounds: Tuple[float, float], N_int: int, chunk_int_size) -> Callable:
    """
    Creates a function to numerically integrate |Psi|^2 over a grid for a subset of coordinates.

    Parameters:
      logpsi: the wavefunction ansatz (function or Flax module)
      params: The current variational ansatz parameters.
      bounds: A tuple (0, L) defining the integration range for each dimension.
      N_int: Number of grid points per dimension.
      chunk_int_size: Batch size for vectorizing the evaluation over grid points.

    Returns:
      marginal_grid_integration_fn: A function that returns the integral of |Psi(x_new)|^2 over the grid at specified indices.
    """
    logpsi = canonicalize_ansatz(logpsi)

    def marginal_grid_integration_fn(x: jnp.ndarray, mask_valid, inds: jnp.ndarray) -> jnp.ndarray:
        marg_logpsi_fn = Marginal_logpsi(logpsi, params)
        marg_psi_squared = lambda x_marg: jnp.exp(2 * marg_logpsi_fn(x, mask_valid, x_marg, inds))

        phys_dim = x.shape[1]  
        low, high = bounds  
        
        grids = [jnp.linspace(low, high, N_int) for _ in range(phys_dim)]
        mesh = jnp.meshgrid(*grids, indexing="ij")
        points = jnp.stack([m.ravel() for m in mesh], axis=-1)
        
        func_vals = vmap(marg_psi_squared, chunk_size=chunk_int_size)(points)
        func_vals = func_vals.reshape([N_int] * phys_dim)

        integral = func_vals
        for dim in range(phys_dim):
            integral = jnp.trapezoid(integral, x=grids[dim], axis=0)
        return integral
    
    return marginal_grid_integration_fn


def Local_ke_density(logpsi, params, m, L, N_int, chunk_int_size):
    """
    Creates a function to evaluate the local kinetic energy density at a given spatial point.

    Parameters:
      logpsi: the wavefunction ansatz (function or Flax module)
      params: The current variational ansatz parameters.
      m: Particle mass.
      L: System size.
      N_int: Number of integration grid points.
      chunk_int_size: Batch size for integration vectorization.

    Returns:
      lcl_ke_density_fn: A function that returns the local kinetic energy density.
    """
    logpsi = canonicalize_ansatz(logpsi)
    marginal_grad_fn = Marginal_gradient_square_sum(logpsi, params)
    bounds = (0, L)
    marginal_grid_integration_fn = marginal_grid_integration(logpsi, params, bounds, N_int, chunk_int_size)
    
    def lcl_ke_density_fn(x, mask_valid, x_eval):
        inds = jnp.argmax(mask_valid)
        n = jnp.sum(mask_valid)
        gradient = marginal_grad_fn(x, mask_valid, x_eval, inds)
        integrals = marginal_grid_integration_fn(x, mask_valid, inds)        
        ke_density_local = (1 / (2 * m) * n * gradient  / integrals)
        return ke_density_local
    return lcl_ke_density_fn


def KE_density(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup, sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size):
    """
    Creates a MC estimator for the local kinetic energy density at a given spatial point.

    Parameters:
      logpsi: The wavefunction ansatz.
      params: Variational parameters.
      m: Particle mass.
      L: System size.
      N_int: Number of integration grid points.
      n_samples: Number of samples per chain.
      n_chains: Number of parallel chains.
      warmup: Number of initial steps to discard.
      sweep_size: Steps between saved samples.
      pm: Probability of proposing an add/remove move.
      n_max: Maximum number of particles.
      phys_dim: Physical dimension.
      w: Width of perturbation.
      chunk_int_size: Batch size for internal integration vectorization.
      chunk_size: Batch size for evaluating samples over chains.

    Returns:
      ke_density_fn: A function `f(x_eval, n_0, rng)` returning (mean, std_err) of KE density.
    """
    logpsi = canonicalize_ansatz(logpsi)
    lcl_KE_density_fn = Local_ke_density(logpsi, params, m, L, N_int, chunk_int_size)
    sampler = jax.jit(MetropolisHastings_sampler(logpsi, MetropolisHastings_proposal, n_samples,
                                                    n_chains, warmup , sweep_size, L, n_max, phys_dim, w, pm))
    
    def ke_density_fn(x_eval, n_0, rng):
        rng, rand1 = jax.random.split(rng)
        x = sampler(params, n_0, rand1)[0]
        x = jnp.concat(x)
        mask_valid = ~jnp.isnan(x)
        mask_valid = jnp.any(mask_valid, axis=2)
        x = jnp.nan_to_num(x)

        ke_density_locals = vmap(lcl_KE_density_fn, in_axes=(0, 0, None), chunk_size=chunk_size)(x, mask_valid, x_eval)

        ke_densities_mean = jnp.mean(ke_density_locals)
        ke_density_locals = ke_density_locals.reshape(n_chains, -1)
        ke_densities_mean_chains = jnp.mean(ke_density_locals, axis=1)        
        ke_densities_std = jnp.std(jnp.array(ke_densities_mean_chains))/ (n_chains ** 0.5)
        return ke_densities_mean, ke_densities_std
    
    return ke_density_fn
    
def KE_density_on_grid(logpsi: Callable, params: Dict, m: float, L: float, n_0: jnp.ndarray, N_int: int, 
                       n_samples: int, n_chains: int, warmup: int, sweep_size: int, pm: float, 
                       n_max: int, phys_dim: int, w: float, bounds: Tuple[float, float], rng: jnp.ndarray, 
                       chunk_int_size: int, chunk_size: int, chunk_plt_size: int, num_points: int):
    """
    Evaluates the kinetic energy (ke) density profile on a regular Cartesian grid.

    Returns:
      ke_dens_mean: Mean ke density values at each grid point.
      ke_dens_std: Standard error of the ke density at each grid point.
      points: The coordinates of the grid points.
    """
    KE_density_fn =  jax.jit(KE_density(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup , sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size))
    low, high = bounds
    grids = jnp.linspace(low, high, num_points)
    mesh = jnp.array(jnp.meshgrid(*([grids] * phys_dim), indexing="ij"))
    points = mesh.reshape(phys_dim, -1).T
    rng = jax.random.split(rng, len(points))
    ke_dens_mean, ke_dens_std = vmap(KE_density_fn, in_axes=(0, None, 0), chunk_size=chunk_plt_size)(points, n_0 ,rng)

    return ke_dens_mean, ke_dens_std, points



