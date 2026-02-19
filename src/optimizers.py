from functools import partial
from typing import Any, Callable, Tuple, Union, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.numpy.linalg import lstsq
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
    
    
def adam_update(logpsi: Callable, N: int, chunk_grads_size: int, solver: Callable=lstsq, **kwargs) -> Callable:
    '''
    Constructs a function to compute the gradient of the energy with respect to parameters 
    using Vector-Jacobian Product (VJP). This is suitable for first-order optimizers like Adam or SGD.

    Parameters:
      logpsi: the wavefunction ansatz (function or Flax module)
      N: total number of MC samples
      chunk_grads_size: batch size for chunked vmap execution to manage GPU memory usage
      solver: ignored for this update type, kept only for API consistency
      kwargs: additional keyword arguments passed to the optimizer

    Returns:
      get_grads_fn: a function that returns the estimated gradients of the energy
    '''
    logpsi = canonicalize_ansatz(logpsi)
    batch_logpsi = vmap(logpsi, in_axes=(None, 0, 0), chunk_size=chunk_grads_size)
    
    def get_grads_fn(params: Any, x: jnp.ndarray, mask_valid: jnp.ndarray, Delta_Energies: jnp.ndarray, diag_shift: float) -> Any:
        '''
        Computes the gradients via VJP.

        Parameters:
          params: current network parameters
          x: a single particle configuration of shape (n_max, phys_dim)
          mask_valid: mask indicating the existing particles in the configuration
          Delta_Energies: centered local energies (E_loc - mean(E_loc))
          diag_shift: unused for standard gradients, kept for API consistency

        Returns:
          updates: the gradient of the energy with respect to params
        '''
        _, back = jax.vjp(lambda p: batch_logpsi(p, x, mask_valid), params)
        updates = back(2*Delta_Energies/N)
        return updates[0]

    return get_grads_fn


def sr_update(logpsi: Callable, Ns: int, chunk_grads_size: int, solver: Callable=lstsq, **kwargs) -> Callable:
    '''
    Constructs a function to estimate the update parameters (Natural Gradients) using Stochastic 
    Reconfiguration (SR).

    Parameters:
    logpsi: the wavefunction ansatz (function or Flax module)
    Ns: total number of MC samples
    chunk_grads_size: batch size for chunked vmap execution to manage GPU memory usage
    solver: linear solver to use for the SR system (default: lstsq)
    kwargs: additional arguments passed to the solver

    Returns:
    get_grads: a function that returns the update parameters (Natural Gradients)
    '''
    logpsi = canonicalize_ansatz(logpsi)
    
    def get_grads(params: Any, x: jnp.ndarray, mask_valid: jnp.ndarray, E_deltas: jnp.ndarray, diag_shift: float) -> Any:
        flat_params, rebuild = ravel_pytree(params)
        logpsi_ = lambda p, x, mask_valid: logpsi(rebuild(p), x, mask_valid)
        jacob_logpsi = vmap(jax.grad(logpsi_, argnums=0), in_axes=(None, 0, 0), chunk_size=chunk_grads_size)
        O = jacob_logpsi(flat_params, x, mask_valid)  
        Obar = (O - O.mean(axis=0, keepdims=True)) / jnp.sqrt(Ns)  
        Ebar = E_deltas / jnp.sqrt(Ns) 
        grads_flat = _solve_sr(Obar, Ebar, diag_shift, solver, **kwargs)
        return rebuild(grads_flat)
    
    return get_grads

def _solve_sr(Obar: jnp.ndarray, Ebar: jnp.ndarray, diag_shift: float=1e-3, solver: Callable=lstsq, **kwargs) -> jnp.ndarray:
    '''
    Solves the SR linear system S * update = F.

    Parameters:
      Obar: the centered log-derivative matrix
      Ebar: the centered energy vector
      diag_shift: regularization parameter added to the diagonal of the S matrix
      solver: the linear solver function (e.g. lstsq or cg)
      kwargs: additional keyword arguments passed to the solver

    Returns:
      updates_flat: the solution vector to the linear system (flattened)
    '''
    S = Obar.T.conj() @ Obar
    b = Obar.T.conj() @ Ebar
    S = S.at[jnp.diag_indices_from(S)].add(diag_shift) 
    return solver(S, b, **kwargs)


def min_sr_update(logpsi: Callable, Ns: int, chunk_grads_size: int, solver: Callable=lstsq, **kwargs) -> Callable:
    '''
    Constructs a function to estimate the update parameters (Natural Gradients) 
    using Min-SR. Efficient when number of parameters >> number of samples.

    Parameters:
    logpsi: the wavefunction ansatz (function or Flax module)
    Ns: total number of MC samples
    chunk_grads_size: batch size for chunked vmap execution to manage GPU memory usage
    solver: linear solver to use (default: lstsq)
    kwargs: arguments passed to the solver

    Returns:
    get_grads: a function that returns the update parameters (Natural Gradients)
    '''
    logpsi = canonicalize_ansatz(logpsi)
    
    def get_grads(params: Any, x: jnp.ndarray, mask_valid: jnp.ndarray, E_deltas: jnp.ndarray, diag_shift: float = 1e-4) -> Any:
        flat_params, rebuild = ravel_pytree(params)
        logpsi_ = lambda p, x, mask_valid: logpsi(rebuild(p), x, mask_valid)
        jacob_logpsi = vmap(jax.grad(logpsi_, argnums=0), in_axes=(None, 0, 0), chunk_size=chunk_grads_size)
        O = jacob_logpsi(flat_params, x, mask_valid) 
        Obar = (O - O.mean(axis=0, keepdims=True)) / jnp.sqrt(Ns)  
        Ebar = E_deltas / jnp.sqrt(Ns) 
        grads_flat = _solve_minsr(Obar, Ebar, diag_shift, solver, **kwargs)
        return rebuild(grads_flat)
    
    return get_grads

def _solve_minsr(Obar: jnp.ndarray, Ebar: jnp.ndarray, diag_shift: float=1e-3, solver: Callable=lstsq, **kwargs) -> jnp.ndarray:
    '''
    Solves the MinSR system
    
    Parameters:
      Obar: the centered log-derivative matrix
      Ebar: the centered energy vector
      diag_shift: regularization parameter
      solver: linear solver function
      kwargs: additional keyword arguments passed to the solver

    Returns:
      updates: the update vector projected back to parameter space
    '''
    T = Obar @ Obar.T
    if solver is lstsq:
        return Obar.T.conj() @ solver(T, Ebar, rcond=diag_shift, **kwargs)[0]
    elif callable(solver):
        T = T.at[jnp.diag_indices_from(T)].add(diag_shift)
        return Obar.T.conj() @ solver(T, Ebar, **kwargs)
    else:
        raise ValueError("solver must be a callable (e.g., lstsq or cg)")
