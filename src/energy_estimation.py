import jax
import jax.numpy as jnp
from typing import Any, Callable, Union, Dict

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
    
    
def Laplacian_sum(logpsi: Callable, params: Dict) -> Callable:
    """
    Creates a function to compute the sum of the Laplacian of logpsi 
    with respect to the positions of particles.
    
    Parameters:
     logpsi: the wavefunction ansatz (function or Flax module)
     params: the current variational ansatz parameters

    Returns:
     laplacian_fn: A function that returns the scalar Laplacian sum.
    """

    logpsi = canonicalize_ansatz(logpsi)
    hess_fn = jax.hessian(logpsi, argnums=1)

    def laplacian_fn(x: jnp.ndarray, mask_valid: jnp.ndarray) -> jnp.ndarray:    
        """
        Computes sum of the Laplacian for a single configuration.
        
        Parameters:
          x: a single particle configuration of shape (n_max, phys_dim)
          mask_valid: mask indicating the existing particles in the configuration

        Returns:
          laplacian_sum: The scalar sum of the Laplacian over all valid particles and dimensions.
        """
        # Compute the Hessian matrix
        hessian_matrix = hess_fn(params, x, mask_valid)  
        # Compute the trace over the spatial dimensions for each particle
        hess_trace = jnp.einsum('ijij->ij', hessian_matrix) 
        # Sum across particles and spatial dimensions
        laplacian_sum = jnp.sum(hess_trace, where=mask_valid[:, None])
        return laplacian_sum

    return laplacian_fn

def Gradient_square_sum(logpsi: Callable, params: Dict) -> Callable:
    """
    Creates a function to compute the sum of squared gradients of logpsi.
    with respect to the positions of particles.

    Parameters:
     logpsi: the wavefunction ansatz (function or Flax module)
     params: the current variational ansatz parameters

    Returns:
      grad_fn: A function that computes the squared gradient sum for a configuration.
    """
    logpsi = canonicalize_ansatz(logpsi)
    std_grad_fn = jax.grad(logpsi, argnums=1)

    def grad_fn(x: jnp.ndarray, mask_valid: jnp.ndarray) -> jnp.ndarray:        
        """
        Computes sum of the squared gradients for a single configuration.
        
        Parameters:
          x: a single particle configuration of shape (n_max, phys_dim)
          mask_valid: mask indicating the existing particles in the configuration

        Returns:
          grad_sum: The scalar sum of the squared gradients over all valid particles and dimensions.
        """
        grad = std_grad_fn(params, x, mask_valid)
        grad_sum = jnp.sum(grad**2, where=mask_valid[:, None])
        return grad_sum
    
    return grad_fn


def Local_Energy(logpsi: Callable, m: float, mu: float, V: Callable, W: Callable) -> Callable:
    """
    Creates a function to compute the total local energy of the system.
    
    Parameters:
      logpsi: The wavefunction ansatz.
      m: Particle mass.
      mu: Chemical potential.
      V: External potential function.
      W: Interaction potential function.

    Returns:
      E_local: A function that computes the total local energy, kinetic, potential, 
               and interaction energies for a given configuration.
    """

    logpsi = canonicalize_ansatz(logpsi)

    def KE_local(params: Dict, x, mask_valid):
        laplacian_fn = Laplacian_sum(logpsi, params)
        grad_fn = Gradient_square_sum(logpsi, params)
        lapl = laplacian_fn(x, mask_valid)
        grad = grad_fn(x, mask_valid)
        return 1 / (2 * m) * (-grad - lapl)
        # return 1 / (2 * m) * (grad)
    
    def PE_local(x, mask_valid):
        """Computes Potential (V) and Interaction (W) energies."""
        return V(x, mask_valid), W(x, mask_valid)
    
    def E_local(params: Dict, x, mask_valid):
        """
        Computes the total local energy and its components.
        
        Parameters:
          params: Variational parameters.
          x: Particle configuration.
          mask_valid: Mask for valid particles.
          
        Returns:
          E_total: Total local energy.
          KE: Kinetic energy component.
          VE: External potential component.
          WE: Interaction potential component.
        """
        KE  = KE_local(params, x, mask_valid)
        VE, WE = PE_local(x, mask_valid)
        return WE+VE+KE, KE, VE, WE
        
    return E_local