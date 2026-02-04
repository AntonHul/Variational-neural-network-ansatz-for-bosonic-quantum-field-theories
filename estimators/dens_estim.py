import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple, Union, Optional, Dict
from src.metropolis_sampling import MetropolisHastings_sampler, MetropolisHastings_proposal
from src.vmap_chunked import vmap
from jax.scipy.special import logsumexp


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


def logspace_trapezoid_nd(log_vals, grids):
    """
    Performs N-dimensional trapezoidal integration on log-domain data.
    
    Parameters:
      log_vals: Input array containing log(f(x)). Shape (D1, D2, ..., Dn).
      grids: List of 1D arrays [x1, x2, ..., xn] defining the grid points 
             for each dimension.

    Returns:
      log_integral: The natural logarithm of the integral result. 
    """

    phys_dim = len(grids)

    for dim in range(phys_dim):
        x = grids[dim]  
        dx = jnp.diff(x)

        w = jnp.concatenate((
            dx[:1] / 2,
            (dx[:-1] + dx[1:]) / 2,
            dx[-1:] / 2
        ))  
        broadcast_shape = (w.shape[0],) + (1,) * (log_vals.ndim - 1)
        w = w.reshape(broadcast_shape)
        log_vals = logsumexp(log_vals + jnp.log(w), axis=0)
    return log_vals

def marginal_grid_integration(logpsi: Callable, params: Dict, bounds: Tuple[float, float], N_int: int, chunk_int_size) -> Callable:
    """
    Creates a function to numerically integrate the squared wavefunction magnitude 
    over the spatial coordinates of a single particle.

    Parameters:
      logpsi: The wavefunction ansatz.
      params: Variational parameters.
      bounds: Tuple (low, high) defining the integration limits (e.g., (0, L)).
      N_int: Number of grid points per spatial dimension.
      chunk_int_size: Batch size for evaluation. 

    Returns:
      marginal_grid_integration_fn: Function that computes the log-integral.
    """
    logpsi = canonicalize_ansatz(logpsi)

    def marginal_grid_integration_fn(x: jnp.ndarray, mask_valid, inds: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the log-integral for the particle specified by `inds`.

        Parameters:
          x: Current configuration of shape (n_max, phys_dim).
          mask_valid: mask indicating the existing particles in the configuration
          inds: Index of the particle to integrate out (must be a single index).
        
        Returns:
          log_normalization: Scalar log-value of the integral.
        """
        marg_logpsi_fn = Marginal_logpsi(logpsi, params)
        marg_psi_squared = lambda x_marg: 2 * marg_logpsi_fn(x, mask_valid, x_marg, inds)
        phys_dim = x.shape[1]  
        low, high = bounds  
        

        grids = [jnp.linspace(low, high, N_int) for _ in range(phys_dim)]
        mesh = jnp.meshgrid(*grids, indexing="ij")
        points = jnp.stack([m.ravel() for m in mesh], axis=-1)
        
        func_vals = vmap(marg_psi_squared, chunk_size=chunk_int_size)(points)
        func_vals = func_vals.reshape([N_int] * phys_dim)

        integral = logspace_trapezoid_nd(func_vals, grids)

        return integral
    
    return marginal_grid_integration_fn


def Local_density(logpsi, params, m, L, N_int, chunk_int_size):
    """
    Creates a function to compute the local probability density estimate for a single configuration.

    Parameters:
      logpsi: Wavefunction ansatz.
      params: Variational parameters.
      m: Particle mass.
      L: System length.
      N_int: Number of grid points for the normalization integral.
      chunk_int_size: Chunk size for the integration grid (0 = no chunking).

    Returns:
      lcl_density_fn: Function returning the density contribution of a sample.
    """
    logpsi = canonicalize_ansatz(logpsi)
    bounds = (0, L)
    marginal_grid_integration_fn = marginal_grid_integration(logpsi, params, bounds, N_int, chunk_int_size)
    
    def lcl_density_fn(x, mask_valid, x_eval):
        inds = jnp.argmax(mask_valid)

        integrals = marginal_grid_integration_fn(x, mask_valid, inds)

        marginal_logpsi_fn =  Marginal_logpsi(logpsi, params)
        log_Psi_numerator_val = marginal_logpsi_fn(x, mask_valid, x_eval, inds)

        num_density_local = jnp.exp(2 * log_Psi_numerator_val - integrals)
        return num_density_local
    return lcl_density_fn

def Density(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup, sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size):
    """
    Creates a function to estimate the system's density profile via Monte Carlo sampling.

    Parameters:
      logpsi: Wavefunction ansatz.
      params: Variational parameters.
      m: Particle mass.
      L: System length.
      N_int: Grid points for normalization integral.
      n_samples: Number of MC samples per chain.
      n_chains: Number of parallel MC chains.
      warmup: Number of warmup (burn-in) steps for the sampler.
      sweep_size: Number of steps between samples to ensure decorrelation.
      pm: Probability of proposing a change in particle number during MCMC steps.
      n_max: Maximum particle number.
      phys_dim: Physical dimensions.
      w: Width parameter for the proposal distribution in the MCMC sampling scheme.
      chunk_int_size: Batch size for integration grid.
      chunk_size: Batch size for vmapping over samples.

    Returns:
      density_fn: Function that returns (mean_density, std_error) at a given point `x_eval`.
    """

    logpsi = canonicalize_ansatz(logpsi)
    lcl_density_fn = Local_density(logpsi, params, m, L, N_int, chunk_int_size)
    sampler = jax.jit(MetropolisHastings_sampler(logpsi, MetropolisHastings_proposal, n_samples,
                                                    n_chains, warmup , sweep_size, L, n_max, phys_dim, w, pm))
    
    def density_fn(x_eval, n_0, rng):
        rng, rand1 = jax.random.split(rng)
        x = sampler(params, n_0, rand1)[0]
        x = jnp.concat(x)
        mask_valid = ~jnp.isnan(x)
        mask_valid = jnp.any(mask_valid, axis=2)
        x = jnp.nan_to_num(x)

        density_locals = vmap(lcl_density_fn, in_axes=(0, 0, None), chunk_size=chunk_size)(x, mask_valid, x_eval)

        densities_mean = jnp.mean(density_locals)
        density_locals = density_locals.reshape(n_chains, -1)
        densities_mean_chains = jnp.mean(density_locals, axis=1)        
        densities_std = jnp.std(jnp.array(densities_mean_chains))/ (n_chains ** 0.5)
        return densities_mean, densities_std
    
    return density_fn

def Density_estim(density_type):
    """
    Function to select the density estimation strategy.

    Parameters:
      density_type: 
        - 'full': Evaluates density on a full Cartesian grid (both for 1D and 2D).
        - 'averaged': Evaluates radially averaged density profile (only for 2D cae).

    Returns:
      The corresponding density evaluation function.
    """
    if density_type=="full":
        return Density_on_grid
    elif density_type=="averaged":
        return Density_averaged
    else:
        raise ValueError("Pick a valid density type: 'full' or 'averaged'")
    
def Density_on_grid(logpsi, params, m, L, n_0, N_int, n_samples, n_chains, warmup , sweep_size, pm, 
                              n_max, phys_dim, w, bounds, rng, chunk_int_size, chunk_size, chunk_plt_size, num_points):
    """
    Computes the density profile on a regular Cartesian grid.

    Returns:
      dens_mean: Mean density values at each grid point.
      dens_std: Standard error of the density at each grid point.
      points: The coordinates of the grid points.
    """
    density_fn =  jax.jit(Density(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup , sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size))
    low, high = bounds
    grids = jnp.linspace(low, high, num_points)
    mesh = jnp.array(jnp.meshgrid(*([grids] * phys_dim), indexing="ij"))
    points = mesh.reshape(phys_dim, -1).T
    rng = jax.random.split(rng, len(points))
    dens_mean, dens_std = vmap(density_fn, in_axes=(0, None, 0), chunk_size=chunk_plt_size)(points, n_0 ,rng)

    return dens_mean, dens_std, points

def Density_averaged(logpsi, params, m, L, n_0, N_int, n_samples, n_chains, warmup , sweep_size, pm, 
                              n_max, phys_dim, w, bounds, rng, chunk_int_size, chunk_size, chunk_plt_size, n_angles, n_dists):
    """
    Computes the radially averaged density profile.
    
    Constructs a polar grid centered in the box, evaluates the density, and 
    averages over the angular component to return rho(r).

    Returns:
      dens_mean_avg: Radially averaged density mean.
      dens_std_avg: Aggregated uncertainty (variance-based combination).
      radial_coords: The radial coordinates (r + center) corresponding to the averages.
    """
    density_fn =  jax.jit(Density(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup , sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size))
    low, high = bounds

    angles = jnp.radians(jnp.linspace(0, 360, n_angles, endpoint=False))
    dists = jnp.linspace(0, (high-low)/2, n_dists)
    center_x, center_y = (high-low)/2, (high-low)/2
    vectors = []
    for dist in dists:
        for angle in angles:
            x = center_x + dist * jnp.cos(angle)
            y = center_y + dist * jnp.sin(angle)
            vectors.append((x, y))
    points = jnp.array(vectors)
    rng = jax.random.split(rng, len(points))
    dens_mean, dens_std = vmap(density_fn, in_axes=(0, None, 0), chunk_size=chunk_plt_size)(points, n_0 ,rng)
    dens_mean_reshaped = dens_mean[:(len(dens_mean) // n_angles) * n_angles].reshape(-1, n_angles)  
    dens_std_reshaped = dens_std[:(len(dens_std) // n_angles) * n_angles].reshape(-1, n_angles)
    dens_mean_avg = dens_mean_reshaped.mean(axis=1) 
    dens_std_avg = (dens_std_reshaped**2).sum(axis=1)/float(n_angles) 
    return dens_mean_avg, dens_std_avg, dists+center_x

