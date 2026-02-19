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


def Local_obdm(logpsi, params, m, L, N_int, chunk_int_size):
    """
    Creates a function to compute the local one-body density matrix (obdm) estimate for a single configuration.

    Parameters:
      logpsi: Wavefunction ansatz.
      params: Variational parameters.
      m: Particle mass.
      L: System length.
      N_int: Number of grid points for the normalization integral.
      chunk_int_size: Chunk size for the integration grid (0 = no chunking).

    Returns:
      lcl_obdm_fn: Function returning the obdm contribution of a sample.
    """
    logpsi = canonicalize_ansatz(logpsi)
    
    def lcl_obdm_fn(x, mask_valid, x_eval):
        inds = jnp.argmax(mask_valid)
        n = jnp.sum(mask_valid)

        marginal_logpsi_fn =  Marginal_logpsi(logpsi, params)
        log_Psi_numerator_val = marginal_logpsi_fn(x, mask_valid, (x_eval + x[inds]) % L, inds)

        num_obdm_local = n*jnp.exp(log_Psi_numerator_val - logpsi(params, x, mask_valid))/L
        return num_obdm_local
    
    return lcl_obdm_fn


def obdm(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup, sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size):
    """
    Creates a function to estimate the system's obdm profile via Monte Carlo sampling.

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
      obdm_fn: Function that returns (obmds_density, obdms_std) at a given point `x_eval`.
    """
    logpsi = canonicalize_ansatz(logpsi)
    lcl_obdm_fn = Local_obdm(logpsi, params, m, L, N_int, chunk_int_size)
    sampler = jax.jit(MetropolisHastings_sampler(logpsi, MetropolisHastings_proposal, n_samples,
                                                    n_chains, warmup , sweep_size, L, n_max, phys_dim, w, pm))
    
    def obdm_fn(x_eval, n_0, rng):
        rng, rand1 = jax.random.split(rng)
        x = sampler(params, n_0, rand1)[0]
        x = jnp.concat(x)
        mask_valid = ~jnp.isnan(x)
        mask_valid = jnp.any(mask_valid, axis=2)
        x = jnp.nan_to_num(x)

        obdm_locals = vmap(lcl_obdm_fn, in_axes=(0, 0, None), chunk_size=chunk_size)(x, mask_valid, x_eval)
    
        obdms_mean = jnp.mean(obdm_locals)
        obdm_locals = obdm_locals.reshape(n_chains, -1)
        obdm_mean_chains = jnp.mean(obdm_locals, axis=1)        
        obdms_std = jnp.std(jnp.array(obdm_mean_chains))/ (n_chains ** 0.5)

        return obdms_mean, obdms_std
    
    return obdm_fn

def obdm_estim(obdm_type):
    """
    Function to select the obdm estimation strategy.

    Parameters:
      obdm_type: 
        - 'full': Evaluates obdm on a full Cartesian grid (both for 1D and 2D).
        - 'averaged': Evaluates obdm averaged obdm profile (only for 2D cae).

    Returns:
      The corresponding obdm evaluation function.
    """
    if obdm_type=="full":
        return obdm_on_grid
    elif obdm_type=="averaged":
        return obdm_averaged
    else:
        raise ValueError("Pick a valid obdm type: 'full' or 'averaged'")
    
def obdm_on_grid(logpsi, params, m, L, n_0, N_int, n_samples, n_chains, warmup , sweep_size, pm, 
                              n_max, phys_dim, w, bounds, rng, chunk_int_size, chunk_size, chunk_plt_size, num_points):
    """
    Computes the obdm profile on a regular Cartesian grid.

    Returns:
      obdm_mean: Mean obdm values at each grid point.
      obdm_std: Standard error of the obdm at each grid point.
      points: The coordinates of the grid points.
    """
    obdm_fn =  jax.jit(obdm(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup , sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size))
    low, high = bounds
    grids = jnp.linspace(low, high, num_points)
    mesh = jnp.array(jnp.meshgrid(*([grids] * phys_dim), indexing="ij"))
    points = mesh.reshape(phys_dim, -1).T
    rng = jax.random.split(rng, len(points))
    obdms_mean, obdms_std = vmap(obdm_fn, in_axes=(0, None, 0), chunk_size=chunk_plt_size)(points, n_0 ,rng)

    return obdms_mean, obdms_std, points

def obdm_averaged(logpsi, params, m, L, n_0, N_int, n_samples, n_chains, warmup , sweep_size, pm, 
                              n_max, phys_dim, w, bounds, rng, chunk_int_size, chunk_size, chunk_plt_size, n_angles, n_dists):
    """
    Computes the radially averaged obdm profile.
    
    Constructs a polar grid centered in the box, evaluates the obdm, and 
    averages over the angular component to return rho(r).

    Returns:
      obdms_mean_avg: Radially averaged obdm mean.
      obdms_std_avg: Aggregated uncertainty (variance-based combination).
      radial_coords: The radial coordinates (r + center) corresponding to the averages.
    """
    obdm_fn =  jax.jit(obdm(logpsi, params,  m, L, N_int, n_samples, n_chains, warmup , sweep_size, pm, n_max, phys_dim, w, chunk_int_size, chunk_size))
    low, high = bounds

    angles = jnp.radians(jnp.linspace(0, 360, n_angles, endpoint=False))
    dists = jnp.linspace(0, (high-low)/2, n_dists)
    center_x, center_y = 0.0, 0.0
    vectors = []
    for dist in dists:
        for angle in angles:
            x = center_x + dist * jnp.cos(angle)
            y = center_y + dist * jnp.sin(angle)
            vectors.append((x, y))
    points = jnp.array(vectors)
    rng = jax.random.split(rng, len(points))
    obdms_mean, obdms_std = vmap(obdm_fn, in_axes=(0, None, 0), chunk_size=chunk_plt_size)(points, n_0 ,rng)
    obdms_mean_reshaped = obdms_mean[:(len(obdms_mean) // n_angles) * n_angles].reshape(-1, n_angles)  
    obdms_std_reshaped = obdms_std[:(len(obdms_std) // n_angles) * n_angles].reshape(-1, n_angles)
    obdms_mean_avg = obdms_mean_reshaped.mean(axis=1) 
    obdms_std_avg = (obdms_std_reshaped**2).sum(axis=1)/float(n_angles) 
    return obdms_mean_avg, obdms_std_avg, dists+center_x
