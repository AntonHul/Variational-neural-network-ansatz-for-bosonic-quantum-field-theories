import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple, Union, Optional, Dict
from src.metropolis_sampling import MetropolisHastings_sampler, MetropolisHastings_proposal
from src.vmap_chunked import vmap
from jax import lax
from jax.scipy.special import factorial


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
    
def hermite_phys(n: int, x: jnp.ndarray) -> jnp.ndarray:    
    """
    Computes the Hermite polynomial H_n(x) using the recurrence relation.
    
    H_n(x) = 2x * H_{n-1}(x) - 2(n-1) * H_{n-2}(x)
    
    Parameters:
      n: The order of the Hermite polynomial.
      x: The input coordinates.

    Returns:
      result: The value of H_n(x).
    """
    def body(i, val):
        Hn_2, Hn_1 = val
        Hn = 2 * x * Hn_1 - 2 * (i - 1) * Hn_2
        return (Hn_1, Hn)

    H0 = jnp.ones_like(x)
    H1 = 2 * x

    result = lax.cond(
        n == 0,
        lambda _: H0,
        lambda _: lax.cond(
            n == 1,
            lambda _: H1,
            lambda _: lax.fori_loop(2, n + 1, body, (H0, H1))[1],
            operand=None,
        ),
        operand=None,
    )
    return result

def get_orbitals_list(n):
    """
    Generates a list of quantum number pairs (n_x, n_y).
    Creates an upper triangular set of indices, used for basis enumeration.
    
    Returns:
      An array of shape (M, 2) containing pairs [i, j] where j >= i.
    """
    return jnp.array([[i, j] for i in range(n) for j in range(i, n)])


def phi_2d(x, n_x, n_y, L, beta=1.0):
    """
    Evaluates the full 2D Quantum Harmonic Oscillator eigenstate.
    
    Psi(x, y) = N * H_{n_x}(beta*x') * H_{n_y}(beta*y') * exp(-beta^2 * r'^2 / 2)
    where x' is the coordinate shifted to the center of the box L/2.

    Parameters:
      x: Position vector [x, y].
      n_x, n_y: Quantum numbers for X and Y dimensions.
      L: System size (used to center the oscillator).
      beta: Inverse length scale (sqrt(m*omega/hbar)).

    Returns:
      psi: The wavefunction amplitude.
    """
    x = x - jnp.array([L/2, L/2])
    hx =  hermite_phys(n_x, beta * x[0])
    hy =  hermite_phys(n_y, beta * x[1])
    psi = beta * hx * hy * jnp.exp(-0.5 *  beta**2 * (x[0]**2 + x[1]**2))/jnp.sqrt(jnp.pi*2**(n_x+n_y)*factorial(n_x)*factorial(n_y))
    return psi

def phi_2d_mod(x, n_x, n_y, L, beta=1.0):
    """
    Evaluates the 2D QHO eigenstate "without" the Gaussian exponential tail.
    
    Returns:
      psi: The polynomial part of the wavefunction * normalization.
    """
    x = x - jnp.array([L/2, L/2])
    hx =  hermite_phys(n_x, beta * x[0])
    hy =  hermite_phys(n_y, beta * x[1])
    psi = beta * hx * hy /jnp.sqrt(jnp.pi*2**(n_x+n_y)*factorial(n_x)*factorial(n_y))
    return psi

def sample_rho_2d(key, L, beta=1.0):
    """
    Samples a 2D coordinate from a Gaussian distribution centered in the box.
    
    Used to initialize walker positions close to the high-probability region
    of the Harmonic Oscillator.

    Parameters:
      key: JAX random key.
      L: Box size (mean = L/2).
      beta: Determines standard deviation (sigma = 1/beta).
    """
    std = 1.0 / beta
    mean = jnp.array([L / 2, L / 2])
    sample = jax.random.normal(key, shape=(2,)) * std + mean
    return sample    

def single_orbital_obdm(logpsi, params, L, orbitals):
    """
    Creates a function to compute the local estimator for a specific OBDM matrix element <i|rho|j>.
    
    This calculates the overlap between the many-body wavefunction and the single-particle
    orbitals i and j.

    Parameters:
      logpsi: The wavefunction ansatz.
      params: Variational parameters.
      L: System size.
      orbitals: Array of quantum numbers (n_x, n_y) for the basis set.

    Returns:
      single_orbital_obdm_fn: A function that returns the local observable value for pair (i, j).
    """
    logpsi = canonicalize_ansatz(logpsi)

    def single_orbital_obdm_fn(x, mask_valid, key, pair):

        i, j = pair[0], pair[1]
        n_x_i, n_y_i = orbitals[i][0], orbitals[i][1]
        n_x_j, n_y_j = orbitals[j][0], orbitals[j][1]

        x_1 = sample_rho_2d(key, L)
        x_new = x.at[0].set(x_1)

        obdm_local = jnp.exp(logpsi(params, x_new, mask_valid) - logpsi(params, x, mask_valid))
        obdm_local *= 2 * jnp.pi * phi_2d(x[0], n_x_i, n_y_i, L) * phi_2d_mod(x_1, n_x_j, n_y_j, L)
        
        return obdm_local
    return single_orbital_obdm_fn


def orbital_obdm(logpsi: Callable, params: Dict, L: float, n_samples: int, n_chains: int, warmup: int, 
                 sweep_size: int, pm: float, n_max: int, phys_dim: int, w: float, orbitals: jnp.ndarray, 
                 chunk_size: int) -> Callable:
    """
    Creates a function to estimate the expectation value of a specific OBDM element via MCMC.

    Parameters:
      logpsi: The wavefunction ansatz.
      params: Variational parameters.
      L: System length.
      n_samples: Number of MC samples per chain.
      n_chains: Number of parallel MC chains.
      warmup: Number of warmup (burn-in) steps for the sampler.
      sweep_size: Number of steps between samples to ensure decorrelation.
      pm: Probability of proposing a change in particle number during MCMC steps.
      n_max: Maximum particle number.
      phys_dim: Physical dimensions.
      w: Width parameter for the proposal distribution in the MCMC sampling scheme.
      orbitals: Array of orbital quantum numbers.
      chunk_size: Batch size for vmapping over samples.

    Returns:
      orbital_obdm_fn: A function f(pair, n_0, rng) -> (orbital_obdms_mean, orbital_obdms_std).
    """
    logpsi = canonicalize_ansatz(logpsi)
    single_orbital_obdm_fn = single_orbital_obdm(logpsi, params, L, orbitals)
    sampler = jax.jit(MetropolisHastings_sampler(logpsi, MetropolisHastings_proposal, n_samples,
                                                    n_chains, warmup , sweep_size, L, n_max, phys_dim, w, pm))
    
    def orbital_obdm_fn(pair, n_0, rng):
        rng, rand1 = jax.random.split(rng)
        x = sampler(params, n_0, rand1)[0]
        x = jnp.concat(x)
        mask_valid = ~jnp.isnan(x)
        mask_valid = jnp.any(mask_valid, axis=-1)
        x = jnp.nan_to_num(x)
        rng_orbitals = jax.random.split(rng, n_samples*n_chains)

        orbital_obdms = vmap(single_orbital_obdm_fn, in_axes=(0, 0, 0, None), chunk_size=chunk_size)(x, mask_valid, rng_orbitals, pair)

        orbital_obdms_mean = jnp.mean(orbital_obdms)
        orbital_obdms_std = jnp.std(orbital_obdms)

        return orbital_obdms_mean, orbital_obdms_std
    
    return orbital_obdm_fn

    
def obdm_orbitals(logpsi: Callable, params: Dict, L: float, n_0: jnp.ndarray, n_samples: int, n_chains: int, 
                  warmup: int, sweep_size: int, pm: float, n_max: int, phys_dim: int, w: float, 
                  rng: jnp.ndarray, chunk_size: int, chunk_plt_size: int, n_orbitals: int):
    """
    Computes the full OBDM projected onto the first `n_orbitals` basis functions.

    Parameters:
      n_orbitals: The number of single-particle basis states to project onto (defining the matrix size).
      chunk_plt_size: Batch size for vectorizing over the matrix elements (orbital pairs).

    Returns:
      orbital_obdms_mean: The values of the OBDM matrix elements.
      orbital_obdms_std: The standard deviations (errors) for these elements.
      matrix_pairs: The indices (i, j) corresponding to the computed elements.
    """
    n_x = jnp.arange(n_orbitals)
    orbitals = jnp.array([(x, y) for x in n_x for y in n_x])
    matrix_size = len(orbitals)
    matrix_pairs = get_orbitals_list(matrix_size)
    rng = jax.random.split(rng, len(matrix_pairs))

    orbital_obdm_fn =  jax.jit(orbital_obdm(logpsi, params, L, n_samples, n_chains, warmup, sweep_size, pm, n_max, phys_dim, w, orbitals, chunk_size))

    orbital_obdms_mean, orbital_obdms_std = vmap(orbital_obdm_fn, in_axes=(0, None, 0), chunk_size=chunk_plt_size)(matrix_pairs, n_0 ,rng)

    return orbital_obdms_mean, orbital_obdms_std, matrix_pairs

