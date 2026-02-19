import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
import jax
from functools import partial

class MCState(NamedTuple):
    """
    Container for the state of the Markov Chain.

    Attributes:
      x: Particle positions array. NaNs indicate absent particles (variable N).
      accepted: Boolean indicating if the last proposal was accepted.
      key: JAX PRNGKey for the next stochastic step.
    """
    x: jax.Array      
    accepted: bool     
    key: jax.Array     

def get_init_state(L: float, n_0: int, n_max: int, phys_dim: int, key):
    """
    Initializes the Markov Chain state.

    Places `n_0` particles uniformly at random in the box [0, L] and pads the 
    rest of the array up to `n_max` with NaNs to represent empty slots.

    Parameters:
      L: System length.
      n_0: Initial number of active particles.
      n_max: Maximum number of the particles.
      phys_dim: Physical dimension.
      key: RNG key.

    Returns:
      Initialized MCState.
    """
    key1, key2  = jax.random.split(key, 2)
    random_positions = L * jax.random.uniform(key1, (n_max, phys_dim))
    mask_array = jnp.full((n_max, phys_dim), jnp.nan)
    condition = jnp.arange(n_max) < n_0
    condition_broadcasted = jnp.broadcast_to(condition[:, None], (n_max, phys_dim))
    x_0 = jax.lax.select(condition_broadcasted, random_positions, mask_array)
    return MCState(x_0, True, key2)

def mh_accept(log_prob, x, x_, key, L_factor):
    """
    Performs the Metropolis-Hastings acceptance test in log-space.

    Parameters:
      log_prob: Function returning the log-amplitude of the wavefunction.
      x: Current configuration.
      x_: Proposed configuration.
      key: RNG key.
      L_factor: The proposal ratio factor q(x|x')/q(x'|x) (e.g., volume corrections).

    Returns:
      Boolean indicating acceptance.
    """
    r = jnp.log(jax.random.uniform(key) / L_factor)
    return jnp.minimum(1, 2 * (log_prob(x_) - log_prob(x))) > r

def step(proposal_fn, log_prob, state, L, w, phys_dim, pm):
    """
    Performs a single Metropolis-Hastings step.
    
    1. Generate a proposal configuration x'.
    2. Calculate acceptance probability.
    3. Accept or reject the move.

    Parameters:
      proposal_fn: Function to generate x' from x.
      log_prob: Function to evaluate the target log-probability.
      state: Current MCState.
      L, w, phys_dim, pm: Simulation parameters passed to proposal_fn.

    Returns:
      Updated MCState.
    """
    key1, key2, key3 = jax.random.split(state.key, 3)
    x_, L_factor = proposal_fn(state.x, key1, L, w, phys_dim, pm)
    accept = mh_accept(log_prob, state.x, x_, key2, L_factor)
    x_new = jnp.where(accept, x_, state.x)
    return MCState(x_new, accept, key3)

def sweep(proposal_fn, log_prob, state, n_steps, L, w, phys_dim, pm):
    """Executes a sweep of multiple MH steps (inner loop)."""
    body_fun = lambda _, state: step(proposal_fn, log_prob, state, L, w, phys_dim, pm)
    return jax.lax.fori_loop(0, n_steps, body_fun, state)

def sample_chain(proposal_fn, log_prob, state, n_samples, sweep_size, L, w, phys_dim, pm):
    """
    Generates a full chain of samples using.

    Parameters:
      n_samples: Number of samples to collect.
      sweep_size: Number of thinning steps between samples (decorrelation).
    
    Returns:
      final_state: State after the last sample.
      samples: Stacked array of collected samples.
    """
    def body_fn(state, _):
        state = sweep(proposal_fn, log_prob, state, sweep_size, L, w, phys_dim, pm)
        return state, state
    return jax.lax.scan(body_fn, init=state, xs=None, length=n_samples)

def MetropolisHastings_sampler(logpsi, proposal_fn, n_samples: int, n_chains: int, warmup: int, sweep_size: int, L: float , n_max: int, phys_dim: int, w: float, pm: float = 0.25):
    """
    Creates a vectorized Metropolis-Hastings sampler for variable particle number systems.

    Parameters:
      logpsi: Wavefunction ansatz.
      proposal_fn: MCMC proposal mechanism (e.g., add/remove/perturb).
      n_samples: Number of samples per chain.
      n_chains: Number of parallel chains.
      warmup: Number of initial steps to discard.
      sweep_size: Steps between saved samples.
      L: System length.
      n_max: Maximum number of the particles.
      phys_dim: Physical dimensions.
      w: Width of perturbation.
      pm: Probability of proposing an add/remove move.

    Returns:
      sample_fn: A function that takes (params, n_0, key) and returns a batch of samples.
    """
    @partial(jax.vmap, in_axes=(None, None, 0))
    def _sample(params, n_0, key):
        log_prob = lambda x: logpsi(params, jnp.nan_to_num(x), jnp.any(~jnp.isnan(x), axis=1))
        init_state = get_init_state(L, n_0, n_max, phys_dim, key)
        if warmup > 0:
            state = sweep(proposal_fn, log_prob, init_state, warmup, L, w, phys_dim, pm)
        _, samples = sample_chain(proposal_fn, log_prob, state, n_samples, sweep_size, L, w, phys_dim, pm)
        return samples

    def sample(params, n_0, key):
        keys = jax.random.split(key, n_chains)
        return _sample(params, n_0, keys)

    return sample

def add_particle(x, key, L, w):
    """
    Proposes adding a new particle at a random location.

    L_factor = L^D (Volume factor for detailed balance).
    """
    phys_dim = x.shape[1]
    nan_mask = jnp.isnan(x[:, 0]) 
    nan_index = jnp.argmax(nan_mask)
    new_position = L * jax.random.uniform(key, (1, phys_dim))
    x = jnp.where(nan_mask.any(), x.at[nan_index, :].set(new_position[0]), x)
    return x, L**phys_dim

def remove_particle(x, key, L, w):
    """
    Proposes removing a random existing particle.

    L_factor = 1/L^D (Inverse volume factor).
    """
    phys_dim = x.shape[1]
    non_nan_mask = ~jnp.isnan(x[:, 0])
    num_non_nan = non_nan_mask.sum()
    random_pos = jax.random.randint(key, (), 0, num_non_nan)
    cumulative_mask = jnp.cumsum(non_nan_mask) == (random_pos + 1)
    cumulative_mask = cumulative_mask[:, None]
    x = jnp.where(cumulative_mask, jnp.nan, x)
    return x, 1/L**phys_dim

def perturb_particle(x, key, L, w):
    """
    Proposes perturbing all active particles by a random shift.

    L_factor = 1.0 (Symmetric proposal).
    """
    non_nan_mask = ~jnp.isnan(x)
    perturbation = jax.random.uniform(key, x.shape, minval=-w / 2, maxval=w / 2)
    x = jnp.where(non_nan_mask, x + perturbation, x)
    x %= L
    return x, 1.0

def MetropolisHastings_proposal(x, key, L, w, phys_dim, pm):
    """
    Selects a proposal move (Add, Remove, or Perturb) based on probability `pm`.

    Probabilities:
      - Add: pm
      - Remove: pm
      - Perturb: 1 - 2*pm
    """
    key1, key2 = jax.random.split(key, 2)
    probs = jnp.array([pm, pm, 1-2*pm])
    actions = [add_particle, remove_particle, perturb_particle]
    action_idx = jax.random.choice(key1, jnp.arange(len(actions)), p=probs)
    return jax.lax.switch(action_idx, actions, x, key2, L, w)
