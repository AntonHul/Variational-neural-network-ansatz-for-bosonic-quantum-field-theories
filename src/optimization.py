import os
from functools import partial
from typing import Any, Callable, Tuple, Union, Optional, Dict

import pandas as pd
import haiku as hk
import optax
from flax import serialization

import jax
import jax.numpy as jnp
from jax.numpy.linalg import lstsq

from src.vmap_chunked import vmap
from src.energy_estimation import Local_Energy
from src.metropolis_sampling import MetropolisHastings_sampler, MetropolisHastings_proposal
from src.optimizers import adam_update, sr_update, min_sr_update

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
    
    

def params_update_step(logpsi: Callable, w: float, L: float, n_max: int, phys_dim: int, n_samples: int, n_chains: int, 
                       warmup: int, sweep_size: int, pm,  m: float, mu: float, V: Callable, W: Callable, 
                       optimizer: Callable, optimizer_type, chunk_size: int, chunk_grads_size, optimization, solver) -> Callable:
    """
    Creates a training step function to update the parameters of the variational ansatz.

    Parameters:
      logpsi: The wavefunction ansatz (function or Flax module).
      w: Width parameter for the proposal distribution in the MCMC sampling scheme.
      L: System length.
      n_max: Maximum number of particles allowed.
      phys_dim: Physical dimension of the local Hilbert space.
      n_samples: Number of samples per MCMC chain.
      n_chains: Number of parallel MCMC chains.
      warmup: Number of warmup (burn-in) steps for the sampler.
      sweep_size: Number of steps between samples to ensure decorrelation.
      pm: Probability of proposing a change in particle number during MCMC steps.
      m: Particle mass.
      mu: Chemical potential.
      V: External potential function.
      W: Interaction potential function.
      optimizer: The Optax optimizer instance.
      optimizer_type: String identifier (e.g., 'adam') to handle specific update signatures.
      chunk_size: Batch size for energy evaluation to manage memory usage.
      chunk_grads_size: Batch size for gradient computation (vmap).
      optimization: Optimization strategy ('adam', 'sr', or 'min_sr').
      solver: Linear solver for Stochastic Reconfiguration (SR) methods (e.g., jax.scipy.linalg.solve).

    Returns:
      train_step_fn: function that performs one optimization step.
    """

    Ns = n_chains*n_samples
    
    logpsi = canonicalize_ansatz(logpsi)

    lcl_energy_fn = jax.jit(Local_Energy(logpsi, m, mu, V, W))

    sampler = jax.jit(MetropolisHastings_sampler(logpsi, MetropolisHastings_proposal, n_samples,
                                                    n_chains, warmup , sweep_size, L, n_max, phys_dim, w, pm))
    
    if optimization == "adam":
        get_grads = adam_update(logpsi, Ns, chunk_grads_size, solver)
    elif optimization == "sr":
        get_grads = sr_update(logpsi, Ns, chunk_grads_size, solver)
    elif optimization == "min_sr":
        get_grads = min_sr_update(logpsi, Ns, chunk_grads_size, solver)
    
    def train_step_fn(params, opt_state, n_0, rng, diag_shift: float = 1e-4):
        """
        Performs a single optimization step, including sampling, energy evaluation, and parameter update.

        Parameters:
          params: The current variational ansatz parameters.
          opt_state: The current state of the optimizer.
          n_0: The initial number of particles for the MCMC chains.
          rng: A JAX PRNGKey for stochastic sampling.
          diag_shift: (Optional) Regularization parameter for the diagonal shift in SR/MinSR methods. 
                      Defaults to 1e-4.

        Returns:
          E_mean: The mean total energy estimated from the batch.
          E_std: The standard deviation of the mean energy. 
          n_mean: The average number of particles in the sampled configurations.
          n_std: The standard deviation of the particle number distribution.
          params: The updated variational parameters.
          opt_state: The updated optimizer state.
          KE_mean: The mean kinetic energy component.
          VE_mean: The mean external potential energy component.
          WE_mean: The mean interaction potential energy component.
        """
        rng, rand1 = jax.random.split(rng)
        x = sampler(params, n_0, rand1)[0]
        x = jnp.concat(x)
        mask_valid = ~jnp.isnan(x)
        mask_valid = jnp.any(mask_valid, axis=-1)
        x = jnp.nan_to_num(x)

        E_locals, KE, VE, WE = vmap(lcl_energy_fn, in_axes=(None, 0, 0), chunk_size=chunk_size)(params, x, mask_valid)
        E_mean = jnp.mean(E_locals)
        KE_mean = jnp.mean(KE)
        VE_mean = jnp.mean(VE)
        WE_mean = jnp.mean(WE)

        ns = jnp.sum(mask_valid, axis=1)
        n_mean = jnp.mean(ns)
        n_std = jnp.std(ns)

        grads = get_grads(params, x, mask_valid, E_locals - E_mean, diag_shift)
        if optimizer_type == "adamw":
            updates, opt_state = optimizer.update(grads, opt_state, params)
        else:
            updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Quantum variance
        var_E = jnp.mean(jnp.abs(E_locals - E_mean)**2)
        E_locals = E_locals.reshape(n_chains, -1)
        E_mean_chains = jnp.mean(E_locals, axis=1)        
        # MCMC std
        E_std = jnp.std(jnp.array(E_mean_chains))/ (n_chains ** 0.5)


        return E_mean, E_std, var_E, n_mean, n_std, params, opt_state, KE_mean, VE_mean, WE_mean
    
    return train_step_fn

def setup_optimizer(params: Dict, lr_q: float, lr: float, optimizer_type: str) -> optax.GradientTransformation:
    """
    Configures an optimizer with different learning rates for specific parameter groups.

    This function defines two sets of parameters:
    1. 'q_n' parameters (e.g., 'q_n_mean', 'sigma', 'lam') which use `lr_q`.
    2. Default parameters (everything else) which use `lr`.

    Parameters:
      params: The current variational ansatz parameters.
      lr_q: Learning rate for the specialized 'q_n_*' parameters.
      lr: Learning rate for the default parameters.
      optimizer_type: String identifier for the optimizer ('adam' or 'sgd').

    Returns:
      optimizer: An `optax.GradientTransformation` combining the masked updates.
    """
    qn_mask = hk.data_structures.map(lambda mname, name, val: name in ['q_n_mean', 'q_n_inv_softplus_width', 'q_n_inv_softplus_slope', 'sigma', 'lam'], params)
    default_mask = jax.tree_util.tree_map(lambda x: not x, qn_mask)

    if optimizer_type == "adam":
        chosen_optimizer = optax.adam
    elif optimizer_type == "sgd":
        chosen_optimizer = optax.sgd
    else:
        raise NotImplementedError("Only 'adam' and 'sgd' are supported.")

    optimizer = optax.chain(
        optax.masked(chosen_optimizer(learning_rate=lr_q), qn_mask),
        optax.masked(chosen_optimizer(learning_rate=lr), default_mask)
    )
    return optimizer



def minimize_energy(logpsi: Callable, w: jnp.ndarray, L: float, n_max: int, n_0: int, phys_dim: int, seed: int, 
                    params: Dict, n_samples: int, n_chains: int, warmup: int, 
                    sweep_size: int, pm, m: float, mu: float, V: Callable, W: Callable, 
                    lr: float, lr_q: float, n_iters: int, chunk_size: int, 
                    chunk_grads_size, optimization, optimizer_type, diag_shift=1e-3, diag_shift_step=None, diag_shift_red=5, solver=lstsq) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Minimizes the energy of the variational ansatz using an iterative VMC optimization loop.

    This function sets up the optimizer, initializes the history arrays, and performs
    the main training loop (sampling -> energy evaluation -> parameter update).

    Parameters:
      logpsi: The wavefunction ansatz (function or Flax module).
      w: Width parameter for the proposal distribution in the MCMC sampling scheme.
      L: System length.
      n_max: Maximum number of particles allowed.
      n_0: The initial number of particles for the MCMC chains.
      phys_dim: Physical dimension of the local Hilbert space.
      seed: Random seed.
      params: Initial variational parameters.
      n_samples: Number of samples per MCMC chain.
      n_chains: Number of parallel MCMC chains.
      warmup: Number of warmup (burn-in) steps for the sampler.
      sweep_size: Number of steps between samples to ensure decorrelation.
      pm: Probability of proposing a change in particle number during MCMC steps.
      m: Particle mass.
      mu: Chemical potential.
      V: External potential function.
      W: Interaction potential function.
      lr: Learning rate for default parameters.
      lr_q: Learning rate for specialized 'q_n' parameters.
      n_iters: Total number of optimization iterations.
      chunk_size: Batch size for energy evaluation.
      chunk_grads_size: Batch size for gradient computation.
      optimization: Optimization strategy ('adam', 'sr', or 'min_sr').
      optimizer_type: specific optimizer backend identifier (e.g., 'adam', 'sgd').
      diag_shift: Initial regularization diagonal shift for SR methods. Defaults to 1e-3.
      diag_shift_step: Step interval to reduce the diagonal shift. If None, shift is constant.
      diag_shift_red: Factor by which to divide the diagonal shift at each step.
      solver: Linear solver for Stochastic Reconfiguration (SR) methods (e.g., jax.scipy.linalg.solve).

    Returns:
      Es: Array of mean energies for each iteration.
      E_stds: Array of energy standard errors for each iteration.
      n_means: Array of mean particle numbers for each iteration.
      n_stds: Array of particle number standard deviations for each iteration.
      params: The final optimized variational parameters.
      KEs: Array of mean kinetic energies.
      VEs: Array of mean external potential energies.
      WEs: Array of mean interaction energies.
    """

    logpsi = canonicalize_ansatz(logpsi)

    Es = jnp.zeros(n_iters)
    KEs = jnp.zeros(n_iters)
    VEs = jnp.zeros(n_iters)
    WEs = jnp.zeros(n_iters)
    E_stds = jnp.zeros(n_iters)
    var_Es = jnp.zeros(n_iters)
    n_means = jnp.zeros(n_iters)
    n_stds = jnp.zeros(n_iters)
    n_means = n_means.at[-1].set(n_0)
    rng = jax.random.PRNGKey(seed)

    optimizer = setup_optimizer(params, lr_q, lr, optimizer_type)
    opt_state = optimizer.init(params)
    train_step_fn = jax.jit(params_update_step(logpsi, w, L, n_max, phys_dim, n_samples, n_chains, warmup,
                            sweep_size, pm, m, mu, V, W, optimizer, optimizer_type, chunk_size, chunk_grads_size, optimization, solver))
    for t in range(1, n_iters + 1):
                    
        rng, rand1 = jax.random.split(rng)

        n_0 = int(n_means[t - 2].item())
        if diag_shift_step is not None and t % int(diag_shift_step) == 0:
            diag_shift /= diag_shift_red
        E_mean, E_std, var_E, n_mean, n_std, params, opt_state, KE_mean, VE_mean, WE_mean =  train_step_fn(params, opt_state, n_0, rand1, diag_shift)

        Es = Es.at[t - 1].set(E_mean)
        KEs = KEs.at[t - 1].set(KE_mean)
        VEs = VEs.at[t - 1].set(VE_mean)
        WEs = WEs.at[t - 1].set(WE_mean)
        E_stds = E_stds.at[t - 1].set(E_std)
        var_Es = var_Es.at[t - 1].set(var_E)
        n_means = n_means.at[t - 1].set(n_mean)
        n_stds = n_stds.at[t - 1].set(n_std)

        print("Iteration: " + str(t) + "/" + str(n_iters))
        print("Energy: " + str(round(E_mean.item(), 3)) + " +- " \
                + str(round(E_std.item(), 3)))
        print("VarE: " + str(round(var_E.item(), 3)))
        print("Number of particles: " + str(round(n_mean, 3)) + \
                " +- " + str(round(n_std, 3)))
        print("KE: " + str(KE_mean))
        print("VE: " + str(VE_mean))
        print("WE: " + str(WE_mean))
        if diag_shift_step is not None:
            print("Diag shift: " + str(diag_shift))
        print("\n")

    return Es, E_stds, var_Es, n_means, n_stds, params, KEs, VEs, WEs

def save_Energy_number(Es, E_stds, n_means, n_stds, name):
    """
    Saves energy and particle number statistics to a CSV file.

    Parameters:
      Es: Array of mean energies.
      E_stds: Array of energy standard deviations/errors.
      n_means: Array of mean particle numbers.
      n_stds: Array of particle number standard deviations.
      name: Base name for the output file (without extension).
    """
    data = {
        "Es": Es,
        "E_stds": E_stds,
        "n_means": n_means,
        "n_stds": n_stds
    }
    df = pd.DataFrame(data)
    output_file = name + ".csv"
    df.to_csv(output_file, index=True)
    print("Eenrgy and particle number saved successfully!")

def save_optimized_params(params: Dict, path: str) -> None:
    """
    Saves variational ansatz parameters to a single binary file using Flax serialization.
    
    Parameters:
      params: The current variational ansatz parameters.
      path: The file path where the parameters should be stored.
    """
    path = os.path.abspath(path)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Parameters saved successfully to {path}!")


def get_optimized_params(path: str, template: Dict) -> Dict:
    """
    Loads variational ansatz parameters from a binary file using a template for structure.
    
    Parameters:
      path: The file path to load from.
      template: A  variational ansatz parameters structure (e.g., initialized params) matching the saved data.
      
    Returns:
      params: The loaded parameter dictionary.
    """
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        byte_data = f.read()
    params = serialization.from_bytes(template, byte_data)
    print(f"Parameters loaded successfully from {path}!")
    return params