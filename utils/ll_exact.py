"""
This code is adapted from the repository:
https://github.com/jmmartyn/Neural-Network-Quantum-Field-States

Original work by J. M. Martyn et al.
"""

import jax
from jax import numpy as jnp
from scipy.optimize import root
from typing import Any, Callable, Tuple, Union, Optional, Dict

def LL_GS_Energy(m: float, g: float, L: float, n: int) -> float:
    """
    Determines the ground state energy of the Lieb-Liniger model in a box.
    
    Parameters:
      m: Particle mass.
      g: Interaction strength.
      L: Length of the box.
      n: Number of particles.

    Returns:
      E: The ground state energy.
    """

    c = 2 * m * g
    n_alpha = jnp.ones(n)  
    def funcs(k_alpha):
        k_alpha = jnp.array(k_alpha)
        difs = (k_alpha[:, None] - k_alpha[None, :])
        difs += 1e-20 * jnp.identity(k_alpha.shape[0])
        sums = (k_alpha[:, None] + k_alpha[None, :])
        sum_vals = jnp.arctan(c / difs) + jnp.arctan(c / sums)
        sum_vals = sum_vals.at[jnp.diag_indices_from(sum_vals)].set(0.)

        f = k_alpha * L - jnp.pi * n_alpha - jnp.sum(sum_vals, axis=1)
        f = list(f)
        return f

    sol = root(funcs, jnp.pi * jnp.arange(1, n + 1) / L)
    k_alpha = sol.x
    E = 1 / (2 * m) * jnp.sum(k_alpha ** 2)

    return E

def LL_ks(m: float, g: float, L: float, n: int) -> jnp.ndarray:
    """
    Solves for the exact quasimomenta (k) of the Lieb-Liniger model ground state.
    
    Parameters:
      m: Particle mass.
      g: Interaction strength.
      L: Length of the box.
      n: Number of particles.


    Returns:
      k_alpha: Array of solved quasimomenta.
    """

    c = 2 * m * g
    n_alpha = jnp.ones(n)  

    def funcs(k_alpha):
        k_alpha = jnp.array(k_alpha)
        difs = (k_alpha[:, None] - k_alpha[None, :])
        difs += 1e-20 * jnp.identity(k_alpha.shape[0])
        sums = (k_alpha[:, None] + k_alpha[None, :])
        sum_vals = jnp.arctan(c / difs) + jnp.arctan(c / sums)
        sum_vals = sum_vals.at[jnp.diag_indices_from(sum_vals)].set(0.)

        f = k_alpha * L - jnp.pi * n_alpha - jnp.sum(sum_vals, axis=1)
        f = list(f)
        return f

    sol = root(funcs, jnp.pi * jnp.arange(1, n + 1) / L)
    k_alpha = sol.x

    return k_alpha



def Exact_energy_and_n(m: float, mu: float, g: float, L: float) -> Tuple[float, int]:
    """
    Determines the exact ground state energy and particle number of the Lieb-Liniger model.
    
    Iteratively increases particle number N until the Grand Potential (E - mu*N) 
    stops decreasing.

    Parameters:
      m: Particle mass.
      mu: Chemical potential.
      g: Interaction strength.
      L: System size.

    Returns:
      E_min: The ground state energy.
      n_opt: The ground state particle number.
    """

    E = 0
    n = 0

    while 1 == 1:
        n_tmp = n + 1
        E_tmp = LL_GS_Energy(m, g, L, n_tmp) - mu * n_tmp

        if E_tmp < E:
            E = E_tmp
            n = n_tmp
        else:
            break

    return E, n



def TG_energy(L: float, m: float, mu: float) -> float:
    """
    Calculates the energy of the Tonks-Girardeau ground state (g -> infinity limit).
    
    Parameters:
      L: System size.
      m: Particle mass.
      mu: Chemical potential.

    Returns:
      E: The ground state energy.
    """
    n = TG_n(L, m, mu)
    b = jnp.pi ** 2 / (2 * m * L ** 2)
    c = jnp.pi ** 2 / (12 * m * L ** 2) - mu
    E = b * (n ** 3 / 3 + n ** 2 / 2 + n / 6) - mu * n
    return E

def TG_n(L: float, m: float, mu: float) -> int:
    """
    Calculates the particle number for the Tonks-Girardeau ground state.

    Returns:
      n: Integer particle number.
    """
    b = jnp.pi ** 2 / (2 * m * L ** 2)
    c = jnp.pi ** 2 / (12 * m * L ** 2) - mu
    n = (-b + (b ** 2 - 4 * b * c) ** 0.5) / (2 * b)
    n = int(jnp.round(n))
    return n

def TG_number_density(x_eval: float, n: int, L: float) -> float:
    """
    Calculates the number density of the Tonks-Girardeau ground state at a point.
    
    Parameters:
      x_eval: The spatial point to evaluate.
      n: Number of particles.
      L: System size.

    Returns:
      num_density_TG: The local density.
    """
    num_density_TG = jnp.sum(jnp.array([2 / L * (jnp.sin(j * jnp.pi * x_eval / L)) ** 2 \
                                        for j in range(1, n + 1)]))
    return num_density_TG

def TG_KE_density(x_eval: float, n: int, L: float, m: float) -> float:
    """
    Calculates the kinetic energy density of the Tonks-Girardeau ground state.
    
    Returns:
      KE_density_TG: The local kinetic energy density.
    """

    KE_density_TG = 1 / (2 * m) * jnp.sum(jnp.array([2 / L * (j * jnp.pi / L * jnp.cos(j * jnp.pi * x_eval / L)) ** 2 \
                                                    for j in range(1, n + 1)]))
    return KE_density_TG

def TG_density_function(N_pts: int, n: int, L: float, m: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates the number density and kinetic energy density profiles of the 
    Tonks-Girardeau ground state across the entire domain.

    Parameters:
      N_pts: Number of spatial points to evaluate.
      n: Number of particles.
      L: System size.
      m: Mass.

    Returns:
      num_densities_TG, KE_densities_TG: Arrays of density profiles.
    """

    xs = jnp.linspace(0, L, N_pts)
    num_densities_TG = jnp.array([TG_number_density(x, n, L) for x in xs])
    KE_densities_TG = jnp.array([TG_KE_density(x, n, L, m) for x in xs])
    return num_densities_TG, KE_densities_TG