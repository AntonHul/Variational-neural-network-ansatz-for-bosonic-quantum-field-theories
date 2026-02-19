# Neural network quantum states in the grand canonical ensemble

This repository contains JAX-based implementations for optimizing Neural Quantum States (NQS) to study bosonic Hamiltonians in one and two dimensions within the Grand Canonical Ensemble (GCE). Designed for performance and scalability, the implemented NQS ansatzes include both the **Deep Sets (DS)** architecture introduced by Martyn et al. (Ref. [2]) and the proposed **Transformer (TF)** based architecture detailed in Ref. [1].

---

## Table of Contents
- [About](#about)
- [Project Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
---

## About

Consider a general number-conserving Hamiltonian in the GCE, $H = H_1 + H_2$ with:

$$
\begin{aligned}
    H_1 &= \int _{\mathbb{R}^d} \mathrm{d}\mathbf{r} \hat \psi^\dagger(\mathbf{r}) \left( -\frac{\nabla^2}{2m} + V(\mathbf{r})-\mu\right) \hat \psi(\mathbf{r}) \\
    H_2 &= \frac{1}{2} \int _{\mathbb{R}^d} \mathrm{d}\mathbf{r} \mathrm{d}\mathbf{r}' W(\mathbf{r}, \mathbf{r}') \hat \psi^\dagger(\mathbf{r})\hat \psi^\dagger(\mathbf{r}')\psi(\mathbf{r}')\psi(\mathbf{r}),
\end{aligned}
$$

formulated as a $d$-dimensional spatial integral over one- and two-body terms, respectively. The Hamiltonian is expressed within the framework of second quantization, where the fields are represented by the creation and annihilation operators $\hat\psi^\dagger(\mathbf{r})$ and $\hat\psi(\mathbf{r})$. It comprises the kinetic energy term, an external potential $V(\mathbf{r})$, and a two-body interaction potential $W(\mathbf{r}, \mathbf{r}')$. The particle number is not fixed but is instead determined by the chemical potential $\mu$.

In this framework, the quantum field states live in Fock space and can be expressed as a superposition of $n$-particle wave functions: 

$$
|\Psi\rangle = \bigoplus_{n=0}^{\infty} \int_{\mathbb{R}^{3n}} \mathrm{d}\mathbf{R}_n \phi_n(\mathbf{R}_n)|\mathbf{R}_n\rangle.
$$

As required in bosonic quantum field theories, the functions are permutation-invariant, unnormalized, and can be obtained by projecting the field state onto the corresponding basis state, $\phi_n(\mathbf{R}_n) = \langle \mathbf{R}_n|\Psi\rangle$.

This repository presents a variational ansatz based on the Transformer (TF) neural network architecture, which is used to parametrize each $n$-particle wave function $\phi_n(\mathbf{R}_n)$. The ansatz is tested against an array of bosonic Hamiltonians in one and two dimensions. The Lieb-Liniger and Calogero-Sutherland models are used as accuracy benchmarks, where we show precise agreement with exact results. We also treat harmonically trapped bosons with short-range interactions, extracting accurate variational wave functions and demonstrating the practical applicability of modern Variational Monte Carlo (VMC) to real-world systems.

---

## Project Structure

The source code is organized into the `src` directory, while usage scripts are located in `examples`.

```
├── src/
│   ├── optimization.py         # Energy minimization for the variational ansatz using an iterative VMC optimization loop
│   ├── optimizers.py           # Adam, SR, and MinSR update rules
│   ├── energy_estimation.py    # Total local energy for a given particle configuration
│   ├── metropolis_sampling.py  # Markov Chain Monte Carlo (MCMC) sampling method
│   ├── transformer_nqfs.py     # NQFS Transformer (TF) ansatz definition
│   ├── embeddings_tf.py        # Periodic and Gaussian TF ansatz embedding functions
│   ├── deep_sets_nqfs.py       # NQFS Deep Sets (DS) ansatz definition
│   ├── embeddings_ds.py        # Periodic and closed-boundary DS ansatz embedding functions 
│   ├── deep_sets.py            # Deep Sets (DS) architecture implementation
│   ├── vmap_chunked.py         # Utility for memory-efficient vectorization
│   └── jastrow_factors.py      # Jastrow factors ensuring correct short-range behavior of the wave function
├── estimators/
│   ├── n_dens_estim.py         # Estimator for the particle number density
│   ├── dens_estim.py           # Estimator for the probability density
│   ├── ke_dens_estim.py        # Estimator for the kinetic energy (KE) density
│   ├── obdm_estim.py           # Estimator for one-body density matrix (OBDM)
│   ├── obdm_orbitals_estim.py  # Estimator for one-body density matrix (OBDM) using orbitals approach
├── utils/
│   ├── cs_exact.py         # Exact solution to Calogero–Sutherland (CS) 1D and 2D models
│   ├── ll_exact.py         # Exact solution to the Lieb–Liniger (LL) 1D model
│   ├── plot_utils.py       # Plotting utilities
├── examples/
│   ├── LL_1D.ipynb    # Example optimization of the Lieb–Liniger (LL) 1D model with TF ansatz
│   ├── CS_1D.ipynb    # Example optimization of the Calogero–Sutherland (CS) 1D model with TF ansatz
│   ├── CS_2D.ipynb    # Example optimization of the Calogero–Sutherland (CS) 2D model with TF ansatz
│   └── harm_2D.ipynb  # Example optimization of the harmonically confined 2D system with TF ansatz
├── README.md
└── requirements.txt
```

---

## Installation

We recommend using a fresh Python environment (Python ≥ 3.10).

### CPU-only installation

```bash
pip install -r requirements.txt
```

### GPU installation (CUDA 12)

```
pip install -r requirements.txt
pip install "jax[cuda12]==0.4.35"
```

---

## References

[1] Hul, A., Medvidović, M., & Carrasquilla, J. (2026). Neural network quantum states in the grand canonical ensemble (Unpublished manuscript).

[2] Martyn, John M., Najafi, Khadijeh, and Luo, Di. "Variational Neural-Network Ansatz for Continuum Quantum Field Theory." Phys. Rev. Lett. 131, no. 8 (2023): 081601. https://doi.org/10.1103/PhysRevLett.131.081601