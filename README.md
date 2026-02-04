# Variational Neural Network Ansatz for Bosonic Quantum Field Theories

This repository contains JAX-based implementations for optimizing Neural Quantum States (NQS) in the context of Bosonic Quantum Field Theories (QFT).
The implemneted NQS ansatz includes both the Deep Sets (DS) architecture introduced by Martyn et. al. in Ref. [2] as well as proposed Transfoermer (TF) based
architecture discussed in details in Ref. [1].

---

## Table of Contents
- [About](#about)
- [Project Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
---

## About
**Variational Monte Carlo (VMC) for Quantum Field Theories**

This project provides a flexible framework for simulating Boexasonic Quantum Field Theories using Neural Network Quantum States. It leverages JAX for high-performance automatic differentiation and hardware acceleration (GPU/TPU).

Key features:
- **Transformer Ansatz:** `transformer_nqfs.py` implements a specialized Transformer (TF) architecture based QNFS ansatz.
- **Deep Sets Ansatz:** `deep_sets.py` implements a specialized Transformer architecture for quantum states.
- **Advanced Optimization:** `optimization.py` includes state-of-the-art VMC optimization strategies including Adam as well Stochastic Reconfiguration (SR) and Min-SR for handling large parameter spaces. 
- **Available Optimization:** `optimizers.py` includes state-of-the-art VMC optimization strategies including Adam as well Stochastic Reconfiguration (SR) and Min-SR for handling large parameter spaces. 
- **Useful observables:** 
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

[1] Hul, A., Medvidović, M., & Carrasquilla, J. (2026). Variational neural network ansatz for bosons in the grand canonical ensemble (Unpublished manuscript).

[2] Martyn, John M., Najafi, Khadijeh, and Luo, Di. "Variational Neural-Network Ansatz for Continuum Quantum Field Theory." Phys. Rev. Lett. 131, no. 8 (2023): 081601. https://doi.org/10.1103/PhysRevLett.131.081601