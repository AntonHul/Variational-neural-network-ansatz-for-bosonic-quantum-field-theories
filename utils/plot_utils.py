import jax.numpy as jnp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any, Callable, Tuple, Union, Optional, Dict, List    
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator

def plot_energy(Es: Union[List, np.ndarray], E_stds: Union[List, np.ndarray], E_exact: Optional[float] = None, output_file: Optional[str] = None):
    """
    Plots the variational energy convergence over iterations, including error bars (standard deviation)
    and an optional exact benchmark line.

    Parameters:
      Es: Array of estimated energy values per iteration.
      E_stds: Array of standard deviations associated with the energy estimates.
      E_exact: The exact ground state energy value (if known) for comparison.
      output_file: File path to save the generated plot (e.g., 'energy.png'). If None, the plot is displayed interactively.
    """
    
    # Define the range of iterations to plot
    start = 0
    end = len(Es)
    iterations = np.arange(start, end)

    # Initialize the figure with high DPI for clarity
    fig, ax = plt.subplots(figsize=(5, 3), dpi=200)

    # Plot the mean energy curve
    ax.plot(iterations, Es[start:end], linewidth=0.8, color='blue', label='VMC Energy')
    
    # Add shaded region representing the standard deviation (uncertainty)
    ax.fill_between(
        iterations,
        (Es - E_stds)[start:end],
        (Es + E_stds)[start:end],
        color='blue',
        alpha=0.2,
        label=r'Energy $\sigma$'
    )
    
    # If provided, plot the exact energy reference line
    if E_exact is not None:
        ax.axhline(E_exact, color='red', linestyle='--', linewidth=1, label='Exact Energy')

    # Apply formatting and labels
    ax.set_title('Energy Optimization History', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=8)
    ax.set_ylabel('Energy', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    
    # Ensure x-axis ticks are integers (iterations are discrete)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', labelsize=7)

    # Output handling: Save to file or show window
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Energy convergence plot saved to {output_file}")
    else:
        plt.show()

def plot_energy_discrepancies(Es: Union[List, np.ndarray], E_stds: Union[List, np.ndarray], E_exact: float, output_file: Optional[str] = None):
    """
    Plots the convergence metrics for energy: the relative error from the exact value 
    and the relative standard deviation (coefficient of variation) over iterations.

    Parameters:
      Es: Array of estimated energy values per iteration.
      E_stds: Array of standard deviations associated with the energy estimates.
      E_exact: The exact ground state energy value (required for relative error).
      output_file: File path to save the generated plot. If None, displays interactively.
    """  
    
    # Define range for plotting
    start = 0
    end = len(Es)
    iterations = np.arange(start, end)

    # Compute convergence metrics
    # Relative Error: |E_est - E_exact| / E_exact
    energy_errors = jnp.abs((Es - E_exact) / E_exact)[start:end]
    # Relative Standard Deviation (Coefficient of Variation): sigma / E_est
    rel_std_dev = jnp.abs(E_stds / Es)[start:end]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(5, 3), dpi=200)

    # Plot metrics on a log scale to visualize convergence rates
    ax.plot(iterations, energy_errors, linewidth=0.8, color='green', label='Relative Error')
    ax.plot(iterations, rel_std_dev, linewidth=0.8, color='orange', label='Relative St. Dev.')
    
    # Use logarithmic scale for y-axis to handle orders of magnitude differences
    ax.set_yscale('log')

    # Formatting
    ax.set_title('Energy Discrepancies vs. Iteration', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=8)
    ax.set_ylabel('Log-Scaled Value', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    
    # Force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', labelsize=7)

    # Save or display
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Discrepancy plot saved to {output_file}")
    else:
        plt.show()


def plot_particle_number(n_means: Union[List, np.ndarray], n_stds: Union[List, np.ndarray], n_exact: Optional[float] = None, output_file: Optional[str] = None):
    """
    Plots the expected particle number <N> vs. iteration with error bands.
    Useful for checking if the Grand Canonical simulation has converged to the correct N sector.

    Parameters:
      n_means: Array of estimated mean particle numbers.
      n_stds: Array of standard deviations for particle number.
      n_exact: The target/exact particle number.
      output_file: File path to save the generated plot.
    """
    
    start = 0
    end = len(n_means)
    iterations = np.arange(start, end)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(5, 3), dpi=200)

    # Plot mean particle number
    ax.plot(iterations, n_means[start:end], linewidth=1, color='blue', label='Particle Number')
    
    # Add shaded error bands for standard deviation
    ax.fill_between(
        iterations,
        (n_means - n_stds)[start:end],
        (n_means + n_stds)[start:end],
        color='blue',
        alpha=0.25,
        label='Standard Deviation'
    )
    
    # Plot exact reference line if provided
    if n_exact is not None:
        ax.axhline(n_exact, color='red', linestyle='--', linewidth=1, label='Exact Particle Number')

    # Formatting
    ax.set_title('Particle Number vs. Iteration', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=8)
    ax.set_ylabel('Particle Number', fontsize=8)
    ax.legend(fontsize=7, loc='lower right')
    ax.tick_params(axis='both', labelsize=7)
    
    # Save or display
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Particle number plot saved to {output_file}")
    else:
        plt.show()


def plot_particle_discrepancies(n_means: Union[List, np.ndarray], n_stds: Union[List, np.ndarray], n_exact: float, output_file: Optional[str] = None):
    """
    Plots particle number discrepancies (relative error and relative standard deviation) vs. iteration.

    Parameters:
      n_means: Array of estimated mean particle numbers.
      n_stds: Array of standard deviations for particle number.
      n_exact: The exact particle number (required for relative error).
      output_file: File path to save the generated plot.
    """
    
    start = 0
    end = len(n_means)
    iterations = np.arange(start, end)

    # Compute convergence metrics
    # Relative Error: |<N> - N_exact| / N_exact
    n_errors = jnp.abs((n_means - n_exact) / n_exact)[start:end]
    # Relative Standard Deviation: sigma_N / <N>
    rel_std_dev = jnp.abs(n_stds / n_means)[start:end]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(5, 3), dpi=200)

    # Plot metrics on log scale
    ax.plot(iterations, n_errors, linewidth=0.8, color='green', label='Relative Error')
    ax.plot(iterations, rel_std_dev, linewidth=0.8, color='orange', label='Relative St. Dev.')
    
    # Use logarithmic scale for y-axis
    ax.set_yscale('log')

    # Formatting
    ax.set_title('Particle Number Discrepancies vs. Iteration', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=8)
    ax.set_ylabel('Log-Scaled Value', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(axis='both', labelsize=7)

    # Save or display
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Discrepancy plot saved to {output_file}")
    else:
        plt.show()

def plot_on_grid_1d(
    mean_vals, std_vals, points, xlabel, ylabel, output_file=None
):
    plt.figure(figsize=(6, 5))
    plt.plot(points, mean_vals, color="blue", linestyle="-", marker="o")
    plt.fill_between(
        points,
        mean_vals - std_vals,
        mean_vals + std_vals,
        color="blue",
        alpha=0.3,
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()

def plot_on_grid_1d(
    mean_vals: Union[List, np.ndarray], 
    std_vals: Union[List, np.ndarray], 
    points: Union[List, np.ndarray], 
    xlabel: str, 
    ylabel: str, 
    output_file: Optional[str] = None
):
    """
    Plots a 1D profile of a quantity (e.g., density) with an error band representing uncertainty.

    Parameters:
      mean_vals: Array of mean values to plot.
      std_vals: Array of standard deviation values (defines the error band).
      points: The x-coordinates (grid points).
      xlabel: Label for the x-axis.
      ylabel: Label for the y-axis.
      output_file: Path to save the plot. If None, displays the plot interactively.
    """
    plt.figure(figsize=(6, 5))
    
    # Plot the central mean line
    plt.plot(points, mean_vals, color="blue", linestyle="-", marker="o", markersize=4, label='Mean')
    
    # Add a shaded region for the standard deviation (Mean ± Std)
    plt.fill_between(
        points,
        mean_vals - std_vals,
        mean_vals + std_vals,
        color="blue",
        alpha=0.3,
        label=r'$\pm 1\sigma$'
    )
    
    # Formatting for readability
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"1D Plot saved to {output_file}")
    plt.show()


def plot_on_grid_2d(
    mean_vals: Union[List, np.ndarray], 
    std_vals: Union[List, np.ndarray], 
    points: Union[List, np.ndarray], 
    label_mean: str, 
    label_std: str, 
    output_file: Optional[str] = None
):
    """
    Creates side-by-side 2D contour plots (heatmaps) for a quantity's mean and standard deviation.
    
    Assumes the input points form a square grid (N x N).

    Parameters:
      mean_vals: Flattened array of mean values.
      std_vals: Flattened array of standard deviation values.
      points: Array of shape (N_total, 2) containing coordinates [x, y].
      label_mean: Label for the colorbar of the mean plot.
      label_std: Label for the colorbar of the std plot.
      output_file: Path to save the plot.
    """
    
    # Style configuration variables
    fontsize_labels = 14
    fontsize_ticks = 12
    fontsize_cbar_label = 14
    fontsize_cbar_ticks = 12
    
    # 1. Reconstruct the Grid Structure
    # Inputs come in as flattened lists from the Monte Carlo sampling.
    # We must infer the grid dimensions (Sqrt(N)) to reshape for contour plotting.
    grid_size = int(jnp.sqrt(len(points)))
    
    # Reshape coordinates: points[:,0] is x, points[:,1] is y
    x = points[:, 0].reshape(grid_size, grid_size)
    y = points[:, 1].reshape(grid_size, grid_size)

    # Reshape data values to match the (grid_size, grid_size) spatial layout
    mean_vals = mean_vals.reshape(grid_size, grid_size)
    std_vals = std_vals.reshape(grid_size, grid_size)


    # 2. Create Plot Layout
    # 1x2 layout: Left for Mean, Right for Standard Deviation
    fig_KE, axs_KE = plt.subplots(1, 2, figsize=(12, 5)) 

    # --- Left Plot: Mean Values ---
    # Use 'viridis' for a perceptually uniform colormap
    mean_plot_KE = axs_KE[0].contourf(x, y, mean_vals, levels=50, cmap='viridis')
    cbar_KE = fig_KE.colorbar(mean_plot_KE, ax=axs_KE[0], label=label_mean)
    
    axs_KE[0].set_xlabel("x$_1$", fontsize=fontsize_labels)
    axs_KE[0].set_ylabel("x$_2$", fontsize=fontsize_labels)
    axs_KE[0].set_title("Mean Value", fontsize=fontsize_labels)

    # Colorbar formatting
    cbar_KE.set_label(label_mean, fontsize=fontsize_cbar_label)
    cbar_KE.ax.tick_params(labelsize=fontsize_cbar_ticks)
    
    # --- Right Plot: Standard Deviation ---
    # Use 'plasma' to distinguish it from the mean plot visually
    std_plot_KE = axs_KE[1].contourf(x, y, std_vals, levels=50, cmap='plasma')
    cbar_std_KE = fig_KE.colorbar(std_plot_KE, ax=axs_KE[1], label=label_std)
    
    axs_KE[1].set_xlabel("x$_1$", fontsize=fontsize_labels)
    axs_KE[1].set_ylabel("x$_2$", fontsize=fontsize_labels)
    axs_KE[1].set_title("Standard Deviation", fontsize=fontsize_labels)

    # Colorbar formatting
    cbar_std_KE.set_label(label_std, fontsize=fontsize_cbar_label)
    cbar_std_KE.ax.tick_params(labelsize=fontsize_cbar_ticks)

    # General axis tick formatting
    axs_KE[0].tick_params(axis='both', labelsize=fontsize_ticks)
    axs_KE[1].tick_params(axis='both', labelsize=fontsize_ticks)
    
    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"2D Grid plot saved to {output_file}")
    plt.show()