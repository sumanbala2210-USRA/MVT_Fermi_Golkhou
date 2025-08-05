import os
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from haar_power_mod import haar_power_mod

# ==============================================================================
# PART 1: COMPONENT FUNCTIONS (Building Blocks)
# These must be defined at the top level to be accessible by child processes.
# ==============================================================================


# --- GRB Pulse Shape & Generation Functions ---
def gaussian(t: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    return A * np.exp(-(t - mu)**2 / (2 * sigma**2))

def fred_pulse(t: np.ndarray, A: float, t_peak: float, rise_sigma: float, decay_tau: float) -> np.ndarray:
    flux = np.zeros_like(t, dtype=float)
    rise_mask = t <= t_peak
    flux[rise_mask] = A * np.exp(-(t[rise_mask] - t_peak)**2 / (2 * rise_sigma**2))
    decay_mask = t > t_peak
    flux[decay_mask] = A * np.exp(-(t[decay_mask] - t_peak) / decay_tau)
    return flux

def generate_rate_function(t: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Generates the smooth rate function from a parameter dictionary."""
    rate = np.full_like(t, params['background_level'])
    for pulse_def in params['pulse_list']:
        pulse_type, (rel_amp, *time_params) = pulse_def
        abs_amp = params['main_amplitude'] * rel_amp
        full_params = (abs_amp, *time_params)
        if pulse_type == 'fred':
            rate += fred_pulse(t, *full_params)
        elif pulse_type == 'gaussian':
            rate += gaussian(t, *full_params)
    return rate

def generate_photon_events(t: np.ndarray, rate_t: np.ndarray) -> np.ndarray:
    """Generates discrete photon arrival times from a rate function."""
    rate_interpolated = interp1d(t, rate_t, kind='linear', bounds_error=False, fill_value=0)
    duration = t[-1] - t[0]
    rate_max = np.max(rate_t)
    if rate_max <= 0: return np.array([])
    num_candidates = np.random.poisson(duration * rate_max * 1.2)
    candidate_times = t[0] + np.random.uniform(0, 1, num_candidates) * duration
    acceptance_probs = rate_interpolated(candidate_times) / rate_max
    return np.sort(candidate_times[np.random.uniform(0, 1, num_candidates) < acceptance_probs])

# ==============================================================================
# PART 2: THE PARALLEL WORKER
# ==============================================================================

def _run_full_simulation_worker(params: Dict[str, Any]) -> Tuple[float, float]:
    """
    WORKER FUNCTION: Executes ONE full simulation cycle from scratch.
    Generation -> Binning -> Analysis.
    """
    # 1. Generate the underlying rate function
    time_array = np.linspace(0, params['duration'], int(params['duration']/0.0001))
    rate_func = generate_rate_function(time_array, params)
    
    # 2. Generate a unique, stochastic set of photon events
    photon_times = generate_photon_events(time_array, rate_func)
    
    # 3. Bin the data into a new light curve
    bins = np.arange(0, params['duration'] + params['bin_width'], params['bin_width'])
    counts, _ = np.histogram(photon_times, bins=bins)


    # 3. Setup for histogram plotting and create plots
    #bins = np.arange(0, duration + bin_width, bin_width)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle('GRB Simulation with Relative Amplitudes', fontsize=18)

    # Top Panel: Underlying rate model
    ax1.plot(time_array, rate_func, color='dodgerblue', label='Underlying Rate Function')
    ax1.set_ylabel('Expected Rate (counts/s)')
    ax1.legend()
    ax1.grid(True)

    # Bottom Panel: Binned "observed" data
    counts_per_bin, _ = np.histogram(photon_times, bins=bins)
    rate_per_bin = counts_per_bin / params['bin_width']
    ax2.step(bins[:-1], rate_per_bin, where='post', color='firebrick', linewidth=1.0, label=f"Binned Data ({params['bin_width']*1000:.1f} ms bins)")
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Observed Rate (counts/s)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{params['output_dir']}/create_realistic_grb.png", dpi=300)
    plt.close(fig)





    
    # 4. Analyze this unique light curve
    errors = np.sqrt(counts)
    errors[errors <= 0] = 0.0 # Avoid sqrt(0) issues
    
    results = haar_power_mod(counts, errors, min_dt=params['bin_width'], afactor=-1., doplot=False, verbose=False)
    mvt_ms = round(float(results[2]) * 1000, 3)
    mvt_error_ms = round(float(results[3]) * 1000, 3)
    
    return mvt_ms, mvt_error_ms

# ==============================================================================
# PART 3: MAIN ORCHESTRATION AND EXECUTION
# ==============================================================================

def run_mvt_consistency_test(
    simulation_params: Dict[str, Any],
    n_iter: int,
    output_folder: Path,
    n_workers: Optional[int] = None
):
    """
    Orchestrates the parallel simulation runs to test MVT consistency.
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)
    print(f"\nStarting consistency test: {n_iter} full simulations using {n_workers} workers...")

    # Create a list of tasks; each task is the same set of simulation parameters
    tasks = [simulation_params] * n_iter
    
    mvt_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_iterator = executor.map(_run_full_simulation_worker, tasks)
        mvt_results = list(tqdm(results_iterator, total=n_iter, desc="Running Simulations"))
    
    print("All simulations complete. Aggregating and saving results...")
    df = pd.DataFrame(mvt_results, columns=['MVT_ms', 'Error_ms'])
    df.index.name = 'Iteration'; df.index += 1
    
    # Save detailed results to CSV
    output_csv = output_folder / "mvt_consistency_results.csv"
    df.to_csv(output_csv)
    
    # Plot MVT distribution
    valid_results = df[df['Error_ms'] > 0]
    fail_count = len(df) - len(valid_results)
    mean_mvt, empirical_std = (np.nan, np.nan)
    if not valid_results.empty:
        mean_mvt = round(valid_results['MVT_ms'].mean(), 3)
        empirical_std = round(valid_results['MVT_ms'].std(), 3)

    plt.figure(figsize=(10, 6))
    valid_results['MVT_ms'].hist(bins=30, edgecolor='black', alpha=0.75, density=True)
    plt.axvline(mean_mvt, color='r', linestyle='--', label=f"Mean = {mean_mvt} ms")
    plt.title(f"MVT Distribution from {n_iter} Independent Simulations")
    plt.xlabel("Minimum Variability Timescale (ms)"); plt.ylabel("Probability Density")
    plt.legend(); plt.grid(axis='y', alpha=0.5); plt.tight_layout()
    
    output_plot = output_folder / "mvt_consistency_plot.png"
    plt.savefig(output_plot, dpi=200)
    plt.close()

    print(f"\nResults saved to {output_csv} and {output_plot}")
    print(f"Number of failed MVT calculations: {fail_count}/{n_iter}")
    print(f"Mean MVT (ms): {mean_mvt}")
    print(f"Empirical std of MVT (ms): {empirical_std}")

if __name__ == '__main__':
    # --- Create a unique output folder for this entire run ---
    # Located in Huntsville, AL. Current date: August 5th, 2025.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"grb_consistency_run_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"All output will be saved in: {output_dir}")

    amp =1
    # --- Define the single set of physical parameters for the GRB model ---
    simulation_params = {
        'duration': 15,
        'bin_width': 0.0001,
        'main_amplitude': 1200.0,
        'background_level': 300.0,
        'output_dir': output_dir,
        'pulse_list': [
            ('fred', (1.0*amp,  6.1, 0.1, 1.2)),

        # Other pulses are fractions (or multiples) of the main amplitude
        ('fred', (0.84*amp, 5.2, 0.08, 0.5)),
        ('fred', (0.76*amp, 5.5, 0.06, 0.8)),
        ('fred', (0.6*amp,  6.4, 0.05, 0.6)),
        ('fred', (0.5*amp,  7.1, 0.09, 0.7)),
        ('fred', (0.3*amp,  7.9, 0.1, 1.0)),
        ('fred', (0.36*amp, 4.5, 0.3, 0.9)),
        #('fred', (0.16*amp, 12.0, 0.4, 2.5)),
        #('fred', (0.14*amp, 15.5, 0.2, 0.8)),
        #('fred', (0.3*amp,  9.0, 2.0, 1.0)),    # Broad base component

        ('gaussian', (0.3*amp,  4.8, 0.01)),
        ('gaussian', (0.44*amp, 6.8, 0.15)),
        ('gaussian', (0.38*amp, 7.5, 0.2)),
        ('gaussian', (0.2*amp,  10.5, 0.9)),
        #('gaussian', (0.12*amp, 14.0, 1.0)),
        ]
    }
    
    # --- (Optional) Plot one example realization for reference ---
    print("\nGenerating one representative GRB plot for reference...")
    time_array = np.linspace(0, simulation_params['duration'], 5000)
    rate_func = generate_rate_function(time_array, simulation_params)
    plt.figure(figsize=(12, 4))
    plt.plot(time_array, rate_func, color='dodgerblue')
    plt.title("Underlying Physical Rate Model")
    plt.xlabel("Time (s)"); plt.ylabel("Expected Rate (counts/s)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(output_dir / "representative_rate_model.png")
    plt.close()

    # --- Run the full consistency test ---
    run_mvt_consistency_test(
        simulation_params=simulation_params,
        n_iter=1000,
        output_folder=output_dir,
        n_workers=os.cpu_count() 
    )
    
    print("\nPipeline finished successfully.")