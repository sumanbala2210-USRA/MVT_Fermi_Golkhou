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

# --- New Function for Direct Poisson Binning ---
def create_lightcurve_direct_poisson(
    params: Dict[str, Any],
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a light curve using the fast Direct Poisson Binning method.

    Args:
        params (Dict): A dictionary of simulation parameters, including the pulse_list.
        random_seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - bin_centers (1D array of time for each bin)
            - observed_counts (1D array of noisy counts in each bin)
            - ideal_rate_cps (1D array of the noiseless underlying rate in counts/sec)
    """
    # 1. Set up the time bins
    bins = np.arange(0, params['duration'] + params['bin_width'], params['bin_width'])
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 2. Calculate the ideal, noiseless rate (in counts/sec) at each bin center
    ideal_rate_cps = generate_rate_function(bin_centers, params)

    # 3. Calculate the expected number of counts for each bin (lambda)
    expected_counts_per_bin = ideal_rate_cps * params['bin_width']

    # 4. Generate observed counts by applying Poisson noise to each bin
    rng = np.random.default_rng(seed=random_seed)
    observed_counts = rng.poisson(expected_counts_per_bin)

    return bin_centers, observed_counts, ideal_rate_cps


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
    time_array = np.linspace(0, params['duration'], int(params['duration']/params['bin_width']))
    # 1. Generate the underlying rate function
    rate_func = generate_rate_function(time_array, params)
    if params['type'] == 'photon_list':
        # 2. Generate a unique, stochastic set of photon events
        photon_times = generate_photon_events(time_array, rate_func)
        
        # 3. Bin the data into a new light curve
        bins = np.arange(0, params['duration'] + params['bin_width'], params['bin_width'])
        counts, _ = np.histogram(photon_times, bins=bins)
        # 3. Setup for histogram plotting and create plots
        #bins = np.arange(0, duration + bin_width, bin_width)
        plot_bins = bins[:-1]
    else:
        
        plot_bins, counts, ideal_rate = create_lightcurve_direct_poisson(params, random_seed=params['random_seed'])


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f'GRB Simulation Factor: {round(params["factor"], 3)}, {params["type"]}', fontsize=14)

    # Top Panel: Underlying rate model
    ax1.plot(time_array, rate_func, color='dodgerblue', label='Underlying Rate Function')
    ax1.set_ylabel('Expected Rate (counts/s)')
    ax1.legend()
    ax1.grid(True)

    ax2.step(plot_bins, counts, where='post', color='firebrick', linewidth=1.0, label=f"Binned Data ({params['bin_width']*1000:.1f} ms bins)")
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Observed Rate (counts/s)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{params['output_dir']}/create_{params['factor']}_grb.png", dpi=300)
    plt.close(fig)


            # 4. Analyze this unique light curve

    errors = np.sqrt(counts)
    errors[errors <= 0] = 0.0 # Avoid sqrt(0) issues

    results = haar_power_mod(counts, errors, min_dt=params['bin_width'], afactor=-1., weight=False, zerocheck=True, doplot=True, verbose=False, file=f"{params['output_dir']}/real_grb_{params['factor']}")
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
) -> Tuple:
    """
    Orchestrates the parallel simulation runs, calculates robust statistics,
    and plots the final distribution.
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)
    print(f"\nStarting consistency test: {n_iter} full simulations using {n_workers} workers...")

    # --- Run simulations in parallel ---
    tasks = [simulation_params] * n_iter
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_iterator = executor.map(_run_full_simulation_worker, tasks)
        mvt_results = list(tqdm(results_iterator, total=n_iter, desc="Running Simulations"))
    
    print("All simulations complete. Aggregating and saving results...")
    df = pd.DataFrame(mvt_results, columns=['MVT_ms', 'Error_ms'])
    df.index.name = 'Iteration'
    df.index += 1
    
    # Save the full, detailed results to CSV
    output_csv = output_folder / f"mvt_{simulation_params['factor']}_results.csv"
    df.to_csv(output_csv)
    
    # --- Filter, Analyze, and Plot ---
    # Select the *Series* of valid MVT values, not the whole DataFrame
    valid_mvt_series = df[df['Error_ms'] > 0]['MVT_ms']
    fail_count = len(df) - len(valid_mvt_series)

    # Check if there are any valid results before proceeding
    if not valid_mvt_series.empty:
        # Calculate robust statistics from the 1D Series
        p16, median_mvt, p84 = np.percentile(valid_mvt_series, [16, 50, 84])
        upper_error = p84 - median_mvt
        lower_error = median_mvt - p16

        # --- Create and Save the Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the histogram using the 1D Series
        ax.hist(valid_mvt_series, bins=30, density=True, color='skyblue',
                edgecolor='black', alpha=0.8, label=f"MVT Distribution ({len(valid_mvt_series)} runs)")

        # Overlay the statistics
        ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2.5,
                   label=f"Median = {median_mvt:.3f} ms")
        ax.axvspan(p16, p84, color='firebrick', alpha=0.2,
                   label=f"68% C.I. Range [{p16:.3f}, {p84:.3f}]")

        # Formatting
        ax.set_title(f"MVT Distribution for Factor: {simulation_params['factor']})", fontsize=16)
        ax.set_xlabel("Minimum Variability Timescale (ms)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(left=max(0, p16 - 3 * lower_error)) # This is now safe
        fig.tight_layout()

        # Save the plot
        output_plot_path = output_folder / f"mvt_{simulation_params['factor']}_plot.png"
        plt.savefig(output_plot_path, dpi=300)
        plt.close(fig)

        print(f"\n------ Factor: {simulation_params['factor']} ------")
        print(f"Number of failed calculations: {fail_count}/{n_iter}")
        print(f"Valid simulations: {len(valid_mvt_series)}")
        print(f"Best Estimate (Median): {median_mvt:.3f} ms")
        print(f"68% Confidence Interval: [{p16:.3f}, {p84:.3f}] ms")
        print(f"Asymmetric Error: +{upper_error:.3f} / -{lower_error:.3f} ms")
        print("-----------------------------------")

        return simulation_params['factor'],  fail_count, round(median_mvt,3), round(p16,3), round(p84,3), round(upper_error,3), round(lower_error,3)

    else:
        # This block runs if all simulations failed
        print("\nWarning: All MVT calculations failed. No plot could be generated.")
        print(f"Number of failed calculations: {fail_count}/{n_iter}")

        return simulation_params['factor'],  fail_count, -100, -100, -100, -100, -100



if __name__ == '__main__':
    # --- Create a unique output folder for this entire run ---
    # Located in Huntsville, AL. Current date: August 5th, 2025.
    sim_type = 'poisson_list'  # 'photon_list' for photon events, 'poisson_list' for direct Poisson binning
    sim_type = 'photon_list'  # 'photon_list' for photon events, 'poisson_list' for direct Poisson binning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"grb_{sim_type}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"All output will be saved in: {output_dir}")
    all_runs_summary = []
    values1 = np.arange(0.1, 1.0, .1)
    values2 = np.arange(1.0, 11.0, 1)
    values = np.concatenate((values1, values2))
    values = [1]
    for fact in values:
        amp = 1
        # --- Define the single set of physical parameters for the GRB model ---
        simulation_params = {
            'factor': fact,
            'type': sim_type,  # 'photon_list' for photon events, 'poisson_list' for direct Poisson binning
            'duration': 15,
            'bin_width': 0.001,
            'main_amplitude': 1200.0,
            'background_level': 300.0,
            'output_dir': output_dir,
            'random_seed': 37,  # Seed for reproducibility
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
            ('fred', (0.3*amp,  9.0, 2.0, 1.0)),    # Broad base component

            ('gaussian', (amp*fact,  2.8, 0.01)),
            ('gaussian', (0.44*amp, 6.8, 0.15)),
            ('gaussian', (0.38*amp, 7.5, 0.2)),
            ('gaussian', (0.2*amp,  10.5, 0.9)),
            #('gaussian', (0.12*amp, 14.0, 1.0)),
            ]
        }

        # --- Run the full consistency test ---

        factor, fail_count, median_mvt, p16, p84, upper_error, lower_error = run_mvt_consistency_test(
            simulation_params=simulation_params,
            n_iter=10,
            output_folder=output_dir,
            n_workers=os.cpu_count()-2
        )

        run_summary = {
            'factor': factor,
            'fail_count': fail_count,
            'median_mvt_ms': median_mvt,
            'p16_mvt_ms': p16,
            'p84_mvt_ms': p84,
            'upper_error': upper_error,
            'lower_error': lower_error
        }
        all_runs_summary.append(run_summary)

    # --- After the loop, save the aggregated summary to a single CSV file ---
    print("\nAll factors processed. Saving summary file...")
    summary_df = pd.DataFrame(all_runs_summary)
    summary_output_path = output_dir / "all_runs_summary.csv"
    summary_df.to_csv(summary_output_path, index=False)
        
    print("\nPipeline finished successfully.")