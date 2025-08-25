import os
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from haar_power_mod import haar_power_mod
from TTE_SIM_v2 import generate_pulse_function, gen_pulse_advanced, constant  # Assuming this is defined in TTE_SIM.py



# =======================================================================
# PART 1: GLOBAL PARAMETERS AND CONSTANTS
# =======================================================================
# --- Simulation Parameters ---
simulation_no_of_iterations = 30  # Number of Monte Carlo iterations for MVT calculation 
# simulation_no_of_iterations does not matter for 'poisson' type, keep it low ~5-10.
sim_type = 'photon'  # 'photon' for photon events, 'poisson' for direct Poisson binning
#sim_type = 'poisson'  # 'photon' for photon events, 'poisson' for direct Poisson binning
random_seed = 30  # Seed for reproducibility
overall_amp = 2000.0
bin_width = 0.000001
duration = 0.3 # 15 for max
back = 1.0  # Background level, 0 or 1 to disable or enable background
back_level = 1000
pos_fact = 1.0  # Position factor for Gaussian pulse, can be adjusted


# ==============================================================================
# PART 2: COMPONENT FUNCTIONS (Building Blocks)
# These must be defined at the top level to be accessible by child processes.
# ==============================================================================

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


def analyze_and_plot_mvt(
    results_df: pd.DataFrame, 
    factor: float, 
    n_iter: int, 
    output_folder: Path
) -> Tuple:
    """
    Analyzes MVT results, prints statistics, and generates a distribution plot.

    Args:
        results_df (pd.DataFrame): DataFrame with MVT and Error results.
        factor (float): The simulation factor for titles and filenames.
        n_iter (int): The total number of iterations that were run.
        output_folder (Path): The directory to save plots and CSVs.

    Returns:
        Tuple: A summary of the calculated statistics.
    """
    print("Aggregating and saving results...")
    
    # Save the full, detailed results to CSV
    output_csv = output_folder / f"mvt_{factor:.1f}_results.csv"
    results_df.to_csv(output_csv)
    
    # Select the Series of valid MVT values
    valid_mvt_series = results_df[results_df['Error_ms'] > 0]['MVT_ms']
    fail_count = n_iter - len(valid_mvt_series)

    if not valid_mvt_series.empty:
        p16, median_mvt, p84 = np.percentile(valid_mvt_series, [16, 50, 84])
        upper_error = p84 - median_mvt
        lower_error = median_mvt - p16

        # --- Create and Save the Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(valid_mvt_series, bins=30, density=True, color='skyblue',
                edgecolor='black', alpha=0.8, label=f"MVT Distribution ({len(valid_mvt_series)} runs)")
        ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2.5,
                   label=f"Median = {median_mvt:.3f} ms")
        ax.axvspan(p16, p84, color='firebrick', alpha=0.2,
                   label=f"68% C.I. Range [{p16:.3f}, {p84:.3f}]")
        ax.set_title(f"MVT Distribution for Factor: {factor:.1f}", fontsize=16)
        ax.set_xlabel("Minimum Variability Timescale (ms)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(left=max(0, p16 - 3 * lower_error))
        fig.tight_layout()
        output_plot_path = output_folder / f"mvt_{factor:.1f}_plot.png"
        plt.savefig(output_plot_path, dpi=300)
        plt.close(fig)

        # --- Print Summary ---
        print(f"\n------ Factor: {factor:.1f} ------")
        print(f"Number of failed calculations: {fail_count}/{n_iter}")
        print(f"Best Estimate (Median): {median_mvt:.3f} ms")
        print(f"68% Confidence Interval: [{p16:.3f}, {p84:.3f}] ms")
        print(f"Asymmetric Error: +{upper_error:.3f} / -{lower_error:.3f} ms")
        print("-----------------------------------")
        
        return factor, fail_count, round(median_mvt, 3), round(p16, 3), round(p84, 3), round(upper_error, 3), round(lower_error, 3)

    else:
        # Handle case where all simulations failed
        print(f"\n------ Factor: {factor:.1f} ------")
        print("Warning: All MVT calculations failed. No plot generated.")
        return factor, fail_count, results_df['MVT_ms'].mean(), results_df['Error_ms'].mean(), -100, -100, -100


# ==============================================================================
# PART 2: THE PARALLEL WORKER
# ==============================================================================

def _run_full_simulation_worker(params: Dict[str, Any]) -> Tuple[float, float]:
    """
    WORKER FUNCTION: Executes ONE full simulation cycle: Generation -> Analysis.
    This function is now corrected and streamlined.
    """
    # Each worker gets its own unique seed from the task list
    worker_seed = params.get('iteration_seed', None)

    if params['type'] == 'photon':
        # Use the powerful TTE-based method. It handles everything internally.
        # We pass the parameters from the dictionary directly.
        #if params['n_iter'] = params['iteration']
        output_file_path = params['output_dir'] / f"lc_{round(params['factor'], 3)}.png"
        """
        _, counts, _, _, _ = gen_pulse_advanced(
            t_start=0.0,
            t_stop=params['duration'],
            func=generate_pulse_function,
            func_par=(params,),  # The entire params dict is the parameter for our rate function
            back_func=constant,      # Background is already included in generate_rate_function
            back_func_par=(params['background_level'],),  # Background level as a tuple
            bin_width=params['bin_width'],
            source_base_rate=1.0,      # Rate is absolute in generate_rate_function
            background_base_rate=1.0,  # Rate is absolute in generate_rate_function
            plot_flag=params.get('plot_flag', False),  # Control plotting
            fig_name=output_file_path,             # No plotting inside the worker
            #plot_rebin_factor= 64*1000/params['bin_width'],
            random_seed=worker_seed,
            title=f"lc {round(params['factor'], 3)}"
        )
        """
        output_file_path = params['output_dir'] / f"lc_{round(params['factor'], 3)}.png"
        _, counts, _, _, _ = gen_pulse_advanced(
            t_start=0.0,
            t_stop=params['duration'],
            func=generate_pulse_function,
            func_par=(params,),  # The entire params dict is the parameter for our rate function
            back_func=constant,      # Background is already included in generate_rate_function
            back_func_par=(params['background_level'],),  # Background level as a tuple
            bin_width=params['bin_width'],
            source_base_rate=1.0,      # Rate is absolute in generate_rate_function
            background_base_rate=1.0,  # Rate is absolute in generate_rate_function
            plot_flag=params.get('plot_flag', False),  # Control plotting
            fig_name=output_file_path,             # No plotting inside the worker
            #plot_rebin_factor= 64*1000/params['bin_width'],
            random_seed=worker_seed,
            title=f"lc {round(params['factor'], 3)}"
        )

    else:  # 'poisson'
        # Use the direct binning method with the unique worker seed
        _, counts, _ = create_lightcurve_direct_poisson(params, random_seed=worker_seed)

    # Analyze this unique light curve
    errors = np.sqrt(counts)
    errors[errors <= 0] = 1.0  # Use 1.0 to avoid issues if counts is 0

    output_file_path = params['output_dir'] / f"haar_plot_factor_{round(params['factor'], 1)}_seed_{worker_seed}.png"

    results = haar_power_mod(
        counts, 
        errors, 
        min_dt=params['bin_width'], 
        afactor=-1., 
        weight=False, 
        zerocheck=True, 
        doplot=False, # Set to False for speed during parallel runs
        verbose=False, 
        file=output_file_path # Optional: save individual plots
    )
    mvt_ms = round(float(results[2]) * 1000, 3)
    mvt_error_ms = round(float(results[3]) * 1000, 3)
    
    return mvt_ms, mvt_error_ms

# ==============================================================================
# PART 3: MAIN ORCHESTRATION AND EXECUTION
# ==============================================================================

def run_parallel_simulations(
    simulation_params: Dict[str, Any],
    n_iter: int,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Executes all Monte Carlo simulations in parallel.
    
    Args:
        simulation_params (Dict): Base parameters for the simulation.
        n_iter (int): The number of simulations to run.
        n_workers (int, optional): The number of CPU cores to use.

    Returns:
        pd.DataFrame: A DataFrame containing the raw MVT and Error results.
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)
    print(f"\nStarting consistency test: {n_iter} full simulations using {n_workers} workers...")

    # Create a list of tasks, each with a unique seed for the worker
    tasks = []
    base_seed = simulation_params.get('random_seed', 42)
    for i in range(n_iter):
        task_params = simulation_params.copy()
        task_params['iteration_seed'] = base_seed + i
        task_params['plot_flag'] = True if i == 0 else False  # Only plot the first simulation
        tasks.append(task_params)
    
    # Run simulations in parallel and collect results
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_iterator = executor.map(_run_full_simulation_worker, tasks)
        mvt_results = list(tqdm(results_iterator, total=n_iter, desc="Running Simulations"))
    
    print("All simulations complete.")
    
    # Convert results to a DataFrame
    df = pd.DataFrame(mvt_results, columns=['MVT_ms', 'Error_ms'])
    df.index.name = 'Iteration'
    df.index += 1
    
    return df




def main():
    # --- Create a unique output folder for this entire run ---
    # Located in Huntsville, AL. Current date: August 5th, 2025.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"grb{simulation_no_of_iterations}_{sim_type}_{duration:.1f}s_{bin_width*1000:.1f}ms_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"All output will be saved in: {output_dir}")
    all_runs_summary = []
    #values1 = np.arange(1, 10.0, 2)
    values2 = np.arange(10, 100, 10)
    #values = np.concatenate((values1, values2))
    values = [100, 70, 50, 20]
    for fact in values:
        # --- Define the single set of physical parameters for the GRB model ---
        sharp_pulse_params = (overall_amp * fact, 2.8 * pos_fact, 0.01)
        sharp_pulse_to_analyze = ('gaussian', sharp_pulse_params)
        simulation_params = {
            'factor': fact,
            'type': sim_type,  # 'photon' for photon events, 'poisson' for direct Poisson binning
            'duration': duration,
            'bin_width': bin_width,
            'main_amplitude': 1.0,
            'background_level': back_level,  # Background level, can be 0 or 1 to disable or enable background
            'output_dir': output_dir,
            'random_seed': random_seed,  # Seed for reproducibility
            'back_factor': back,  # Background factor, 0 or 1 to disable or enable background
            'pulse_list': [
                ('fred', (1.0*overall_amp,  6.1, 0.1, 1.2)),

            # Other pulses are fractions (or multiples) of the main amplitude
            ('fred', (0.84*overall_amp, 5.2, 0.08, 0.5)),
            ('fred', (0.76*overall_amp, 5.5, 0.06, 0.8)),
            ('fred', (0.6*overall_amp,  6.4, 0.05, 0.6)),
            ('fred', (0.5*overall_amp,  7.1, 0.09, 0.7)),
            ('fred', (0.3*overall_amp,  7.9, 0.1, 1.0)),
            ('fred', (0.36*overall_amp, 4.5, 0.3, 0.9)),
            #('fred', (0.16*overall_amp, 12.0, 0.4, 2.5)),
            #('fred', (0.14*overall_amp, 15.5, 0.2, 0.8)),
            ('fred', (0.3*overall_amp,  9.0, 2.0, 1.0)),    # Broad base component

            ('gaussian', (overall_amp*fact,  0.2*pos_fact, 0.0001)),
            ('gaussian', (0.44*overall_amp, 6.8, 0.15)),
            ('gaussian', (0.38*overall_amp, 7.5, 0.2)),
            ('gaussian', (0.2*overall_amp,  10.5, 0.9)),
            #('gaussian', (0.12*overall_amp, 14.0, 1.0)),
            ]
        }


        # --- Generate and Plot ONE Representative Light Curve per Factor ---
        print(f"\n--- Processing Factor: {fact:.2f} ---")
         # --- Generate and Plot ONE Representative Light Curve ---

        # --- Run the full consistency test ---
        # Step 1: Run all simulations and get the raw data
        results_df = run_parallel_simulations(
            simulation_params=simulation_params,
            n_iter=simulation_no_of_iterations,
            n_workers=os.cpu_count() - 2
        )

        # Step 2: Analyze the data, plot the results, and get the summary
        factor, fail_count, median_mvt, p16, p84, upper_error, lower_error = analyze_and_plot_mvt(
            results_df=results_df,
            factor=round(simulation_params['factor'], 3),
            n_iter=simulation_no_of_iterations,
            output_folder=output_dir
        )


        run_summary = {
            'factor': round(factor, 3),
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
    
    print(f"Summary saved to: {summary_output_path}")
    print("\nPipeline finished successfully.")


if __name__ == '__main__':
    # This now simply calls the main function
    main()
    