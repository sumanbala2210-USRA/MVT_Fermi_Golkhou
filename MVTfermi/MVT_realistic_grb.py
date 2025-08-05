"""
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th June 2025: Including Fermi GBM simulation of same functions.
08.05.2025: Modified to compute MVT from the light curve create_realistic_grb.py

"""
from haar_power_mod import haar_power_mod



import os
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # For a nice progress bar



def _run_analysis_worker(counts: np.ndarray) -> Tuple[float, float]:
    """
    Worker function for a single analysis of a light curve.
    It is now a "pure" function: it only operates on its inputs.

    Args:
        counts (np.ndarray): The array of photon counts for one light curve.
                             NOTE: The same counts array is passed each time.
                             The randomness comes from the analysis function if it has
                             any stochastic elements, or from bootstrapping the data.
                             If haar_power_mod is deterministic, you'll get the same result
                             every time. A common pattern is to bootstrap (resample) the counts here.
                             For this example, we assume the randomness is internal to haar_power_mod.

    Returns:
        tuple: (mvt_ms, mvt_error_ms) for one analysis run.
    """
    # If your analysis requires bootstrapping (resampling with replacement)
    # you would do it here:
    # bootstrapped_counts = np.random.choice(counts, size=len(counts), replace=True)
    
    # Process the data. Note: errors can be passed as another argument if they vary.
    # The error on counts is typically sqrt(counts).
    errors = np.sqrt(counts)
    errors[errors == 0] = 1.0 # Avoid division by zero if counts can be 0

    results = haar_power_mod(counts, np.sqrt(counts), min_dt= 0.0001, doplot=True, file= 'real_grb', afactor=-1.0, verbose=False)
    #plt.close('all')  # Close all plots to avoid memory issues in parallel processing
    mvt_ms = round(float(results[2]) * 1000, 3)
    mvt_error_ms = round(float(results[3]) * 1000, 3)

    return mvt_ms, mvt_error_ms


def run_monte_carlo_mvt(
    light_curve_counts: np.ndarray,
    n_iter: int = 1000,
    n_workers: Optional[int] = None,
    output_prefix: str = "mvt_distribution"
):
    """
    Computes the MVT distribution for a given light curve using parallel processing.

    Args:
        light_curve_counts (np.ndarray): The binned counts of the light curve to analyze.
        n_iter (int): The number of Monte Carlo iterations.
        n_workers (int, optional): Number of parallel processes. Defaults to os.cpu_count() - 2.
        output_prefix (str): Prefix for the output .csv and .png files.
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)
    print(f"Analyzing light curve over {n_iter} iterations using {n_workers} workers...")

    # The data (light_curve_counts) is passed to each worker.
    # We create a list of arguments for the map function.
    tasks = [light_curve_counts] * n_iter

    mvt_results = []
    # Execute tasks in parallel with a progress bar
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Use tqdm to visualize progress
        results_iterator = executor.map(_run_analysis_worker, tasks)
        mvt_results = list(tqdm(results_iterator, total=n_iter, desc="Simulations"))
    
    print("All iterations complete. Aggregating and saving results...")

    # --- Use pandas for robust data handling and saving ---
    df = pd.DataFrame(mvt_results, columns=['MVT_ms', 'Error_ms'])
    df.index.name = 'Iteration'
    df.index += 1

    # Save results to a CSV file
    output_csv = Path(f"{output_prefix}_results.csv")
    df.to_csv(output_csv)
    
    # --- Analysis and Plotting ---
    valid_results = df[df['Error_ms'] > 0]
    fail_count = len(df) - len(valid_results)

    if not valid_results.empty:
        mean_mvt = round(valid_results['MVT_ms'].mean(), 3)
        empirical_std = round(valid_results['MVT_ms'].std(), 3)
        original_error = round(valid_results['Error_ms'].mean(), 3)
    else:
        mean_mvt = float('nan')
        empirical_std = float('nan')
        print("Warning: All MVT calculations failed.")

    # Plot distribution
    plt.figure(figsize=(10, 6))
    valid_results['MVT_ms'].hist(bins=20, edgecolor='black', alpha=0.75)
    plt.axvline(mean_mvt, color='r', linestyle='--', label=f"Mean = {mean_mvt} ms")
    plt.title(f"MVT Distribution ({len(valid_results)} Valid Runs)")
    plt.xlabel("Minimum Variability Timescale (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    
    output_plot = Path(f"{output_prefix}_plot.png")
    plt.show()
    plt.savefig(output_plot)
    #plt.close()

    print(f"\nResults saved to {output_csv} and {output_plot}")
    print(f"Number of failed calculations: {fail_count}/{n_iter}")
    print(f"Mean MVT (ms): {mean_mvt}")
    print(f"Empirical std MVT (ms): {empirical_std}")
    print(f"Original error (ms): {original_error}")


if __name__ == '__main__':
    # --- Step 1: Prepare the Data (Load or Simulate ONCE) ---
    # This section is now separate from the analysis function.
    # You can easily switch between loading a file or generating a new GRB.
    
    # Option A: Load the data from the .npz file
    try:
        data_file = Path('grb_light_curve.npz')
        if not data_file.exists():
             raise FileNotFoundError(f"Data file not found: {data_file}")
        print(f"Loading data from {data_file}...")
        data = np.load(data_file)
        grb_counts = data['counts']
            # --- Step 2: Run the Monte Carlo Analysis on the Prepared Data ---
        run_monte_carlo_mvt(
            light_curve_counts=grb_counts,
            n_iter=30,
            output_prefix="real_GRB_mvt_distribution"
        )
        
    except FileNotFoundError as e:
        print(e)
        # Option B: Or generate it on the fly if the file doesn't exist
        print("Data file not found. Generating a mock GRB for demonstration.")
        exit()


