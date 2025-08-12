"""
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th June 2025: Including Fermi GBM simulation of same functions.

"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import scipy.integrate as spi
import gdt.core
import yaml  # Import the JSON library for parameter logging
import warnings
from astropy.io.fits.verify import VerifyWarning

# --- GDT Core Imports ---
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.plot.lightcurve import Lightcurve
from gdt.core.simulate.profiles import tophat, constant, norris, quadratic, linear, gaussian
from gdt.core.simulate.tte import TteBackgroundSimulator, TteSourceSimulator
from gdt.core.simulate.pha import PhaSimulator
from gdt.core.spectra.functions import DoubleSmoothlyBrokenPowerLaw, Band
from gdt.missions.fermi.gbm.response import GbmRsp2
from gdt.missions.fermi.gbm.tte import GbmTte
#from lib_sim import write_yaml
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
import matplotlib.pyplot as plt
from gdt.core.plot.lightcurve import Lightcurve
from gdt.core.plot.model import ModelFit
from gdt.core.tte import PhotonList
from gdt.core.plot.spectrum import Spectrum
from haar_power_mod import haar_power_mod
from sim_functions import constant2, gaussian2, triangular # type: ignore
from TTE_SIM import gen_pulse, gen_GBM_pulse
# Suppress a common FITS warning
import concurrent.futures


warnings.simplefilter('ignore', VerifyWarning)

MAX_WORKERS = os.cpu_count() - 2
const_par = (1, )
fred_par = (0.5, 0.0, 0.05, 0.1)
gauss_params = (5000, 0.0, 0.01)


def _run_single_simulation(args):
    """
    Worker function for a single simulation iteration.
    It's designed to be called by a parallel process pool.
    
    Args:
        args (tuple): A tuple containing all necessary parameters.

    Returns:
        tuple: (mvt_ms, mvt_error_ms) for one simulation.
    """
    # Unpack all arguments for clarity
    iteration_num, trigger, det, angle, gauss_params, const_par = args

    # Simulate the pulse (assuming gen_GBM_pulse is defined elsewhere)
    """
    t_bins, counts, src_max, back_avg, SNR = gen_GBM_pulse(
        trigger, det, angle, 
        func=gaussian2, 
        func_par=gauss_params, 
        back_func=constant, 
        back_func_par=const_par, 
        simulation=True
    )
    """
    t_bins, counts, src_max, back_avg, SNR = gen_pulse( t_start=-10.0, t_stop=10.0, func=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par, simulation=True)
    # Process the data (assuming haar_power_mod is defined elsewhere)
    # Note: doplot must be False inside the worker to avoid conflicts
    results = haar_power_mod(counts, np.sqrt(counts), min_dt=0.0001, doplot=False, afactor=-1., file=' ', verbose=False)

    mvt_ms = round(float(results[2]) * 1000, 3)
    mvt_error_ms = round(float(results[3]) * 1000, 3)

    return mvt_ms, mvt_error_ms


def compute_GBM_mvt_distribution_parallel(
    trigger, det, angle, gauss_params, const_par,
    n_iter=100,
    n_workers=None,
    output_txt="mvt_distribution_results.txt",
    output_plot="mvt_distribution_plot.png"
):
    """
    Computes the MVT distribution in parallel for a given trigger and detector.
    
    Args:
        trigger (str): The trigger identifier.
        det (str): The detector identifier.
        angle (float): The angle associated with the data.
        gauss_params (tuple): Parameters for the Gaussian pulse function.
        const_par (tuple): Parameters for the constant background function.
        n_iter (int): The number of iterations for the simulation.
        n_workers (int, optional): The number of parallel processes (nodes) to use. 
                                   Defaults to the number of available CPU cores.
        output_txt (str): The filename for the output text file.
        output_plot (str): The filename for the output plot.
        
    Returns:
        tuple: Mean MVT, empirical standard deviation, and count of failed calculations.
    """
    if n_workers is None:
        n_workers = MAX_WORKERS 
        print(f"Using {n_workers} available CPU cores.")

    # Prepare the arguments for each parallel task
    tasks = [(i, trigger, det, angle, gauss_params, const_par) for i in range(n_iter)]

    mvt_results = []
    print(f"Starting {n_iter} simulations across {n_workers} workers...")
    
    # Execute tasks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # map() efficiently distributes tasks and collects results
        mvt_results = list(executor.map(_run_single_simulation, tasks))
    
    print("All simulations complete. Aggregating results...")
    
    # Unpack the results from the parallel runs
    mvt_ms_list = [res[0] for res in mvt_results]
    mvt_error_list = [res[1] for res in mvt_results]

    # Write all results to the file at once
    with open(output_txt, "w") as f:
        f.write("Iteration\tMVT(ms)\tError(ms)\n")
        for i, (mvt, err) in enumerate(mvt_results):
            f.write(f"{i+1}\t{mvt}\t{err}\n")

    # --- The rest of the function for analysis and plotting is the same ---
    mvt_ms_array = np.array(mvt_ms_list)
    mvt_error_array = np.array(mvt_error_list)
    
    fail_count = np.sum(mvt_error_array == 0.0)
    valid_mvt_ms_array = mvt_ms_array[mvt_error_array > 0]

    # Handle case where all calculations might fail
    if len(valid_mvt_ms_array) > 0:
        mean_mvt = round(np.mean(valid_mvt_ms_array), 3)
        empirical_std = round(np.std(valid_mvt_ms_array), 3)
    else:
        mean_mvt = float('nan')
        empirical_std = float('nan')
        print("Warning: All MVT calculations failed.")

    # Plot distribution
    plt.figure(figsize=(8, 5))
    plt.hist(valid_mvt_ms_array, bins=15, edgecolor='black', alpha=0.7)
    plt.axvline(mean_mvt, color='red', linestyle='dashed', linewidth=1, label=f"Mean = {mean_mvt} ms")
    plt.title(f"MVT Distribution (ms)\nTrigger: {trigger}, Det: {det}, Angle: {angle}")
    plt.xlabel("MVT (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"\nNumber of failed calculations: {fail_count}/{n_iter}")
    print(f"Mean MVT (ms): {mean_mvt}")
    print(f"Empirical std MVT (ms): {empirical_std}")

    return mean_mvt, empirical_std, fail_count

def compute_mvt_distribution(
    n_iter=100,
    output_txt="mvt_distribution_results.txt",
    output_plot="mvt_distribution_plot.png"
):
    """ Computes the MVT distribution for a given trigger and detector.
    Args:                       
        trigger (str): The trigger identifier.
        det (str): The detector identifier.
        angle (float): The angle associated with the data.
        n_iter (int): The number of iterations for the simulation.
        output_txt (str): The filename for the output text file.
        output_plot (str): The filename for the output plot.
    Returns:
        tuple: Mean MVT, empirical standard deviation, and count of failed calculations.
    """
    mvt_ms_list = []
    mvt_error_list = []
    fail_count = 0

    with open(output_txt, "w") as f:
        for i in range(n_iter):
            t_bins, counts, src_max, back_avg, SNR = gen_pulse( t_start=-10.0, t_stop=10.0, ffunc=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par, simulation=True)

            results = haar_power_mod(counts, np.sqrt(counts), min_dt=0.0001, doplot=True, afactor=-1., file=' ', verbose=False)

            mvt_ms = round(float(results[2]) * 1000, 3)
            mvt_error_ms = round(float(results[3]) * 1000, 3)

            if mvt_error_ms == 0.0:
                fail_count += 1

            mvt_ms_list.append(mvt_ms)
            mvt_error_list.append(mvt_error_ms)

            f.write(f"{i+1}\tMVT(ms): {mvt_ms}\tError(ms): {mvt_error_ms}\n")

    # Filter out failed results
    mvt_ms_array = np.array(mvt_ms_list)
    valid_mvt_ms_array = mvt_ms_array[np.array(mvt_error_list) > 0]

    mean_mvt = round(np.mean(valid_mvt_ms_array), 3)
    empirical_std = round(np.std(valid_mvt_ms_array), 3)

    # Plot distribution.  
    plt.figure(figsize=(8, 5))
    plt.hist(valid_mvt_ms_array, bins=15, edgecolor='black', alpha=0.7)
    plt.axvline(mean_mvt, color='red', linestyle='dashed', linewidth=1, label=f"Mean = {mean_mvt} ms")
    #plt.title(f"MVT Distribution (ms)\nTrigger: {trigger}, Det: {det}, Angle: {angle}")
    plt.xlabel("MVT (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"\nNumber of failed calculations (mvt_error_ms = 0): {fail_count}/{n_iter}")
    print(f"Mean MVT(ms): {mean_mvt}")
    print(f"Empirical std MVT(ms): {empirical_std}")

    return mean_mvt, empirical_std, fail_count



def compute_GBM_mvt_distribution(
    trigger, det, angle,
    n_iter=100,
    output_txt="mvt_distribution_results.txt",
    output_plot="mvt_distribution_plot.png"
):
    """ Computes the MVT distribution for a given trigger and detector.
    Args:                       
        trigger (str): The trigger identifier.
        det (str): The detector identifier.
        angle (float): The angle associated with the data.
        n_iter (int): The number of iterations for the simulation.
        output_txt (str): The filename for the output text file.
        output_plot (str): The filename for the output plot.
    Returns:
        tuple: Mean MVT, empirical standard deviation, and count of failed calculations.
    """
    mvt_ms_list = []
    mvt_error_list = []
    fail_count = 0

    with open(output_txt, "w") as f:
        for i in range(n_iter):
            t_bins, counts, src_max, back_avg, SNR = gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par, simulation=True)

            results = haar_power_mod(counts, np.sqrt(counts), min_dt=0.0001, doplot=True, afactor=-1., file=' ', verbose=False)

            mvt_ms = round(float(results[2]) * 1000, 3)
            mvt_error_ms = round(float(results[3]) * 1000, 3)

            if mvt_error_ms == 0.0:
                fail_count += 1

            mvt_ms_list.append(mvt_ms)
            mvt_error_list.append(mvt_error_ms)

            f.write(f"{i+1}\tMVT(ms): {mvt_ms}\tError(ms): {mvt_error_ms}\n")

    # Filter out failed results
    mvt_ms_array = np.array(mvt_ms_list)
    valid_mvt_ms_array = mvt_ms_array[np.array(mvt_error_list) > 0]

    mean_mvt = round(np.mean(valid_mvt_ms_array), 3)
    empirical_std = round(np.std(valid_mvt_ms_array), 3)

    # Plot distribution.  
    plt.figure(figsize=(8, 5))
    plt.hist(valid_mvt_ms_array, bins=15, edgecolor='black', alpha=0.7)
    plt.axvline(mean_mvt, color='red', linestyle='dashed', linewidth=1, label=f"Mean = {mean_mvt} ms")
    plt.title(f"MVT Distribution (ms)\nTrigger: {trigger}, Det: {det}, Angle: {angle}")
    plt.xlabel("MVT (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"\nNumber of failed calculations (mvt_error_ms = 0): {fail_count}/{n_iter}")
    print(f"Mean MVT(ms): {mean_mvt}")
    print(f"Empirical std MVT(ms): {empirical_std}")

    return mean_mvt, empirical_std, fail_count




def compute_mvt_distribution_from_data(counts,
    trigger, det, angle,
    n_iter=100,
    output_txt="mvt_distribution_results.txt",
    output_plot="mvt_distribution_plot.png"
):
    """ Computes the MVT distribution from provided counts data.
    Args:
        counts (np.array): The counts data to analyze.
        trigger (str): The trigger identifier.
        det (str): The detector identifier.
        angle (float): The angle associated with the data.
        n_iter (int): The number of iterations for the simulation.
        output_txt (str): The filename for the output text file.
        output_plot (str): The filename for the output plot.
    Returns:
        tuple: Mean MVT, empirical standard deviation, and count of failed calculations."""
    mvt_ms_list = []
    mvt_error_list = []
    fail_count = 0

    with open(output_txt, "w") as f:
        for i in range(n_iter):
            results = haar_power_mod(counts, np.sqrt(counts), min_dt=0.0001, doplot=True, afactor=-1., file=' ', verbose=False)

            mvt_ms = round(float(results[2]) * 1000, 3)
            mvt_error_ms = round(float(results[3]) * 1000, 3)

            if mvt_error_ms == 0.0:
                fail_count += 1

            mvt_ms_list.append(mvt_ms)
            mvt_error_list.append(mvt_error_ms)

            f.write(f"{i+1}\tMVT(ms): {mvt_ms}\tError(ms): {mvt_error_ms}\n")

    # Filter out failed results
    mvt_ms_array = np.array(mvt_ms_list)
    valid_mvt_ms_array = mvt_ms_array[np.array(mvt_error_list) > 0]

    mean_mvt = round(np.mean(valid_mvt_ms_array), 3)
    empirical_std = round(np.std(valid_mvt_ms_array), 3)

    # Plot distribution
    plt.figure(figsize=(8, 5))
    plt.hist(valid_mvt_ms_array, bins=15, edgecolor='black', alpha=0.7)
    plt.axvline(mean_mvt, color='red', linestyle='dashed', linewidth=1, label=f"Mean = {mean_mvt} ms")
    plt.title(f"MVT Distribution (ms)\nTrigger: {trigger}, Det: {det}, Angle: {angle}")
    plt.xlabel("MVT (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"\nNumber of failed calculations (mvt_error_ms = 0): {fail_count}/{n_iter}")
    print(f"Mean MVT(ms): {mean_mvt}")
    print(f"Empirical std MVT(ms): {empirical_std}")

    return mean_mvt, empirical_std, fail_count

if __name__ == '__main__':
    #gauss_params = (.5, 0.0, 0.2)
    #tri_par = (0.01, -1., 0.0, 1.)
    #const_par = (1, )
    #fred_par = (0.5, 0.0, 0.05, 0.1)  # amp, tstart, trise, tdecay
    trigger_info = [
    {'trigger': '250709653', 'det': '6', 'angle': 10.73}, #10
    {'trigger': '250709653', 'det': '3', 'angle': 39.2}, #40
    {'trigger': '250709653', 'det': '9', 'angle': 59.42}, #60
    {'trigger': '250709653', 'det': '1', 'angle': 89.63}, #90
    {'trigger': '250709653', 'det': '2', 'angle': 129.77}, #130
    {'trigger': '250717158', 'det': '3', 'angle': 30.38}, #30
    {'trigger': '250717158', 'det': '0', 'angle': 72.9}, #70
    {'trigger': '250717158', 'det': '6', 'angle': 50.41}, #50
    {'trigger': '250717158', 'det': '9', 'angle': 99.28}, #100
    {'trigger': '250723551', 'det': '1', 'angle': 81.81}, #80
    {'trigger': '250723551', 'det': '3', 'angle': 22.82}, #20
    {'trigger': '250723551', 'det': '2', 'angle': 122.52}, #120
    {'trigger': '250723551', 'det': 'a', 'angle': 141.17}, #140
    ]

    """
    for trigger in trigger_info:
        print(f"Processing trigger {trigger['trigger']} with detector {trigger['det']}")
        gen_GBM_pulse(trigger['trigger'], trigger['det'], trigger['angle'], -10.0, 10.0, func=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par)
    """
    #gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par)
    #gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=triangular, func_par=tri_par, back_func=constant, back_func_par=const_par)
    #gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=fred, func_par=tri_par, back_func=constant, back_func_par=const_par)
    test = 0
    if test == 1:
        compute_GBM_mvt_distribution_parallel( '250709653', '6', 10.73, gauss_params, const_par,
                n_iter=1000,
                output_txt="mvt_distribution_results.txt",
                output_plot="mvt_distribution_plot.png"
            )

        """
        t_bins, counts, src_max, back_avg, SNR = gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=fred, func_par=fred_par, back_func=constant, back_func_par=const_par)


        # Call the function 100 times on this data
        compute_mvt_distribution_from_data(
            counts=counts,
            trigger='250709653',
            det='6',
            angle=10.73
        )
        """
    else:
        #gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par)
        t_bins, counts, src_max, back_avg, SNR = gen_pulse(-10.0, 10.0, func=gaussian, func_par=gauss_params, back_func=constant, back_func_par=const_par)
        compute_mvt_distribution_from_data(
            counts=counts,
            trigger='250709653',
            det='6',
            angle=10.73
        )
