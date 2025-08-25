"""
# TTE_SIM_v2.py
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th June 2025: Including Fermi GBM simulation of same functions.
14th August 2025: Added GBM and normal functions together.

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
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import re
from typing import List

# --- GDT Core Imports ---
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.plot.lightcurve import Lightcurve
#from gdt.core.simulate.profiles import tophat, constant, norris, quadratic, linear, gaussian
from gdt.core.simulate.tte import TteBackgroundSimulator, TteSourceSimulator
from gdt.core.simulate.pha import PhaSimulator
from gdt.core.spectra.functions import Band
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
from sim_functions import constant2, gaussian2, triangular, fred, constant, norris, gaussian, lognormal
# Suppress a common FITS warning
import concurrent.futures
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)

#warnings.simplefilter('ignore', VerifyWarning)

MAX_WORKERS = os.cpu_count() - 2
const_par = (1, )
fred_par = (0.5, 0.0, 0.05, 0.1)
gauss_params = (.5, 0.0, 0.1)



from typing import Dict, Any, Tuple, Callable



def print_nested_dict(d, indent=0):
    """
    Recursively prints a nested dictionary with simple values (int, str, list of primitives)
    printed on the same line as their key.
    """
    spacing = "  " * indent

    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, (str, int, float, bool)) or (
                isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value)
            ):
                print(f"{spacing}{repr(key)}: {value}")
            else:
                print(f"{spacing}{repr(key)}:")
                print_nested_dict(value, indent + 1)

    elif isinstance(d, list):
        for i, item in enumerate(d):
            if isinstance(item, (str, int, float, bool)) or (
                isinstance(item, list) and all(isinstance(v, (str, int, float, bool)) for v in item)
            ):
                print(f"{spacing}- [Index {i}]: {item}")
            else:
                print(f"{spacing}- [Index {i}]")
                print_nested_dict(item, indent + 1)
    else:
        print(f"{spacing}{repr(d)}")

from typing import Dict, Any

def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a dictionary by one level.

    It takes a dictionary like {'a': 1, 'b': {'c': 2, 'd': 3}} and
    returns a flat dictionary {'a': 1, 'c': 2, 'd': 3}.

    Args:
        d (Dict): The dictionary to flatten.

    Returns:
        Dict: A new, flattened dictionary.
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, unpack its items
            flat_dict.update(value)
        else:
            # Otherwise, just add the key-value pair
            flat_dict[key] = value
    return flat_dict


def check_param_consistency(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    dict1_name: str = 'sim_params',
    dict2_name: str = 'base_params'
) -> List[str]:
    """
    Finds all keys common to two dictionaries and checks if their values are equal.
    Handles standard types, floats (with tolerance), and NumPy arrays.

    Args:
        dict1 (Dict): The first dictionary.
        dict2 (Dict): The second dictionary.
        dict1_name (str): A descriptive name for the first dictionary for logging.
        dict2_name (str): A descriptive name for the second dictionary for logging.

    Returns:
        List[str]: A list of strings describing any discrepancies found. An
                   empty list means no discrepancies were found.
    """
    #discrepancies = []
    
    # Find the set of keys that exist in both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    for key in sorted(list(common_keys)):
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Use different comparison methods based on the data type
        are_different = False
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                are_different = True
        elif isinstance(val1, float) or isinstance(val2, float):
            if not np.isclose(val1, val2):
                are_different = True
        elif val1 != val2:
            are_different = True
            
        if are_different:
            msg = (
                f"Discrepancy found for key '{key}':\n"
                f"  - {dict1_name}['{key}'] = {val1} (type: {type(val1).__name__})\n"
                f"  - {dict2_name}['{key}'] = {val2} (type: {type(val2).__name__})"
            )
            #discrepancies.append(msg)
            print("########### ERROR ##########")
            print(msg)
            exit()


def calculate_src_interval(params: Dict) -> Tuple[float, float]:
    """
    Calculates the 'true' source interval directly from the pulse model parameters.
    """
    pulse_shape = params['pulse_shape'] #, pulse_shape)
    if pulse_shape == 'gaussian':
        # For a Gaussian, the interval containing >99.7% of the flux is +/- 3-sigma
        sigma = params['sigma']
        center = params['center_time']
        return center - 3 * sigma, center + 3 * sigma

    elif pulse_shape == 'triangular':
        # For a triangular pulse, the start and stop are explicitly defined
        width = params['width']
        center = params['center_time']
        peak_ratio = params['peak_time_ratio']
        t_start = center - (width * peak_ratio)
        t_stop = t_start + width
        return t_start, t_stop

    elif pulse_shape in ['norris', 'fred']:
        # For pulses with long tails, we can define a practical window,
        # e.g., from the start time to where the pulse has significantly decayed.
        # Here, we approximate this as the peak + 7 decay timescales.
        t_rise = params['rise_time']
        t_decay = params['decay_time']
        t_start = params['start_time']
        # Rough peak time is a few rise times after the start
        peak_time_approx = t_start + 2 * t_rise 
        t_stop = peak_time_approx + 7 * t_decay
        return t_start, t_stop
    
    # Add other pulse shapes as needed...

    # Fallback if shape is unknown
    return params.get('src_start'), params.get('src_stop')






def parse_intervals_from_csv(value_from_csv: str) -> List[List[float]]:
    """
    Parses a string like '-20.0, -5.0, 75.0, 200.0' from a CSV cell
    into a list of pairs, e.g., [[-20.0, -5.0], [75.0, 200.0]].
    """
    # Ensure the input is treated as a string
    s = str(value_from_csv).strip()
    
    # Use a regular expression to split by comma and/or any amount of space
    # This is very robust against formatting like "-20, -5" or "-20 -5"
    parts = re.split(r'[,\s]+', s)
    
    # Convert non-empty parts to floats
    vals = [float(p) for p in parts if p]
    
    # Check for an even number of values
    if len(vals) % 2 != 0:
        raise ValueError(
            f"background_intervals must contain an even number of values, but got {len(vals)} from '{value_from_csv}'"
        )
        
    # Group the flat list into a list of [start, end] pairs
    return [vals[i:i+2] for i in range(0, len(vals), 2)]






def calculate_adaptive_simulation_params(pulse_shape: str, params: Dict) -> Dict:
    """
    Calculates optimal t_start, t_stop, and grid_resolution based on pulse parameters.
    """
    t_start, t_stop, grid_res = None, None, None
    padding = 10.0  # Seconds of padding before and after the pulse

    if pulse_shape == 'gaussian':
        sigma = params['sigma']
        center = params['center_time']
        grid_res = sigma / 10.0
        # The pulse is significant within ~5-sigma of the center
        t_start = center - 5 * sigma - padding * grid_res * 20
        t_stop = center + 5 * sigma + padding * grid_res * 20
        # Rule of Thumb: Grid must be ~10x finer than the narrowest feature

    elif pulse_shape == 'lognormal':
        sigma = params['sigma']
        center = params['center_time']
        timescale = min(center, sigma * center)
        grid_res = timescale / 10.0
        # Use a similar 5-sigma rule, but in log-space
        t_start = center * np.exp(-5 * sigma - padding * grid_res * 20)
        t_stop = center * np.exp(5 * sigma + padding * grid_res * 20)
        # A rough characteristic timescale for lognormal
        
    elif pulse_shape in ['norris', 'fred']:
        t_rise = params['rise_time']
        t_decay = params['decay_time']
        start = params['start_time']
        # For FRED/Norris, the rise time is the narrowest feature
        grid_res = t_rise / 10.0
        # Determine the peak time to set a reasonable end point
        # A simple approximation for the peak time
        peak_time_approx = start + t_rise * 2 
        # End the simulation after the pulse has decayed significantly (~10x decay time)
        t_start = start - padding * grid_res * 20 - 2 * t_rise - 1 * t_decay
        t_stop = peak_time_approx + 10 * t_decay + padding * grid_res * 20

    elif pulse_shape == 'triangular':
        width = params['width']
        center = params['center_time']
        peak_ratio = params['peak_time_ratio']
        rise_duration = width * peak_ratio
        fall_duration = width * (1.0 - peak_ratio)
        grid_res = min(rise_duration, fall_duration) / 10.0
        # Start and stop are explicitly defined by the parameters
        t_start = center - (width * peak_ratio) - padding * grid_res * 20
        #t_stop = t_start + width + padding + (padding*2)
        t_stop = t_start + width + padding * grid_res * 20
        # Narrowest feature is the shorter of the rise or fall time

    # --- Safety Net ---
    # Ensure grid resolution is within reasonable bounds to prevent memory errors
    # or inaccurate simulations.
    if grid_res is not None:
        grid_res = np.clip(grid_res, a_min=1e-6, a_max=0.001) # e.g., 1µs to 1ms
    else:
        # Fallback for unknown shapes
        t_start, t_stop, grid_res = -5.0, 5.0, 0.0001
        
    return {'t_start': t_start, 't_stop': t_stop, 'grid_resolution': grid_res}







def _format_params_for_annotation(func: Callable, func_par: Tuple) -> str:
    """Formats function and parameters into a concise string for a plot title."""
    if not func or not func_par:
        return "No Source Model"

    # This handles our specific case where func is generate_pulse_function
    # and func_par is a tuple containing a single parameter dictionary.
    if func.__name__ == 'generate_pulse_function' and isinstance(func_par[0], dict):
        params_dict = func_par[0]
        pulse_strings = []
        
        # Loop through the pulse definitions in the dictionary
        for pulse_def in params_dict.get('pulse_list', []):
            p_type, p_params = pulse_def
            # Format numbers to 3 decimal places to keep the title clean
            param_str = ", ".join([f"{round(p,3)}" for p in p_params])
            pulse_strings.append(f"{p_type}({param_str})\n")
        
        if not pulse_strings:
            return "Empty Pulse List"
        # Join multiple pulses with a plus sign
        return " ".join(pulse_strings)

    # This is a generic fallback for other function types
    else:
        try:
            func_name = func.__name__
            param_str = ", ".join([f"{round(p,3)}" for p in func_par])
            return f"{func_name}({param_str})"
        except TypeError:
            # Fallback for non-numeric or complex parameters
            return f"{func.__name__}{func_par}"




def _calculate_multi_timescale_snr(
    total_counts: np.ndarray,
    sim_bin_width: float,
    back_avg_cps: float,
    search_timescales: List[float]
) -> Dict[str, float]:
    """
    Calculates SNR by finding the peak in the total light curve and
    subtracting the expected background.

    Args:
        total_counts (np.ndarray): High-resolution binned light curve of TOTAL (source + bkg) events.
        sim_bin_width (float): The bin width of the high-resolution light curve (in seconds).
        back_avg_cps (float): The average background rate in counts per second.
        search_timescales (List[float]): A list of timescales (in seconds) to search.

    Returns:
        A dictionary of SNR values for each timescale.
    """
    snr_results = {}

    for timescale in search_timescales:
        try:
            factor = max(1, int(round(timescale / sim_bin_width)))
            end = (len(total_counts) // factor) * factor
            if end == 0:
                snr_results[f'S{int(timescale*1000)}'] = 0.0
                continue
            
            # Re-bin the TOTAL counts
            rebinned_total_counts = np.sum(total_counts[:end].reshape(-1, factor), axis=1)
            
            # Find the total number of counts in the brightest time window
            counts_in_peak_bin = np.max(rebinned_total_counts)
            
            # <<< NEW: Calculate Signal and Noise via background subtraction >>>
            expected_bkg_in_bin = back_avg_cps * timescale
            
            # The signal is the excess counts above the background
            signal = counts_in_peak_bin - expected_bkg_in_bin
            
            # The noise is the Poisson error on the total counts in that bin
            noise = np.sqrt(counts_in_peak_bin)
            
            snr = signal / noise if noise > 0 else 0
            snr_results[f'S{int(timescale*1000)}'] = round(snr, 2)
            
        except Exception:
            snr_results[f'S{int(timescale*1000)}'] = -1.0
            continue

    return snr_results




def create_final_plot(
    source_events: np.ndarray,
    background_events: np.ndarray,
    model_info: Dict,
    output_info: Dict
):
    """
    A self-contained function that takes raw event data and creates a
    final, styled, and richly annotated representative plot.
    """
    try:
        # --- 1. Unpack all necessary data and parameters ---
        params = model_info['base_params']
        #print_nested_dict(params)
        func_to_use = model_info['func']
        func_par = model_info['func_par']
        fig_name = output_info['file_path'] / f"LC_{output_info['file_name']}.png"
        base_title = f" LC {output_info['file_name']}"
        t_start, t_stop = params['t_start'], params['t_stop']
        background_level_cps = params['background_level']* params.get('scale_factor', 1.0)
        
        # --- 2. Prepare Data for Plotting (Binning) ---
        total_events = np.sort(np.concatenate([source_events, background_events]))
        bin_width = params.get('bin_width_for_plot', 0.01)
        bins = np.arange(t_start, t_stop + bin_width, bin_width)
        times = bins[:-1] + bin_width / 2.0
        total_counts, _ = np.histogram(total_events, bins=bins)
        source_only_counts, _ = np.histogram(source_events, bins=bins)
        
        # <<< NEW: Calculate multi-timescale SNR >>>
        duration = t_stop - t_start
        total_counts_fine, _ = np.histogram(total_events, bins=int(duration / 0.001))
        snr_results_dict = _calculate_multi_timescale_snr(
            total_counts=total_counts_fine,
            sim_bin_width=0.001,
            back_avg_cps=background_level_cps,
            search_timescales= model_info['snr_analysis']
        )
        
        # <<< NEW: Format SNR results for the title >>>
        snr_annotation_parts = []
        for ts, snr_val in snr_results_dict.items():
            label = f"S$_{{{ts[1:]}}}$" # Use LaTeX for subscript
            value = f"{snr_val:.1f}"
            snr_annotation_parts.append(f"{label}={value}")
        SNR_text = "; ".join(snr_annotation_parts)
        final_title = f"{base_title}\n{SNR_text}"

        # <<< NEW: Format model parameters for the annotation box >>>
        annotation_text = _format_params_for_annotation(func_to_use, func_par)

        # --- 3. Define the plot data (for 'decomposed' plot type) ---
        ideal_background_counts = background_level_cps * bin_width
        plot_data = [
            {'x': times, 'y': total_counts, 'label': 'Total Signal (Simulated)', 'color': 'rosybrown', 'fill_alpha': 0.6},
            {'x': times, 'y': source_only_counts, 'label': 'Source Signal (Simulated)', 'color': 'darkgreen', 'fill_alpha': 0.4}
        ]
        h_lines = [{'y': ideal_background_counts, 'label': f'Ideal Background ({background_level_cps:.1f} cps)', 'color': 'k'}]
        ylabel = f"Counts per {bin_width*1000:.1f} ms Bin"
            
        # --- 4. Create the Plot (Core Matplotlib Logic) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        for data in plot_data:
            ax.step(data['x'], data['y'], where='mid', label=data.get('label'), color=data.get('color'), lw=data.get('lw', 1.5))
            if 'fill_alpha' in data:
                ax.fill_between(data['x'], data['y'], step="mid", color=data.get('color'), alpha=data.get('fill_alpha'))
        for line in h_lines:
            ax.axhline(y=line['y'], color=line.get('color'), linestyle='--', label=line.get('label'))

        ax.set_title(final_title, fontsize=12) # Use new dynamic title
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper right')
        ax.set_xlim(t_start, t_stop)
        ax.set_ylim(bottom=0)

        # Use the new dynamic annotation text
        if annotation_text:
            props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
            ax.text(0.03, 0.97, annotation_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
        
        fig.tight_layout()
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)
        #logging.info(f"Representative plot saved to {fig_name}")

    except Exception as e:
        print(f"Failed to generate representative plot. Error: {e}")
        pass
        #logging.error(f"Failed to generate representative plot. Error: {e}")


def create_final_gbm_plot(
                src_event_file,
                back_event_file,
                model_info: Dict,
                output_info: Dict):
    params = model_info['base_params']
    #params = model_info.get('params', {})
    #print_nested_dict(params)
    
    
    en_lo = params.get('en_lo', 8.0)
    en_hi = params.get('en_hi', 900.0)

    #fig_name = params.get('fig_name', None)
    t_start = params['t_start'] if 't_start' in params else -5.0
    t_stop = params['t_stop'] if 't_stop' in params else 5.0

    #print(params)
    trigger_number = params['trigger_number'] if 'trigger_number' in params else 0
    det = params['det'] if 'det' in params else 'nn'
    angle = params['angle'] if 'angle' in params else 0

    #analysis_settings = model_info['snr_analysis']

    func_to_use = model_info.get('func', None)
    func_par = model_info.get('func_par', {})
    fig_name = output_info['file_path'] / f"LC_{output_info['file_name']}.png"
    base_title = f" LC {output_info['file_name']}"

    energy_range_nai = (en_lo, en_hi)

    # Open the files
    tte_src_all = GbmTte.open(src_event_file)
    tte_bkgd_all = GbmTte.open(back_event_file)

    tte_src = tte_src_all.slice_time([t_start, t_stop])
    tte_bkgd = tte_bkgd_all.slice_time([t_start, t_stop])
    total_bkgd_counts = tte_bkgd.data.size
    #print("Tstart:", t_start, "Tstop:", t_stop)
    bkgd_cps = total_bkgd_counts/(t_stop - t_start)
    #print(f"Background counts: {total_bkgd_counts}, Background CPS: {bkgd_cps}")

    # merge the background and source
    tte_total = GbmTte.merge([tte_src, tte_bkgd])

    try:
        fine_bw = 0.001
        phaii = tte_total.to_phaii(bin_by_time, fine_bw)

        phaii = tte_total.to_phaii(bin_by_time, fine_bw)
        phii_src = tte_src.to_phaii(bin_by_time, fine_bw)
        phii_bkgd = tte_bkgd.to_phaii(bin_by_time, fine_bw)
        lc_tot = phaii.to_lightcurve(energy_range=energy_range_nai)
        lc_src = phii_src.to_lightcurve(energy_range=energy_range_nai)
        lc_bkgd = phii_bkgd.to_lightcurve(energy_range=energy_range_nai)

        try:
            #lcplot = Lightcurve(data=phaii.to_lightcurve(energy_range=energy_range_nai))
            lcplot = Lightcurve(data=lc_tot)
            lcplot.add_selection(lc_src)
            lcplot.add_selection(lc_bkgd)
            lcplot.selections[1].color = 'pink'
            lcplot.selections[0].color = 'green'
            lcplot.selections[0].alpha = 1
            lcplot.selections[1].alpha = 0.5

            #x_low = func_par[1] - func_par[1]
            #x_high = func_par[1] + func_par[1]
            #plt.xlim(x_low, x_high)
            lcplot.errorbars.hide()


            ######### SNR Calculation #########
        except Exception as e:
            print(f"Error during plotting: {e}")
            lcplot = None

        snr_results_dict = _calculate_multi_timescale_snr(
                    total_counts=lc_tot.counts, sim_bin_width=0.001,
                    back_avg_cps= bkgd_cps,
                    search_timescales=model_info['snr_analysis']
                )
        
        # <<< NEW: Format SNR results for the title >>>
        snr_annotation_parts = []
        for ts, snr_val in snr_results_dict.items():
            label = f"S$_{{{ts[1:]}}}$" # Use LaTeX for subscript
            value = f"{snr_val:.1f}"
            snr_annotation_parts.append(f"{label}={value}")
        SNR_text = "; ".join(snr_annotation_parts)
        final_title = f'Bn{trigger_number}, n{det}, {angle}deg,' + f"{base_title}\n{SNR_text}"

        # <<< NEW: Format model parameters for the annotation box >>>
        if func_to_use is not None:
            annotation_text = _format_params_for_annotation(func_to_use, func_par)
            if annotation_text:
                props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
                plt.text(0.03, 0.97, annotation_text, transform=plt.gca().transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='left', bbox=props)


        if fig_name is None:
            fig_name = f'lc_{trigger_number}_n{det}_{angle}deg.png'

        

        #plt.show()

        plt.title(final_title, fontsize=10)

        plt.savefig(fig_name, dpi=300)
        plt.close()
    
    except Exception as e:
        print(f"Failed to generate representative GBM plot. Error: {e}")




def Function_MVT_analysis(input_info: Dict,
                           output_info: Dict):
    #params = input_info['base_params']

    src_event_files = input_info['src_event_files']
    back_event_files = input_info['back_event_files']

    sim_params_file = input_info['sim_par_file']

    if type(sim_params_file) is dict:
        sim_params = sim_params_file
    else:
        sim_params = yaml.safe_load(open(sim_params_file, 'r'))

    data_src = np.load(src_event_files[0], allow_pickle=True)
    data_back = np.load(back_event_files[0], allow_pickle=True)
    #sim_params = data_src['params'].item()
    
    background_level = sim_params['background_level']
    scale_factor = sim_params['scale_factor']
  

    t_start = sim_params['t_start']
    t_stop = sim_params['t_stop']
    det = sim_params.get('det', 'nn')
    angle = sim_params.get('angle', 0)
   
    base_params = input_info['base_params']
    NN_analysis = base_params['num_analysis']
    snr_timescales = input_info.get('snr_timescales', [0.010, 0.016, 0.032, 0.064, 0.128])
    analysis_bin_widths_ms = input_info['analysis_bin_widths_ms']


    source_event_realizations_all = data_src['realizations']
    background_event_realizations = data_back['realizations']
    sim_params = data_src['params'].item()
    
    #background_level = sim_params['background_level']
    #scale_factor = sim_params['scale_factor']

    #background_level_cps = background_level * scale_factor
    src_start, src_stop = calculate_src_interval(sim_params)
    src_duration = src_stop - src_start

    duration = sim_params['t_stop'] - sim_params['t_start']

    iteration_results = []
    

    NN = len(source_event_realizations_all)
    NN_back = len(background_event_realizations)
    if NN != NN_back:
        logging.warning(f"Mismatch in realization counts: {NN} (source) vs {NN_back} (background)")
        return []
    if NN < NN_analysis:
        logging.warning(f"Insufficient realizations: {NN} (source) < {NN_analysis} (required)")
        return []
    
    if NN_analysis < NN:
        source_event_realizations = source_event_realizations_all[:NN_analysis]
        NN = NN_analysis
    else:
        source_event_realizations = source_event_realizations_all

    for i, source_events in enumerate(source_event_realizations):
        try:
            background_events = background_event_realizations[i]
            total_events = np.sort(np.concatenate([source_events, background_events]))
            iteration_seed = sim_params['random_seed'] + i

            total_src_counts = len(source_events)
            total_bkgd_counts = len(background_events)

            background_level_cps = total_bkgd_counts / duration
            background_counts = background_level_cps * src_duration
            try:
                snr_fluence = total_src_counts / np.sqrt(background_counts)
            except ZeroDivisionError:
                snr_fluence = 0
            #snr_fluence = total_src_counts / sigma_bkgd_counts
            # Calculate per-realization metrics that are independent of bin width
            total_counts_fine, _ = np.histogram(total_events, bins=int(duration / 0.001))
            snr_dict = _calculate_multi_timescale_snr(
                total_counts=total_counts_fine, sim_bin_width=0.001,
                back_avg_cps=background_level_cps,
                search_timescales=snr_timescales
            )

            base_iter_detail = {
                'iteration': i + 1,
                'random_seed': iteration_seed,
                'back_avg_cps': round(background_level_cps, 2),
                'bkgd_counts': int(background_counts),
                'src_counts': total_src_counts,
                'S_flu': round(snr_fluence, 2),
                **snr_dict,
            }

            if i == 1:
                create_final_plot(source_events=source_events,
                                  background_events=background_events,
                                    model_info={
                                        'func': None,
                                        'func_par': None,
                                        'base_params': sim_params,
                                        'snr_analysis': snr_timescales
                                    },
                                    output_info= output_info
                                )

            # Loop through analysis bin widths
            for bin_width_ms in analysis_bin_widths_ms:
                bin_width_s = bin_width_ms / 1000.0
                bins = np.arange(sim_params['t_start'], sim_params['t_stop'] + bin_width_s, bin_width_s)
                counts, _ = np.histogram(total_events, bins=bins)


                mvt_res = haar_power_mod(counts, np.sqrt(counts), min_dt=bin_width_s, doplot=False, afactor=-1.0, verbose=False)
                plt.close('all')
                mvt_val = float(mvt_res[2]) * 1000
                mvt_err = float(mvt_res[3]) * 1000


                iter_detail = {**base_iter_detail,
                                'analysis_bin_width_ms': bin_width_ms,
                                'mvt_ms': round(mvt_val, 4),
                                'mvt_err_ms': round(mvt_err, 4),
                                **base_params}
                iteration_results.append(iter_detail)

        except Exception as e:
            logging.warning(f"Failed analysis on realization {i} in {src_event_files[0].name}. Error: {e}")
            for bin_width_ms in analysis_bin_widths_ms:
                iteration_results.append({'iteration': i + 1,
                                                'random_seed': sim_params['random_seed'] + i,
                                                'analysis_bin_width_ms': bin_width_ms,
                                                'mvt_ms': -100,
                                                'mvt_err_ms': -200,
                                                'back_avg_cps': -100,
                                                'bkgd_counts': -100,
                                                'src_counts': -100,
                                                'S_flu': -100,
                                                **base_params,
                                                **snr_dict})
    return iteration_results, NN




def GBM_MVT_analysis(input_info: Dict,
                output_info: Dict):
    #params = input_info['base_params']

    src_event_files = input_info['src_event_files']
    back_event_files = input_info['back_event_files']
    snr_timescales = input_info.get('snr_timescales', [0.010, 0.016, 0.032, 0.064, 0.128, 0.256])
    analysis_bin_widths_ms = input_info['analysis_bin_widths_ms']
    sim_params_file = input_info['sim_par_file']

    if type(sim_params_file) is dict:
        sim_params = sim_params_file
    else:
        sim_params = yaml.safe_load(open(sim_params_file, 'r'))
    #sim_params = yaml.safe_load(open(sim_params_file[0], 'r'))
    base_params = input_info['base_params']
    #print_nested_dict(base_params)

    en_lo = sim_params.get('en_lo', 8.0)
    en_hi = sim_params.get('en_hi', 900.0)
  
   
    t_start = sim_params['t_start']
    t_stop = sim_params['t_stop']
    det = sim_params.get('det', 'nn')
    angle = sim_params.get('angle', 0)

    energy_range_nai = (en_lo, en_hi)
    src_start, src_stop = calculate_src_interval(sim_params)
    src_duration = src_stop - src_start
    duration = sim_params['t_stop'] - sim_params['t_start']

    iteration_results = []
    NN = len(src_event_files)
    for i, src_file in enumerate(src_event_files):
        iteration_seed = sim_params['random_seed'] + i
        bkgd_file = back_event_files[i]

        # Open the files
        tte_src_all = GbmTte.open(src_file)
        tte_bkgd_all = GbmTte.open(bkgd_file)

        tte_src = tte_src_all.slice_time([t_start, t_stop])
        tte_bkgd = tte_bkgd_all.slice_time([t_start, t_stop])

        total_src_counts = tte_src.data.size
        total_bkgd_counts = tte_bkgd.data.size
        background_level_cps = total_bkgd_counts / duration
        background_counts = background_level_cps * src_duration
        snr_fluence = total_src_counts / np.sqrt(background_counts)

        # merge the background and source
        tte_total = GbmTte.merge([tte_src, tte_bkgd])

        #try:
        fine_bw = 0.001
        phaii = tte_total.to_phaii(bin_by_time, fine_bw)
        lc_total = phaii.to_lightcurve(energy_range=energy_range_nai)

        snr_results_dict = _calculate_multi_timescale_snr(
                    total_counts=lc_total.counts, sim_bin_width=0.001,
                    back_avg_cps=total_bkgd_counts/(t_stop - t_start),
                    search_timescales=snr_timescales
                )
        
        #except Exception as e:
        #    print(f"Error during SNR computing: {e}")

        base_iter_detail = {
                    'iteration': i + 1,
                    'random_seed': iteration_seed,
                    'back_avg_cps': round(background_level_cps, 2),
                    'bkgd_counts': int(background_counts),
                    'src_counts': total_src_counts,
                    'S_flu': round(snr_fluence, 2),
                    **snr_results_dict,
                }

        if i == 1:
            create_final_gbm_plot(
                                src_file,
                                bkgd_file,
                                model_info={
                                    'func': None,
                                    'func_par': None,
                                    'base_params': sim_params,
                                    'snr_analysis': snr_timescales
                                },
                                output_info= output_info
                            )

        # Loop through analysis bin widths
        for bin_width_ms in analysis_bin_widths_ms:
            try:
                bin_width_s = bin_width_ms / 1000.0
                phaii_hi = tte_total.to_phaii(bin_by_time, bin_width_s)
                phaii = phaii_hi.slice_energy(energy_range_nai)
                data = phaii.to_lightcurve()
        
                mvt_res = haar_power_mod(data.counts, np.sqrt(data.counts), min_dt=bin_width_s, doplot=False, afactor=-1.0, verbose=False)
                plt.close('all')
                mvt_val = float(mvt_res[2]) * 1000
                mvt_err = float(mvt_res[3]) * 1000
            except Exception as e:
                print(f"Error during MVT calculation for bin width {bin_width_ms} ms: {e}")
                mvt_val = -100
                mvt_err = -100

            iter_detail = {**base_iter_detail,
                            'analysis_bin_width_ms': bin_width_ms,
                            'mvt_ms': round(mvt_val, 4),
                            'mvt_err_ms': round(mvt_err, 4),
                            **base_params}
            iteration_results.append(iter_detail)


    return iteration_results, NN






def generate_function_events(
    func: Callable,
    func_par: Tuple,
    back_func: Callable,
    back_func_par: Tuple,
    params: Dict[str, Any],
    back_flag: bool = True,
    source_flag: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a TTE simulation and returns source and/or background events based on flags.

    Args:
        ...
        back_flag (bool): If True, simulate and return background events.
        source_flag (bool): If True, simulate and return source events.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing (source_events, background_events).
                                       An array will be empty if its flag was False.
    """
    # If neither is requested, return empty arrays immediately
    if not source_flag and not back_flag:
        return np.array([]), np.array([])

    # Unpack necessary parameters from the dictionary
    t_start = params['t_start']
    t_stop = params['t_stop']
    random_seed = params['random_seed']
    source_base_rate = params.get('scale_factor', 1.0)
    background_base_rate = params.get('scale_factor', 1.0)
    grid_resolution = params.get('grid_resolution', 0.0001)

    np.random.seed(random_seed)

    # 1. Efficiently define the total rate function based on flags
    def total_rate_func(t):
        source_rate = func(t, *func_par) * source_base_rate if source_flag and func else 0
        background_rate = back_func(t, *back_func_par) * background_base_rate if back_flag and back_func else 0
        return source_rate + background_rate

    # 2. Simulate the total required event stream
    grid_times = np.arange(t_start, t_stop, grid_resolution)
    rate_on_grid = total_rate_func(grid_times)
    cumulative_counts = np.cumsum(rate_on_grid) * grid_resolution
    total_expected_counts = cumulative_counts[-1] if len(cumulative_counts) > 0 else 0
    num_events = np.random.poisson(total_expected_counts)
    random_counts = np.random.uniform(0, total_expected_counts, num_events)
    total_event_times = np.interp(random_counts, cumulative_counts, grid_times)

    # 3. Conditionally assign or split events
    source_event_times = np.array([])
    background_event_times = np.array([])

    if source_flag and back_flag:
        # If we need both, we must do the probabilistic split
        source_rate_at_events = func(total_event_times, *func_par) * source_base_rate
        background_rate_at_events = back_func(total_event_times, *back_func_par) * background_base_rate
        total_rate_at_events = source_rate_at_events + background_rate_at_events

        p_source = np.divide(source_rate_at_events, total_rate_at_events,
                             out=np.zeros_like(total_rate_at_events),
                             where=total_rate_at_events > 0)
        
        is_source_event = np.random.rand(num_events) < p_source
        source_event_times = np.sort(total_event_times[is_source_event])
        background_event_times = np.sort(total_event_times[~is_source_event])

    elif source_flag:
        # If we only simulated the source, all events are source events
        source_event_times = np.sort(total_event_times)

    elif back_flag:
        # If we only simulated the background, all events are background events
        background_event_times = np.sort(total_event_times)

    return source_event_times, background_event_times





def generate_gbm_events(
        event_file_path: Path,
        func: Callable,
        func_par: Tuple,
        back_func: Callable,
        back_func_par: Tuple,
        params: Dict[str, Any],
        back_flag: bool = True,
        source_flag: bool = True,
        det_flag: bool = False):

    det = params['det']
    trigger_number = params['trigger_number']
    angle = params['angle']
    en_lo = params['en_lo']
    en_hi = params['en_hi']
    t_start = params['t_start']
    t_stop = params['t_stop']

    select_time = (t_start, t_stop)
    random_seed = params['random_seed']
    grid_resolution = params.get('grid_resolution', 0.0001) # Use a fixed, fine

    energy_range_nai = (en_lo, en_hi)
    #print(f"angle: {type(params['angle2'])}")
    raw_intervals = params['background_intervals']
    bkgd_times = parse_intervals_from_csv(raw_intervals)  #[(-20.0, -5.0), (75.0, 200.0)]
    #print(f"Background intervals: {type(bkgd_times)}")
    #print(type(bkgd_times))

    # Fixed spectral Model
    band_params = (0.1, 300.0, -1.0, -2.5)

    #tte = GbmTte.open('glg_tte_n6_bn250612519_v00.fit')
    folder_path = os.path.join(os.getcwd(), f'bn{trigger_number}')

    tte_pattern = f'{folder_path}/glg_tte_n{det}_bn{trigger_number}_v*.fit'
    tte_files = glob.glob(tte_pattern)

    if not tte_files:
        raise FileNotFoundError(f"No TTE file found matching pattern: {tte_pattern}")
    tte_file = tte_files[0]  # Assuming only one file/version per det/trigger_number

    # Find the RSP2 file (e.g., glg_cspec_n3_bn230307053_v03.rsp2)
    rsp2_pattern = f'{folder_path}/glg_cspec_n{det}_bn{trigger_number}_v*.rsp2'
    rsp2_files = glob.glob(rsp2_pattern)

    if not rsp2_files:
        raise FileNotFoundError(f"No RSP2 file found matching pattern: {rsp2_pattern}")
    rsp2_file = rsp2_files[0]  # Assuming only one file/version per det/trigger_number
    rsp2 = GbmRsp2.open(rsp2_file)
    rsp = rsp2.extract_drm(atime=np.average(select_time))

    # Use the .name property to get the descriptive base filename
    base_filename = event_file_path.name
    src_filename = f"{base_filename}_src.fits"
    bkgd_filename = f"{base_filename}_bkgd.fits"

    # Open the files
    tte = GbmTte.open(tte_file)
    tte_demo = tte.slice_time([-50,-49.99])


    if source_flag:
    # source simulation
        tte_sim = TteSourceSimulator(rsp, Band(), band_params, func, func_par, deadtime=1e-6, sample_period=grid_resolution, rng=np.random.default_rng(random_seed))
        tte_src = tte_sim.to_tte(t_start, t_stop)
        tte_gbm_src = GbmTte.merge([tte_demo, tte_src])
        # Construct the new FITS filenames
        
        # Save the files to the correct directory with the new names
        tte_gbm_src.write(filename=src_filename, directory=event_file_path.parent, overwrite=True)

    if back_flag:
        #bin to 1.024 s resolution, reference time is trigger time
        phaii = tte.to_phaii(bin_by_time, 1.024, time_ref=0.0)
        bkgd_times = bkgd_times
        backfitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_times)
        backfitter.fit(order=1)
        bkgd = backfitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
        
        # the background model integrated over the source selection time
        spec_bkgd = bkgd.integrate_time(*select_time)
        
        # background simulation
        tte_sim = TteBackgroundSimulator(spec_bkgd, 'Gaussian', back_func, back_func_par, deadtime=1e-6, sample_period=grid_resolution, rng=np.random.default_rng(random_seed))
        tte_bkgd = tte_sim.to_tte(t_start, t_stop)
        tte_gbm_bkgd = GbmTte.merge([tte_demo, tte_bkgd])
        tte_gbm_bkgd.write(filename=bkgd_filename, directory=event_file_path.parent, overwrite=True)
        src_path = event_file_path.parent / src_filename
        bkgd_path = event_file_path.parent / bkgd_filename

    return src_path, bkgd_path



if __name__ == '__main__':
    gauss_params = (0.5, 0.0, 0.2)
    tri_par = (0.01, -1., 0.0, 1.)
    const_par = (1, )
    fred_par = (0.5, 0.0, 0.05, 0.1)  # amp, tstart, trise, tdecay
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
    #results = gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par, random_seed=37)
    #gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=triangular, func_par=tri_par, back_func=constant, back_func_par=const_par)
    #gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=fred, func_par=tri_par, back_func=constant, back_func_par=const_par)
    
    results = gen_pulse(
        t_start=-10.0,
        t_stop=10.0,
        func=gaussian,
        func_par=gauss_params,
        back_func=constant,
        back_func_par=const_par,
        bin_width=0.0001,
        fig_name='test_gaussian.png',
        simulation=True
    )

    print(f"Generated pulse with times: {results[0]}, counts: {results[1]},source max cps: {results[2]}, background avg cps: {results[3]}, SNR: {results[4]}")



"""

# 1. Define the parameters for your complex model
pulse_params = {
    'background_level': 100.0,
    'main_amplitude': 5000.0,
    'pulse_list': [
        ('fred', (1.0, 0.0, 0.1, 1.0)),      # A strong FRED pulse
        ('gaussian', (0.5, 2.5, 0.2)) # A weaker Gaussian pulse later
    ]
}

# 2. Call the advanced simulation function
# Notice func_par now just contains the single dictionary
times, counts, _, _, _ = gen_pulse_advanced(
    t_start=-5.0,
    t_stop=10.0,
    func=generate_rate_function,
    func_par=(pulse_params,), # Pass the dictionary as a tuple
    back_func=None,           # Background is handled inside the new function
    bin_width=0.01,
    source_base_rate=1.0,     # The rates are now absolute, so base rate is 1
    fig_name='complex_grb_pulse.png'
)
"""