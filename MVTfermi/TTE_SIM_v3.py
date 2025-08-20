"""
# TTE_SIM_v3.py
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th June 2025: Including Fermi GBM simulation of same functions.
14th August 2025: Added GBM and normal functions together.
16th August 2025: Improved to separately write event files for each pulse type.

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

# --- GDT Core Imports ---
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.plot.lightcurve import Lightcurve
#from gdt.core.simulate.profiles import tophat, constant, norris, quadratic, linear, gaussian
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
from sim_functions import constant2, gaussian2, triangular, fred, constant, norris, gaussian, lognormal
# Suppress a common FITS warning
import concurrent.futures
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)



# lib.py
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

#warnings.simplefilter('ignore', VerifyWarning)

const_par = (1, )
fred_par = (0.5, 0.0, 0.05, 0.1)
gauss_params = (.5, 0.0, 0.1)



def calculate_adaptive_simulation_params(pulse_shape: str, params: Dict) -> Dict:
    """
    Calculates optimal t_start, t_stop, and grid_resolution based on pulse parameters.
    """
    t_start, t_stop, grid_res = None, None, None
    padding = 2.0  # Seconds of padding before and after the pulse

    if pulse_shape == 'gaussian':
        sigma = params['sigma']
        center = params['center_time']
        # The pulse is significant within ~5-sigma of the center
        t_start = center - 5 * sigma - padding
        t_stop = center + 5 * sigma + padding
        # Rule of Thumb: Grid must be ~10x finer than the narrowest feature
        grid_res = sigma / 10.0

    elif pulse_shape == 'lognormal':
        sigma = params['sigma']
        center = params['center_time']
        # Use a similar 5-sigma rule, but in log-space
        t_start = center * np.exp(-5 * sigma) - padding
        t_stop = center * np.exp(5 * sigma) + padding
        # A rough characteristic timescale for lognormal
        timescale = min(center, sigma * center)
        grid_res = timescale / 10.0

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
        t_start = start - padding
        t_stop = peak_time_approx + 10 * t_decay + padding

    elif pulse_shape == 'triangular':
        width = params['width']
        center = params['center_time']
        peak_ratio = params['peak_time_ratio']
        # Start and stop are explicitly defined by the parameters
        t_start = center - (width * peak_ratio) - padding
        t_stop = t_start + width + padding + (padding*2)
        # Narrowest feature is the shorter of the rise or fall time
        rise_duration = width * peak_ratio
        fall_duration = width * (1.0 - peak_ratio)
        grid_res = min(rise_duration, fall_duration) / 10.0
    
    # --- Safety Net ---
    # Ensure grid resolution is within reasonable bounds to prevent memory errors
    # or inaccurate simulations.
    if grid_res is not None:
        grid_res = np.clip(grid_res, a_min=1e-6, a_max=0.001) # e.g., 1µs to 1ms
    else:
        # Fallback for unknown shapes
        t_start, t_stop, grid_res = -10.0, 10.0, 0.0001
        
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
    source_counts: np.ndarray,
    sim_bin_width: float,
    back_avg_cps: float,
    search_timescales: List[float]
) -> Dict[float, float]:
    """
    Calculates the SNR for a list of timescales and returns the results
    as a dictionary.
    """
    snr_results = {} # Use a dictionary for clarity

    for timescale in search_timescales:
        try:
            factor = max(1, int(round(timescale / sim_bin_width)))
            end = (len(source_counts) // factor) * factor
            if end == 0:
                snr_results[timescale] = 0.0
                continue
            
            rebinned_counts = np.sum(source_counts[:end].reshape(-1, factor), axis=1)
            signal = np.max(rebinned_counts)
            background_counts_in_bin = back_avg_cps * timescale
            noise = np.sqrt(background_counts_in_bin)
            snr = signal / noise if noise > 0 else np.inf
            snr_results[timescale] = snr
        except Exception:
            snr_results[timescale] = 0.0 # Assign 0 on error
            continue

    return snr_results

def create_and_save_plot(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    plot_data: List[Dict[str, Any]],
    h_lines: Optional[List[Dict[str, Any]]] = None,
    # ... other styling arguments
    annotation_text: Optional[str] = None,
    figsize: tuple = (12, 6),
    style: str = 'seaborn-v0_8-whitegrid',
    title_fontsize: int = 16,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
    grid_alpha: float = 0.6,
    xlim: Optional[tuple] = None,
    ylim_bottom: float = 0
):
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)
    for data in plot_data:
        ax.step(data['x'], data['y'], where='mid', label=data.get('label'),
                color=data.get('color', 'black'), linewidth=data.get('lw', 1.5))
        if 'fill_alpha' in data:
            ax.fill_between(data['x'], data['y'], step="mid",
                            color=data.get('color', 'gray'), alpha=data.get('fill_alpha', 0.5))
    if h_lines:
        for line in h_lines:
            ax.axhline(y=line['y'], color=line.get('color', 'k'),
                        linestyle=line.get('linestyle', '--'), linewidth=line.get('lw', 1.5),
                        label=line.get('label'), zorder=line.get('zorder', 3))
    ax.set_title(title, fontsize=title_fontsize, pad=10)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=grid_alpha)
    if xlim:
        ax.set_xlim(*xlim)
        # --- NEW: Add the annotation text box if text is provided ---
    if annotation_text:
        # Define the style of the box
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
        # Place the text box in the top-right corner of the plot area
        ax.text(0.97, 0.97, annotation_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=props)
    ax.set_ylim(bottom=ylim_bottom)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    plt.close('all')
    #print(f"Plot saved to: {output_path}")


# Assume generate_pulse_function, constant, and create_and_save_plot are defined
# lib.py


# NOTE: This library would also contain your other helpers:
# _calculate_multi_timescale_snr, _format_params_for_annotation, create_and_save_plot

def generate_function_events(
    event_file_path: Path,
    func: Callable,
    func_par: Tuple,
    back_func: Callable,
    back_func_par: Tuple,
    params: Dict[str, Any]
):
    """
    Runs a full TTE simulation, probabilistically splits the events into
    source and background, and saves them to a compressed .npz file.
    """
    # Unpack necessary parameters from the dictionary
    t_start = params['t_start']
    t_stop = params['t_stop']
    random_seed = params['random_seed']
    source_base_rate = params.get('source_base_rate', 1000.0)
    background_base_rate = params.get('background_base_rate', 1000.0)
    grid_resolution = params.get('grid_resolution', 0.0001) # Use a fixed, fine resolution for the integral

    np.random.seed(random_seed)

    # 1. Simulate the total event stream
    def total_rate_func(t):
        source_rate = func(t, *func_par) * source_base_rate if func else 0
        background_rate = back_func(t, *back_func_par) * background_base_rate if back_func else 0
        return source_rate + background_rate

    grid_times = np.arange(t_start, t_stop, grid_resolution)
    rate_on_grid = total_rate_func(grid_times)
    cumulative_counts = np.cumsum(rate_on_grid) * grid_resolution
    total_expected_counts = cumulative_counts[-1] if len(cumulative_counts) > 0 else 0
    num_events = np.random.poisson(total_expected_counts)
    random_counts = np.random.uniform(0, total_expected_counts, num_events)
    total_event_times = np.interp(random_counts, cumulative_counts, grid_times)

    # 2. Probabilistically split events into source and background
    source_rate_at_events = func(total_event_times, *func_par) * source_base_rate
    background_rate_at_events = back_func(total_event_times, *back_func_par) * background_base_rate
    total_rate_at_events = source_rate_at_events + background_rate_at_events

    p_source = np.divide(source_rate_at_events, total_rate_at_events, 
                         out=np.zeros_like(total_rate_at_events), 
                         where=total_rate_at_events > 0)
    
    is_source_event = np.random.rand(num_events) < p_source
    
    source_event_times = np.sort(total_event_times[is_source_event])
    background_event_times = np.sort(total_event_times[~is_source_event])

    # 3. Save the classified events to a single .npz file
    np.savez_compressed(
        event_file_path,
        source_events=source_event_times,
        background_events=background_event_times,
        params=params # Save the simulation parameters for later reference
    )



def generate_gbm_events(
        event_file_path: Path,
        func: Callable,
        func_par: Tuple,
        back_func: Callable,
        back_func_par: Tuple,
        params: Dict[str, Any]):

    det = params['det']
    trigger_number = params['trigger_number']
    angle = params['angle']
    en_lo = params['en_lo']
    en_hi = params['en_hi']
    t_start = params['t_start']
    t_stop = params['t_stop']
    random_seed = params['random_seed']
    grid_resolution = params.get('grid_resolution', 0.0001) # Use a fixed, fine

    energy_range_nai = (en_lo, en_hi)
    bkgd_times = [(-20.0, -5.0), (75.0, 200.0)]

    # Fixed spectral Model
    band_params = (0.1, 300.0, -1.0, -2.5)

    #tte = GbmTte.open('glg_tte_n6_bn250612519_v00.fit')
    folder_path = os.path.join(os.getcwd(), 'data_rmf')

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

    # Open the files
    tte = GbmTte.open(tte_file)
    rsp2 = GbmRsp2.open(rsp2_file)

    # bin to 1.024 s resolution, reference time is trigger time
    phaii = tte.to_phaii(bin_by_time, 1.024, time_ref=0.0)
    bkgd_times = bkgd_times
    backfitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_times)
    
    backfitter.fit(order=1)
    bkgd = backfitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
    
    select_time = (t_start, t_stop)
    # the background model integrated over the source selection time
    spec_bkgd = bkgd.integrate_time(*select_time)
    rsp = rsp2.extract_drm(atime=np.average(select_time))
    
    # source simulation
    tte_sim = TteSourceSimulator(rsp, Band(), band_params, func, func_par, deadtime=1e-6, sample_period=grid_resolution, rng=np.random.default_rng(random_seed))

    tte_src = tte_sim.to_tte(t_start, t_stop)
    
    # background simulation
    tte_sim = TteBackgroundSimulator(spec_bkgd, 'Gaussian', back_func, back_func_par, deadtime=1e-6, sample_period=grid_resolution, rng=np.random.default_rng(random_seed))
    tte_bkgd = tte_sim.to_tte(t_start, t_stop)

    tte_demo = tte.slice_time([-50,-49.99])
    tte_gbm_src = GbmTte.merge([tte_demo, tte_src])
    tte_gbm_bkgd = GbmTte.merge([tte_demo, tte_bkgd])

    # Use the .name property to get the descriptive base filename
    base_filename = event_file_path.name
    
    # Construct the new FITS filenames
    src_filename = f"{base_filename}_src.fits"
    bkgd_filename = f"{base_filename}_bkgd.fits"

    # Save the files to the correct directory with the new names
    tte_gbm_src.write(filename=src_filename, directory=event_file_path.parent, overwrite=True)
    tte_gbm_bkgd.write(filename=bkgd_filename, directory=event_file_path.parent, overwrite=True)




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
    params = {
        'trigger_number': '250709653',
        'det': '6',
        'angle': 10.73,
        'en_lo': 8,
        'en_hi': 900,
        't_start': -2.0,
        't_stop': 2.0,
        'grid_resolution': 0.0001,
        'random_seed': 37,
    }
    os.getcwd()
    event_file_path = Path(f"simulated_gbm_events_{params['trigger_number']}_{params['det']}_{params['angle']}")
    generate_gbm_events( event_file_path=event_file_path, func=triangular, func_par=tri_par, back_func=constant, back_func_par=const_par, params=params)
    #gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=fred, func_par=tri_par, back_func=constant, back_func_par=const_par)
    



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