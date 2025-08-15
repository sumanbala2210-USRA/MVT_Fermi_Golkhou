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

#warnings.simplefilter('ignore', VerifyWarning)

MAX_WORKERS = os.cpu_count() - 2
const_par = (1, )
fred_par = (0.5, 0.0, 0.05, 0.1)
gauss_params = (.5, 0.0, 0.1)



from typing import Dict, Any, Tuple, Callable

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


def generate_pulse_function(t: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Generates only the pulse components from a parameter dictionary.
    This function excludes the background level.

    Args:
        t (np.ndarray): Array of times.
        params (Dict[str, Any]): A dictionary containing 'main_amplitude' and 'pulse_list'.

    Returns:
        np.ndarray: An array containing the sum of all generated pulses.
    """
    # Initialize an array of zeros to accumulate the pulses
    pulses = np.zeros_like(t, dtype=float)

    # Ensure 'pulse_list' exists to avoid errors
    if 'pulse_list' not in params:
        return pulses

    # Iterate through each pulse defined in the parameter list
    for pulse_def in params.get('pulse_list', []):
        pulse_type, (rel_amp, *time_params) = pulse_def

        # Calculate the absolute amplitude based on the main amplitude
        abs_amp = params.get('main_amplitude', 1.0) * rel_amp
        full_params = (abs_amp, *time_params)

        # Add the corresponding pulse shape to the total
        if pulse_type == 'fred':
            pulses += fred(t, *full_params)
        elif pulse_type == 'norris':
            pulses += norris(t, *full_params)
        elif pulse_type == 'gaussian':
            pulses += gaussian(t, *full_params)
        elif pulse_type == 'lognormal':
            pulses += lognormal(t, *full_params)
        else:
            # Optionally, warn the user about an unknown pulse type
            print(f"Warning: Unknown pulse type '{pulse_type}' encountered. Skipping.")

    return pulses

# Assume generate_pulse_function, constant, and create_and_save_plot are defined

def gen_pulse_advanced(
    t_start=-10.0,
    t_stop=10.0,
    func=None,
    func_par=(),
    back_func=None,
    back_func_par=(),
    bin_width=0.0001,
    source_base_rate=1000.0,
    background_base_rate=1000.0,
    fig_name=None,
    plot_flag=True,
    random_seed=42,
    # NEW: Control which plot to generate
    plot_type: str = 'decomposed',
    plot_rebin_factor: Optional[int] = 100,
    title: Optional[str] = None
):
    """
    Generates a high-fidelity pulse profile and can create one of two plot types.
    - 'diagnostic': Compares simulated rate to the ideal model rates.
    - 'decomposed': Shows the simulated total and source-only components.

    Returns:
        tuple: (times, total_counts, source_only_counts, src_max_cps, back_avg_cps, SNR)
    """
    np.random.seed(random_seed)
    search_timescales = [0.010, 0.016, 0.032, 0.064, 0.128]

    # --- 1. TTE Event Generation (Same as before) ---
    def total_rate_func(t):
        source_rate = func(t, *func_par) * source_base_rate if func else 0
        background_rate = back_func(t, *back_func_par) * background_base_rate if back_func else 0
        return source_rate + background_rate

    grid_resolution = bin_width / 10.0
    grid_times = np.arange(t_start, t_stop, grid_resolution)
    rate_on_grid = total_rate_func(grid_times)
    cumulative_counts = spi.cumulative_trapezoid(rate_on_grid, grid_times, initial=0)
    total_expected_counts = cumulative_counts[-1]
    num_events = np.random.poisson(total_expected_counts)
    random_counts = np.random.uniform(0, total_expected_counts, num_events)
    event_times = np.interp(random_counts, cumulative_counts, grid_times)

    # --- 2. Bin Total Events (Same as before) ---
    bins = np.arange(t_start, t_stop + bin_width, bin_width)
    total_counts, _ = np.histogram(event_times, bins=bins)
    times = bins[:-1] + bin_width / 2.0
    
    # --- NEW: 3. Probabilistic Splitting of Events ---
    # At the precise time of each event, what was the probability it was a source event?
    source_rate_at_events = func(event_times, *func_par) * source_base_rate
    background_rate_at_events = back_func(event_times, *back_func_par) * background_base_rate
    total_rate_at_events = source_rate_at_events + background_rate_at_events
    
    # Calculate probability, avoiding division by zero
    p_source = np.divide(
        source_rate_at_events, total_rate_at_events,
        out=np.zeros_like(source_rate_at_events), where=total_rate_at_events != 0
    )
    
    # For each event, "roll a die" to classify it as source or background
    is_source_event = np.random.rand(num_events) < p_source
    source_event_times = event_times[is_source_event]
    
    # Bin the source-only events
    source_only_counts, _ = np.histogram(source_event_times, bins=bins)

    # --- 4. Calculate Metrics (Same as before) ---
    source_rate_ideal = func(times, *func_par) * source_base_rate
    background_rate_ideal = back_func(times, *back_func_par) * background_base_rate
    src_max_cps = np.max(source_rate_ideal)
    back_avg_cps = np.mean(background_rate_ideal)

    snr_results_dict = _calculate_multi_timescale_snr(
        source_counts=source_only_counts,
        sim_bin_width=bin_width,
        back_avg_cps=back_avg_cps,
        search_timescales=search_timescales
    )



        # Find the single best SNR for the main title
    max_snr = max(snr_results_dict.values()) if snr_results_dict else 0
    int_snr = int(np.round(max_snr))
    # --- 5. Optional Plotting (Now with choices!) ---
    if plot_flag:
        annotation_text = _format_params_for_annotation(func, func_par)

        snr_annotation_parts = []
        for ts, snr_val in snr_results_dict.items():
            # Format timescale in ms for the label, e.g., 0.016 -> s16
            label = f"S$_{{{int(ts * 1000)}}}$"
            # Format SNR value to one decimal place
            value = f"{snr_val:.1f}"
            snr_annotation_parts.append(f"{label}={value}")
        
        # Join the parts with a semicolon
        SNR_text = "; ".join(snr_annotation_parts)
        #title = "; ".join(snr_annotation_parts)
        SNR_text = "; ".join(snr_annotation_parts)
        
        if title is None:
            title = "Sim " + SNR_text
        else:
            title = title + "; " + SNR_text

        if fig_name is None:
            fig_name = f"simulation_plot_{int_snr}.png"
        rebin = plot_rebin_factor is not None and plot_rebin_factor > 1
        if rebin:
            factor = plot_rebin_factor
            end = (len(total_counts) // factor) * factor
            
            # Re-bin the data by summing counts
            plot_total_counts = np.sum(total_counts[:end].reshape(-1, factor), axis=1)
            plot_source_counts = np.sum(source_only_counts[:end].reshape(-1, factor), axis=1)
            
            # Calculate the new time bins for the plot
            plot_bin_width = bin_width * factor
            plot_times = np.arange(t_start, t_start + len(plot_total_counts) * plot_bin_width, plot_bin_width) + plot_bin_width / 2.0
        else:
            # If not rebinning, use the original simulation data
            plot_total_counts = total_counts
            plot_source_counts = source_only_counts
            plot_times = times
            plot_bin_width = bin_width


        # --- Call the plotter with the (possibly rebinned) data ---
        if plot_type == 'diagnostic':
            # Note: For diagnostic plots, we still show the original ideal rates for comparison
            plot_data = [
                {'x': plot_times, 'y': plot_total_counts / plot_bin_width, 'label': f'Simulated Rate (Binned to {plot_bin_width*1000:.1f}ms)', 'color': 'black'},
                {'x': times, 'y': source_rate_ideal + background_rate_ideal, 'label': 'Ideal Total Rate', 'color': 'red', 'lw': 1},
            ]
            create_and_save_plot(output_path=Path(fig_name), title=title, annotation_text=annotation_text,
                                 xlabel="Time (s)", ylabel="Rate (counts/sec)", plot_data=plot_data, xlim=(t_start, t_stop))
        
        elif plot_type == 'decomposed':
            ideal_background_counts = back_avg_cps * plot_bin_width
            plot_data = [
                {'x': plot_times, 'y': plot_total_counts, 'label': 'Total Signal (Simulated)', 'color': 'rosybrown', 'fill_alpha': 0.6},
                {'x': plot_times, 'y': plot_source_counts, 'label': 'Source Signal (Simulated)', 'color': 'darkgreen', 'fill_alpha': 0.4}
            ]
            h_lines_data = [{'y': ideal_background_counts, 'label': f'Ideal Background', 'color': 'k'}]
            create_and_save_plot(output_path=Path(fig_name), title=title, annotation_text=annotation_text,
                                 xlabel="Time (s)", ylabel=f"Counts per {plot_bin_width*1000:.1f} ms Bin",
                                 plot_data=plot_data, h_lines=h_lines_data, xlim=(t_start, t_stop))


    # UPDATED return statement
    return times, total_counts, int(np.round(src_max_cps)), int(np.round(back_avg_cps)), int_snr

def gen_GBM_pulse(trigger_number,
                  det,
                  angle=0,
                  t_start=-10.0,
                  t_stop=10.0,
                  func = None,
                  func_par = (0,0,0,0),
                  back_func = constant,
                  back_func_par = (1.,),
                  bkgd_times=[(-20.0, -5.0), (75.0, 200.0)],
                  en_lo=8.0,
                  en_hi=900.0,
                  bin_width = 0.0001,
                  random_seed=42,
                  fig_name=None,
                  plot_flag=False):
    energy_range_nai = (en_lo, en_hi)
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
    #tte = GbmTte.open(f'glg_tte_{det}_bn{trigger_number}_v00.fit')
    #rsp2 = GbmRsp2.open(f'glg_cspec_{det}_bn{trigger_number}_v03.rsp2')

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
    tte_sim = TteSourceSimulator(rsp, Band(), band_params, func, func_par, deadtime=1e-6, rng=np.random.default_rng(random_seed))
    
    tte_src = tte_sim.to_tte(t_start, t_stop)
    
    # background simulation
    #tte_sim = TteBackgroundSimulator(spec_bkgd, 'Gaussian', quadratic, quadratic_params)
    tte_sim = TteBackgroundSimulator(spec_bkgd, 'Gaussian', back_func, back_func_par, deadtime=1e-6, rng=np.random.default_rng(random_seed))
    tte_bkgd = tte_sim.to_tte(t_start, t_stop)
    
    # merge the background and source
    #tte_total = GbmTte.merge([tte_bkgd, tte_src])
    #tte_total = GbmTte.merge([tte_src, tte_bkgd])
    tte_total = PhotonList.merge([tte_src, tte_bkgd])
    src_max = 1
    back_avg = 1
    SNR = 1

    try:
        plot_bw = 0.1
        phaii = tte_total.to_phaii(bin_by_time, plot_bw)
        
        phaii = tte_total.to_phaii(bin_by_time, plot_bw)
        phii_src = tte_src.to_phaii(bin_by_time, plot_bw)
        phii_bkgd = tte_bkgd.to_phaii(bin_by_time, plot_bw)
        lc_tot = phaii.to_lightcurve(energy_range=energy_range_nai)
        lc_src = phii_src.to_lightcurve(energy_range=energy_range_nai)
        lc_bkgd = phii_bkgd.to_lightcurve(energy_range=energy_range_nai)

        """
        lcplot = Lightcurve(data=lc_tot, background=lc_bkgd)
        _= lcplot.add_selection(lc_src)
        lcplot.selections[1].color = 'pink'
        """

        src_max = max(lc_src.counts)
        back_avg = np.mean(lc_bkgd.counts)
        SNR = src_max / np.sqrt(back_avg)
    
    except Exception as e:
        print(f"Error during SNR computing")
        lc_tot = None
        lc_src = None
        lc_bkgd = None


    if plot_flag:
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

            if fig_name is None:
                fig_name = f'lc_{trigger_number}_n{det}_{angle}deg.png'

                #plt.show()
            
            for i in range(len(func_par)):
                if i == 0:
                    func_txt = f"amp: {func_par[i]:.2f},"
                elif i == 2:
                    func_txt += f" rise/sig/tp: {func_par[i]:.2f},"
                elif i == 3:
                    func_txt += f" decay/stop: {func_par[i]:.2f}"
            plt.title(f'Bn{trigger_number}, n{det}, {angle}deg, back {back_func_par[0]},\n {func_txt}', fontsize=10)

            plt.savefig(fig_name, dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error during plotting: {e}")
            lcplot = None
        

    phaii_hi = tte_total.to_phaii(bin_by_time, bin_width)
    phaii = phaii_hi.slice_energy(energy_range_nai)
    data = phaii.to_lightcurve()
    return data.centroids, data.counts, int(np.round(src_max)), int(np.round(back_avg)), int(np.round(SNR))



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