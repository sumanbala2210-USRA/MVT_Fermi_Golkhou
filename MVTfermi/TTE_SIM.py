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
from sim_functions import constant2, gaussian2, triangular, fred, constant, norris, gaussian
# Suppress a common FITS warning
import concurrent.futures

#warnings.simplefilter('ignore', VerifyWarning)

MAX_WORKERS = os.cpu_count() - 2
const_par = (1, )
fred_par = (0.5, 0.0, 0.05, 0.1)
gauss_params = (.5, 0.0, 0.1)




def generate_tte_events(
    t_start=-10.0,
    t_stop=10.0,
    func=None,
    func_par=(),
    back_func=constant,
    back_par=()
):
    """Generates an unbinned list of TTE event arrival times.

    Args:
        t_start (float): The start time of the observation window.
        t_stop (float): The end time of the observation window.
        func (function): The source pulse rate shape (in cps).
        func_kwargs (dict): Keyword arguments for the source function.
        back_func (function): The background rate shape (in cps).
        back_kwargs (dict): Keyword arguments for the background function.

    Returns:
        np.array: A sorted array of event arrival times (TTE data).
    """
    # Define the total rate function (source + background)
    def total_rate_func(t):
        source_rate = func(t, *func_par) if func else 0
        background_rate = back_func(t, *back_par) if back_func else 0
        return source_rate + background_rate

    # 1. Calculate the total expected number of events by integrating the rate
    total_expected_counts, _ = spi.quad(total_rate_func, t_start, t_stop)

    # 2. Draw the actual number of events from a Poisson distribution
    num_events = np.random.poisson(total_expected_counts)

    # 3. Distribute events in time using the inverse transform sampling method
    # This is a standard and efficient way to generate events for a given rate model.
    events = []
    while len(events) < num_events:
        # Propose a random time and test against the peak rate
        # This "acceptance-rejection" method is robust for complex shapes.
        # We need a reasonable estimate of the peak rate for efficiency.
        peak_rate_estimate = total_rate_func(func_kwargs.get('center', (t_start+t_stop)/2)) * 1.2
        
        # Generate a batch of candidate events for efficiency
        batch_size = int(num_events - len(events))
        candidate_times = np.random.uniform(t_start, t_stop, size=batch_size)
        candidate_rates = np.random.uniform(0, peak_rate_estimate, size=batch_size)
        
        # Accept candidates that fall under the true rate curve
        actual_rates = total_rate_func(candidate_times)
        accepted_mask = candidate_rates < actual_rates
        events.extend(candidate_times[accepted_mask])

    # Ensure we have exactly the right number of events and sort them
    final_events = np.sort(np.array(events[:num_events]))

    return final_events


# You would still have your pulse and background functions (gaussian, constant, etc.) here.

def generate_tte_events_faster(
    t_start=-10.0,
    t_stop=10.0,
    func=None,
    func_par=(),
    back_func=constant,
    back_par={'cps': 100},
    grid_resolution=0.0001
):
    """
    Generates TTE event times using the faster integral inversion method.

    Args:
        t_start (float): Start time of the observation window.
        t_stop (float): End time of the observation window.
        func (function): Source pulse rate shape (in cps).
        func_kwargs (dict): Keyword arguments for the source function.
        back_func (function): Background rate shape (in cps).
        back_kwargs (dict): Keyword arguments for the background function.
        grid_resolution (float): The time resolution for the internal integration grid.

    Returns:
        np.array: A sorted array of event arrival times.
    """
    # 1. Create a fine time grid for numerical integration
    # For high-intensity pulses, this grid can be coarser than the bin width.
    grid_times = np.arange(t_start, t_stop, grid_resolution)
    
    # 2. Define and calculate the rate on the grid
    def total_rate_func(t):
        source_rate = func(t, *func_par) if func else 0
        background_rate = back_func(t, *back_par) if back_func else 0
        return source_rate + background_rate

    rate_on_grid = total_rate_func(grid_times)

    # 3. Create the cumulative event count curve (integrated rate)
    cumulative_counts = spi.cumulative_trapezoid(rate_on_grid, grid_times, initial=0)
    total_expected_counts = cumulative_counts[-1]

    # 4. Determine the total number of events to simulate
    num_events = np.random.poisson(total_expected_counts)

    # 5. Generate random uniform values along the y-axis (counts)
    random_counts = np.random.uniform(0, total_expected_counts, num_events)

    # 6. Invert the cumulative function using fast interpolation
    # This maps the random counts back to the time axis.
    event_times = np.interp(random_counts, cumulative_counts, grid_times)

    # Return the sorted list of event times
    return np.sort(event_times)


def bin_events_to_lightcurve(
    event_times,
    t_start=-10.0,
    t_stop=10.0,
    bin_width=0.1
):
    """Bins a list of TTE events into a light curve.

    Args:
        event_times (np.array): A sorted array of event arrival times.
        t_start (float): The start time for binning.
        t_stop (float): The end time for binning.
        bin_width (float): The desired time resolution of the light curve.

    Returns:
        tuple: A tuple containing:
            - times (np.array): The center of the time bins.
            - counts (np.array): The number of counts in each bin.
    """
    # Define the time bins
    bins = np.arange(t_start, t_stop + bin_width, bin_width)
    
    # Use numpy.histogram to efficiently count events in each bin
    counts, _ = np.histogram(event_times, bins=bins)
    
    # Calculate the center of each time bin for plotting
    times = bins[:-1] + bin_width / 2.0
    
    return times, counts

def gen_pulse(
    t_start=-10.0,
    t_stop=10.0,
    func=None,
    func_par=(),
    back_func=None,
    back_func_par=(),
    bin_width=0.0001,
    source_base_rate=1000.0,   # NEW: Base rate for the source pulse
    background_base_rate=1000.0, # NEW: Base rate for the background
    fig_name=None,
    random_seed=42,
    simulation=True
):
    """
    Generates a pulse profile where amplitude parameters are scalers.

    Args:
        t_start (float): The start time of the observation window.
        t_stop (float): The end time of the observation window.
        func (function): The function for the source pulse shape.
        func_par (tuple): Parameters for the source function, e.g.,
                          (amp_scaler, center, sigma). amp_scaler=1 gives a
                          peak rate of source_base_rate.
        back_func (function): The function for the background shape.
        back_func_par (tuple): Parameters for the background function, e.g.,
                               (amp_scaler,). amp_scaler=1 gives a
                               background rate of background_base_rate.
        bin_width (float): The width of each time bin in seconds.
        source_base_rate (float): The rate in cps for a source amp_scaler of 1.
        background_base_rate (float): The rate in cps for a background amp_scaler of 1.
        fig_name (str, optional): If provided, saves a plot to this filename.
        simulation (bool): If True, performs a Poisson simulation for counts.

    Returns:
        tuple: (times, counts, src_max_cps, back_avg_cps, SNR)
    """
    # 1. Create time bins and their centers
    time_bins = np.arange(t_start, t_stop, bin_width)
    times = time_bins + bin_width / 2.0

    # 2. Calculate the unitless shape of the source and background
    background_shape = np.zeros_like(times)
    if back_func is not None:
        background_shape = back_func(times, *back_func_par)

    source_shape = np.zeros_like(times)
    if func is not None:
        source_shape = func(times, *func_par)

    # 3. Scale the shapes by the base rates to get rates in cps
    background_rate_cps = background_shape * background_base_rate
    source_rate_cps = source_shape * source_base_rate
    
    # 4. Convert rates (cps) to expected counts per bin
    total_expected_counts = (source_rate_cps + background_rate_cps) * bin_width

    # 5. Calculate metrics from the cps rates
    back_avg_cps = np.mean(background_rate_cps)
    src_max_cps = np.max(source_rate_cps)
    snr = src_max_cps / np.sqrt(back_avg_cps) if back_avg_cps > 0 else np.inf

    # 6. Create the final light curve
    #if simulation:
    np.random.seed(random_seed)
    counts = np.random.poisson(total_expected_counts)#, random_seed=random_seed)


    # 7. Optional plotting
    if simulation:
        plt.figure(figsize=(12, 7))
        #plt.step(times, counts, where='mid', label='Simulated Rate (cps)')
        plt.plot(times, np.random.poisson(source_rate_cps), 'b-', alpha=0.8, label='Source Rate (cps)')
        plt.plot(times, np.random.poisson(background_rate_cps), 'g--', alpha=0.8, label='Background Rate (cps)')
        #plt.plot(times, source_rate_cps + background_rate_cps, 'r--',
        #         alpha=0.8, label='Ideal Rate (cps)')
        plt.plot(times, np.random.poisson(source_rate_cps + background_rate_cps), 'r--',
                 alpha=0.8, label='Total Rate (cps)')
        plt.xlabel("Time (s)")
        plt.ylabel("Rate (counts/sec)")
        plt.title(f"Simulated Pulse Profile (SNR ≈ {int(np.round(snr))})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        if fig_name is None:
            fig_name = f'lc_{func.__name__}_pulse.png'
        plt.savefig(fig_name)
        plt.close()
        #print(f"✅ Plot saved to {fig_name}")

    return times, counts, int(np.round(src_max_cps)), int(np.round(back_avg_cps)), int(np.round(snr))



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
    random_seed=42,
    simulation=True
):
    """
    Generates a high-fidelity pulse profile using the TTE event generation method.

    This function simulates individual event arrival times based on the integrated
    rate function and then bins them, providing a more physically accurate light
    curve than the direct-to-counts method.

    Args:
        t_start (float): The start time of the observation window.
        t_stop (float): The end time of the observation window.
        func (function): The function for the source pulse shape (unitless).
        func_par (tuple): Parameters for the source function.
        back_func (function): The function for the background shape (unitless).
        back_func_par (tuple): Parameters for the background function.
        bin_width (float): The width of each time bin in seconds.
        source_base_rate (float): The rate in cps for a source shape of 1.
        background_base_rate (float): The rate in cps for a background shape of 1.
        fig_name (str, optional): If provided, saves a plot to this filename.
        simulation (bool): If True, performs Poisson simulation and plotting.

    Returns:
        tuple: (times, counts, src_max_cps, back_avg_cps, SNR)
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # --- 1. TTE Event Generation using Integral Inversion ---

    # Define the total rate function in counts per second (cps)
    def total_rate_func(t):
        source_rate = 0
        if func is not None:
            source_rate = func(t, *func_par) * source_base_rate
        
        background_rate = 0
        if back_func is not None:
            background_rate = back_func(t, *back_func_par) * background_base_rate
            
        return source_rate + background_rate

    # Create a fine grid for accurate numerical integration.
    # The grid should be much finer than the final bin width.
    grid_resolution = bin_width / 10.0
    grid_times = np.arange(t_start, t_stop, grid_resolution)
    
    # Calculate the rate on the fine grid
    rate_on_grid = total_rate_func(grid_times)

    # Create the cumulative event count curve (the integrated rate)
    cumulative_counts = spi.cumulative_trapezoid(rate_on_grid, grid_times, initial=0)
    total_expected_counts = cumulative_counts[-1]

    # Determine the total number of events to simulate from a single Poisson draw
    num_events = np.random.poisson(total_expected_counts)

    # Generate random uniform values along the y-axis (the counts axis)
    random_counts = np.random.uniform(0, total_expected_counts, num_events)

    # Invert the cumulative function to map random counts back to the time axis
    event_times = np.interp(random_counts, cumulative_counts, grid_times)

    # --- 2. Bin Events into a Light Curve ---

    # Define the final time bins for the output light curve
    bins = np.arange(t_start, t_stop + bin_width, bin_width)
    
    # Use numpy.histogram to efficiently count events in each bin
    counts, _ = np.histogram(event_times, bins=bins)
    
    # Calculate the center of each time bin for the final x-axis
    times = bins[:-1] + bin_width / 2.0
    
    # --- 3. Calculate Metrics from Ideal Rates ---

    # For consistent metrics, calculate them from the ideal model, not the simulation
    source_rate_ideal = np.zeros_like(times)
    if func is not None:
        source_rate_ideal = func(times, *func_par) * source_base_rate

    background_rate_ideal = np.zeros_like(times)
    if back_func is not None:
        background_rate_ideal = back_func(times, *back_func_par) * background_base_rate

    src_max_cps = np.max(source_rate_ideal)
    back_avg_cps = np.mean(background_rate_ideal)
    snr = src_max_cps / np.sqrt(back_avg_cps) if back_avg_cps > 0 else np.inf

    # --- 4. Optional Plotting ---
    if simulation and fig_name is not None:
        plt.figure(figsize=(12, 7))
        
        # Plot the final simulated light curve (as a rate)
        plt.step(times, counts / bin_width, where='mid', label=f'Simulated Rate (Binned to {bin_width}s)', color='black', lw=1.5)
        
        # Plot the ideal underlying rate functions for comparison
        plt.plot(times, source_rate_ideal, 'b-', alpha=0.7, label='Ideal Source Rate')
        plt.plot(times, background_rate_ideal, 'g--', alpha=0.7, label='Ideal Background Rate')
        plt.plot(times, source_rate_ideal + background_rate_ideal, 'r--', alpha=0.7, label='Ideal Total Rate')
        
        plt.xlabel("Time (s)")
        plt.ylabel("Rate (counts/sec)")
        plt.title(f"Simulated Pulse Profile (SNR ≈ {int(np.round(snr))})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(fig_name)
        plt.close()

    return times, counts, int(np.round(src_max_cps)), int(np.round(back_avg_cps)), int(np.round(snr))





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
                  simulation=False):
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


    if simulation:
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