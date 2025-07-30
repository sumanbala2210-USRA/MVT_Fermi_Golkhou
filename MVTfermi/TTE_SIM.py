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
import gdt.core
import yaml  # Import the JSON library for parameter logging
import warnings
from astropy.io.fits.verify import VerifyWarning

# --- GDT Core Imports ---
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.plot.lightcurve import Lightcurve
from gdt.core.simulate.profiles import tophat, constant, norris, quadratic, gaussian, triangular,linear
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
# Suppress a common FITS warning
warnings.simplefilter('ignore', VerifyWarning)

def gaussian2(x, amp, center, sigma):
    r"""A Gaussian (normal) profile.
    
    The functional form is:
    
    :math:`f(x) = A e^{-\frac{(x - \mu)^2}{2\sigma^2}}`
    
    where:
    
    * :math:`A` is the pulse amplitude
    * :math:`\mu` is the center of the peak
    * :math:`\sigma` is the standard deviation (width)
    
    Args:
        x (np.array): Array of times.
        amp (float): The amplitude of the pulse, A.
        center (float): The center time of the pulse, :math:`\mu`.
        sigma (float): The standard deviation of the pulse, :math:`\sigma`.
    
    Returns:
        (np.array)
    """
    return amp * np.exp(-((x - center)**2) / (2 * sigma**2))

def triangular(x, amp, tstart, tpeak, tstop):
    r"""A triangular pulse function.
    
    The functional form is a piecewise linear function:
    
    :math:`f(x) = \begin{cases} 
    A \frac{x - t_{start}}{t_{peak} - t_{start}} & t_{start} \le x < t_{peak} \\ 
    A \frac{t_{stop} - x}{t_{stop} - t_{peak}} & t_{peak} \le x \le t_{stop} \\ 
    0 & \text{otherwise} 
    \end{cases}`
    
    where:
    
    * :math:`A` is the pulse amplitude
    * :math:`t_{start}` is the start time of the pulse
    * :math:`t_{peak}` is the time of the peak amplitude
    * :math:`t_{stop}` is the end time of the pulse
    
    Args:
        x (np.array): Array of times.
        amp (float): The amplitude of the peak, A.
        tstart (float): The start time of the pulse.
        tpeak (float): The time of the peak.
        tstop (float): The end time of the pulse.
        
    Returns:
        (np.array)
    """
    return np.interp(x, [tstart, tpeak, tstop], [0, amp, 0])

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
                  fig_name=None):
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
    tte_sim = TteSourceSimulator(rsp, Band(), band_params, func, func_par)
    tte_src = tte_sim.to_tte(t_start, t_stop)
    
    # background simulation
    #tte_sim = TteBackgroundSimulator(spec_bkgd, 'Gaussian', quadratic, quadratic_params)
    tte_sim = TteBackgroundSimulator(spec_bkgd, 'Gaussian', back_func, back_func_par)
    tte_bkgd = tte_sim.to_tte(t_start, t_stop)
    
    # merge the background and source
    #tte_total = GbmTte.merge([tte_bkgd, tte_src])
    #tte_total = GbmTte.merge([tte_src, tte_bkgd])
    tte_total = PhotonList.merge([tte_src, tte_bkgd])

    plot_bw = 0.1
    phaii = tte_total.to_phaii(bin_by_time, plot_bw)

    
    phii_src = tte_src.to_phaii(bin_by_time, plot_bw)
    
    phii_bkgd = tte_bkgd.to_phaii(bin_by_time, plot_bw)
    lc_tot = phaii.to_lightcurve(energy_range=energy_range_nai)
    lc_src = phii_src.to_lightcurve(energy_range=energy_range_nai)
    lc_bkgd = phii_bkgd.to_lightcurve(energy_range=energy_range_nai)

    plot_bw = 0.1
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
    lc_tot = phaii.to_lightcurve(energy_range=energy_range_nai)

    src_max = max(lc_src.counts)
    back_avg = np.mean(lc_bkgd.counts)
    SNR = src_max / np.sqrt(back_avg)


    #lcplot = Lightcurve(data=phaii.to_lightcurve(energy_range=energy_range_nai))
    lcplot = Lightcurve(data=lc_tot)
    lcplot.add_selection(lc_src)
    lcplot.add_selection(lc_bkgd)
    lcplot.selections[1].color = 'pink'
    lcplot.selections[0].color = 'green'

    #x_low = func_par[1] - func_par[1]
    #x_high = func_par[1] + func_par[1]
    #plt.xlim(x_low, x_high)
    lcplot.errorbars.hide()


    ######### SNR Calculation #########

    if fig_name is None:
        fig_name = f'lc_{trigger_number}_n{det}_{angle}deg.png'
        plt.show()
    plt.title(f'Bn{trigger_number}, n{det}, {angle}deg, back {back_func_par[0]}, Peak {func_par[0]}')

    plt.savefig(fig_name, dpi=300)
        

    phaii_hi = tte_total.to_phaii(bin_by_time, bin_width)
    phaii = phaii_hi.slice_energy(energy_range_nai)
    data = phaii.to_lightcurve()
    return data.centroids, data.counts, src_max, back_avg, SNR

if __name__ == '__main__':
    gauss_params = (.2, 0.0, 0.2)
    const_par = (2, )
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
    gen_GBM_pulse('250709653', '6', 10.73, -10.0, 10.0, func=gaussian2, func_par=gauss_params, back_func=constant, back_func_par=const_par)