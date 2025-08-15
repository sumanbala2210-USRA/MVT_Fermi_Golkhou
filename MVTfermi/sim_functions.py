"""
# sim_functions.py
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th June 2025: Including Fermi GBM simulation of same functions.
3rd August 2025: Including functions for simulating light curves.
14th August 2025: Added more pulse shapes and utility functions.

"""
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def constant2(x, value):
    """
    A constant background function.
    
    Args:
        x (np.array): Array of times (not used, but required for compatibility).
        value (float): The constant value to return for all times.
    
    Returns:
        (np.array)
    """
    return np.full_like(x, value, dtype=float)



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


def fred(x, amp, tstart, t_rise, t_decay):
    r"""A Fast Rise, Exponential Decay (FRED) pulse.
    
    The functional form is a double exponential:
    $f(t) = A (e^{-(t-t_{start})/\tau_{decay}} - e^{-(t-t_{start})/\tau_{rise}})$
    
    Args:
        x (np.array): Array of times.
        amp (float): The normalization amplitude, A.
        tstart (float): The start time of the pulse.
        t_rise (float): The rise timescale of the pulse, $\tau_{rise}$.
        t_decay (float): The decay timescale of the pulse, $\tau_{decay}$.
    
    Returns:
        (np.array)
    """
    x = np.asarray(x)
    fxn = np.zeros_like(x, dtype=float)
    mask = x > tstart
    
    time_shifted = x[mask] - tstart
    fxn[mask] = amp * (np.exp(-time_shifted / (t_decay+t_rise)) - np.exp(-time_shifted / t_rise))
    
    return fxn


def norris(x, amp, tstart, t_rise, t_decay):
    r"""A Norris pulse-shape function:

    :math:`I(t) = A \lambda e^{-\tau_1/t - t/\tau_2} \text{ for } t > 0;\\ 
    \text{ where } \lambda = e^{2\sqrt(\tau_1/\tau_2)};`
    
    and where
    
    * :math:`A` is the pulse amplitude
    * :math:`\tau_1` is the rise time
    * :math:`\tau_2` is the decay time
    
    References:
        `Norris, J. P., et al. 2005 ApJ 627 324
        <https://iopscience.iop.org/article/10.1086/430294>`_
    
    Args:
        x (np.array): Array of times
        amp (float): The amplitude of the pulse
        tstart (float): The start time of the pulse
        t_rise (float): The rise timescal of the pulse
        t_decay (flaot): The decay timescale of the pulse
    
    Returns:
        (np.array)
    """
    x = np.asarray(x)
    fxn = np.zeros_like(x)
    mask = (x > tstart)
    lam = amp * np.exp(2.0 * np.sqrt(t_rise / t_decay))
    fxn[mask] = lam * np.exp(
        -t_rise / (x[mask] - tstart) - (x[mask] - tstart) / t_decay)
    return fxn

# Helper functions (no changes needed for these)
def constant(x, amp):
    fxn = np.empty(x.size)
    fxn.fill(amp)
    return fxn

def gaussian(x, amp, center, sigma):
    return amp * np.exp(-((x - center)**2) / (2 * sigma**2))

def lognormal(x, amp, center, sigma):
    r"""A log-normal profile.
    
    The functional form is:
    
    :math:`f(x) = A e^{-\frac{(\ln(x/x_c))^2}{2\sigma^2}}`
    
    This parameterization is chosen so that the amplitude :math:`A` is the
    peak value of the function, which occurs at :math:`x=x_c`.
    
    where:
    
    * :math:`A` is the pulse amplitude
    * :math:`x_c` is the center of the peak (the median of the distribution)
    * :math:`\sigma` is the standard deviation of the *logarithm* of the variable, which controls the width and skewness.
    
    Args:
        x (np.array): Array of independent variable values.
        amp (float): The amplitude of the pulse, A.
        center (float): The center (peak location) of the pulse, :math:`x_c`. Must be > 0.
        sigma (float): The width parameter, :math:`\sigma`. Must be > 0.
    
    Returns:
        (np.array)
    """
    x = np.asarray(x, dtype=float)
    fxn = np.zeros_like(x)
    
    # Create a mask to handle the function's domain (x > 0).
    # The function will return 0 for any x <= 0.
    mask = x > 0
    
    # Calculate the function only for positive values of x.
    # This avoids np.log() errors on non-positive numbers.
    fxn[mask] = amp * np.exp(-(np.log(x[mask] / center)**2) / (2 * sigma**2))
    
    return fxn

