"""
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th June 2025: Including Fermi GBM simulation of same functions.
3rd August 2025: Including functions for simulating light curves.

"""
import numpy as np


def generate_gaussian_light_curve(center_time, sigma, peak_amplitude, bin_width,
                                  background_level, pre_post_background_time=2.0, random_seed=None):
    """Generates a Gaussian light curve with Poisson noise."""
    rng = np.random.default_rng(seed=random_seed)
    full_start = center_time - pre_post_background_time - 4 * sigma
    full_end = center_time + pre_post_background_time + 4 * sigma
    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    gaussian_signal = peak_amplitude * np.exp(-0.5 * ((time_bins - center_time) / sigma) ** 2)
    noisy_signal = rng.poisson(gaussian_signal)
    noisy_background = rng.poisson(background_level, size=time_bins.size)
    observed_counts = noisy_signal + noisy_background

    return time_bins, observed_counts, noisy_signal, noisy_background


def generate_triangular_light_curve_with_fixed_peak_amplitude(
    width,
    start_time,
    peak_time,
    peak_amplitude,    # new parameter
    peak_time_ratio,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates:
    - Time bins
    - Observed counts (triangle + noisy background)
    - Triangle-only counts (no noise)
    - Noisy background-only counts

    Keeps the peak amplitude of the triangle constant, regardless of width.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Time bins
    full_start = start_time - pre_post_background_time
    full_end = start_time + width + pre_post_background_time

    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Triangle-only (no noise)
    end_time = start_time + width
    in_rise = (time_bins >= start_time) & (time_bins < peak_time)
    in_fall = (time_bins >= peak_time) & (time_bins < end_time)

    rise_slope = peak_amplitude / (peak_time - start_time) if peak_time != start_time else 0
    fall_slope = peak_amplitude / (end_time - peak_time) if end_time != peak_time else 0

    triangle_counts = np.zeros_like(time_bins)
    triangle_counts[in_rise] = (time_bins[in_rise] - start_time) * rise_slope
    triangle_counts[in_fall] = (end_time - time_bins[in_fall]) * fall_slope

    # Noisy background (Poisson noise)
    background_noisy_counts = rng.poisson(background_level, size=time_bins.size)

    # Observed counts = triangle (noiseless) + noisy background
    observed_counts = triangle_counts + background_noisy_counts

    # Round to nearest integer (counts must be integers)
    observed_counts = np.round(observed_counts).astype(int)

    return time_bins, observed_counts, triangle_counts, background_noisy_counts

# Your original Gaussian function for reference
def generate_gaussian_light_curve(center_time, sigma, peak_amplitude, bin_width,
                                  background_level, pre_post_background_time=2.0, random_seed=None):
    """
    Generates a Gaussian light curve with Poisson noise.
    
    This version follows the more physically accurate model where the signal and background
    are combined before the Poisson noise is applied.
    """
    rng = np.random.default_rng(seed=random_seed)
    full_start = center_time - pre_post_background_time - 5 * sigma
    full_end = center_time + pre_post_background_time + 5 * sigma
    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate the ideal, noiseless signal
    ideal_signal = peak_amplitude * np.exp(-0.5 * ((time_bins - center_time) / sigma) ** 2)
    
    # The total expected rate is the sum of the signal and the background
    total_expected_rate = ideal_signal + background_level
    
    # The observed counts are a Poisson realization of the total expected rate
    observed_counts = rng.poisson(total_expected_rate)
    
    # For comparison, we can also return the ideal background component
    ideal_background = np.full_like(time_bins, background_level)

    return time_bins, observed_counts, ideal_signal, ideal_background

# A slightly revised and more robust version of your Triangle function
def generate_triangular_light_curve(
    width,
    start_time,
    peak_time_ratio,
    peak_amplitude,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates a triangular light curve with a physically consistent noise model.

    Args:
        width (float): The total width of the triangle pulse (from start to end).
        start_time (float): The start time of the pulse.
        peak_time_ratio (float): The fraction of the width at which the peak occurs (0 to 1).
        peak_amplitude (float): The peak amplitude of the triangle.
        bin_width (float): The width of each time bin.
        background_level (float): The average background counts per bin.
        pre_post_background_time (float): Time to include before and after the pulse for context.
        random_seed (int, optional): Seed for the random number generator.

    Returns:
        tuple: (time_bins, observed_counts, ideal_signal, ideal_background)
    """
    rng = np.random.default_rng(seed=random_seed)

    # Define time properties of the pulse
    end_time = start_time + width
    peak_time = start_time + width * peak_time_ratio

    # Time bins
    full_start = start_time - pre_post_background_time
    full_end = end_time + pre_post_background_time
    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate ideal, noiseless triangle signal
    ideal_signal = np.zeros_like(time_bins, dtype=float)
    
    # Define boolean masks for rise and fall periods
    in_rise = (time_bins >= start_time) & (time_bins < peak_time)
    in_fall = (time_bins >= peak_time) & (time_bins <= end_time)

    # Calculate rise and fall slopes
    rise_duration = peak_time - start_time
    fall_duration = end_time - peak_time
    
    if rise_duration > 0:
        rise_slope = peak_amplitude / rise_duration
        ideal_signal[in_rise] = (time_bins[in_rise] - start_time) * rise_slope
        
    if fall_duration > 0:
        fall_slope = peak_amplitude / fall_duration
        ideal_signal[in_fall] = (end_time - time_bins[in_fall]) * fall_slope

    # The total expected rate is the sum of the signal and the background
    total_expected_rate = ideal_signal + background_level
    
    # The observed counts are a Poisson realization of the total expected rate
    observed_counts = rng.poisson(total_expected_rate)
    
    # Ideal background component for plotting/comparison
    ideal_background = np.full_like(time_bins, background_level)

    return time_bins, observed_counts, ideal_signal, ideal_background

# --- NEW FUNCTIONS ---

def generate_norris_light_curve(
    peak_time,
    peak_amplitude,
    rise_time,
    decay_time,
    pulse_shape_nu,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates a Norris pulse light curve with Poisson noise.

    The Norris pulse is defined by the function:
    I(t) = A * exp( -( |t-t_peak| / sigma )^nu )
    where sigma is different for the rise (sigma_r) and decay (sigma_d) phases.
    This is a very common pulse shape for Gamma-Ray Bursts (GRBs).
    """
    rng = np.random.default_rng(seed=random_seed)
    
    # Define time range, ensuring it's wide enough to capture the pulse tails
    full_start = peak_time - pre_post_background_time - 5 * rise_time
    full_end = peak_time + pre_post_background_time + 5 * decay_time
    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate ideal, noiseless Norris pulse signal
    ideal_signal = np.zeros_like(time_bins, dtype=float)
    time_relative = time_bins - peak_time
    
    # Define boolean masks for rise and decay periods
    rise_mask = time_relative < 0
    decay_mask = time_relative >= 0
    
    # Rise phase
    if np.any(rise_mask) and rise_time > 0:
        ideal_signal[rise_mask] = peak_amplitude * np.exp(
            -(np.abs(time_relative[rise_mask]) / rise_time) ** pulse_shape_nu
        )
    
    # Decay phase
    if np.any(decay_mask) and decay_time > 0:
        ideal_signal[decay_mask] = peak_amplitude * np.exp(
            -(time_relative[decay_mask] / decay_time) ** pulse_shape_nu
        )

    # The total expected rate is the sum of the signal and the background
    total_expected_rate = ideal_signal + background_level
    
    # The observed counts are a Poisson realization of the total expected rate
    observed_counts = rng.poisson(total_expected_rate)
    
    # Ideal background component for plotting/comparison
    ideal_background = np.full_like(time_bins, background_level)
    
    return time_bins, observed_counts, ideal_signal, ideal_background


def generate_fred_light_curve(
    peak_time,
    peak_amplitude,
    rise_time,
    decay_time,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates a FRED (Fast Rise, Exponential Decay) light curve.

    This is a special case of the Norris pulse where the pulse shape
    parameter nu=1, resulting in an exponential rise and exponential decay.
    """
    return generate_norris_light_curve(
        peak_time=peak_time,
        peak_amplitude=peak_amplitude,
        rise_time=rise_time,
        decay_time=decay_time,
        pulse_shape_nu=1.0,  # FRED is defined by nu=1
        bin_width=bin_width,
        background_level=background_level,
        pre_post_background_time=pre_post_background_time,
        random_seed=random_seed
    )

def generate_lognormal_light_curve(
    peak_time,
    peak_amplitude,
    sigma,
    tau,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates a lognormal pulse light curve with Poisson noise.

    The lognormal pulse is defined by the function:
    I(t) = A * exp( - (ln(1 + (t-t_peak)/tau))^2 / (2*sigma^2) )
    The pulse is defined for t > t_peak - tau.
    """
    rng = np.random.default_rng(seed=random_seed)
    
    # Define time range
    # Pulse starts at t_peak - tau
    pulse_start_time = peak_time - tau
    # The decay can be long, so we use a generous factor of tau for the end time
    full_start = pulse_start_time - pre_post_background_time
    full_end = peak_time + pre_post_background_time + 15 * tau * sigma
    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate ideal, noiseless lognormal signal
    ideal_signal = np.zeros_like(time_bins, dtype=float)
    
    # Argument of the logarithm must be positive
    log_argument = 1 + (time_bins - peak_time) / tau
    valid_mask = log_argument > 0
    
    # Calculate pulse only where it is defined
    log_term = np.log(log_argument[valid_mask])
    ideal_signal[valid_mask] = peak_amplitude * np.exp(-0.5 * (log_term / sigma) ** 2)

    # The total expected rate is the sum of the signal and the background
    total_expected_rate = ideal_signal + background_level
    
    # The observed counts are a Poisson realization of the total expected rate
    observed_counts = rng.poisson(total_expected_rate)
    
    # Ideal background component for plotting/comparison
    ideal_background = np.full_like(time_bins, background_level)

    return time_bins, observed_counts, ideal_signal, ideal_background






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

