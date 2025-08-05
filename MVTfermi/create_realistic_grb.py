import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Pulse Shape Functions (Unchanged) ---
def gaussian(t, A, mu, sigma):
    return A * np.exp(-(t - mu)**2 / (2 * sigma**2))

def fred_pulse(t, A, t_peak, rise_sigma, decay_tau):
    flux = np.zeros_like(t, dtype=float)
    rise_mask = t <= t_peak
    flux[rise_mask] = A * np.exp(-(t[rise_mask] - t_peak)**2 / (2 * rise_sigma**2))
    decay_mask = t > t_peak
    flux[decay_mask] = A * np.exp(-(t[decay_mask] - t_peak) / decay_tau)
    return flux

# --- Core Simulation Functions (Updated) ---
def generate_rate_function(t, pulses, main_amplitude, background_level=0.0):
    """
    Generates the smooth rate function, scaling pulses by a main amplitude.

    Args:
        pulses (list): List of pulses where the first parameter of each pulse
                       is a *relative* amplitude factor.
        main_amplitude (float): The absolute amplitude of the brightest pulse,
                                used to scale all others.
    """
    rate = np.full_like(t, background_level)

    for pulse_definition in pulses:
        pulse_type = pulse_definition[0]
        # The first parameter is now relative amplitude
        relative_amp, *time_params = pulse_definition[1]
        
        # Calculate the absolute amplitude for this pulse
        absolute_amplitude = main_amplitude * relative_amp
        
        # Combine absolute amplitude with the rest of the time parameters
        params = (absolute_amplitude, *time_params)
        
        if pulse_type == 'fred':
            rate += fred_pulse(t, *params)
        elif pulse_type == 'gaussian':
            rate += gaussian(t, *params)
    return rate

def generate_photon_events(t, rate_t):
    """Generates discrete photon arrival times from a rate function."""
    rate_interpolated = interp1d(t, rate_t, kind='linear', bounds_error=False, fill_value=0)
    duration = t[-1] - t[0]
    rate_max = np.max(rate_t)
    if rate_max == 0: return np.array([])
    num_candidates = np.random.poisson(duration * rate_max * 1.2)
    candidate_times = t[0] + np.random.uniform(0, 1, num_candidates) * duration
    actual_rates_at_candidates = rate_interpolated(candidate_times)
    acceptance_probs = actual_rates_at_candidates / rate_max
    accepted_times = candidate_times[np.random.uniform(0, 1, num_candidates) < acceptance_probs]
    return np.sort(accepted_times)

# --- Example Usage ---
if __name__ == '__main__':
    # --- Simulation Control Panel ---
    duration = 20  # seconds
    bin_width = 0.0001 # 0.1 ms binning
    
    # This single value now controls the entire burst's brightness!
    main_burst_amplitude = 800.0  # counts/s for the brightest pulse
    background_flux = 200.0 # counts/s

    # Define pulse list with RELATIVE amplitudes
    # Format: ('type', (Relative_Amp, time_param_1, ...))
    amp = 2.0
    relative_pulse_list = [
        # The main, brightest pulse has a relative amplitude of 1.0
        ('fred', (1.0*amp,  6.1, 0.1, 1.2)),

        # Other pulses are fractions (or multiples) of the main amplitude
        ('fred', (0.84*amp, 5.2, 0.08, 0.5)),
        ('fred', (0.76*amp, 5.5, 0.06, 0.8)),
        ('fred', (0.6*amp,  6.4, 0.05, 0.6)),
        ('fred', (0.5*amp,  7.1, 0.09, 0.7)),
        ('fred', (0.3*amp,  7.9, 0.1, 1.0)),
        ('fred', (0.36*amp, 4.5, 0.3, 0.9)),
        #('fred', (0.16*amp, 12.0, 0.4, 2.5)),
        #('fred', (0.14*amp, 15.5, 0.2, 0.8)),
        #('fred', (0.3*amp,  9.0, 2.0, 1.0)),    # Broad base component

        ('gaussian', (0.3*amp,  4.8, 0.01)),
        ('gaussian', (0.44*amp, 6.8, 0.15)),
        ('gaussian', (0.38*amp, 7.5, 0.2)),
        ('gaussian', (0.2*amp,  10.5, 0.9)),
        #('gaussian', (0.12*amp, 14.0, 1.0)),
    ]

    # --- Run the Simulation ---
    time_array_smooth = np.linspace(0, duration, int(duration/0.001))

    # 1. Generate the underlying smooth rate function
    rate_func = generate_rate_function(time_array_smooth, relative_pulse_list,
                                       main_amplitude=main_burst_amplitude,
                                       background_level=background_flux)

    # 2. Generate the discrete photon arrival times
    photon_arrival_times = generate_photon_events(time_array_smooth, rate_func)
    print(f"Generated {len(photon_arrival_times)} total photon events.")


    # --- Bin the Data ---
    bins = np.arange(0, duration + bin_width, bin_width)
    counts_per_bin, _ = np.histogram(photon_arrival_times, bins=bins)

     # +++ NEW SECTION: SAVE DATA TO FILE +++

    # 1. Calculate the center time for each bin
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 2. Save the data to your preferred format
    
    # Option A: Save as a human-readable text file (.dat)
    output_txt_filename = 'grb_light_curve.dat'
    # Stack arrays into columns: [bin_center, counts]
    data_to_save = np.vstack((bin_centers, counts_per_bin)).T
    header_text = (f'GRB Light Curve Data\n'
                   f'Bin Width: {bin_width} s\n'
                   f'Column 1: Bin Center (s)\n'
                   f'Column 2: Photon Counts per Bin')
    np.savetxt(output_txt_filename, data_to_save, header=header_text, fmt=['%.6f', '%d'])
    print(f"\nBinned data saved to text file: {output_txt_filename}")

    # Option B: Save as a compressed NumPy .npz file (more efficient)
    output_npz_filename = 'grb_light_curve.npz'
    np.savez_compressed(output_npz_filename,
                        bin_centers=bin_centers,
                        counts=counts_per_bin,
                        bin_width=bin_width)
    print(f"Binned data saved to npz file: {output_npz_filename}")
    

    # 3. Setup for histogram plotting and create plots
    #bins = np.arange(0, duration + bin_width, bin_width)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle('GRB Simulation with Relative Amplitudes', fontsize=18)

    # Top Panel: Underlying rate model
    ax1.plot(time_array_smooth, rate_func, color='dodgerblue', label='Underlying Rate Function')
    ax1.set_ylabel('Expected Rate (counts/s)')
    ax1.legend()
    ax1.grid(True)

    # Bottom Panel: Binned "observed" data
    counts_per_bin, _ = np.histogram(photon_arrival_times, bins=bins)
    rate_per_bin = counts_per_bin / bin_width
    ax2.step(bins[:-1], rate_per_bin, where='post', color='firebrick', linewidth=1.0, label=f'Binned Data ({bin_width*1000:.1f} ms bins)')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Observed Rate (counts/s)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.savefig('create_realistic_grb.png', dpi=300)
    plt.close(fig)

    """
    from haar_power_mod import haar_power_mod
    results = haar_power_mod(counts_per_bin, np.sqrt(counts_per_bin), min_dt=0.0001, doplot=True, file='realistic_grb', afactor=-1.0, verbose=False)

    print(f"'mvt_ms': {round(float(results[2]) * 1000, 3)},\n 'mvt_error_ms': {round(float(results[3]) * 1000, 3)}")
    """