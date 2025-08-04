import numpy as np
import matplotlib.pyplot as plt

def gaussian(t, A, mu, sigma):
    """Generates a single symmetric Gaussian pulse."""
    return A * np.exp(-(t - mu)**2 / (2 * sigma**2))

def fred_pulse(t, A, t_peak, rise_sigma, decay_tau):
    """Generates a single asymmetric FRED pulse."""
    flux = np.zeros_like(t, dtype=float)
    rise_mask = t <= t_peak
    flux[rise_mask] = A * np.exp(-(t[rise_mask] - t_peak)**2 / (2 * rise_sigma**2))
    decay_mask = t > t_peak
    flux[decay_mask] = A * np.exp(-(t[decay_mask] - t_peak) / decay_tau)
    return flux

def generate_mixed_grb(t, pulses, background_level=0.0, noise_std_dev=0.0):
    """Generates a GRB light curve by summing a mix of FRED and Gaussian pulses."""
    flux = np.full_like(t, background_level)

    for pulse_definition in pulses:
        pulse_type, params = pulse_definition
        if pulse_type == 'fred':
            flux += fred_pulse(t, *params)
        elif pulse_type == 'gaussian':
            flux += gaussian(t, *params)
            
    noise = np.random.normal(0, noise_std_dev, t.shape)
    flux += noise
    
    return flux

# --- Example with a Densely Populated Pulse List ---
if __name__ == '__main__':
    time_array = np.linspace(0, 25, 5000)

    # A much denser list of pulses for a highly complex light curve
    dense_pulse_list = [
        # --- Broad underlying structure ---
        ('fred', (150, 9.0, 2.0, 8.0)),      # A very broad base component

        # --- Initial rising complex ---
        ('fred',     (180, 4.5, 0.3, 0.9)),
        ('gaussian', (150, 4.8, 0.2)),

        # --- Main, dense spiky complex ---
        ('fred',     (420, 5.2, 0.08, 0.5)),
        ('fred',     (380, 5.5, 0.06, 0.8)),
        ('fred',     (500, 6.1, 0.1, 1.2)),    # Main peak
        ('fred',     (300, 6.4, 0.05, 0.6)),
        ('gaussian', (220, 6.8, 0.15)),
        ('fred',     (250, 7.1, 0.09, 0.7)),
        ('gaussian', (190, 7.5, 0.2)),
        ('fred',     (150, 7.9, 0.1, 1.0)),

        # --- Decaying tail with late-time spikes ---
        ('gaussian', (100, 10.5, 0.9)),
        ('fred',     (80, 12.0, 0.4, 2.5)),
        ('gaussian', (60, 14.0, 1.5)),
        ('fred',     (70, 15.5, 0.2, 1.8)),  # A final, late sharp pulse
    ]

    background_flux = 100.0
    noise_level = np.sqrt(background_flux)

    # Generate the light curve
    light_curve = generate_mixed_grb(time_array, dense_pulse_list,
                                     background_level=background_flux,
                                     noise_std_dev=noise_level)

    # Plot the result
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_array, light_curve, color='mediumvioletred', linewidth=1.5, label='Dense GRB Simulation')
    ax.set_title('Dense & Complex GRB Simulation', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Detector Counts / Flux', fontsize=12)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 25)
    ax.set_ylim(bottom=0)

    plt.show()