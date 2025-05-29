import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.patheffects as pe




class ExponentialFloat(float):
    """
    A float subclass for formatting numbers in scientific notation with readable precision.

    Automatically expresses a float as:
        (coefficient)e±exponent
    Example:
        0.039366 → 39.366e-3

    Parameters:
    ----------
    value : float
        The numeric value to format.
    n : int, optional (default = 3)
        Number of decimal places for the coefficient (mantissa).
    pow : int or None, optional (default = None)
        If specified, forces formatting with this fixed exponent (10^pow).
        If None, it auto-calculates the exponent so the coefficient lies in [1, 10).
    """

    def __new__(cls, value, n=2, pow=None):
        obj = super().__new__(cls, float(value))
        obj.n = n
        obj.pow = pow if pow is not None else cls.get_pow(value)
        return obj

    def __format__(self, format_spec):
        return super().__format__(format_spec)

    def __str__(self):
        scale = 10 ** self.pow
        scaled_val = self / scale
        return f"{scaled_val:.{self.n}f}e{self.pow:+d}"

    def scaled_str(self):
        """
        Returns only the scaled coefficient as a string with `n` decimals.
        Useful for formatting uncertainty pairs like: (value ± error)e−n
        """
        scale = 10 ** self.pow
        return f"{self / scale:.{self.n}f}"

    @staticmethod
    def get_pow(val):
        """
        Calculates the base-10 exponent that normalizes a float into scientific notation.
        For example:
            0.000234 → -4
            23444    → 4
        """
        val = np.abs(val)
        if val == 0:
            return 0
        return int(np.floor(np.log10(val)))




def calculate_window_size(x, bw):
    """
    Calculate the window size as a function of x and bw.
    The window size scales proportionally to x / bw.
    """
    # Calculate window size directly as x / bw
    window_size = int(np.round(x / bw)/4)
    
    # Ensure window size is within reasonable bounds
    min_window_size = 1
    max_window_size = 1000
    window_size = min(max(window_size, min_window_size), max_window_size)
    
    return window_size

def variable_window_moving_average(x, y, bw):
    moving_avgs = []

    for i in range(len(x)):
        window_size = calculate_window_size(x[i], bw)

        if i < window_size:
            avg = np.mean(y[:i + 1])  # Average of all available elements up to i
        else:
            avg = np.mean(y[i - window_size + 1:i + 1])  # Average of the window size elements
        
        moving_avgs.append(avg)
    
    return np.array(moving_avgs)

def generate_random_hist_values(mean_value):
    poisson_values = np.random.poisson(mean_value)
    return poisson_values

def convert_res_coarse(x, h, fact):
    nn = len(h)
    n = (nn + 1) // fact
    
    x1 = np.zeros(n + 1, dtype=np.float64)
    h1 = np.zeros(n + 1, dtype=np.float64)
    
    for i in range(0, nn - fact, fact):
        x1[i // fact] = x[i]
        if i + fact - 1 >= nn:
            return x1, h1
        h1[i // fact] = np.sum(h[i : i + fact])
    
    return x1, h1


def find_optimum_resolution_diff(trigger_number, grb_range, grb_count, bkg_range, bkg_count, tt1, tt2, bw, N,f1,f2,k,path, src_start_n, src_end_n, bkg_start_n, bkg_end_n, Fig):

    N_iter= int(1/bw)
    bin_width = np.zeros(N_iter, dtype=np.float64)
    ratio_var = np.zeros(N_iter, dtype=np.float64)
    num = np.zeros(N_iter, dtype=np.float64)
    num1 = np.zeros(N_iter, dtype=np.float64)
    denom = np.zeros(N_iter, dtype=np.float64)
    signal = np.zeros(N_iter, dtype=np.float64)
    
    ratio_var_list = np.zeros((N, N_iter), dtype=np.float64)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    

    delt = tt2-tt1
    
    if Fig:
        file_name =f'{trigger_number}_{k}.pdf'
        if path:
            fig_path = os.path.join(path, file_name)
        else:
            fig_path = file_name
    else:
        file_name = None
  
    for j in range(N):
        # Use initial_xn and initial_h to avoid changing them
        hh_grb = generate_random_hist_values(grb_count)
        h_grb=hh_grb.copy()
        xn_grb = grb_range.copy()
        hh_bkg = generate_random_hist_values(bkg_count)
        h_bkg=hh_bkg.copy()
        xn_bkg = bkg_range.copy()
        
        for i in range(N_iter):
            n1 = src_start_n[i]
            n2 = src_end_n[i]
            n1_bkg = bkg_start_n[i] # n1_bkg_array[i]
            n2_bkg = bkg_end_n[i] # n3_array[i]
            grb = h_grb[n1:n2+1]

            bgnd = h_bkg[n1_bkg:n2_bkg]
            
            if len(grb) > len(bgnd):
                grb = grb[:len(bgnd)]
            else:
                bgnd = bgnd[:len(grb)]

            bin_width_tem = xn_grb[1] - xn_grb[0]
            #if bin_width_tem>0.09:
            #    print('bin_width_tem=', bin_width_tem)
            #if len(grb) <= 2:# or bin_width_tem <= 0.1:
            if len(grb) <= 2:
                continue
            #bin_width[i] = xn[1] - xn[0]
            dif_grb = np.diff(grb)
            num[i] = np.var(dif_grb)
            num1[i] = np.var(np.diff(grb - bgnd))
            denom[i] = np.var(np.diff(bgnd))

            if np.sum(bgnd) >0:
                signal[i] = np.sum(grb - bgnd)/np.sqrt(np.sum(bgnd))
            else:
                signal[i] = 999
            if denom[i]==0:
                ratio_var[i] = 0

            else:
                ratio_var[i] = num[i] / denom[i] / (i + 1)
            if j == 0:
                bin_width[i] = xn_grb[1] - xn_grb[0]
    
            ratio_var_list[j, i] = ratio_var[i]
            #print('\n\nHI 1\n\n')
            xn_grb, h_grb = convert_res_coarse(grb_range, hh_grb, i + 1)  # Pass current xn and h for modification
            xn_bkg, h_bkg = convert_res_coarse(bkg_range, hh_bkg, i + 1)
        #h = hist_value.copy()
        #plt.figure(figsize=(10, 6))
        if j == 0:
            l_bw = len(bin_width[bin_width > 0])
    ratio_var_list_truncated = ratio_var_list[:, 1:l_bw]
    bin_width_sel = bin_width[1:l_bw]
    
    signal_sel = signal[1:l_bw]
    mean_ratio_var = np.mean(ratio_var_list_truncated, axis=0)
    rms_ratio_var = np.sqrt(np.mean(np.square(ratio_var_list_truncated - mean_ratio_var), axis=0))

    moving_avg = variable_window_moving_average(bin_width_sel, mean_ratio_var, bw)
    min_index = np.argmin(moving_avg)
    corresponding_bin_width = bin_width_sel[min_index]

    dt1 = corresponding_bin_width / f1
    dt2 = corresponding_bin_width * f2
    k1 = np.searchsorted(bin_width_sel, dt1) if np.searchsorted(bin_width_sel, dt1) < len(bin_width_sel) else 0
    k2 = np.searchsorted(bin_width_sel, dt2) if np.searchsorted(bin_width_sel, dt2) < len(bin_width_sel) else len(bin_width)

    # Filter out data beyond the dominant error point
    bin_width_valid = np.log(bin_width_sel[k1:k2])
    
    signal_valid = signal_sel[k1:k2]
    mean_ratio_var_valid = np.log(mean_ratio_var[k1:k2])
    rms_ratio_var_valid = rms_ratio_var[k1:k2] / mean_ratio_var[k1:k2]

    # Define a function to fit to your data (example quadratic function)
    def model_func(x, a, b, c):
        return a * x**2 + b * x + c

    bounds = (0, [np.inf, np.inf, np.inf])
    
    # Fit the model to the valid data
    popt, pcov = curve_fit(model_func, bin_width_valid, mean_ratio_var_valid, sigma=rms_ratio_var_valid, bounds=bounds)
    
    # Extract the parameters and their uncertainties
    a, b, c = popt
    perr = np.sqrt(np.diag(pcov))
    delta_a_fit, delta_b_fit, delta_c_fit = np.sqrt(np.diag(pcov))


    # Calculate the x-coordinate of the minimum
    x_min = -b / (2 * a)
    
    # Calculate the uncertainty (error) on x_min
    partial_x_partial_a = -b / (2 * a**2)
    partial_x_partial_b = -1 / (2 * a)
    
    delta_x = np.sqrt((partial_x_partial_a * delta_a_fit)**2 + (partial_x_partial_b * delta_b_fit)**2)
    error_x = np.exp(x_min)-np.exp(x_min-delta_x)
    
    # Generate fitted curve values for plotting
    fit_curve = model_func(bin_width_valid, a, b, c)
    
    min_index = np.argmin(fit_curve)

    optimal_bin_width = bin_width_valid[min_index]
    opt_signal = signal_valid[min_index]

    val_fmt = ExponentialFloat(np.exp(x_min), n=2, pow=-3)
    err_fmt = ExponentialFloat(error_x, n=2, pow=val_fmt.pow)  # match exponent

    mvt_str = f'({val_fmt.scaled_str()} ± {err_fmt.scaled_str()})ms'#e{val_fmt.pow:+d}'
    print(f"MVT: {mvt_str.rjust(20)}, Δt: {tt1:6.2f} to {tt2:6.2f}, Sec:  SNR = {opt_signal:6.2f}")

    if Fig:
        # --- Color palette for consistency ---
        #color_full_data = 'deepgray'         # Raw data
        color_filtered = 'seagreen'           # Primary filtered data
        color_fit = 'mediumseagreen'          # Fit curve (softer green in same family)
        color_mvt = 'darkseagreen'            # MVT line & span (same hue, lighter/duller)
        color_avg = 'orangered'             # Moving average
        color_ref_line = 'red'            # Reference vertical line
        text_box_color = 'white'#'lavender'           # Annotation background

        color_full_data = 'dimgray'        # Background data
        color_filtered = 'teal'              # Sigma-filtered signal data
        color_avg = 'darkorange'             # Moving average
        color_fit = 'black'# 'mediumseagreen'         # Fitted model curve
        color_mvt = 'limegreen'              # MVT + error
        color_ref_line = 'crimson'           # Reference vertical

        
        # --- Create plot ---
        mpl.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')  # Or any color like '#f5f5f5', 'whitesmoke', etc.
        # Full data (light gray)
        ax.errorbar(
            bin_width_sel, mean_ratio_var, yerr=rms_ratio_var,
            markersize=2, linestyle='none', ecolor=color_full_data, alpha=0.4, label='Full data'
        )
        
        # Vertical reference line (black/dimgray)
        ax.axvline(x=corresponding_bin_width, color=color_ref_line, linestyle='--', linewidth=1.2)
        
        # Sigma-filtered data
        ax.errorbar(
            np.exp(bin_width_valid), np.exp(mean_ratio_var_valid),
            yerr=rms_ratio_var_valid * np.exp(mean_ratio_var_valid),
            fmt='o', markersize=2, linestyle=' ',
            color=color_filtered, linewidth=1, ecolor=color_filtered, alpha=0.8,
            label=f'({f1}, {f2})σ data'
        )
        
        ax.plot(
            np.exp(bin_width_valid), np.exp(fit_curve),
            color=color_fit,
            label='Fitted Curve',
            linewidth=1.5,
            zorder=5,
            path_effects=[pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()]
        )
        
        # Moving average
        ax.plot(
            bin_width_sel, moving_avg,
            linestyle='-', color=color_avg, linewidth=1.2, label='Moving Average'
        )
        
        # MVT vertical line and shaded region
        ax.axvline(x=np.exp(x_min), color=color_mvt, linestyle='--', linewidth=1.5, label=f'MVT ± error = {mvt_str}')
        ax.axvspan(np.exp(x_min - delta_x), np.exp(x_min + delta_x), alpha=0.2, color=color_mvt)
        
        # Annotation box
        if opt_signal < 0:
            snr_color = '#D62728'
            #label = 'SNR < 0σ'
        elif opt_signal < 3:
            snr_color = '#FFA500'  
            #label = 'SNR < 3σ'
        else:
            snr_color = '#006400'  # '#2CA02C'
            #label = 'SNR ≥ 3σ'
        
        ax.text(
            0.5, 0.93,
            f"Time range: {tt1:.2f} to {tt2:.2f} sec\nMinima: {mvt_str}\nSNR={opt_signal:.1f}",
            color='black', ha='center', va='center', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=snr_color, edgecolor='gray', alpha=0.7)
        )
        
        # Labels and formatting
        ax.set_xlabel("Bin Width (s)")
        ax.set_ylabel("Mean Ratio Variance")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Legend box
        ax.legend(
            fontsize=12,
            loc='lower left',
            frameon=True,
            facecolor=text_box_color,
            edgecolor='gray',
            framealpha=0.9
        )
        
        # Layout and save
        plt.tight_layout()
        plt.savefig(fig_path,facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

    return np.exp(x_min), error_x, ExponentialFloat(opt_signal), k, file_name
    



