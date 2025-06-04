import numpy as np
import os
import matplotlib.pyplot as plt # Keep for other functions that plot
# from scipy.stats import median_abs_deviation # Already in _helper_robust_sigma
from numpy.polynomial import polynomial as P
# pandas might not be needed if pyloess 0.1.0 returns numpy array directly
# import pandas as pd 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.patheffects as pe

try:
    import pyloess # Corrected import
except ImportError:
    print("CRITICAL WARNING: 'pyloess' package not found. Functions requiring LOESS will not work correctly.")
    pyloess = None

from .find_opt_res_fermi import find_optimum_resolution_diff

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

def find_var_py(array_in):
    arr = np.asarray(array_in)
    if arr.size < 2: return np.nan
    diff_arr = np.diff(arr)
    if diff_arr.size == 0: return np.nan
    return np.var(diff_arr)

def convert_res_coarse_py(x_in, h_in, coarsening_factor):
    x, h = np.asarray(x_in), np.asarray(h_in)
    fact = int(coarsening_factor)
    nn = len(h)
    if fact <= 0: raise ValueError("Factor 'fact' must be positive.")
    if nn == 0: return np.array([], dtype=x.dtype), np.array([], dtype=h.dtype)
    
    n_alloc_idx_max = int((float(nn) + 1.0) / fact)
    alloc_size = n_alloc_idx_max + 1
    
    x_out = np.zeros(alloc_size, dtype=x.dtype)
    h_out = np.zeros(alloc_size, dtype=h.dtype)
    output_idx = 0
    if nn >= fact:
        for i_start in range(0, nn - fact + 1, fact):
            if output_idx < alloc_size:
                x_out[output_idx] = x[i_start]
                h_out[output_idx] = np.sum(h[i_start : i_start + fact])
                output_idx += 1
            else: break
    return x_out, h_out



# Ensure sigma_py is defined (as provided previously)
def sigma_py(x_in):
    x = np.asarray(x_in)
    valid_x = x[np.isfinite(x)]
    if valid_x.size < 1: return np.nan
    if valid_x.size == 1: return 0.0
    return np.std(valid_x)

import numpy as np
from scipy.stats import median_abs_deviation, norm # norm is for the scaling factor

def _helper_robust_sigma(data_in, zero=False, eps_val=1e-20):
    """
    Calculates a robust standard deviation using Median Absolute Deviation (MAD).
    Revised to handle issues with scipy's median_abs_deviation center=0.0.

    Args:
        data_in (np.ndarray): Input data array (residuals).
        zero (bool): If True, calculate MAD relative to zero.
        eps_val (float): A small epsilon value.
    Returns:
        float: Robust standard deviation.
    """
    data_arr = np.asarray(data_in)
    
    if data_arr.size == 0: 
        return eps_val 

    finite_data = data_arr[np.isfinite(data_arr)] # Work with finite data for calculations
    if finite_data.size == 0: 
        return eps_val 
    
    sigma = np.nan
    # Scaling factor to make MAD estimate std dev for a normal distribution
    # k = 1 / Phi^-1(3/4) where Phi is the CDF of standard normal. Approx 1.4826
    SCALING_FACTOR_FOR_NORMAL = 1.0 / norm.ppf(0.75) 

    if zero:
        # Calculate MAD relative to zero manually
        # mad_val = median(|X_i - 0|) = median(|X_i|)
        mad_val = np.median(np.abs(finite_data)) 
        sigma = mad_val * SCALING_FACTOR_FOR_NORMAL
    else:
        # For non-zero center, calculate median of data and pass it as the center.
        # median_abs_deviation should handle a float center correctly.
        # If this also fails, we might need to pass np.median as a callable.
        center_of_data = np.median(finite_data)
        try:
            # Using finite_data ensures no NaNs are passed if nan_policy='raise' or 'propagate' internally
            sigma = median_abs_deviation(finite_data, center=center_of_data, scale='normal')
        except Exception as e: # Catch if this specific call fails too
            print(f"Warning: median_abs_deviation with calculated center failed: {e}. Using manual MAD.")
            mad_val = np.median(np.abs(finite_data - center_of_data))
            sigma = mad_val * SCALING_FACTOR_FOR_NORMAL

    # Fallback if sigma is still problematic (NaN or too small)
    if np.isnan(sigma) or sigma < eps_val:
        center_for_fallback = 0.0 if zero else np.median(finite_data) # Recalculate for safety
        avg_abs_dev = np.mean(np.abs(finite_data - center_for_fallback))
        # IDL's LOWESS uses TOTAL(ABS(R))/.8/N_ELEMENTS(R) as a fallback
        fallback_sigma = avg_abs_dev / 0.8 if avg_abs_dev > eps_val else eps_val
        final_sigma = max(fallback_sigma, eps_val) # Ensure it's at least eps_val
        # print(f"_helper_robust_sigma: Fallback sigma used: {final_sigma}") # For debugging
        return final_sigma
        
    return sigma


def find_error_on_mvt_py(ccf01_in, noise_in, ntr_in, lag_in, 
                         nm1_idx_start, nm2_idx_inclusive_end):
    if pyloess is None or not hasattr(pyloess, 'loess'):
        print("Error: pyloess package not available for find_error_on_mvt_py.")
        return np.nan, np.nan
        
    ntr = int(ntr_in)
    ccf01_orig, noise_orig, lag_orig = map(np.asarray, [ccf01_in, noise_in, lag_in])

    if not (len(ccf01_orig) == len(noise_orig) == len(lag_orig)):
        raise ValueError("Inputs ccf01, noise, lag must have same length.")

    positive_noise_mask = noise_orig > 0.0
    ccf01_flt, noise_flt, lag_flt = ccf01_orig[positive_noise_mask], noise_orig[positive_noise_mask], lag_orig[positive_noise_mask]
    
    kk_len = len(ccf01_flt)
    if kk_len == 0: return np.nan, np.nan

    safe_nm1 = max(0, int(nm1_idx_start))
    safe_nm2_incl = min(int(nm2_idx_inclusive_end), kk_len - 1)

    if safe_nm1 > safe_nm2_incl or kk_len < 2: return np.nan, np.nan

    peaks_lag_values = np.full(ntr, np.nan, dtype=float)

    for i in range(ntr):
        random_noise_component = np.random.randn(kk_len)
        cc01_noisy_iter = ccf01_flt + noise_flt * random_noise_component
        
        ysmooth_iter_values = np.copy(cc01_noisy_iter) # Fallback
        if len(lag_flt) == len(cc01_noisy_iter) and len(lag_flt) > 0:
            lowess_ndeg_param, lowess_window_param = 2, 50
            n_lowess_pts_iter = len(lag_flt)

            if n_lowess_pts_iter > lowess_ndeg_param :
                actual_window_pts = min(max(lowess_window_param, lowess_ndeg_param + 2), n_lowess_pts_iter)
                span_iter = float(actual_window_pts) / n_lowess_pts_iter
                span_iter = np.clip(span_iter, (lowess_ndeg_param + 1.0) / n_lowess_pts_iter if n_lowess_pts_iter > 0 else 0.1, 1.0)
                try:
                    loess_result = pyloess.loess(lag_flt, cc01_noisy_iter, span=span_iter, degree=lowess_ndeg_param)
                    if loess_result.ndim == 2 and loess_result.shape[0] == kk_len and loess_result.shape[1] >= 2:
                        # Assuming result[:,1] is smoothed y, and order matches input lag_flt or lag_flt is sorted
                        #print(f"find_error_on_mvt_py: pyloess returned 2D array, taking column 1 as smoothed Y for iter {i}.")
                        ysmooth_iter_values = loess_result[:, 1]
                    elif loess_result.ndim == 1 and loess_result.shape[0] == kk_len:
                        ysmooth_iter_values = loess_result
                    else: # Unexpected shape
                        print(f"Warning: pyloess returned unexpected shape {loess_result.shape} in find_error_on_mvt_py. Using noisy data for iter {i}.")
                        # ysmooth_iter_values remains copy of cc01_noisy_iter
                except Exception as e:
                    print(f"pyloess smoothing failed in find_error_on_mvt_py iter {i}: {e}. Using noisy data.")
                    # ysmooth_iter_values remains copy of cc01_noisy_iter
            # else: ysmooth_iter_values remains copy if not enough points
        else:
            peaks_lag_values[i] = np.nan; continue

        if np.all(np.isnan(ysmooth_iter_values)): peaks_lag_values[i] = np.nan; continue
            
        ys_roi = ysmooth_iter_values[safe_nm1 : safe_nm2_incl + 1]
        lag_roi = lag_flt[safe_nm1 : safe_nm2_incl + 1]

        if ys_roi.size == 0: peaks_lag_values[i] = np.nan; continue

        pos_ys_mask = ys_roi > 0.0
        ys_pos, lag_pos = ys_roi[pos_ys_mask], lag_roi[pos_ys_mask]
        
        if ys_pos.size > 0: peaks_lag_values[i] = lag_pos[np.argmin(ys_pos)]
        else: peaks_lag_values[i] = np.nan

    valid_peaks = peaks_lag_values[~np.isnan(peaks_lag_values)]
    if valid_peaks.size < 2:
        mean_mvt, error_mvt = (np.mean(valid_peaks) if valid_peaks.size > 0 else np.nan), np.nan
    else:
        sig_peaks, mean_peaks = sigma_py(valid_peaks), np.mean(valid_peaks)
        if np.isnan(sig_peaks) or np.isnan(mean_peaks): peaks_filt = valid_peaks
        else:
            mask = (valid_peaks >= mean_peaks - sig_peaks) & (valid_peaks <= mean_peaks + sig_peaks)
            peaks_filt = valid_peaks[mask]
        if peaks_filt.size > 0: mean_mvt, error_mvt = np.mean(peaks_filt), sigma_py(peaks_filt)
        else: mean_mvt, error_mvt = mean_peaks, sig_peaks
            
    #print(f"FIND_ERROR_ON_MVT Py (using pyloess): Mean MVT (s) = {mean_mvt:.4f}, Error = {error_mvt:.4f}")
    print(f"Mean MVT (s) = {mean_mvt:.4f}, Error = {error_mvt:.4f}")
    return mean_mvt, error_mvt




# find_optimum_resolution_new_py definition
# Ensure basic helpers like find_var_py, convert_res_coarse_py are defined above.

# find_optimum_resolution_new_py definition
# Ensure basic helpers like find_var_py, convert_res_coarse_py are defined above.

def find_optimum_resolution_new_py(bn_title, xn0_in, hh_in, t1, tt1, tt2, tt20, t2,
                                   initial_bin_width_bw, 
                                   f1, f2,
                                   nm1_idx, nm2_idx_exclusive,k, output_plot_dir):
    if pyloess is None or not hasattr(pyloess, 'loess'):
        print("Error: pyloess package not available for find_optimum_resolution_new_py.")
        return np.nan, np.nan, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    #print(f"Running find_optimum_resolution_new_py for {bn_title} with signal window {tt1} to {tt2} (using pyloess)")
   
    print(f"Signal window {tt1} to {tt2}")
    nmax_iterations = 1000
    # Initialize result arrays
    bin_width_hist, ratio_var_hist, num_hist, num1_hist, denom_hist, signal = [np.full(nmax_iterations, np.nan) for _ in range(6)]
    xn_iter_orig, h_iter_orig = np.copy(np.asarray(xn0_in)), np.copy(np.asarray(hh_in))
    xn_current, h_current = np.copy(xn_iter_orig), np.copy(h_iter_orig)
    actual_iterations = 0

    for i in range(nmax_iterations): # Main coarsening loop
        actual_iterations = i + 1
        if len(xn_current) <= 1: print(f"Iter {i}: xn_current too short. Stop coarsening."); break
        bin_width_hist[i] = xn_current[1] - xn_current[0] if len(xn_current) > 1 else initial_bin_width_bw
        
        idx_t1=np.searchsorted(xn_current,t1,side='left'); idx_tt1=np.searchsorted(xn_current,tt1,side='left')
        idx_tt2=np.searchsorted(xn_current,tt2,side='left'); idx_tt20=np.searchsorted(xn_current,tt20,side='left')
        idx_t2=np.searchsorted(xn_current,t2,side='left'); len_xn_c=len(xn_current)
        grb_data=h_current[min(idx_tt1,len_xn_c):min(idx_tt2,len_xn_c)]
        bgnd1=h_current[min(idx_t1,len_xn_c):min(idx_tt1,len_xn_c)]
        bgnd2=h_current[min(idx_tt20,len_xn_c):min(idx_t2,len_xn_c)]
        bgnd_list=[d for d in [bgnd1,bgnd2] if d.size>0]
        bgnd_data=np.concatenate(bgnd_list) if bgnd_list else np.array([])
        min_len=0; grb_proc,bgnd_proc_resized=grb_data,bgnd_data # Renamed bgnd_data to bgnd_proc_resized
        if grb_proc.size>0 and bgnd_proc_resized.size>0:
            min_len=min(grb_proc.size,bgnd_proc_resized.size); grb_proc,bgnd_proc_resized=grb_proc[:min_len],bgnd_proc_resized[:min_len]

        
        
        if min_len <= 2: print(f"Iter {i}: GRB/BGND too short (len {min_len}).")
        else:
            num_hist[i]=find_var_py(grb_proc)
            num1_hist[i]=find_var_py(grb_proc-bgnd_proc_resized)
            denom_hist[i]=find_var_py(bgnd_proc_resized)
            if denom_hist[i]!=0 and not np.isnan(denom_hist[i]) and not np.isnan(num_hist[i]):
                ratio_var_hist[i]=num_hist[i]/denom_hist[i]/(i+1.0)
            else: ratio_var_hist[i]=np.nan
        
        if np.sum(bgnd_proc_resized) >0:
                signal[i] = np.sum(grb_proc - bgnd_proc_resized)/np.sqrt(np.sum(bgnd_proc_resized))
        else:
            signal[i] = 999

        if len(xn_iter_orig)>0 and len(h_iter_orig)>0:
            xn_current,h_current=convert_res_coarse_py(xn_iter_orig,h_iter_orig,i+2)
            if len(xn_current)<=1: actual_iterations=i+1; break
        else: break
            
    bin_w_out, ratio_v_out, signal_v_out = bin_width_hist[:actual_iterations], ratio_var_hist[:actual_iterations], signal[:actual_iterations]
    #signal_sel = 
    num_out, num1_out, denom_out = num_hist[:actual_iterations], num1_hist[:actual_iterations], denom_hist[:actual_iterations]
    valid_mask = np.isfinite(ratio_v_out) & np.isfinite(bin_w_out) & (ratio_v_out > 0)
    bin_width_sel, ratio_v_proc, signal_sel = bin_w_out[valid_mask], ratio_v_out[valid_mask], signal_v_out[valid_mask]
    #print(f"Iter {actual_iterations}: Valid bins: {len(bin_width_sel)}, Ratio Variance: {len(ratio_v_proc)}, Signal: {len(signal_sel)}\n")
    #print(f"Iter {actual_iterations}: Bin Widths: {bin_width_sel}, Ratio Variance: {ratio_v_proc}, Signal: {signal_sel}")   

    bw_opt_final, sig_bw_opt_final = np.nan, np.nan

    mean_ratio_var = np.mean(ratio_v_proc, axis=0)
    rms_ratio_var = np.sqrt(np.mean(np.square(ratio_v_proc - mean_ratio_var), axis=0))
    #print(f"Iter {actual_iterations}: Mean Ratio Variance: {len(mean_ratio_var)}, RMS Ratio Variance: {len(rms_ratio_var)}")

    moving_avg = variable_window_moving_average(bin_width_sel, mean_ratio_var, initial_bin_width_bw)
    min_index = np.argmin(ratio_v_proc)
    corresponding_bin_width = bin_width_sel[min_index]

    dt1 = corresponding_bin_width / f1
    dt2 = corresponding_bin_width * f2
    k1 = np.searchsorted(bin_width_sel, dt1) if np.searchsorted(bin_width_sel, dt1) < len(bin_width_sel) else 0
    k2 = np.searchsorted(bin_width_sel, dt2) if np.searchsorted(bin_width_sel, dt2) < len(bin_width_sel) else len(bin_width_sel)

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
    filename = f"{bn_title}_opt_resolution_iter_{k}.png"
    fig_path = os.path.join(output_plot_dir, filename) if output_plot_dir else filename

    Fig= True
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

    return np.exp(x_min), error_x, ExponentialFloat(opt_signal), k, filename
    


    return (bw_opt_final, sig_bw_opt_final, ratio_v_out, bin_w_out, num_out, num1_out, denom_out)

# evolve_optimum_resolution_new_py would be defined as before,
# calling this updated find_optimum_resolution_new_py
# (Code for evolve_optimum_resolution_new_py is the same as last provided)




'''

def find_optimum_resolution_new_py(bn_title, xn0_in, hh_in, t1, tt1, tt2, tt20, t2,
                                   initial_bin_width_bw, 
                                   f1_factor, f2_factor,
                                   nm1_idx, nm2_idx_exclusive,k, output_plot_dir):
    if pyloess is None or not hasattr(pyloess, 'loess'):
        print("Error: pyloess package not available for find_optimum_resolution_new_py.")
        return np.nan, np.nan, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    #print(f"Running find_optimum_resolution_new_py for {bn_title} with signal window {tt1} to {tt2} (using pyloess)")
    print(f"Signal window {tt1} to {tt2}")
    nmax_iterations = 1000
    # Initialize result arrays
    bin_width_hist, ratio_var_hist, num_hist, num1_hist, denom_hist = [np.full(nmax_iterations, np.nan) for _ in range(5)]
    xn_iter_orig, h_iter_orig = np.copy(np.asarray(xn0_in)), np.copy(np.asarray(hh_in))
    xn_current, h_current = np.copy(xn_iter_orig), np.copy(h_iter_orig)
    actual_iterations = 0

    for i in range(nmax_iterations): # Main coarsening loop
        actual_iterations = i + 1
        if len(xn_current) <= 1: print(f"Iter {i}: xn_current too short. Stop coarsening."); break
        bin_width_hist[i] = xn_current[1] - xn_current[0] if len(xn_current) > 1 else initial_bin_width_bw
        
        idx_t1=np.searchsorted(xn_current,t1,side='left'); idx_tt1=np.searchsorted(xn_current,tt1,side='left')
        idx_tt2=np.searchsorted(xn_current,tt2,side='left'); idx_tt20=np.searchsorted(xn_current,tt20,side='left')
        idx_t2=np.searchsorted(xn_current,t2,side='left'); len_xn_c=len(xn_current)
        grb_data=h_current[min(idx_tt1,len_xn_c):min(idx_tt2,len_xn_c)]
        bgnd1=h_current[min(idx_t1,len_xn_c):min(idx_tt1,len_xn_c)]
        bgnd2=h_current[min(idx_tt20,len_xn_c):min(idx_t2,len_xn_c)]
        bgnd_list=[d for d in [bgnd1,bgnd2] if d.size>0]
        bgnd_data=np.concatenate(bgnd_list) if bgnd_list else np.array([])
        min_len=0; grb_proc,bgnd_proc_resized=grb_data,bgnd_data # Renamed bgnd_data to bgnd_proc_resized
        if grb_proc.size>0 and bgnd_proc_resized.size>0:
            min_len=min(grb_proc.size,bgnd_proc_resized.size); grb_proc,bgnd_proc_resized=grb_proc[:min_len],bgnd_proc_resized[:min_len]
        
        if min_len <= 2: print(f"Iter {i}: GRB/BGND too short (len {min_len}).")
        else:
            num_hist[i]=find_var_py(grb_proc)
            num1_hist[i]=find_var_py(grb_proc-bgnd_proc_resized)
            denom_hist[i]=find_var_py(bgnd_proc_resized)
            if denom_hist[i]!=0 and not np.isnan(denom_hist[i]) and not np.isnan(num_hist[i]):
                ratio_var_hist[i]=num_hist[i]/denom_hist[i]/(i+1.0)
            else: ratio_var_hist[i]=np.nan
        
        if len(xn_iter_orig)>0 and len(h_iter_orig)>0:
            xn_current,h_current=convert_res_coarse_py(xn_iter_orig,h_iter_orig,i+2)
            if len(xn_current)<=1: actual_iterations=i+1; break
        else: break
            
    bin_w_out, ratio_v_out = bin_width_hist[:actual_iterations], ratio_var_hist[:actual_iterations]
    num_out, num1_out, denom_out = num_hist[:actual_iterations], num1_hist[:actual_iterations], denom_hist[:actual_iterations]
    valid_mask = np.isfinite(ratio_v_out) & np.isfinite(bin_w_out) & (ratio_v_out > 0)
    bin_w_proc, ratio_v_proc = bin_w_out[valid_mask], ratio_v_out[valid_mask]

    bw_opt_final, sig_bw_opt_final = np.nan, np.nan
    
    # Initialize y_smooth and noise_for_mvt to be 1D and compatible with ratio_v_proc
    # These will be used by plotting and FIND_ERROR_ON_MVT even if loess fails
    y_smooth_for_plot = np.copy(ratio_v_proc) 
    noise_for_plot_and_mvt = np.zeros_like(ratio_v_proc)

    if len(ratio_v_proc) == 0:
        print("FIND_OPTIMUM: No valid ratio_var data. Cannot find optimum or plot.")
    else:
        # Attempt LOWESS using pyloess
        if len(bin_w_proc) > 0: # Check if there's data to smooth
            lowess_ndeg, lowess_win_pts = 2, 50
            n_l_pts = len(bin_w_proc)

            if n_l_pts > lowess_ndeg:
                actual_win = min(max(lowess_win_pts, lowess_ndeg + 2), n_l_pts)
                span = float(actual_win) / n_l_pts
                span = np.clip(span, (lowess_ndeg + 1.0) / n_l_pts if n_l_pts > 0 else 0.1, 1.0)
                
                try:
                    # It's good practice to sort x before calling loess if it expects/returns sorted data
                    sort_indices_loess = np.argsort(bin_w_proc)
                    bin_w_proc_sorted = bin_w_proc[sort_indices_loess]
                    ratio_v_proc_sorted_for_loess = ratio_v_proc[sort_indices_loess]

                    loess_result_array = pyloess.loess(bin_w_proc_sorted, ratio_v_proc_sorted_for_loess, 
                                                       span=span, degree=lowess_ndeg)
                    
                    temp_smoothed_y_sorted = np.nan # Initialize
                    if loess_result_array.ndim == 1 and loess_result_array.shape[0] == n_l_pts:
                        temp_smoothed_y_sorted = loess_result_array
                    elif loess_result_array.ndim == 2 and loess_result_array.shape[0] == n_l_pts and loess_result_array.shape[1] >= 2:
                        #print("find_optimum_resolution_new_py: pyloess returned 2D array, taking column 1 as smoothed Y.")
                        temp_smoothed_y_sorted = loess_result_array[:, 1]
                    else:
                        raise ValueError(f"Unexpected pyloess output shape: {loess_result_array.shape}")

                    # Unsort to match original bin_w_proc order
                    y_smooth_for_plot_temp = np.empty_like(temp_smoothed_y_sorted)
                    y_smooth_for_plot_temp[sort_indices_loess] = temp_smoothed_y_sorted
                    y_smooth_for_plot = y_smooth_for_plot_temp # Now 1D and in correct order

                    residuals = ratio_v_proc - y_smooth_for_plot # Should work now
                    global_noise_sigma = _helper_robust_sigma(residuals, zero=True, eps_val=1e-9)
                    if np.isnan(global_noise_sigma) or global_noise_sigma == 0:
                        mean_abs_smooth = np.nanmean(np.abs(y_smooth_for_plot))
                        global_noise_sigma = 0.01 * mean_abs_smooth if not np.isnan(mean_abs_smooth) and mean_abs_smooth != 0 else 0.01
                    noise_for_plot_and_mvt = np.full_like(ratio_v_proc, global_noise_sigma)

                except Exception as e: 
                    print(f"pyloess call or subsequent noise estimation failed: {e}. Using raw ratio_var for y_smooth.")
                    # y_smooth_for_plot and noise_for_plot_and_mvt retain pre-try defaults (raw ratio_v_proc and zeros)
            else: # Not enough points for pyloess
                 print("Not enough points for pyloess smoothing, using raw ratio_var.")
                 # y_smooth_for_plot and noise_for_plot_and_mvt retain pre-try defaults
        
        # Plotting
        plt.figure(figsize=(10,6))
        min_bw_p=np.min(bin_w_proc) if bin_w_proc.size>0 else initial_bin_width_bw
        if min_bw_p<=0:min_bw_p=initial_bin_width_bw
        max_bw_p=np.max(bin_w_proc) if bin_w_proc.size>0 else initial_bin_width_bw*actual_iterations
        plt.plot(bin_w_proc, ratio_v_proc, marker='o', linestyle='-', label='Ratio Var/(Iter+1)', ms=4)
        plt.plot(bin_w_proc, y_smooth_for_plot, linestyle='--', color='c', label='LOESS Smooth (pyloess)')
        plt.fill_between(bin_w_proc, y_smooth_for_plot - noise_for_plot_and_mvt, 
                         y_smooth_for_plot + noise_for_plot_and_mvt, color='c', alpha=0.3, label='Est. Noise Band')
        plt.xlabel('Bin Width (s)'); plt.ylabel('Ratio of Variances / (Iteration+1)'); 
        plt.title(f"MVT between {tt1:.3g}s and {tt2:.3g}s"); 
        if bin_w_proc.size > 0 : plt.xlim(min_bw_p, max_bw_p)

        # FIND_ERROR_ON_MVT call
        if len(y_smooth_for_plot) > 0:
            len_ys = len(y_smooth_for_plot)
            safe_nm1_mvt = min(nm1_idx, len_ys-1) if len_ys>0 else 0
            safe_nm2_mvt_incl = min(nm2_idx_exclusive-1, len_ys-1) if len_ys>0 else 0
            if safe_nm1_mvt > safe_nm2_mvt_incl and len_ys > 0 : safe_nm1_mvt, safe_nm2_mvt_incl = 0, len_ys-1
            
            mean_mvt, error_mvt = find_error_on_mvt_py(y_smooth_for_plot, noise_for_plot_and_mvt, 
                                                       100, bin_w_proc, 
                                                       safe_nm1_mvt, safe_nm2_mvt_incl)
            bw_opt_final, sig_bw_opt_final = mean_mvt, error_mvt
            if not np.isnan(bw_opt_final):
                min_py_ax, max_py_ax = plt.ylim()
                plt.plot([bw_opt_final]*2,[min_py_ax,max_py_ax],'--g',lw=2,label=f'Opt BW: {bw_opt_final:.3g}s')
                if not np.isnan(sig_bw_opt_final):
                    plt.plot([bw_opt_final-sig_bw_opt_final]*2,[min_py_ax,max_py_ax],':m',lw=2,label=f'+/- {sig_bw_opt_final:.3g}s')
                    plt.plot([bw_opt_final+sig_bw_opt_final]*2,[min_py_ax,max_py_ax],':m',lw=2)
        filename = f"{bn_title}_opt_resolution_iter_{k}.png"
        file_path = os.path.join(output_plot_dir, filename) if output_plot_dir else filename
        plt.xscale('log'); plt.yscale('log')
        plt.legend(); plt.grid(True); plt.savefig(file_path)

    return (bw_opt_final, sig_bw_opt_final, ratio_v_out, bin_w_out, num_out, num1_out, denom_out)
''' 





def evolve_optimum_resolution_new_py(bn_title, xn0_in, hh_in, 
                                     t1_global, initial_tt1, t2_global,
                                     initial_bw_ref, 
                                     f1, f2, 
                                     nm1, nm2_exclusive, 
                                     nn_evol_steps, delt_evol, t90, 
                                     output_plot_dir=""):
    """ Python translation of EVOLVE_OPTIMUM_RESOLUTION_NEW.
        Calls find_optimum_resolution_new_py which now uses pyloess (direct y_smooth version).
    """
    # (Full code for evolve_optimum_resolution_new_py as provided in the previous
    #  comprehensive response, with updated print/title to reflect pyloess usage)
    print(f"Running evolve_optimum_resolution_new_py for {bn_title} (using pyloess via FIND_OPTIMUM)")
    tt20_fixed_bg_start = t90 + 20.0

    if not (len(xn0_in) == len(hh_in)):
        raise ValueError("Input xn0_in and hh_in must have the same dimensions.")

    op_tim_hist = np.full(nn_evol_steps, np.nan, dtype=float)
    err_op_tim_hist = np.full(nn_evol_steps, np.nan, dtype=float)
    tr_param_hist = np.full(nn_evol_steps, np.nan, dtype=float)
    actual_steps_done = 0

    for k in range(nn_evol_steps):
        actual_steps_done = k + 1
        tt2_curr_sig_end = (k + 1.0) * delt_evol
        if tt2_curr_sig_end <= initial_tt1:
            tt2_curr_sig_end = initial_tt1 + (k + 1.0) * delt_evol
        
        curr_tr_param = tt2_curr_sig_end + initial_tt1
        tr_param_hist[k] = curr_tr_param
        
        print(f"Evolve Iter #{k}: Sig Win: {initial_tt1} to {tt2_curr_sig_end} (tr_param={curr_tr_param}), T90={t90}")

        if curr_tr_param > (t90 / 0.9):
            print("Stop: tr_param > t90/0.9."); actual_steps_done = k; break 
        
        bw_opt_k, sig_bw_opt_k, _, _, _, _, _ = \
            find_optimum_resolution_new_py( # This now calls the pyloess version
                f"{bn_title}_evol_iter{k}", xn0_in, hh_in, 
                t1_global, initial_tt1, tt2_curr_sig_end, 
                tt20_fixed_bg_start, t2_global, 
                initial_bw_ref, 
                f1, f2, nm1, nm2_exclusive, k, output_plot_dir)
        
        op_tim_hist[k] = bw_opt_k
        err_op_tim_hist[k] = sig_bw_opt_k
            
    tr_final, op_tim_final, err_op_tim_final = \
        tr_param_hist[:actual_steps_done], op_tim_hist[:actual_steps_done], err_op_tim_hist[:actual_steps_done]
    
    valid_mask = (op_tim_final > 0)&np.isfinite(op_tim_final)&np.isfinite(err_op_tim_final)&np.isfinite(tr_final)
    tr_plot, op_tim_plot, err_op_tim_plot = tr_final[valid_mask], op_tim_final[valid_mask], err_op_tim_final[valid_mask]

    mvt_val, sig_mvt_val, epoch_at_mvt = np.nan, np.nan, np.nan
    if len(op_tim_plot) == 0:
        print("EVOLVE_OPTIMUM: No valid optimum time scales found.")
    else: 
        fig1, ax1 = plt.subplots(figsize=(10, 6.5))
        ax1.errorbar(tr_plot, op_tim_plot, yerr=err_op_tim_plot, fmt='o', ms=5, capsize=3, label='Opt. Time Scale')
        ax1.set_xlabel('Signal Epoch Parameter (tr = tt2+tt1) (s)'); ax1.set_ylabel('Optimum Time Scale (s)')
        ax1.set_title(f"{bn_title}: Evolution of Optimum Time Scale (pyloess)")
        if op_tim_plot.size > 0:
            min_y = np.nanmin(op_tim_plot - err_op_tim_plot) if np.any(np.isfinite(op_tim_plot - err_op_tim_plot)) else np.nanmin(op_tim_plot)
            max_y = np.nanmax(op_tim_plot + err_op_tim_plot) if np.any(np.isfinite(op_tim_plot + err_op_tim_plot)) else np.nanmax(op_tim_plot)
            if not (np.isnan(min_y) or np.isnan(max_y)): ax1.set_ylim(min_y, max_y)
        ax1.set_xlim(1.0e-1, 1.1 * t90); ax1.grid(True); ax1.legend(); 
        # plt.show() # Show plots if running interactively or at the end

        if len(op_tim_plot) >= 3 :
            op_slice, err_slice = op_tim_plot[1:], err_op_tim_plot[1:]
            if op_slice.size > 0:
                combined_metric = op_slice + err_slice
                idx_in_slice = np.argmin(combined_metric)
                min_comb_val = combined_metric[idx_in_slice]
                orig_idx_kk = idx_in_slice + 1
                if orig_idx_kk + 1 < len(op_tim_plot):
                    mvt_val = min_comb_val - err_op_tim_plot[orig_idx_kk + 1]
                    sig_mvt_val = err_op_tim_plot[orig_idx_kk + 1]
                    epoch_at_mvt = tr_plot[orig_idx_kk]
                    print(f"EVOLVE_OPTIMUM: MVT = {mvt_val:.4f} +/- {sig_mvt_val:.4f} at tr_param {epoch_at_mvt:.4f}")
        
        fig2, ax2 = plt.subplots(figsize=(9.75, 4.5))
        ax2.errorbar(tr_plot, op_tim_plot, yerr=err_op_tim_plot, fmt='o', ms=5, capsize=3, elinewidth=1.5, label='Opt. Time Scale')
        ax2.set_xlabel('Signal Epoch Parameter (tr = tt2+tt1) (s)', fontsize=18); ax2.set_ylabel('Optimum Time Scale (s)', fontsize=18)
        ax2.set_title(f"{bn_title} (pyloess)", fontsize=21)
        if op_tim_plot.size > 0:
             min_y2 = np.nanmin(op_tim_plot - err_op_tim_plot) if np.any(np.isfinite(op_tim_plot - err_op_tim_plot)) else np.nanmin(op_tim_plot)
             max_y2 = np.nanmax(op_tim_plot + err_op_tim_plot) if np.any(np.isfinite(op_tim_plot + err_op_tim_plot)) else np.nanmax(op_tim_plot)
             if not (np.isnan(min_y2) or np.isnan(max_y2)): ax2.set_ylim(min_y2, max_y2)
        ax2.set_xlim(8.0e-2, 1.1 * t90)
        ax2.tick_params(axis='both', which='major', width=2, length=7, labelsize=18)
        for spine in ax2.spines.values(): spine.set_linewidth(3)
        ax2.grid(True); ax2.legend()
        
        filename = f"{bn_title}_INT_MVT_Distn.png"
        file_path = os.path.join(output_plot_dir,filename)
        try: plt.savefig(file_path); print(f"Plot saved: {filename}")
        except Exception as e: print(f"Error saving plot {filename}: {e}")
        
        plt.close(fig1); plt.close(fig2) # Close after saving
            
    return mvt_val, sig_mvt_val, epoch_at_mvt, tr_plot, op_tim_plot, err_op_tim_plot






