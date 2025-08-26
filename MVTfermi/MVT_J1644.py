from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

# --- Load FITS file ---
with fits.open('J1644_450258_1ms_lc.fits') as hdul:
    data_hdu = hdul[1]
    data_table = Table(data_hdu.data)

# Extract columns
met_start = data_table['MET_START']
met_stop = data_table['MET_STOP']
rate = data_table['RATE']
rate_err = data_table['RATE_ERR']

# Original bin width (s)
orig_bin_width = np.mean(met_stop - met_start)
print("Original bin width (s):", orig_bin_width)

# === Original data mid-times relative to t0 ===
t0 = float(met_start.min())
met_mid = (met_start + met_stop) / 2.0 - t0

# === Rebin parameters ===
bin_width_ms = 1000#*10   # desired rebin size in ms
bin_width = bin_width_ms / 1000.0  # in seconds

t_max_rel = float(met_stop.max() - t0)
new_bins = np.arange(0.0, t_max_rel + bin_width, bin_width)
indices = np.digitize(met_mid, new_bins) - 1

binned_time, binned_rate, binned_rate_err = [], [], []
for i in range(len(new_bins) - 1):
    in_bin = (indices == i)
    if np.any(in_bin):
        # Bin center
        bin_center = 0.5 * (new_bins[i] + new_bins[i+1])
        binned_time.append(bin_center)

        # Convert rate → counts
        counts = rate[in_bin] * (met_stop[in_bin] - met_start[in_bin])
        total_counts = counts.sum()

        # Rebin to new rate
        new_rate = total_counts / bin_width
        new_rate_err = np.sqrt(total_counts) / bin_width

        binned_rate.append(new_rate)
        binned_rate_err.append(new_rate_err)

binned_time = np.array(binned_time)
binned_rate = np.array(binned_rate)
binned_rate_err = np.array(binned_rate_err)

# === Plot: Original + Rebinned ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Original
ax1.errorbar(met_mid, rate, yerr=rate_err, fmt='.', markersize=2,
             ecolor='gray', elinewidth=0.5, capsize=1, label=f'Original ({orig_bin_width*1000:.1f} ms)')
ax1.set_ylabel('Count Rate (counts/s)')
ax1.set_title('Original Light Curve')
ax1.grid(True)
ax1.legend()

# Rebinned
ax2.errorbar(binned_time, binned_rate, yerr=binned_rate_err, fmt='o', markersize=3,
             ecolor='gray', elinewidth=1, capsize=2, label=f'Rebinned ({bin_width_ms} ms)')
ax2.set_xlabel('Time since start (s)')
ax2.set_ylabel('Count Rate (counts/s)')
ax2.set_title('Rebinned Light Curve')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()


#exit()

from haar_power_mod import haar_power_mod

"""
results = haar_power_mod(
                rate, rate_err, min_dt=lc_bin_width, max_dt=100., tau_bg_max=0.01, nrepl=2,
                doplot=True, bin_fac=4, zerocheck=False, afactor=1., snr=3.,
                verbose=True, weight=True, file=file_name)
"""

def run_haar_power_segments(rate, rate_err, NN, lc_bin_width, file_name, **kwargs):
    """
    Run haar_power_mod on the full rate array or segmented based on NN.

    Parameters:
        rate        : array-like, count rate
        rate_err    : array-like, rate errors
        NN          : int, determines number of segments
        lc_bin_width: float, min_dt value
        file_name   : str, used in haar_power_mod

        kwargs      : additional parameters to pass to haar_power_mod

    Returns:
        results_list: list of results from haar_power_mod per segment
    """
    results_list = []


    # Determine number of segments
    if NN <= 1:
        num_segments = 1
    elif NN == 2:
        num_segments = 2
    else:
        num_segments = (NN - 1) * 2

    print(f"Splitting data into {num_segments} segment(s)")

    # Get segment size
    total_len = len(rate)
    seg_len = total_len // num_segments

    for i in range(num_segments):
        file_name = file_name + f"_{i}_{NN}"
        start_idx = i * seg_len
        end_idx = (i + 1) * seg_len if i < num_segments - 1 else total_len

        rate_seg = rate[start_idx:end_idx]
        rate_err_seg = rate_err[start_idx:end_idx]

        print(f"\nRunning segment {i+1}/{num_segments} (data points {start_idx}:{end_idx})")

        results = haar_power_mod(
            rate_seg,
            rate_err_seg,
            min_dt=lc_bin_width,
            max_dt=100.,
            tau_bg_max=0.01,
            nrepl=2,
            doplot=True,
            bin_fac=4,
            zerocheck=False,
            afactor=-1.,
            snr=3.,
            verbose=True,
            weight=True,
            file=file_name,
        )
        plt.close('all')
        results_list.append(results)

    return results_list

def run_haar_power_segments_all(rate, rate_err, NN, lc_bin_width, file_name, **kwargs):
    """
    Run haar_power_mod on multiple segmentations: [1, 2, (NN-1)*2]

    Parameters:
        rate        : array-like, count rate
        rate_err    : array-like, rate errors
        NN          : int, maximum segmentation level
        lc_bin_width: float, min_dt for haar_power_mod
        file_name   : str, used in haar_power_mod
        kwargs      : other parameters passed to haar_power_mod

    Returns:
        results_dict: dict with keys = number of segments, values = list of results per segment
    """
    results_dict = {}

    # Compute the segmentation levels up to NN
    segment_counts = [1]
    if NN >= 2:
        segment_counts.append(2)
    if NN >= 3:
        segment_counts.extend([(i - 1) * 2 for i in range(3, NN + 1)])

    total_len = len(rate)

    for num_segments in segment_counts:
        print(f"\n======= Processing {num_segments} segment(s) =========")
        seg_len = total_len // num_segments
        segment_results = []

        for i in range(num_segments):
            start_idx = i * seg_len
            end_idx = (i + 1) * seg_len if i < num_segments - 1 else total_len

            rate_seg = rate[start_idx:end_idx]
            rate_err_seg = rate_err[start_idx:end_idx]

            print(f"  Segment {i+1}/{num_segments}: indices {start_idx}:{end_idx}")

            result = haar_power_mod(
                rate_seg,
                rate_err_seg,
                min_dt=lc_bin_width,
                max_dt=100.,
                tau_bg_max=0.01,
                nrepl=2,
                doplot=True,
                bin_fac=4,
                zerocheck=False,
                afactor=1.,
                snr=3.,
                verbose=True,
                weight=True,
                file=file_name + f"_seg{num_segments}_{i+1}",
            )
            plt.close('all')
            segment_results.append(result)

        results_dict[num_segments] = segment_results

    return results_dict

#run_haar_power_segments(rate, rate_err, NN=2, lc_bin_width=lc_bin_width, file_name=file_name)

run_haar_power_segments_all(rate, rate_err, NN=3, lc_bin_width=orig_bin_width, file_name='Swift_J1644')
