import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams

from datetime import datetime
import numpy as np 
import csv
import pandas as pd

from scipy.interpolate import interp1d

from .evolve_opt_res_fermi import evolve_optimum_resolution_diff
from .find_opt_res_fermi import ExponentialFloat



def write_mvt_csv_row(
    csv_path,  # this will be 'delt'
    delt,
    tr,
    min_mvt,
    error_mvt,
    significance_z,
    significance_weighted,
    snr
):
    headers = [
        'delt',
        'Time',
        'MVT',
        'error',
        'Significance (z-score)',
        'Significance (weighted mean)',
        'SNR'
    ]

    values = [
        round(delt, 4),
        round(tr, 3),
        round(min_mvt, 7),
        round(error_mvt, 7),
        round(significance_z, 2),
        round(significance_weighted, 2),
        round(snr, 2)
    ]

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow(values)



def MVT_delt_plot(file_name, plot_name, path=None):

    # Setup file paths
    if path:
        file_path = os.path.join(path, file_name)
        plot_path = os.path.join(path, plot_name)
    else:
        file_path = file_name
        plot_path = plot_name

    # Load data
    df = pd.read_csv(file_path)
    df = df.sort_values('delt')

    delt = df['delt'].values
    mvt = df['MVT'].values * 1000  # ms
    error = df['error'].values * 1000  # ms
    delt_err = 0.05 * delt  # 5% error placeholder

    # Significance methods
    sig_dict = {
        'z-score': df['Significance (z-score)'].values,
        'weighted mean': df['Significance (weighted mean)'].values,
    }

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(12,6), dpi=300)
    rcParams.update({'font.size': 13})
    axes = axes.flatten()

    # Significance bins and styling
    sig_bins = [(0, 1), (1, 3), (3, 5), (5, np.inf)]
    markers = ['o', 's', 'D', '*']  # circle, square, diamond, triangle
    colors = ['k', 'dimgray', 'lightgray', 'white']
    labels = ['σ < 1', '1 ≤ σ < 3', '3 ≤ σ < 5', 'σ ≥ 5']

    for i, (label, sig) in enumerate(sig_dict.items()):
        ax = axes[i]

        # Interpolation to find delt at 3σ and 5σ
        interp = interp1d(sig, delt, bounds_error=False, fill_value='extrapolate')
        delt_at_3 = interp(3.0)
        delt_at_5 = interp(5.0)

        # Plot points by significance range with distinct markers/colors
        for (low, high), marker, color, label_text in zip(sig_bins, markers, colors, labels):
            mask = (sig >= low) & (sig < high)
            if np.any(mask):
                ax.scatter(delt[mask], mvt[mask], marker=marker, c=color,
                           s=80, edgecolor='k', label=label_text, zorder=3)

        # Error bars
        ax.errorbar(delt, mvt, yerr=error, xerr=delt_err, fmt='none',
                    ecolor='gray', alpha=0.6, capsize=3, zorder=2)

        # Vertical lines for 3σ and 5σ
        ax.axvline(delt_at_3, color='red', linestyle='--', linewidth=1.2,
                   label=f'Δ at 3σ ≈ {delt_at_3:.3f}', zorder=4)
        ax.axvline(delt_at_5, color='green', linestyle='--', linewidth=1.2,
                   label=f'Δ at 5σ ≈ {delt_at_5:.3f}', zorder=4)

        # Axis settings
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Δ (bin width)')
        ax.set_ylabel('MVT (ms)')
        ax.set_title(f'Significance: {label}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Show only custom legend entries (avoid duplicates)
        if i == 0:
            legend_elements = [
                Line2D([0], [0], marker=m, color='w', label=lbl, markerfacecolor=c,
                       markersize=8, markeredgecolor='k')
                for m, c, lbl in zip(markers, colors, labels)
            ] + [
                Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='Δ at 3σ'),
                Line2D([0], [0], color='green', lw=1.5, linestyle='--', label='Δ at 5σ'),
            ]
            ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

    # Super title and layout
    fig.suptitle("MVT vs Δ for Different Significance Methods", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path)
    plt.close()



def grb_mvt_significance(
    delta,
    trigger_number,
    T0,
    T90,
    bw,tt1,
    time_edges, counts, back_counts,
    output_dir,
    output_file_path,
    start_padding,
    end_padding,
    N,
    cores,
    f1,
    f2,
    en,
    all_fig
):
    delt= round(delta, 2)
    print(f'\n$$$$$$$$$$$$$  Starting MVT calculation for delta = {delt} $$$$$$$$$$$$$')
    
    try:
        #print('full_grb_time_lo_edge', full_grb_time_lo_edge[0])
        #print('full_grb_time_lo_edge max', full_grb_time_lo_edge[-1])
        tr, min_mvt, error_mvt, significance_z, significance_weighted, snr = evolve_optimum_resolution_diff(
            trigger_number,
            en,
            time_edges,
            counts,
            time_edges,
            back_counts,
            T0,
            T90,
            start_padding,
            end_padding,
            delt,
            bw,
            N,
            cores,
            f1,
            f2,
            path=output_dir,
            all_fig=all_fig,
        )
    
    except Exception as e:
        print(f"\n!!!!!!!! Error computing MVT for {trigger_number} at delta={delt} !!!!!!!!\n{e}")
        
        # Set all values to NaN or fallback
        tr = -100
        min_mvt = error_mvt = np.nan
        significance_z = significance_weighted = snr = np.nan
    
    write_mvt_csv_row(
        csv_path=output_file_path,
        delt=delt,
        tr=tr,
        min_mvt=min_mvt,
        error_mvt=error_mvt,
        significance_z=significance_z,
        significance_weighted=significance_weighted,
        snr=snr
    )
  # or float('-inf') to clearly mark a failure
        

    print(f'####################  Done for delta = {delt} ####################')
    plt.close('all')

    return significance_weighted, min_mvt, error_mvt, tr  # This is the value used in binary search


def binary_search_mvt(valid_deltas, trigger_number, T0, T90, tt1,
                    bw,time_edges,   counts, back_counts, output_path,
                    output_file_path, start_padding, end_padding, N,
                    cores,f1, f2, en, threshold=5.0, all_delta=False, all_fig=False, limit=True):
    

    cache = {}
    

    if all_delta:
        results = []
        for delta in valid_deltas:
            significance, mvt, mvt_error, tr = grb_mvt_significance(
                delta, trigger_number, T0, T90, bw,tt1, time_edges, counts, back_counts, output_path, output_file_path, 
                start_padding, end_padding, N, cores, f1, f2, en, all_fig=all_fig
                
            )
            results.append({
                "delta": delta,
                "significance": significance,
                "mvt": mvt,
                "mvt_error": mvt_error,
                "tr": tr
            })
        return {"all_results": results}

    if limit:
    # -------- Upper Limit Check --------
        upper_limit_delta = round(min(max(valid_deltas[-1], T90), 4.0), 2) 
        print(f'Running MVT for Upper limit delta = {upper_limit_delta}')
        significance, mvt, mvt_error, tr = grb_mvt_significance(
                    upper_limit_delta, trigger_number, T0, T90, bw,tt1, time_edges, counts, back_counts, output_path, output_file_path, 
                    start_padding, end_padding, N, cores, f1, f2, en, all_fig=all_fig
                    
                )
        if significance <= threshold:
            return {
                "tr": tr,
                "delta": upper_limit_delta,
                "mvt": mvt,
                "mvt_error": mvt_error,
                "significance": significance,
                "is_upper_limit": True
            }

        cache[upper_limit_delta] = (significance, mvt, mvt_error, tr)
    # -------- Binary Search Only --------
    low, high = 0, len(valid_deltas) - 1
    result = {
        "tr": None,
        "delta": None,
        "mvt": None,
        "mvt_error": None,
        "significance": None,
        "is_upper_limit": False
    }

    last_tried_delta = None
    last_significance = None
    last_mvt = None
    last_mvt_error = None
    last_tr = None

    while low <= high:
        mid = (low + high) // 2
        delta = valid_deltas[mid]

        if delta not in cache:
            significance, mvt, mvt_error, tr = grb_mvt_significance(
                delta, trigger_number, T0, T90, bw, tt1, time_edges, counts, back_counts,
                output_path, output_file_path, start_padding, end_padding, N,
                cores, f1, f2, en, all_fig=all_fig
            )
            cache[delta] = (significance, mvt, mvt_error, tr)
        else:
            significance, mvt, mvt_error, tr = cache[delta]

        # Track the last tried values
        last_tried_delta = delta
        last_significance = significance
        last_mvt = mvt
        last_mvt_error = mvt_error
        last_tr = tr

        if significance > threshold:
            result.update({
                "tr": tr,
                "delta": delta,
                "mvt": mvt,
                "mvt_error": mvt_error,
                "significance": significance
            })
            high = mid - 1
        else:
            low = mid + 1

    # If result was never updated (no delta passed the threshold)
    if result["delta"] is None:
        result.update({
            "tr": last_tr,
            "delta": last_tried_delta,
            "mvt": last_mvt,
            "mvt_error": last_mvt_error,
            "significance": last_significance,
            "is_upper_limit": True  # optional: indicate that this is a fallback
        })

    return result




def run_mvt_analysis(trigger_number, time_edges, counts, back_counts, T0, T90, tt1, bw,
                     valid_deltas, start_padding, end_padding, N, cores, f1, f2, en,
                     output_folder=None, all_delta=False, all_fig = True, delta=None, limit=True):
    
    time_now = datetime.now().strftime("%y_%m_%d_%H:%M:%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = f'BN{trigger_number}_MVT'#_{time_now}'   ############## Change this to your desired output directory name
    output_path = os.path.join(output_folder or script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    output_file_path = os.path.join(output_path, output_dir + '.csv')
    output_plot = output_dir + '.pdf'
    output_plot_path = os.path.join(output_path, output_plot)

    # --- CASE 1: Run for a specific delta ---
    if delta is not None:
        print(f"\n@@@@@@@@@@@@@@ Running MVT analysis for specific delta = {delta:.2f}s @@@@@@@@@@@@@@@")
        significance, mvt, mvt_error, tr = grb_mvt_significance(
            delta, trigger_number, T0, T90, bw, tt1, time_edges, counts, back_counts,
            output_path, output_file_path, start_padding, end_padding, N, cores, f1, f2, en, all_fig=all_fig
        )
        val_fmt = ExponentialFloat(mvt, n=2, pow=-3)
        err_fmt = ExponentialFloat(mvt_error, n=2, pow=val_fmt.pow)
        formatted = f'({val_fmt.scaled_str()} ± {err_fmt.scaled_str()})ms'

        print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(f'MVT = {formatted}')
        print(f'Δ = {delta:.2f}, Tr = {tr:.2f} – {tr + delta:.2f}s')
        print(f'Significance = {significance:.2f}')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(f'\n@@@@@@@@@@@@@@@@@ Analysis SAVED in {output_dir} @@@@@@@@@@@@@@@@@\n')

        return tr, delta, mvt, mvt_error, significance, False

    # --- CASE 2: Full binary search or all deltas ---
    result = binary_search_mvt(
        valid_deltas, trigger_number, T0, T90, tt1, bw, time_edges, counts, back_counts,
        output_path, output_file_path, start_padding, end_padding, N, cores, f1, f2, en,
        threshold=5.0, all_delta=all_delta, all_fig=all_fig, limit=limit
    )
    
    MVT_delt_plot(output_file_path, output_plot_path)
    print(f'\nPlots are saved to \n{output_plot}')

    if all_delta:
        print('\n@@@@@@@@@ Full Delta Scan Results @@@@@@@@@')
        for res in result["all_results"]:
            val_fmt = ExponentialFloat(res["mvt"], n=2, pow=-3)
            err_fmt = ExponentialFloat(res["mvt_error"], n=2, pow=val_fmt.pow)
            print(f'Δ={res["delta"]:>4.2f}s: '
                  f'MVT=({val_fmt.scaled_str()}±{err_fmt.scaled_str()})ms, '
                  f'Sig={res["significance"]:.2f}, '
                  f'Tr={res["tr"]:.2f}–{res["tr"] + res["delta"]:.2f}s')
        print(f'\n@@@@@@@@@@@@@@@@@@@@@@@ Analysis SAVED in {output_dir} @@@@@@@@@@@@@@@@@@@@@\n')
        return result["all_results"]

    # Binary search result
    val_fmt = ExponentialFloat(result["mvt"], n=2, pow=-3)
    err_fmt = ExponentialFloat(result["mvt_error"], n=2, pow=val_fmt.pow)
    formatted = f'({val_fmt.scaled_str()} ± {err_fmt.scaled_str()})ms'

    if result["is_upper_limit"]:
        print('\n<<<<<<<<<<<<<<<<<  NO MVT >>>>>>>>>>>>>>>>>>>')
        print(f'MVT > {formatted}')
        print(f'Delta = {result["delta"]:.2f} (upper limit)')
    else:
        print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(f'MVT = {formatted}')
        print(f'Best delta = {result["delta"]:.2f}')

    print(f"Δt: {result['tr']:.2f} – {(result['tr'] + result['delta']):.2f}s")
    print(f'Significance = {result["significance"]:.2f}')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'\n@@@@@@@@@@@@@@@@@ Analysis SAVED in {output_dir} @@@@@@@@@@@@@@@@@\n')
    
    return result["tr"], result["delta"], result["mvt"], result["mvt_error"], result["significance"], result["is_upper_limit"]
