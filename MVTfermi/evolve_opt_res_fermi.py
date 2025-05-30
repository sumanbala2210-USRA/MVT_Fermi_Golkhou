import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np 
from datetime import datetime

from .find_opt_res_fermi import find_optimum_resolution_diff, convert_res_coarse, ExponentialFloat

import concurrent.futures
import PyPDF2



def truncate_xn_h(xn1, h):
    xn1 = np.asarray(xn1)
    h = np.asarray(h)
    
    # Compute where the differences are non-negative
    diffs = np.diff(xn1)
    
    # Find the last index where diff < 0 (non-increasing)
    bad_indices = np.where(diffs < 0)[0]
    
    if len(bad_indices) == 0:
        return xn1, h  # Fully increasing
    
    # Truncate at the first bad point from the right
    truncate_idx = bad_indices[-1] + 1
    return xn1[:truncate_idx], h[:truncate_idx]

def e_n(number):
    if number == 0:
        return "0"  # Special case for zero
    
    # Take the absolute value of the number
    abs_number = abs(number)
    
    # Convert number to scientific notation string
    scientific_notation = "{:.0e}".format(abs_number)
    
    # Split the scientific notation into base and exponent parts
    base, exponent = scientific_notation.split('e')
    
    # Convert the exponent to an integer and take its absolute value
    exponent = int(exponent)
    abs_exponent = abs(exponent)
    
    # Format the output as e_{exponent}
    formatted_output = f"{base}e_{abs_exponent}"
    
    return formatted_output



def combine_pdfs(pdf_list, output_filename):
    pdf_writer = PyPDF2.PdfWriter()

    for pdf in pdf_list:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])
        try:
            os.remove(pdf)
        except FileNotFoundError:
            pass

    with open(output_filename, 'wb') as out_file:
        pdf_writer.write(out_file)


def combine_pdfs_path(pdf_list, output_filename, path):
    pdf_writer = PyPDF2.PdfWriter()

    for pdf in pdf_list:
        #print(pdf_list)
        full_pdf_path = os.path.join(path, pdf)
        pdf_reader = PyPDF2.PdfReader(full_pdf_path)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)
        
        # Optionally remove the original file
        try:
            os.remove(full_pdf_path)
        except FileNotFoundError:
            pass

    output_path = os.path.join(path, output_filename)
    with open(output_path, 'wb') as out_file:
        pdf_writer.write(out_file)

def time_to_bin_index_uniform(time, start_time, bw, side='left'):
    """Convert time to bin index for uniformly spaced bins."""
    offset = (time - start_time) / bw
    if side == 'left':
        return int(np.floor(offset))
    elif side == 'right':
        return int(np.ceil(offset))
    else:
        return int(round(offset))
      
    

# Ensure this function is at the top level of your script
def process_iteration(params):
    try:
        trigger_number, grb_range, grb_count, bkg_range, bkg_count, tt1, tt2, bw, N, f1, f2, k, path, n1_array, n2_array, n3_array, n4_array, all_fig  = params
        return find_optimum_resolution_diff(trigger_number, grb_range, grb_count, bkg_range, bkg_count, tt1, tt2, bw, N, f1, f2, k, path, n1_array, n2_array, n3_array, n4_array, all_fig )
    except Exception as e:
        print(f'Error executing find_optimum_resolution_diff !!!!!!!!!!!!!!!\n{e}')
        return None
    
    
def slice_grb_data(tstart, tend, time_edges, counts, bkg_lo_edges, back_counts):
    """
    Slices the high-res GRB arrays into coarser bins for a given delt_i.
    Uses compute_grb_time_bounds internally.
    """
    mask = (time_edges >= tstart) & (time_edges < tend)
    return time_edges[mask], counts[mask], bkg_lo_edges[mask], back_counts[mask]



def compute_grb_time_bounds(T0, T90, delta, start_padding=False, end_padding=False, end_t90=2.0):
    """
    Compute the widest time range needed for all delta values, using delt_max.
    
    Returns:
        tuple: (tt1, t0, tend) to be used in trigger_process.
    """

    if start_padding:
        tt1 = T0 - min((start_padding+1)*delta,20)
    else:
        tt1 = T0 - 20

    t0  = tt1+ 5*delta

    if end_padding:
        tend = max(
            T0 + T90 * end_t90,
            T0 + T90 + (end_padding+1) * delta
        )
    else:
        tend = max(
            T0 + T90 * end_t90,
            T0 + T90 + 10*delta
        )

    #print(f"tt1 = {tt1}, t0 = {t0}, tend = {tend} delta = {delta}")
    return tt1, t0, tend

    

def evolve_optimum_resolution_diff(trigger_number,en, time_edges, counts, back_edges, back_counts, T0, T90, start_padding, end_padding, delt, bw, N, cores, f1=5,f2=3, path = None, tr_fixed=None, all_fig=False):
    tt1, t0, tend = compute_grb_time_bounds(T0, T90, delt, start_padding, end_padding)
   
    
    time_interval = T90 + (tend*delt)+(start_padding*delt)+T0
    nn = int(time_interval/delt+1)
    N_iter= int(1/bw)
    n1_array = np.zeros(N_iter, dtype=int)
    n2_array = np.zeros(N_iter, dtype=int)
    n3_array = np.zeros(N_iter, dtype=int)
    n4_array = np.zeros(N_iter, dtype=int)
    #print(f"tt1 = {tt1}, T0 = {T0}, T90 = {T90}, tend = {tend}, delt = {delt}, bw = {ExponentialFloat(bw)}, nn = {nn}, f1={f1}, f2={f2}")
    #info_trig = f' BN: {trigger_number}, {en}, T90={T90}, bw= {bw}, delt= {delt}, T0= {T0}, ({f1}, {f2})$\sigma$'
    op_tim = np.zeros(nn, dtype=np.float64)
    err_op_tim = np.zeros(nn, dtype=np.float64)
    signal = np.zeros(nn, dtype=np.float64)
    fig_list = [None] * nn
    tr = np.zeros(nn, dtype=np.float64)

    add_index = int(delt/bw)

    padding = bw *10
    grb_start = np.searchsorted(time_edges, tt1-padding)
    #grb_start = int((tt1 - padding - time_edges[0]) / bw)+1
    #grb_end = np.searchsorted(lc_lo_edges, tt1+delt+padding)
    grb_end = grb_start + int((delt+2*padding)/bw)
    #grb_end1 = grb_start1 + int((delt+2*padding)/bw)
    #print('max time_edges=', max(time_edges))
    #print('min time_edges=', min(time_edges))
    #print('min counts=', min(counts))
    #print('max counts=', max(counts))
    #print('min back_counts=', min(back_counts))
    #print('max back_counts=', max(back_counts))
    #print('tt1=', tt1)
    #print('tmax=', tt1+delt+padding)
    #print('tend=', tend)
    #print(f'grb_start= {grb_start}, grb_end = {grb_end}')
    #print(len(time_edges), len(counts), len(back_counts))
    #print(f'grb_start= {time_edges[grb_start]}, grb_end = {time_edges[grb_end]}')
    #print(f'grb_start1= {time_edges[grb_start1]}, grb_end1 = {time_edges[grb_end1]}')
    #print(f'New grb end = {end_index}')
    #print(f'time_edges bw = {time_edges[grb_start]-time_edges[grb_start+1]}')

    e_bw = e_n(bw)

    now = datetime.now()
    # Format the date and time as a string
    time_now = now.strftime("%d_%m_%H:%M:%S")
    file_name = f'MVT_bn{trigger_number}_{en}_bw_{e_bw}_delt_{round(delt,2)}s_{time_now}'
    pdf_name = file_name+'.pdf'
    npz_name = file_name + '.npz'
    csv_name = file_name + '.csv'
        
    log_MVT_fig = f'LOG_MVT_bn{trigger_number}_{time_now}.pdf'
    if path:
        #pdf_path = os.path.join(path, pdf_name)
        npz_path = os.path.join(path, npz_name)
        log_MVT_fig_path = os.path.join(path, log_MVT_fig)
        csv_path = os.path.join(path, csv_name)

    else:
        #pdf_path = pdf_name 
        npz_path = npz_name
        log_MVT_fig_path = log_MVT_fig
        csv_path = csv_name
    
    tasks = []
    
    for k in range(nn):
        tt1 += delt
        tt2 = tt1 + delt

        if tt2 > T0 + T90 + (delt * tend) or tt2 + delt > max(time_edges):
            break
        tr[k] = tt1
        

        grb_start += add_index 
        grb_end += add_index
        bkg_end = grb_end*1.03
        grb_range = time_edges[int(grb_start):int(grb_end)]
        grb_count = counts[int(grb_start):int(grb_end)]

        bkg_range = time_edges[int(grb_start):int(bkg_end)]
        bkg_count = back_counts[int(grb_start):int(bkg_end)]
        #if k == 0:
            #print(f"tt1 = {tt1}, tt2 = {tt2}")
            #print(f"Processing iteration {k+1}/{nn}: tt1 = {tt1}, tt2 = {tt2}, bw = {bw}, delt = {delt}")
            #print(f'grb_range = {grb_range[0]} to {grb_range[-1]}, grb_count = {len(grb_count)}\n')
        #print(f'bkg_range = {bkg_range[0]} - {bkg_range[-1]}, bkg_count = {len(bkg_count)}')
        #print(f'grb_start = {grb_start}, grb_end = {grb_end}, bkg_end = {bkg_end}')
        #print(f'grb_count max = {np.max(grb_count)}, grb_count min = {np.min(grb_count)}')
        #print(f'bkg_count max = {np.max(bkg_count)}, bkg_count min = {np.min(bkg_count)}')
        #params = (grb_range, grb_count, bkg_range, bkg_count, tt1, tt2, bw, N, f1, f2)
        #print(f'starting for {k
        '''
        if k ==0:
            xn = bkg_range.copy()  # Initial bin edges
            print('K =',0)
            for i in range(N_iter):
                n1 = np.searchsorted(xn, tt1, side='left')
                n2 = np.searchsorted(xn, tt2, side='right')
                n1_array[i] = n1
                n2_array[i] = n2
                
                
                # Matching your grb slicing logic: grb = grb_range[n1:n2+1]
                if (n2 - n1 + 1) <= 2:
                    continue  # skip this resolution level
                xn,_ = convert_res_coarse(bkg_range, bkg_count, i + 1)
        '''
        if k ==0:
            xn_bkg = bkg_range.copy()  # Initial bin edges
            xn_src = grb_range.copy()
            #print('K =',0)
            for i in range(N_iter):
                n1 = np.searchsorted(xn_src, tt1, side='left')
                n2 = np.searchsorted(xn_src, tt2, side='right')
                n3 = np.searchsorted(xn_bkg, tt1, side='left')
                n4 = np.searchsorted(xn_bkg, tt2, side='right')
                n1_array[i] = n1
                n2_array[i] = n2
                n3_array[i] = n3 
                n4_array[i] = n4 
                
                
                # Matching your grb slicing logic: grb = grb_range[n1:n2+1]
                if (n2 - n1 + 1) <= 2:
                    continue  # skip this resolution level
                xn_bkg,_ = convert_res_coarse(bkg_range, bkg_count, i + 1)
                xn_src,_ = convert_res_coarse(grb_range, grb_count, i + 1)
        #print(f"n1_array = {max(n1_array)}, n2_array = {max(n2_array)}, n3_array = {max(n3_array)}, n4_array = {max(n4_array)}\n")

        tasks.append((trigger_number, grb_range, grb_count, bkg_range, bkg_count, tt1, tt2, bw, N, f1, f2, k, path, n1_array, n2_array, n3_array, n4_array, all_fig))
    exit()
    max_workers = min(cores, len(tasks))  # Use fewer workers if needed
    print(f"len(tasks)= {len(tasks)}, nn={nn}")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_iteration, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    bw_opt, bin_err, signal_out, k, fig_name = result
                    signal[k] = signal_out
                      
                    op_tim[k] = bw_opt
                    err_op_tim[k] = bin_err
                    results.append(result)
                    if fig_name:
                        fig_list[k] = fig_name
                else:
                    # Handle the case where result is None due to an exception
                    op_tim[k] = np.nan
                    err_op_tim[k] = np.nan
                    signal[k] = np.nan
            except Exception as e:
                pass

    fig_list=fig_list[:len(tasks)]
    tr = tr[:len(tasks)]
    op_tim = op_tim[:len(tasks)]
    err_op_tim = err_op_tim[:len(tasks)]
    signal = signal[:len(tasks)]
    
    kk = np.where((op_tim >= 1.e-7) & (op_tim <= 1.0))[0]
    op_tim = op_tim[kk]
    tr = tr[kk]
    err_op_tim = err_op_tim[kk]
    signal = signal[kk]

    xn1, h1 = convert_res_coarse(time_edges, counts, int(delt / (bw*10)))

    xn, h = truncate_xn_h(xn1,h1)
    #h = h1[:-1]
    yerr = np.sqrt(h)
    # Calculate effective op_tim considering errors
    effective_op_tim = op_tim + err_op_tim


    ##### Find GRB MVT (signal > 0) ############
    mask = signal > 0

    op_masked = op_tim[mask]
    err_masked = err_op_tim[mask]
    
    if op_masked.size > 0:
        mean_op = np.mean(op_masked)
        z_scores = (op_masked - mean_op) / err_masked
        masked_min_index = np.argmin(z_scores)
        min_index = np.where(mask)[0][masked_min_index]
    
        if tr_fixed:
            min_index = np.where(np.isclose(tr, tr_fixed))[0][0]
    
        min_op_tim = op_tim[min_index]
        min_tr = tr[min_index]
        min_err_op_tim = err_op_tim[min_index]
    else:
        min_op_tim = np.nan
        min_tr = np.nan
        min_err_op_tim = np.nan
    
    mask_bkg = signal < 0
    op_masked_bkg = op_tim[mask_bkg]
    err_masked_bkg = err_op_tim[mask_bkg]
    
    if op_masked_bkg.size > 0:
        ## 1. Z-score method
        mean_op_bkg = np.mean(op_masked_bkg)
        z_scores_bkg = (op_masked_bkg - mean_op_bkg) / err_masked_bkg
        masked_min_index_bkg = np.argmin(z_scores_bkg)
        min_index_bkg = np.where(mask_bkg)[0][masked_min_index_bkg]
    
        min_op_tim_bkg = op_tim[min_index_bkg]
        min_err_op_tim_bkg = err_op_tim[min_index_bkg]
    
        # Significance using z-score method
        total_err_z = np.sqrt(err_op_tim**2 + min_err_op_tim_bkg**2)
        significance_z = (min_op_tim_bkg - op_tim) / total_err_z

        ## 3. Significance using weighted mean
        weights = 1 / err_masked_bkg**2
        weighted_mean_bkg = np.sum(op_masked_bkg * weights) / np.sum(weights)
        weighted_mean_err = np.sqrt(1 / np.sum(weights))
        total_err_weighted = np.sqrt(err_op_tim**2 + weighted_mean_err**2)
        significance_weighted = (weighted_mean_bkg - op_tim) / total_err_weighted

    else:
        significance_z = significance_weighted = np.full_like(op_tim, np.nan)
        min_index_bkg = 0  # dummy index to avoid errors if used later
    
    # Significance values at the detected minimum
    significance_min_z = significance_z[min_index]
    significance_min_weighted = significance_weighted[min_index]

    signal_val = signal[min_index]


    val_fmt = ExponentialFloat(min_op_tim, n=2, pow=-3)
    err_fmt = ExponentialFloat(min_err_op_tim, n=2, pow=val_fmt.pow)  # match exponent

    min_mvt_final = f'({val_fmt.scaled_str()} ± {err_fmt.scaled_str()})ms'

    mpl.rcParams.update({'font.size': 14})
    min_text = (
                f"MVT: {min_mvt_final}\n"
                f"Δt: {min_tr:.2f} – {(min_tr + delt):.2f}s\n"
                f"significance (z-score): {significance_min_z:.2f}\n"
                f"significance (weighted mean): {significance_min_weighted:.2f}\n"
                f"SNR: {signal_val:.2f}"
            )
    
    tr = np.append(tr, tr[-1] + delt)
    tr_centers = 0.5 * (tr[:-1] + tr[1:])
    text_box_color = 'lavender'    
    
    # Plot 2: New plot with linear scale on ax2
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    
    ax1.plot(xn, h, color='k', linestyle='-', alpha=0.2, label='Lightcurve')
    ax1.fill_between(xn, h - yerr, h + yerr, color='lightgray', alpha=0.5)
    
    # Text box with MVT result
    ax1.text(
        0.98, 0.97, min_text,
        transform=ax1.transAxes,
        ha='right', va='top',  # Position box at top-right
        multialignment='left',  # Center-align text within box
        bbox=dict(boxstyle='round,pad=0.5', facecolor=text_box_color, edgecolor='gray', alpha=0.8),
        fontsize=14,
        linespacing=1.2
    )
    
    # Primary y-axis
    ax1.set_xlabel('Time since Trigger (s)', fontsize=14)
    ax1.set_ylabel('Counts per bin', color='k', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='k')
    
    # Secondary y-axis for optimum timescale
    tr_error = np.full_like(tr_centers, delt/2)
    
    ax2 = ax1.twinx()
    for x, y, err, x_err, s in zip(tr_centers, op_tim, err_op_tim, tr_error, signal):
        if s < 0:
            color = '#D62728'
            label = 'SNR < 0'
            fmt  = 'v'
        elif s < 3:
            color = '#FFA500'  
            label = 'SNR < 3'
            fmt = 'D'
        else:
            color = '#006400'  # '#2CA02C'
            label = 'SNR ≥ 3'
            fmt = 'o'
        ax2.errorbar(x, y, yerr=err, xerr=x_err, fmt=fmt, color=color, label=label, alpha=0.8)

    min_color = 'cyan'
    ax2.plot(
        min_tr+delt/2, min_op_tim,
        marker='*', markersize=15,
        markerfacecolor=min_color,
        markeredgecolor='black',
        label='Min MVT',
        zorder=10  # Keep it on top of others
    )
    # Add vertical line at min_mvt_final
    #ax2.axvline(min_mvt_final, color='blue', linestyle='--', linewidth=1.5, label='Min MVT')
    
    
    #ax2.annotate('MVT', (highlight_x, highlight_y), textcoords="offset points", xytext=(0, 10), ha='center', color='blue')
    
    # Axis settings
    ax2.set_ylabel('MVT (s)', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_yscale('log')
    #ax2.set_ylim(bottom=1e-4)
    #ax2.set_ylim(1e-4, 1e3)
    
    # x-limits and grid
    ax1.set_xlim(tr.min() - 0.5, tr.max() + 0.5)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.grid(True, which='both', axis='x', linestyle='--', alpha=0.2)
    
    # Merge and deduplicate legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    combined = dict(zip(labels2, handles2))
    ax1.legend(handles1 + list(combined.values()), labels1 + list(combined.keys()), loc='lower right')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_MVT_fig_path)
  
    #fig_list.append(linear_MVT_fig)
    fig_list.append(log_MVT_fig)
    fig_list = [fig for fig in fig_list if fig is not None]

    if path: 
        combine_pdfs_path(fig_list, pdf_name,path)
    else:
        combine_pdfs(fig_list, pdf_name)
    print(f"$$$$$$$$$$$$$$$  delata = {delt}  $$$$$$$$$$$$$$$")
    print(min_text)
    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"MVT w.r.t Lightcurve: {pdf_name} ")
    
    plt.close()
    np.savez(npz_path,
         time_bin=tr_centers,
         time_bin_error=tr_error,
         mvt=op_tim,
         mvt_error=err_op_tim,
         significance_z=significance_z,
         significance_weighted=significance_weighted,
         snr=signal)

    print(f"MVT analysis npz File saved: {npz_name} ")
    
    # Stack the data including all significance methods
    data = np.column_stack((tr_centers, tr_error, op_tim, err_op_tim,
                            significance_z, significance_weighted,
                            signal))
    
    # File name and saving the data as CSV
    np.savetxt(csv_path, data, delimiter=",",header="time_bin, time_bin_error, mvt, mvt_error, significance_z, significance_weighted, snr", comments='', fmt='%.7g')

    print(f"MVT analysis CSV File saved: {csv_name} ")

    return min_tr, min_op_tim, min_err_op_tim, significance_min_z, significance_min_weighted, signal_val


    








 
