import os
import sys
import yaml

from gdt.missions.fermi.gbm.tte import GbmTte

from gdt.missions.fermi.gbm.finders import TriggerFtp
from gdt.core.binning.unbinned import bin_by_time
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection
from gdt.missions.fermi.time import *
from gdt.core.phaii import Phaii
from datetime import datetime
import numpy as np 
import glob

from pathlib import Path

def trigger_file_list(trigger_dir, file_type, trigger_number, nai_dets=None):
    """
    Generate lists of file absolute paths and file names based on trigger directory, file type, and trigger number.

    Args:
    trigger_dir (str): The directory where the files are located.
    file_type (str): The type of files to include.
    trigger_number (str): The trigger number to filter files.

    Returns:
    tuple: A tuple containing two lists:
           - List of file absolute paths.
           - List of file names.
    """
    # Generate file absolute paths using glob based on file type and trigger number
    file_abs_path_list = glob.glob(trigger_dir + "/*glg_" + file_type + "_*_bn" + trigger_number + "*.fit")
    
    # Extract file names from absolute paths
    file_name_list = []
    for file in file_abs_path_list:  
        file_name_list.append(file.split('/')[-1])
    
    # Return the lists of file absolute paths and file names
    return file_abs_path_list, file_name_list


def trigger_process(
    file_write,
    trigger_directory,
    trigger_number,
    bw,
    bkgd_range,
    tt1,
    tend,
    t_del,
    en_lo,
    en_hi,
    nai_dets,
    t0
):
    """
    Processes GRB trigger data to extract signal and background light curves.

    Parameters:
        file_write (str): Path to output file.
        trigger_directory (str): Directory containing trigger files.
        trigger_number (int): Trigger number ID.
        bw (float): Bin width for lightcurve.
        bkgd_range (list of tuples): Background fit time ranges.
        tt1 (float): GRB start time.
        tend (float): GRB end time.
        t_del (float): Delay time for binning.
        en_lo (float): Lower energy bound.
        en_hi (float): Upper energy bound.
        bin_by_time (bool): Whether to bin by time.
        par_in (dict): Parameter input containing at least 'poly_order'.
        nai_dets (list): List of NaI detector names.
        t0 (float): GRB trigger time.

    Returns:
        tuple: (full_grb_time_lo_edge, full_grb_counts, full_back_counts) or (None, None, None) on failure.
    """
    try:
        print(f"Generating and saving data to \n{file_write}")
        energy_range_nai = (en_lo, en_hi)
        trange_bkg = (bkgd_range[0][0] - 10, bkgd_range[1][1] + 10)
        trange = (tt1 - 20, tend + 20)

        tte_list_path, _ = trigger_file_list(trigger_directory, "tte", trigger_number)
        tte_list_path.sort()

        #if len(tte_list_path) < 2:
        #    print("Insufficient TTE files for processing.")
        #    return None, None, None

        # Background selection
        tte_dict_bkg = []
        for i in tte_list_path[1:]:
            tte_time_selection = GbmTte.open(i).slice_time(trange_bkg)
            tte_dict_bkg.append(tte_time_selection)
        final_tte_bkg = GbmTte.merge(tte_dict_bkg)

        phaii_bkg_full_en = final_tte_bkg.to_phaii(bin_by_time, t_del)
        phaii_bkg = phaii_bkg_full_en.slice_energy(energy_range_nai)
        #data_bkg = phaii_bkg.to_lightcurve()

        backfitters = [BackgroundFitter.from_phaii(phaii_bkg, Polynomial, time_ranges=bkgd_range)]
        backfitters = GbmDetectorCollection.from_list(backfitters, dets=nai_dets[0:1])
        #backfitters.fit(order=int(par_in['poly_order']))
        backfitters.fit(order=2)
        print("DONE BackgroundFitter")

    except Exception as e:
        print("\n\n!!! Error in background FITTING !!!\n", str(e), "\n")
        return None, None, None

    try:
        # Signal selection
        tte_dict = []
        for i in tte_list_path[1:]:
            tte_time_selection = GbmTte.open(i).slice_time(trange)
            tte_dict.append(tte_time_selection)
        final_tte = GbmTte.merge(tte_dict)

        phaii_full_en = final_tte.to_phaii(bin_by_time, bw)
        phaii = phaii_full_en.slice_energy(energy_range_nai)
        data = phaii.to_lightcurve()

        # Compute full_grb_start index
        if t0 < data.lo_edges[0]:
            full_grb_start = 0
        else:
            full_grb_start = int(round((tt1 - data.lo_edges[0]) / bw))
        #print("full_grb_start", full_grb_start)

        # Number of bins between t0 and tend

        #print(tend, max(t0, data.lo_edges[0]), bw, (tend - max(t0,data.lo_edges[0])))
        #print("n_bins", n_bins)
        # Compute full_grb_end
        #full_grb_end = full_grb_start + n_bins/2
        full_grb_end = int(round((tend - data.lo_edges[0]) / bw))
        #print("full_grb_end", full_grb_end)

        sl = slice(full_grb_start, full_grb_end)

        full_grb_time_lo_edge = data.lo_edges[sl]
        full_grb_counts = data.counts[sl]
        full_grb_exposure = data.exposure[sl]
        print("DONE Data Selection")

    except Exception as e:
        print("\n\n!!! Error in DATA SELECTION !!!\n", str(e), "\n")
        return None, None, None

    try:
        # Interpolate background to GRB time bins
        bkgds = backfitters.interpolate_bins(full_grb_time_lo_edge, full_grb_time_lo_edge + bw)
        print("DONE interpolate_bins")
        bkgds = GbmDetectorCollection.from_list(bkgds, dets=nai_dets[0:1])

        lc_all = bkgds.integrate_energy(nai_args=energy_range_nai)
        print("DONE integrate_energy")

        full_back_counts = lc_all[0].rates * full_grb_exposure
        print("DONE Background interpolation")

        full_back_counts = np.maximum(0, np.array(full_back_counts))
  # prevent negatives

    except Exception as e:
        print("\n\n!!! Error in BACKGROUND INTERPOLATION !!!\n", str(e), "\n")
        return None, None, None

    try:
        file_write_path = os.path.join(trigger_directory, file_write)
        
        np.savez_compressed(
            file_write_path,
            full_grb_time_lo_edge=full_grb_time_lo_edge,
            full_grb_counts=full_grb_counts,
            full_back_counts=full_back_counts
        )
        print(f"File written: {file_write}")
        
    except Exception as e:
        print(f"\n\n!!! Error saving file {file_write} !!!\n", str(e), "\n")
        return None, None, None
    #print('full_grb_time_lo_edge', full_grb_time_lo_edge[0])
    #print('full_grb_time_lo_edge', full_grb_time_lo_edge[-1])
    #print('t0', t0)
    #print('tend', tend)

    return full_grb_time_lo_edge, full_grb_counts, full_back_counts

def trigger_process_int(
    file_write,
    trigger_directory,
    trigger_number,
    bw,
    bkgd_range,
    tt1,
    tend,
    en_lo,
    en_hi,
    nai_dets,
    t0
):
    """
    Processes GRB trigger data to extract signal and background light curves.

    Parameters:
        file_write (str): Path to output file.
        trigger_directory (str): Directory containing trigger files.
        trigger_number (int): Trigger number ID.
        bw (float): Bin width for lightcurve.
        bkgd_range (list of tuples): Background fit time ranges.
        tt1 (float): GRB start time.
        tend (float): GRB end time.
        t_del (float): Delay time for binning.
        en_lo (float): Lower energy bound.
        en_hi (float): Upper energy bound.
        bin_by_time (bool): Whether to bin by time.
        par_in (dict): Parameter input containing at least 'poly_order'.
        nai_dets (list): List of NaI detector names.
        t0 (float): GRB trigger time.

    Returns:
        tuple: (full_grb_time_lo_edge, full_grb_counts, full_back_counts) or (None, None, None) on failure.
    """
    try:
        print(f"Generating and saving data to \n{file_write}")
        energy_range_nai = (en_lo, en_hi)
        trange_bkg = (bkgd_range[0][0] - 10, bkgd_range[1][1] + 10)
        trange = (tt1 - 20, tend + 20)

        tte_list_path, _ = trigger_file_list(trigger_directory, "tte", trigger_number, nai_dets)
        tte_list_path.sort()

        #if len(tte_list_path) < 2:
        #    print("Insufficient TTE files for processing.")
        #    return None, None, None

        # Background selectio
        #data_bkg = phaii_bkg.to_lightcurve()

    except Exception as e:
        print("\n\n!!! Error in background FITTING !!!\n", str(e), "\n")
        return None, None, None

    try:
        # Signal selection
        tte_dict = []
        for i in tte_list_path[1:]:
            tte_time_selection = GbmTte.open(i).slice_time(trange)
            tte_dict.append(tte_time_selection)
        final_tte = GbmTte.merge(tte_dict)

        phaii_full_en = final_tte.to_phaii(bin_by_time, bw)
        phaii = phaii_full_en.slice_energy(energy_range_nai)
        data = phaii.to_lightcurve()

        # Compute full_grb_start index
        if t0 < data.lo_edges[0]:
            full_grb_start = 0
        else:
            full_grb_start = int(round((tt1 - data.lo_edges[0]) / bw))
        #print("full_grb_start", full_grb_start)

        # Number of bins between t0 and tend

        #print(tend, max(t0, data.lo_edges[0]), bw, (tend - max(t0,data.lo_edges[0])))
        #print("n_bins", n_bins)
        # Compute full_grb_end
        #full_grb_end = full_grb_start + n_bins/2
        full_grb_end = int(round((tend - data.lo_edges[0]) / bw))
        #print("full_grb_end", full_grb_end)

        sl = slice(full_grb_start, full_grb_end)

        full_grb_time_lo_edge = data.lo_edges[sl]
        full_grb_counts = data.counts[sl]
        full_back_counts = np.maximum(0, np.array(full_back_counts))
        print("DONE Data Selection")

    except Exception as e:
        print("\n\n!!! Error in DATA SELECTION !!!\n", str(e), "\n")
        return None, None, None

    try:
        file_write_path = os.path.join(trigger_directory, file_write)
        
        np.savez_compressed(
            file_write_path,
            full_grb_time_lo_edge=full_grb_time_lo_edge,
            full_grb_counts=full_grb_counts,
            full_back_counts=full_back_counts
        )
        print(f"File written: {file_write}")
        
    except Exception as e:
        print(f"\n\n!!! Error saving file {file_write} !!!\n", str(e), "\n")
        return None, None, None
    #print('full_grb_time_lo_edge', full_grb_time_lo_edge[0])
    #print('full_grb_time_lo_edge', full_grb_time_lo_edge[-1])
    #print('t0', t0)
    #print('tend', tend)

    return full_grb_time_lo_edge, full_grb_counts, full_back_counts