import os
import sys
import yaml


from gdt.missions.fermi.gbm.finders import TriggerFtp

from gdt.missions.fermi.time import *
from gdt.core.phaii import Phaii
from datetime import datetime
import numpy as np 
import csv
import pandas as pd
#from find_opt_res_pap import find_optimum_resolution_diff, convert_res_coarse
from mvt_analysis import run_mvt_analysis
from evolve_opt_res_fermi import compute_grb_time_bounds
from trigger_process import trigger_process

#from multiprocessing import Pool, cpu_count
#from concurrent.futures import ProcessPoolExecutor, as_completed


import csv
import os

import os
import csv


def main():
    flag_data_read = True
    trigger_config_file = 'config_MVT.yaml'
    with open(trigger_config_file, 'r') as f:
        config_trigger = yaml.safe_load(f)

    trigger_number = config_trigger['trigger_number']
    print(f"Trigger Number = {trigger_number}")
    bw = config_trigger['bw']
    delt = config_trigger['delt']
    T0 = config_trigger['T0']
    start_padding = config_trigger['start_padding']
    end_padding = config_trigger['end_padding']
    N = config_trigger['N']
    f1 = config_trigger['f1']
    f2 = config_trigger['f2']
    en_lo = config_trigger['en_lo']
    en_hi = config_trigger['en_hi']
    cores = config_trigger['cores']
    data_path = config_trigger['data_path']
    output_path = config_trigger['output_path']

    dets = config_trigger['det_list']
    T90 = 4  # or config_trigger['T90']
    bkgd_range = config_trigger['background_intervals']
    nai_dets = [d for d in dets if d.startswith('n')]
    config_file = config_trigger['data_path']
    en = f'{en_lo}to{en_hi}keV'


    temp_trigger_directory = "bn" + trigger_number
    trigger_directory = os.path.join(data_path, temp_trigger_directory)

    

    t_del = 0.064 if T90 <= 4.0 else 1.024

    delta_list = np.concatenate((
        np.arange(0.1, 1.0, 0.1),
        np.arange(1.0, 5.0, 1.0)
    ))

    T90_rounded = round(T90, 2)
    if np.max(delta_list) > T90_rounded and T90_rounded not in delta_list:
        delta_list = np.append(delta_list, T90_rounded)
        delta_list = np.sort(delta_list)

    valid_deltas = delta_list[delta_list <= T90_rounded]
    if valid_deltas.size == 0:
        raise ValueError("No valid delta ≤ T90")

    tt1, t0, tend = compute_grb_time_bounds(T0, T90, max(delta_list), start_padding, end_padding, end_t90=2.0)

    file_write = f"all_arrays_{trigger_number}_bw_{str(bw)}_delt_{delt}.npz"
    print(f'File Name = {file_write}\n')
    file_write_path = os.path.join(trigger_directory, file_write)

    if flag_data_read and os.path.exists(file_write_path):
        print(f"Reading data from {file_write}")
        data = np.load(file_write_path)
        time_edges = data["full_grb_time_lo_edge"]
        counts = data["full_grb_counts"]
        back_counts = data["full_back_counts"]
        #print('time_edges min = ', time_edges.min())
        #print('time_edges max = ', time_edges.max())
    else:
        time_edges, counts, back_counts = trigger_process(
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
        )

    run_mvt_analysis(
        trigger_number,
        time_edges,
        counts,
        back_counts,
        T0,
        T90,
        tt1,
        bw,
        valid_deltas,
        start_padding,
        end_padding,
        N,
        cores,
        f1,
        f2,
        en,
        output_folder=output_path
    )


if __name__ == '__main__':
    main()

#print(f'\n @@@@@@@@@@@@@@@@@@@@@@@ Analysis SAVED in {output_dir} @@@@@@@@@@@@@@@@@@@@@\n')