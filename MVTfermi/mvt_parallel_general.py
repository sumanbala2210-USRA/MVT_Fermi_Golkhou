import os
import sys
import yaml
import argparse
import numpy as np 

from .mvt_analysis import run_mvt_analysis
from .evolve_opt_res_fermi import compute_grb_time_bounds


from .core import (
    str2bool,
    parse_args_general,
    load_and_merge_config
)



def mvtgeneral(
    delta=None, limit=None, trigger_number=None, bw=None, T0=None,
    T90=None, start_padding=None, end_padding=None, N=None, f1=None, f2=None,
    en_lo=None, en_hi=None, cores=None, file_path=None, output_path=None,
    all_delta=None, time_edges=None, counts=None, back_counts=None, config=None
):
    func_args = {
        "delta": delta, "limit": limit, "bw": bw, "T0": T0,
        "T90": T90, "start_padding": start_padding, "end_padding": end_padding,
        "N": N, "f1": f1, "f2": f2, "cores": cores, "output_path": output_path,
        "file_path": file_path, "config": config
    }
    func_args = {
                k: v for k, v in locals().items()
                if k not in {'time_edges', 'counts', 'back_counts'}
            }
    default_cfg_file = "config_MVT_general.yaml"  # grabs all function args as dict
    config = load_and_merge_config(func_args, cli_args=None, default_config_file=default_cfg_file, parse_fn=parse_args_general)
    

    # continue with specific logic for mvtfermi using merged config dict
    # 2. Post-process delta

    delta_raw = str(config.get('delta')).strip().lower() if config.get('delta') is not None else None

    if delta_raw == 'all':
        config['delta'] = None
        config['all_delta'] = True
    elif delta_raw in (None, 'none'):
        config['delta'] = None
        config['all_delta'] = False
    else:
        try:
            config['delta'] = float(config['delta'])
            config['all_delta'] = False
        except (ValueError, TypeError):
            raise ValueError(f"Invalid delta value: {config['delta']}")

    config['all_delta'] = all_delta or config.get('all_delta', False)
    config['limit'] = str2bool(config.get('limit', True))

    T90 =config['T90']# config.get('T90', 4)
    en = f"{config['en_lo']}to{config['en_hi']}keV"

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

    tt1, t0, tend = compute_grb_time_bounds(
        config['T0'], T90, max(delta_list),
        config['start_padding'], config['end_padding'], end_t90=2.0
    )
    # Load data if not provided as function inputs
    #if config['time_edges'] is None or config['counts'] is None or config['back_counts'] is None:
    if os.path.exists(config['file_path']):
        print(f"Reading data from {config['file_path']}")
        data = np.load(config['file_path'])
        time_edges = data["full_grb_time_lo_edge"]
        counts = data["full_grb_counts"]
        back_counts = data["full_back_counts"]
    else:
        raise ValueError("No input arrays provided and no .npz file found.")

    print('\n')
    print("Final config:".center(20,'*'))
    for k, v in config.items():
        print(f"{k}: {v}")

    #exit()
    # Finally run the analysis
    run_mvt_analysis(
        config['trigger_number'],
        time_edges,
        counts,
        back_counts,
        config['T0'],
        T90,
        tt1,
        config['bw'],
        valid_deltas,
        config['start_padding'],
        config['end_padding'],
        config['N'],
        config['cores'],
        config['f1'],
        config['f2'],
        en,
        output_folder=config['output_path'],
        all_delta=config['all_delta'],
        delta=config['delta'],
        limit=config['limit']
    )

if __name__ == "__main__":
    mvtgeneral()


