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


'''
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
'''
def mvtgeneral(
    delta=None, limit=None, trigger_number=None, bw=None, T0=None,
    T90=None, start_padding=None, end_padding=None, N=None, f1=None, f2=None,
    en_lo=None, en_hi=None, cores=None, file_path=None, output_path=None,
    all_delta=None, time_edges=None, counts=None, back_counts=None, config=None
):
    skip_keys = {'time_edges', 'counts', 'back_counts', 'skip_keys'}
    
    # Grab all locals except skip_keys and None values
    func_args = {
        k: v for k, v in locals().items()
        if k not in skip_keys and v is not None and k != "config"
    }
    # Add config only if it is set (not None)
    if config is not None:
        func_args['config'] = config

    default_cfg_file = "config_MVT_general.yaml"
    config_dic = load_and_merge_config(
        func_args,
        cli_args=None,
        default_config_file=default_cfg_file,
        parse_fn=parse_args_general
    )
    # rest of your function ...
    # continue with specific logic for mvtfermi using merged config dict
    # 2. Post-process delta

    delta_raw = str(config_dic.get('delta')).strip().lower() if config_dic.get('delta') is not None else None

    if delta_raw == 'all':
        config_dic['delta'] = None
        config_dic['all_delta'] = True
    elif delta_raw in (None, 'none'):
        config_dic['delta'] = None
        config_dic['all_delta'] = False
    else:
        try:
            config_dic['delta'] = float(config_dic['delta'])
            config_dic['all_delta'] = False
        except (ValueError, TypeError):
            raise ValueError(f"Invalid delta value: {config_dic['delta']}")

    config_dic['all_delta'] = all_delta or config_dic.get('all_delta', False)
    config_dic['limit'] = str2bool(config_dic.get('limit', True))

    T90 =config_dic['T90']# config_dic.get('T90', 4)
    en = f"{config_dic['en_lo']}to{config_dic['en_hi']}keV"

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
        config_dic['T0'], T90, max(delta_list),
        config_dic['start_padding'], config_dic['end_padding'], end_t90=2.0
    )
    # Load data if not provided as function inputs
    if time_edges is None or counts is None or back_counts is None:
        if os.path.exists(config_dic['file_path']):
            print(f"Reading data from {config_dic['file_path']}")
            data = np.load(config_dic['file_path'])
            time_edges = data["full_grb_time_lo_edge"]
            counts = data["full_grb_counts"]
            back_counts = data["full_back_counts"]
        else:
            raise ValueError("No input arrays provided and no .npz file found.")
        
    temp_bw = np.round(time_edges[1] - time_edges[0],6)  # Assuming uniform bin width

    if not np.allclose(config_dic['bw'], temp_bw, rtol=1e-9, atol=1e-12):
        print("\n!!!!!!  WARNING: Input bin width does not match Data bin width. !!!!!!!")
        print(f"Data bin width: {temp_bw}, Provided bin width: {config_dic['bw']}")
        config_dic['bw'] = temp_bw
        print(f"Using Data bin width {temp_bw} instead.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print('\n')
    print("Final config:".center(20,'*'))
    for k, v in config_dic.items():
        print(f"{k}: {v}")
    exit()
    # Finally run the analysis
    return run_mvt_analysis(
        config_dic['trigger_number'],
        time_edges,
        counts,
        back_counts,
        config_dic['T0'],
        T90,
        tt1,
        config_dic['bw'],
        valid_deltas,
        config_dic['start_padding'],
        config_dic['end_padding'],
        config_dic['N'],
        config_dic['cores'],
        config_dic['f1'],
        config_dic['f2'],
        en,
        output_folder=config_dic['output_path'],
        all_delta=config_dic['all_delta'],
        delta=config_dic['delta'],
        limit=config_dic['limit']
    )

def mvtgeneral_cli():
    mvtgeneral()  # don't return, suppress output
'''    
if __name__ == "__main__":
    args = parse_args_general(argv=[])
    mvtgeneral(args)
'''
if __name__ == "__main__":
    args = parse_args_general(argv=[])
    mvtgeneral_cli(args)


