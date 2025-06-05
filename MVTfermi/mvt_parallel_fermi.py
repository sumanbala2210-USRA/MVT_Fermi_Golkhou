
import os
import sys
import numpy as np 

from .mvt_analysis import run_mvt_analysis
from .evolve_opt_res_fermi import compute_grb_time_bounds
from .trigger_process import trigger_process

from .core import (
    str2bool,
    parse_args_fermi,
    load_and_merge_config,
    normalize_background_intervals,
    normalize_det_list
)

'''
def mvtfermi(
    delta=None, limit=None, trigger_number=None, bw=None, T0=None,
    T90=None, start_padding=None, end_padding=None, N=None, f1=None, f2=None,
    en_lo=None, en_hi=None, cores=None, data_path=None, output_path=None,
    all_delta=None, background_intervals=None, det_list=None, config=None
):
    func_args = {
            k: v for k, v in {
                "delta": delta, "limit": limit, "trigger_number": trigger_number, "bw": bw,
                "T0": T0, "T90": T90, "start_padding": start_padding, "end_padding": end_padding,
                "N": N, "f1": f1, "f2": f2, "en_lo": en_lo, "en_hi": en_hi, "cores": cores,
                "data_path": data_path, "output_path": output_path, "all_delta": all_delta,
                "background_intervals": background_intervals, "det_list": det_list,
                **({"config": config} if config else {})
            }.items() if v is not None
        }
    
    default_cfg_file = "config_MVT_fermi.yaml"
    config_dic = load_and_merge_config(func_args, cli_args=None, default_config_file=default_cfg_file, parse_fn=parse_args_fermi)
'''

def mvtfermi(
    delta=None, limit=None, trigger_number=None, bw=None, T0=None,
    T90=None, start_padding=None, end_padding=None, N=None, f1=None, f2=None,
    en_lo=None, en_hi=None, cores=None, data_path=None, output_path=None,
    all_delta=None, background_intervals=None, det_list=None, config=None
):
    skip_keys = {'skip_keys'}  # or add keys to skip if any

    func_args = {
        k: v for k, v in locals().items()
        if k not in skip_keys and v is not None and k != "config"
    }
    if config is not None:
        func_args['config'] = config

    default_cfg_file = "config_MVT_fermi.yaml"
    config_dic = load_and_merge_config(
        func_args,
        cli_args=None,
        default_config_file=default_cfg_file,
        parse_fn=parse_args_fermi
    )
    # rest of your function ...


    config_dic['background_intervals'] = normalize_background_intervals(config_dic.get('background_intervals'))
    #print('Here I am\n')
    config_dic["det_list"] = normalize_det_list(config_dic["det_list"])


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


    #exit()
    # 3. Set up
    T90 = config_dic['T90']
    nai_dets = [d for d in config_dic['det_list'] if d.startswith('n')]
    en = f"{config_dic['en_lo']}to{config_dic['en_hi']}keV"
    t_del = 0.064 if T90 <= 4.0 else 1.024
    trigger_directory = os.path.join(config_dic['data_path'], "bn" + config_dic['trigger_number'])

    delta_list = np.concatenate((
        np.arange(0.1, 1.0, 0.1),
        np.arange(1.0, 7.0, 1.0)
    ))
    T90_rounded = round(T90, 2)
    '''
    if np.max(delta_list) > T90_rounded and T90_rounded not in delta_list:
        delta_list = np.append(delta_list, T90_rounded)
        delta_list = np.sort(delta_list)
    '''
    # Ensure T90_rounded is in the list
    if T90_rounded not in delta_list:
        delta_list = np.append(delta_list, T90_rounded)
    
    # Keep only values ≤ T90_rounded, and sort
    #delta_list = np.sort(delta_list[delta_list <= T90_rounded])

    valid_deltas = delta_list#[delta_list <= T90_rounded]
    valid_deltas = np.sort(delta_list)
    if valid_deltas.size == 0:
        raise ValueError("No valid delta ≤ T90")

    tt1, t0, tend = compute_grb_time_bounds(
        config_dic['T0'], T90, max(delta_list),
        config_dic['start_padding'], config_dic['end_padding'], end_t90=2.0
    )

    if config_dic['delta'] is None or config_dic['delta'] < max(delta_list):
        file_write = f"all_arrays_{config_dic['trigger_number']}_bw_{config_dic['bw']}_delt_{max(delta_list)}.npz"
    else:
        file_write = f"all_arrays_{config_dic['trigger_number']}_bw_{config_dic['bw']}_delt_{np.round(config_dic['delta'],2)}.npz"
        
    #file_write = f"all_arrays_{config_dic['trigger_number']}_bw_{config_dic['bw']}_delt_1.0.npz"
    #print(f"File to write: {file_write}")
    file_write_path = os.path.join(trigger_directory, file_write)

    print(f"\n@@@@@@@@@@@@@@@@@ Starting Analysis for {config_dic['trigger_number']} @@@@@@@@@@@@@@@@@")

    if os.path.exists(file_write_path):
        print(f"Reading data from {file_write}")
        data = np.load(file_write_path)
        time_edges = data["full_grb_time_lo_edge"]
        counts = data["full_grb_counts"]
        back_counts = data["full_back_counts"]
    else:
        
        time_edges, counts, back_counts = trigger_process(
            file_write,
            trigger_directory,
            config_dic['trigger_number'],
            config_dic['bw'],
            config_dic['background_intervals'],
            tt1,
            tend,
            t_del,
            config_dic['en_lo'],
            config_dic['en_hi'],
            nai_dets,
            t0
        )

    temp_bw = np.round(time_edges[1] - time_edges[0],6)  # Assuming uniform bin width
    if not np.allclose(config_dic['bw'], temp_bw, rtol=1e-9, atol=1e-12):
        print("\n!!!!!!  WARNING: Input bin width does not match Data bin width. !!!!!!!")
        print(f"Data bin width: {temp_bw}, Provided bin width: {config['bw']}")
        config_dic['bw'] = temp_bw
        print(f"Using Data bin width {temp_bw} instead.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    #print('\n')
    print("\n************ Final config_dic ************")
    for k, v in config_dic.items():
        print(f"{k}: {v}")
    #exit()

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


def mvtfermi_cli():
    mvtfermi()  # don't return, suppress output



if __name__ == "__main__":
    # Necessary for macOS multiprocessing to avoid recursive execution
    mvtfermi_cli()
