
import os
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


def mvtfermi(
    delta=None, limit=None, trigger_number=None, bw=None, T0=None,
    T90=None, start_padding=None, end_padding=None, N=None, f1=None, f2=None,
    en_lo=None, en_hi=None, cores=None, data_path=None, output_path=None,
    all_delta=None, background_intervals=None, det_list=None, config=None
):
    func_args = {"delta": delta, "limit": limit, "trigger_number": trigger_number, "bw": bw,
                 "T0": T0, "T90": T90, "start_padding": start_padding, "end_padding": end_padding, "N": N,
                "f1": f1, "f2": f2, "en_lo": en_lo, "en_hi": en_hi, "cores": cores, "data_path": data_path,
                "output_path": output_path, "all_delta": all_delta, "background_intervals": background_intervals,
                "det_list": det_list, "config": config}

    default_cfg_file = "config_MVT_fermi.yaml"
    config = load_and_merge_config(func_args, cli_args=None, default_config_file=default_cfg_file, parse_fn=parse_args_fermi)
    config['background_intervals'] = normalize_background_intervals(config.get('background_intervals'))

    config["det_list"] = normalize_det_list(config["det_list"])


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


    #exit()
    # 3. Set up
    T90 = config['T90']
    nai_dets = [d for d in config['det_list'] if d.startswith('n')]
    en = f"{config['en_lo']}to{config['en_hi']}keV"
    t_del = 0.064 if T90 <= 4.0 else 1.024
    trigger_directory = os.path.join(config['data_path'], "bn" + config['trigger_number'])

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

    file_write = f"all_arrays_{config['trigger_number']}_bw_{config['bw']}_delt_{max(delta_list)}.npz"
    file_write = f"all_arrays_{config['trigger_number']}_bw_{config['bw']}_delt_1.0.npz"
    #print(f"File to write: {file_write}")
    file_write_path = os.path.join(trigger_directory, file_write)

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
            config['trigger_number'],
            config['bw'],
            config['background_intervals'],
            tt1,
            tend,
            t_del,
            config['en_lo'],
            config['en_hi'],
            nai_dets,
            t0
        )

    print('\n')
    print("Final config:".center(20,'*'))
    for k, v in config.items():
        print(f"{k}: {v}")

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
    mvtfermi()
