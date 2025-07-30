
import os
import sys
import yaml
import argparse
import numpy as np 

from haar_power_mod import haar_power_mod


from core import (
    str2bool,
    parse_args_general,
    load_and_merge_config
)


def mvtgeneral(
    trigger_number=None, bw=None, T0=None, T90=None,
    en_lo=None, en_hi=None, cores=None, file_name=None, output_path=None,
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

    default_cfg_file = "config_MVT_fermi.yaml"
    config_dic = load_and_merge_config(
        func_args,
        cli_args=None,
        default_config_file=default_cfg_file,
        parse_fn=parse_args_general
    )
    # rest of your function ...
    # continue with specific logic for mvtfermi using merged config dict
    # 2. Post-process delta


    T90 =config_dic['T90']# config_dic.get('T90', 4)
    en = f"{config_dic['en_lo']}to{config_dic['en_hi']}keV"
    trigger_directory = os.path.join(config_dic['data_path'], "bn" + config_dic['trigger_number'])

    data_write_path = os.path.join(trigger_directory, config_dic['file_name'])

    # Load data if not provided as function inputs
    if time_edges is None or counts is None or back_counts is None:
        if os.path.exists(data_write_path):
            #print(f"Reading data from {config_dic['file_name']}")
            data = np.load(data_write_path)
            time_edges = data["full_grb_time_lo_edge"]
            counts = data["full_grb_counts"]
            #back_counts = data["full_back_counts"]
        else:
            raise ValueError("No input arrays provided and no .npz file found.")
        
    counts[counts < 0] = 0 
    temp_bw = np.round(time_edges[1] - time_edges[0],6)  # Assuming uniform bin width

    if not np.allclose(config_dic['bw'], temp_bw, rtol=1e-9, atol=1e-12):
        print("\n!!!!!!  WARNING: Input bin width does not match Data bin width. !!!!!!!")
        print(f"Data bin width: {temp_bw}, Provided bin width: {config_dic['bw']}")
        config_dic['bw'] = temp_bw
        print(f"Using Data bin width {temp_bw} instead.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    #print(config_dic['output_path'])
    if not os.path.exists(config_dic['output_path']):
        print("\nXXXXXXXXXXXXXXX  Please check the 'output_path'   XXXXXXXXXXXXXX")
        print(f"'output_path': {config_dic['output_path']}")
        print('Does NOT exists')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')
        raise RuntimeError(f"'output_path' does not exist")
    
    
    
    #print('\n')
    #print("\n************ Final config_dic ************")
    #for k, v in config_dic.items():
    #    print(f"{k}: {v}")
    #exit()
    # Finally run the analysis
    file_name = os.path.join(config_dic['output_path'], config_dic['trigger_number'])
    print(f'Plotting results to {file_name}')
    return  haar_power_mod(
    counts, np.sqrt(counts), min_dt=config_dic['bw'], max_dt=100., tau_bg_max=0.01, nrepl=2,
    doplot=True, bin_fac=4, zerocheck=False, afactor=-1., snr=3.,
    verbose=True, weight=True, file=file_name)

def mvtgeneral_cli():
    mvtgeneral()  # don't return, suppress output
'''    
if __name__ == "__main__":
    args = parse_args_general(argv=[])
    mvtgeneral(args)
'''
if __name__ == "__main__":
    #args = parse_args_general(argv=[])
    mvtgeneral_cli()
