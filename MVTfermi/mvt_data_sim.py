
import os
import sys
import numpy as np 
import yaml
import subprocess

#from .mvt_analysis import run_mvt_analysis
#from evolve_opt_res_fermi import compute_grb_time_bounds
from trigger_process import trigger_process
from haar_power_mod import haar_power_mod

from core import (
    str2bool,
    parse_args_fermi,
    load_and_merge_config,
    normalize_background_intervals,
    normalize_det_list
)


def format_par_as_yaml(dict_data, dict_name, count_dict=0, indent=0):
    """
    Formats a dictionary into a YAML-like string.
    """
    yaml_str = ''
    
    # Add the dictionary name with correct indentation
    if dict_name:
        yaml_str += ' ' * indent + dict_name + ':\n'
        indent += 4  # Increase indentation for child elements

    # Convert dictionary items to a list for iteration
    items = list(dict_data.items())
    count_dict += 1  # Increase depth level

    for i, (key, value) in enumerate(items):
        if isinstance(value, dict):
            # Recursive call, passing increased indent
            yaml_str += format_par_as_yaml(value, key, count_dict=count_dict, indent=indent)
        else:
            # Format the value based on its type
            if isinstance(value, str):
                value_str = f"'{value}'"
            elif isinstance(value, list):
                value_str = yaml.dump(value, default_flow_style=True).strip()
            else:
                value_str = str(value)
            
            yaml_str += f"{' ' * indent}{key}: {value_str}\n"

    return yaml_str


def mvtfermi(trigger_number=None, bw=None, T0=None,
    T90=None, N=None, en_lo=None, en_hi=None, cores=None, data_path=None, output_path=None,
    background_intervals=None, det_list=None, config=None
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




    #exit()
    # 3. Set up
    T90 = config_dic['T90']
    nai_dets = [d for d in config_dic['det_list'] if d.startswith('n')]
    en = f"{config_dic['en_lo']}to{config_dic['en_hi']}keV"
    t_del = 1.024
    trigger_directory = os.path.join(config_dic['data_path'], "bn" + config_dic['trigger_number'])

    #T90_rounded = round(T90, 2)




    det_string = "_".join(nai_dets)

    data_name = f"bn{config_dic['trigger_number']}_bw_{config_dic['bw']}"
    #file_write = f"all_arrays_{config_dic['trigger_number']}_bw_{config_dic['bw']}_dets_{det_string}.npz"
        
    #file_write = f"all_arrays_{config_dic['trigger_number']}_bw_{config_dic['bw']}_delt_1.0.npz"
    #print(f"File to write: {file_write}")
    data_write_path = os.path.join(trigger_directory, data_name)+'.npz'

    print(f"\n@@@@@@@@@@@@@@@@@ Starting Analysis for {config_dic['trigger_number']} @@@@@@@@@@@@@@@@@")
    print(f'file_write_path: {data_write_path}')
    #exit()
    config_dic['file_name'] = data_name + '.npz'
      #print('\n')
    #print("\n************ Final config_dic ************")
    #for k, v in config_dic.items():
    #    print(f"{k}: {v}")

    if os.path.exists(data_write_path):
        print(f"Reading data from {data_name}.npz")
        data = np.load(data_write_path)
        time_edges = data["full_grb_time_lo_edge"]
        counts = data["full_grb_counts"]
        #back_counts = data["full_back_counts"]
    else:
        
        time_edges, counts = trigger_process(
            data_name,
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
    counts[counts < 0] = 0
    temp_bw = np.round(time_edges[1] - time_edges[0],6)  # Assuming uniform bin width
    if not np.allclose(config_dic['bw'], temp_bw, rtol=1e-9, atol=1e-12):
        print("\n!!!!!!  WARNING: Input bin width does not match Data bin width. !!!!!!!")
        print(f"Data bin width: {temp_bw}, Provided bin width: {config['bw']}")
        config_dic['bw'] = temp_bw
        print(f"Using Data bin width {temp_bw} instead.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    yaml_file_name = f"config_MVT_fermi_{config_dic['trigger_number']}.yaml"
    yaml_path = os.path.join(trigger_directory, yaml_file_name)
    #write_yaml(config_trigger, yaml_path, comments=[])
    yaml_content = format_par_as_yaml(config_dic, '')
    # Open the file for writing
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    #exit()
    print(f"\nConfiguration saved to: \t {yaml_file_name}")
    print("Running MVT analysis...")
    subprocess.run(["/Users/sbala/anaconda3/bin/python",
                   "mvt_analysis_fermi.py",
                   "--config", yaml_path], check=True)


def mvtfermi_cli():
    mvtfermi()  # don't return, suppress output



if __name__ == "__main__":
    # Necessary for macOS multiprocessing to avoid recursive execution
    mvtfermi_cli()
