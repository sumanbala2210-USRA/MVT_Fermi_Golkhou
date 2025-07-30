import os
import numpy as np 
import csv
import pandas as pd
from astropy.table import Table
from datetime import datetime
import yaml
#from BAduty.Notebook.NB_lib_notebook import *

#from MVTfermi.mvt_integral_fermi import mvtintegral

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

def save_mvt_result(csv_path, trigger_number, T90, tr, delta, mvt, mvt_error, significance, is_upper_limit):
    row = {
        'trigger_number': str(trigger_number),
        'T90': round(float(T90),3) if T90 is not None else np.nan,
        'mvt_ms': round(float(mvt) * 1000, 3) if mvt is not None else np.nan,
        'mvt_error_ms': round(float(mvt_error) * 1000, 3) if mvt_error is not None else np.nan,
        'delta_used': round(float(delta), 2) if delta is not None else np.nan,
        'tr': round(float(tr), 2) if tr is not None else np.nan,
        'significance': round(float(significance), 2) if significance is not None else np.nan,
        'upper_limit': bool(is_upper_limit)
    }

    if not os.path.exists(csv_path):
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(csv_path, index=False)



data_path= '/home/sbala/work/GRB_catalog_v4/all_grb_data'
grb_list_file = '/home/sbala/work/GRB_catalog_v4/all_grb_data/complete_grbs_06.04.25.txt'

trigger_config_file = 'config_MVT_fermi_int.yaml'
with open(trigger_config_file, 'r') as f:
        config_trigger = yaml.safe_load(f)

#T0=config_trigger['T0']
#start_padding = config_trigger['start_padding']
#end_padding = config_trigger['end_padding']
#N = config_trigger['N']
#f1 = config_trigger['f1']
#f2 = config_trigger['f2']
#en_lo = config_trigger['en_lo']
#en_hi = config_trigger['en_hi']
#cores = config_trigger['cores']
#config_file = config_trigger['grb_catalog_output']

now = datetime.now()
    # Format the date and time as a string
time_now = now.strftime("%d_%m_%H:%M:%S")
with open(grb_list_file, 'r') as f:
    grb_list = f.read().splitlines()


script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = f'Trigger_number_vs_mvt_{time_now}'
print('\noutput_dir=', output_dir)
#print('\n')
# Define a relative folder path from the script location
output_path = os.path.join(script_dir, output_dir)
os.makedirs(output_dir, exist_ok=True)

output_csv = output_dir +'.csv'
output_csv_path = os.path.join(output_path, output_csv)


for trigger_number in grb_list:
    temp_trigger_directory ="bn"+trigger_number
    trigger_directory = os.path.join(data_path, temp_trigger_directory)
    #trigger_directory
    par_file_list,_ = par_file_list_fun(trigger_directory, trigger_number)
    par_all = yaml.safe_load(open(par_file_list[-1], 'r'))
    par_in=get_par(par_all, 'tte', par_all['det_list'][0])
    
    dets = par_in['det_list']
    #bkgd_range = par_in['background_intervals']
    T90 = par_in['T90']
    nai_dets = [item for item in dets if item.startswith('n')]
    bkgd_range = par_in['background_intervals']
    
    config_trigger['trigger_number']=trigger_number
    config_trigger['T90'] = T90
    config_trigger['det_list'] = nai_dets
    config_trigger['background_intervals']= bkgd_range
    config_trigger['output_path'] = output_path
    config_trigger['delta'] = min(T90/10,1.0)
    config_trigger['bw'] = 0.001
    print(f"T90 before start = {T90}")
    if config_trigger['T90'] < 2.0:
        config_trigger['bw'] = 0.0001
    
    yaml_file_name = f'config_MVT_fermi_{trigger_number}.yaml'
    yaml_path = os.path.join(script_dir, output_dir, yaml_file_name)
    #write_yaml(config_trigger, yaml_path, comments=[])
    yaml_content = format_par_as_yaml(config_trigger, '')
    # Open the file for writing
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    #with open(yaml_path, "w") as file:
    #    yaml.dump(config_trigger, file, sort_keys=False)

    #exit()
    
    try: 
        #tr, delta, mvt, mvt_error, significance, UL_flag = mvtfermi(config=yaml_path, limit = 0)
        tr, delta, mvt, mvt_error, significance, UL_flag = mvtintegral(config=yaml_path, delta=config_trigger['delta'])
    
    except Exception as e:
        print(f'Error: {e}')
        print(f"\n\n!!!!!!!!!!!!!!!!!! \t Error computing MVT for {trigger_number} \t !!!!!!!!!!!!!!!!!! \n\n")
        tr, delta, mvt, mvt_error, significance, UL_flag = -100, 0, np.nan, np.nan, 0, True
    save_mvt_result(output_csv_path, str(trigger_number), config_trigger['T90'], tr, delta, mvt, mvt_error, significance, UL_flag)
    print('\n\n')
    #print(f'\n##########  Done for {trigger_number} #########')
    


#file.close()
print(f'\n@@@@@@@@@@@  Analysis completed!!:  {output_dir} @@@@@@@@@@@ \n\n')
    