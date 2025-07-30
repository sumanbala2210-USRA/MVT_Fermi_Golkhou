import os
import numpy as np 
import csv
import pandas as pd
from astropy.table import Table
from datetime import datetime
from BAduty.Notebook.NB_lib_notebook import *

from MVTfermi.mvt_integral_fermi import mvtintegral


def print_nested_dict(d, indent=0):
    """
    Recursively prints a nested dictionary with simple values (int, str, list of primitives)
    printed on the same line as their key.
    """
    spacing = "  " * indent

    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, (str, int, float, bool)) or (
                isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value)
            ):
                print(f"{spacing}{repr(key)}: {value}")
            else:
                print(f"{spacing}{repr(key)}:")
                print_nested_dict(value, indent + 1)

    elif isinstance(d, list):
        for i, item in enumerate(d):
            if isinstance(item, (str, int, float, bool)) or (
                isinstance(item, list) and all(isinstance(v, (str, int, float, bool)) for v in item)
            ):
                print(f"{spacing}- [Index {i}]: {item}")
            else:
                print(f"{spacing}- [Index {i}]")
                print_nested_dict(item, indent + 1)
    else:
        print(f"{spacing}{repr(d)}")


def e_n(number):
    if number == 0:
        return "0"
    
    abs_number = abs(number)
    
    # If number is "simple" (0.01 ≤ |number| ≤ 1000), use fixed-point
    if 1e-2 <= abs_number <= 1e3:
        if float(number).is_integer():
            return str(int(number))
        else:
            return str(number)
    
    # Otherwise, use scientific notation with em/e style
    scientific_notation = "{:.1e}".format(abs_number)
    base, exponent = scientific_notation.split('e')
    exponent = int(exponent)
    abs_exponent = abs(exponent)
    
    # Remove unnecessary '.0' if base is an integer
    if float(base).is_integer():
        base = str(int(float(base)))
    
    # Format with e/em style
    if exponent < 0:
        return f"{base}em{abs_exponent}"
    else:
        return f"{base}e{abs_exponent}"
import numpy as np

def generate_gaussian_light_curve_with_noise(
    center_time,
    sigma,
    peak_amplitude,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates:
    - Time bins
    - Observed counts (noisy Gaussian + noisy background)
    - Gaussian-only counts (noisy)
    - Noisy background-only counts

    Both the Gaussian signal and background have Poisson noise.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Define the time range (centered around the Gaussian)
    full_start = center_time - pre_post_background_time - 4 * sigma
    full_end = center_time + pre_post_background_time + 4 * sigma

    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Gaussian signal (noiseless)
    gaussian_counts_noiseless = peak_amplitude * np.exp(-0.5 * ((time_bins - center_time) / sigma) ** 2)

    # Add Poisson noise to Gaussian signal
    gaussian_counts_noisy = rng.poisson(gaussian_counts_noiseless)

    # Noisy background (Poisson)
    background_noisy_counts = rng.poisson(background_level, size=time_bins.size)

    # Observed = noisy Gaussian + noisy background
    observed_counts = gaussian_counts_noisy + background_noisy_counts

    return time_bins, observed_counts, gaussian_counts_noisy, background_noisy_counts


def generate_triangular_light_curve_with_fixed_peak_amplitude(
    width,
    start_time,
    peak_time,
    peak_amplitude,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates:
    - Time bins
    - Observed counts (noisy triangle + noisy background)
    - Triangle-only counts (noisy)
    - Noisy background-only counts

    Both triangle and background now have Poisson noise.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Time bins
    full_start = start_time - pre_post_background_time
    full_end = start_time + width + pre_post_background_time

    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Triangle-only (before noise)
    end_time = start_time + width
    in_rise = (time_bins >= start_time) & (time_bins < peak_time)
    in_fall = (time_bins >= peak_time) & (time_bins < end_time)

    rise_slope = peak_amplitude / (peak_time - start_time) if peak_time != start_time else 0
    fall_slope = peak_amplitude / (end_time - peak_time) if end_time != peak_time else 0

    triangle_counts_noiseless = np.zeros_like(time_bins)
    triangle_counts_noiseless[in_rise] = (time_bins[in_rise] - start_time) * rise_slope
    triangle_counts_noiseless[in_fall] = (end_time - time_bins[in_fall]) * fall_slope

    # Add Poisson noise to triangle
    triangle_counts_noisy = rng.poisson(triangle_counts_noiseless)

    # Noisy background (Poisson noise)
    background_noisy_counts = rng.poisson(background_level, size=time_bins.size)

    # Observed counts = noisy triangle + noisy background
    observed_counts = triangle_counts_noisy + background_noisy_counts

    return time_bins, observed_counts, triangle_counts_noisy, background_noisy_counts


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

def save_mvt_result(csv_path, trigger_number, PA, T90, delta, mvt, mvt_error):
    row = {
        'trigger_number': str(trigger_number),
        'Peak_amp': round(float(PA),3) if PA is not None else np.nan,
        'Sigma': round(float(T90),3) if T90 is not None else np.nan,
        'mvt_ms': round(float(mvt) * 1000, 3) if mvt is not None else np.nan,
        'mvt_error_ms': round(float(mvt_error) * 1000, 3) if mvt_error is not None else np.nan,
        'delta_used': round(float(delta), 2) if delta is not None else np.nan,
    }

    if not os.path.exists(csv_path):
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(csv_path, index=False)

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'Gauss_MVT')
os.makedirs(path, exist_ok=True)
data_path = os.path.join(path, 'data')
os.makedirs(data_path, exist_ok=True)


trigger_config_file = 'config_MVT_SIM_int.yaml'
with open(trigger_config_file, 'r') as f:
        config_trigger = yaml.safe_load(f)


now = datetime.now()
    # Format the date and time as a string
time_now = now.strftime("%d_%m_%H:%M:%S")


#script_dir = os.path.dirname(os.path.abspath(__file__))

output_name = f'Gauss_sigma_PA_vs_mvt_{time_now}'
print('\noutput_dir=', output_name)
#print('\n')
# Define a relative folder path from the script location
output_path = os.path.join(path, output_name)
os.makedirs(output_path, exist_ok=True)

output_csv = output_name +'.csv'
output_csv_path = os.path.join(output_path, output_csv)



#first_part = np.arange(.01, .1, .01)      # [0.01, 0.02, ..., 0.09]
second_part = np.arange(0.1, 1.0, 0.1)   # [0.1, 0.2, ..., 0.9]
third_part = np.arange(2, 11, 1)         # [2, 3, ..., 10]
values = np.concatenate((second_part, third_part))  # sigma values

for peak_amp in np.arange(10, 110, 10):

    # Loop over sigma values
    for sigma in values:
        peak_amplitude = peak_amp
    
        # Adjust bin width like you did for triangles
        T90 = 5*sigma
        if T90 < 2.0:
            bin_width = 0.0001
        else:
            bin_width = 0.001
    
        background_level = peak_amplitude / 10
        center_time = 0.0
        start_time = center_time - 4 * sigma
    
        # Ensure sufficient pre/post background coverage
        pre_post_background_time = 10 * sigma + max(10, 2 * sigma)
    
        t_bins, obs_counts, tri_counts, bkg_counts = generate_gaussian_light_curve_with_noise(
            center_time=center_time,
            sigma=sigma,
            peak_amplitude=peak_amplitude,
            bin_width=bin_width,
            background_level=background_level,
            pre_post_background_time=pre_post_background_time,
            random_seed=42
        )
    
        sim_name = f"Gauss_s_{sigma}_pa_{peak_amplitude}"
        file_name = sim_name +".npz"
        #print(file_name)
        file_path = os.path.join(data_path, file_name)
        np.savez_compressed(
                file_path,
                full_grb_time_lo_edge=t_bins,
                full_grb_counts=obs_counts,
                full_back_counts=bkg_counts
            )
    
        #bkgd_range = par_in['background_intervals']
        trigger_number = sim_name 
        
        config_trigger['trigger_number'] = trigger_number
    
        
        
        config_trigger['file_name'] = file_name
        config_trigger['data_path'] = data_path
        config_trigger['output_path'] = output_path
        config_trigger['delta'] = min(T90/10,1.0)
        config_trigger['T0'] = start_time
        config_trigger['bw'] = 0.001
        config_trigger['T90'] = T90+2*config_trigger['delta']
     
        if T90 < 2.0:
            config_trigger['bw'] = bin_width
        
        yaml_file_name = f'config_MVT_{trigger_number}.yaml'
        yaml_path = os.path.join(output_path, yaml_file_name)
        #write_yaml(config_trigger, yaml_path, comments=[])
        yaml_content = format_par_as_yaml(config_trigger, '')
        # Open the file for writing
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print_nested_dict(config_trigger)
    
        #exit()
        
        try: 
            #tr, delta, mvt, mvt_error, significance, UL_flag = mvtfermi(config=yaml_path, limit = 0)
            tr, delta, mvt, mvt_error, _,_ = mvtintegral(config=yaml_path, delta=config_trigger['delta'])
        
        except Exception as e:
            print(f'Error: {e}')
            print(f"\n\n!!!!!!!!!!!!!!!!!! \t Error computing MVT for {trigger_number} \t !!!!!!!!!!!!!!!!!! \n\n")
            tr, delta, mvt, mvt_error = -100, 0, np.nan, np.nan
        save_mvt_result(output_csv_path, str(trigger_number), peak_amplitude, sigma, delta, mvt, mvt_error)
        print('\n\n')
        #print(f'\n##########  Done for {trigger_number} #########')
    


#file.close()
print(f'\n@@@@@@@@@@@  Analysis completed!!:  {output_name} @@@@@@@@@@@ \n\n')
    