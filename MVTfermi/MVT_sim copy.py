import os
import numpy as np
import pandas as pd
import yaml
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits
from mvt_data_fermi import mvtfermi, format_par_as_yaml
from trigger_process import trigger_file_list,  get_dets_list
import smtplib
from email.message import EmailMessage
from haar_power_mod import haar_power_mod

from BAduty.Notebook.NB_lib_notebook import *


# ========= USER SETTINGS =========
MAX_WORKERS = 2  # You can change this to 16 if needed
BATCH_WRITE_SIZE = 2  # Number of results to write to CSV at once
DATA_PATH = '/GBMdata/triggers'
GRB_LIST_FILE = 'grb_list.txt'
TRIGGER_CONFIG_FILE = 'config_MVT_fermi.yaml'
GMAIL_FILE = 'config_mail.yaml' 
# =================================



def send_email(input='!!'):
    msg = EmailMessage()
    msg['Subject'] = 'Python Script Completed'
    msg['From'] = '2210sumaanbala@gmail.com'
    msg['To'] = 'sumanbala2210@gmail.com'
    msg.set_content(f'Hey, your script has finished running!\n{input}')

    with open(GMAIL_FILE, 'r') as f:
        config_mail = yaml.safe_load(f)

    # Use your Gmail App Password here
    gmail_user = config_mail['gmail_user']
    gmail_password = config_mail['gmail_password']

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(gmail_user, gmail_password)
        smtp.send_message(msg)


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
    triangle_counts_noiseless[in_rise] = peak_amplitude / (peak_time - start_time)#  (time_bins[in_rise] - start_time) * rise_slope
    #triangle_counts_noiseless[in_fall] = (end_time - time_bins[in_fall]) * fall_slope

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

def save_mvt_result(csv_path, trigger_number, T90, delta, mvt, mvt_error):
    row = {
        'trigger_number': str(trigger_number),
        'T90': round(float(T90),3) if T90 is not None else np.nan,
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


# ========= CORE FUNCTIONS =========


def process_grb(center_time, sigma, peak_amplitude, bin_width, background_level, pre_post_background_time, trigger_number, output_path):
    config = config_template.copy()
    try:
        t_bins, counts, tri_counts, bkg_counts  = generate_gaussian_light_curve_with_noise(
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
        file_path = os.path.join(output_path, file_name)
        np.savez_compressed(
                file_path,
                full_grb_time_lo_edge=t_bins,
                full_grb_counts=obs_counts,
                full_back_counts=bkg_counts
            )
        trigger_number = sim_name 
        file_name = os.path.join(config_dic['output_path'], config_dic['trigger_number'])
        print(f'Plotting results to {file_name}')
        return  haar_power_mod(
        counts, np.sqrt(counts), min_dt=config_dic['bw'], max_dt=100., tau_bg_max=0.01, nrepl=2,
        doplot=True, bin_fac=4, zerocheck=False, afactor=-1., snr=3.,
        verbose=True, weight=True, file=file_name)
    except:
        T90, T50, PF64, PFLX, FLU = -1  # Use -1 to indicate failure in getting T90

    
    
    config.update({
        'trigger_number': trigger_number,
        'T90': T90,
        'det_list': 'all',
        'background_intervals': [[0, 0], [0, 0]],
        'output_path': output_path,
        'bw': 0.0001,  # Set bin width to 0.0001
    })

    yaml_file_name = f'config_MVT_fermi_{trigger_number}.yaml'
    yaml_path = os.path.join(output_path, yaml_file_name)
    yaml_content = format_par_as_yaml(config, '')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(yaml_content)

    try:
        mvt_path = mvtfermi(config=yaml_path)
        with open(mvt_path, 'r') as f:
            mvt_dic = yaml.safe_load(f)

        mvt = mvt_dic['tmin']
        mvt_error = mvt_dic['dtmin']

        return {
            'trigger_number': str(trigger_number),
            'mvt_ms': round(float(mvt) * 1000, 3) if mvt else 0,
            'mvt_error_ms': round(float(mvt_error) * 1000, 3) if mvt_error else 0,
            'T90': round(float(T90), 3) if T90 else 0,
            'T50': round(float(T50), 3) if T50 else 0,
            'PF64': round(float(PF64), 3) if PF64 else 0,
            'PFLX': round(float(PFLX), 3) if PF64 else 0,
            'FLUxe6': round(float(FLU), 3) if FLU else 0,
        }

    except Exception as e:
        print(f"\nError in {trigger_number}: {e}")
        traceback.print_exc()
        return {
            'trigger_number': str(trigger_number),
            'mvt_ms': -100,
            'mvt_error_ms': -100,
            'T90': round(float(T90), 3) if T90 else 0,
            'T50': round(float(T50), 3) if T50 else 0,
            'PF64': round(float(PF64), 3) if PF64 else 0,
            'PFLX': round(float(PFLX), 3) if PF64 else 0,
            'FLUxe6': round(float(FLU), 3) if FLU else 0,
        }


# ========= MAIN PARALLEL LOGIC =========



# Load trigger config
with open(TRIGGER_CONFIG_FILE, 'r') as f:
    config_template = yaml.safe_load(f)

# Read GRB trigger list
with open(GRB_LIST_FILE, 'r') as f:
    grb_list = f.read().splitlines()

# Setup output folder
script_dir = os.path.dirname(os.path.abspath(__file__))
now = datetime.now().strftime("%d_%m_%H:%M:%S")
output_dir = f'SIM_vs_mvt_{now}'
output_path = os.path.join(script_dir, output_dir)
os.makedirs(output_path, exist_ok=True)

# CSV path
output_csv = f"{output_dir}.csv"
output_csv_path = os.path.join(output_path, output_csv)






def main():
    print(f'\nStarting MVT analysis for {len(grb_list)} GRBs using {MAX_WORKERS} workers...')
    print(f'Output directory: {output_dir}\n')

    results_batch = []
    header_written = os.path.exists(output_csv_path)

    for peak_amp in np.arange(10, 110, 10):

        # Loop over sigma values
        for sigma in values:
            peak_amplitude = peak_amp
        
            # Adjust bin width like you did for triangles
            T90 = 5*sigma
            bin_width = 0.0001
        
            background_level = peak_amplitude / 10
            center_time = 0.0
            start_time = center_time - 4 * sigma
        
            # Ensure sufficient pre/post background coverage
            pre_post_background_time = 10 * sigma + max(10, 2 * sigma)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_grb, trig) for trig in grb_list]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results_batch.append(result)

            print(f"[{i}/{len(grb_list)}] Done: {result['trigger_number']}")

            # Write batch to CSV every N results
            if len(results_batch) >= BATCH_WRITE_SIZE:
                df_batch = pd.DataFrame(results_batch)
                df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)
                header_written = True
                results_batch = []

    # Write any remaining results
    if results_batch:
        df_batch = pd.DataFrame(results_batch)
        df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)
    
    #send_email(input=f"Analysis completed for {len(grb_list)} GRBs! \nResults saved to {output_csv_path}")

    print(f'\nAll GRBs processed! Results saved to:\n{output_csv_path}\n')







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

output_name = f'SIM_width_vs_mvt_{time_now}'
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
        bin_width = 0.0001
    
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


"""
#first_part = np.arange(.01, .1, .01)      # [0.01, 0.02, ..., 0.09]
second_part = np.arange(.1, 1., 0.1)      # [0.1, 0.2, ..., 0.9]
third_part = np.arange(2, 6, 1)

values = np.concatenate((second_part, third_part))
values = [.1]
for width in values:# np.arange(10, 110, 10):
    peak_amplitude = 1
    bin_width = 0.0001
    #bin_width = 0.0001
    background_level = peak_amplitude/10
    peak_time = 0.1
    start_time = 0.0
    pre_post_background_time = width + max(10,2*width)

    t_bins, obs_counts, tri_counts, bkg_counts = generate_triangular_light_curve_with_fixed_peak_amplitude(
        width,
        start_time,
        peak_time,
        peak_amplitude,
        bin_width,
        background_level,
        pre_post_background_time=pre_post_background_time
    )

    

    sim_name = f"SIM_w_{width}"
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
    T90 = width
    config_trigger['trigger_number'] = trigger_number

    
    
    config_trigger['file_name'] = file_name
    config_trigger['data_path'] = data_path
    config_trigger['output_path'] = output_path
    config_trigger['delta'] = min(T90/10,1.0)
    config_trigger['T0'] = 0-3*config_trigger['delta']
    config_trigger['bw'] = 0.001
    config_trigger['T90'] = T90+2*config_trigger['delta']
 
    if T90 < 2.0:
        config_trigger['bw'] = 0.0001
    
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
    save_mvt_result(output_csv_path, str(trigger_number), config_trigger['T90'], delta, mvt, mvt_error)
    print('\n\n')
    #print(f'\n##########  Done for {trigger_number} #########')
    


#file.close()
print(f'\n@@@@@@@@@@@  Analysis completed!!:  {output_name} @@@@@@@@@@@ \n\n')
    
"""
if __name__ == '__main__':
    main()

