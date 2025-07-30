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


# ========= CORE FUNCTIONS =========


def process_grb(center_time, sigma, peak_amplitude, bw, background_level, pre_post_background_time, output_path):
    #config = config_template.copy()
    sim_name = f"Gauss_s_{e_n(sigma)}_pa_{e_n(peak_amplitude)}"
    try:
        t_bins, counts, tri_counts, bkg_counts  = generate_gaussian_light_curve_with_noise(
            center_time=center_time,
            sigma=sigma,
            peak_amplitude=peak_amplitude,
            bin_width=bw,
            background_level=background_level,
            pre_post_background_time=pre_post_background_time,
            random_seed=42
        )
        
        #file_name = sim_name +".npz"
        #print(file_name)
        #file_path = os.path.join(output_path, sim_name)

        #trigger_number = sim_name 
        file_name = os.path.join(output_path, sim_name)
        print(f'Plotting results to {file_name}')
        try:
            results = haar_power_mod(
                counts, np.sqrt(counts), min_dt=bw, max_dt=100., tau_bg_max=0.01, nrepl=2,
                doplot=True, bin_fac=4, zerocheck=False, afactor=-1., snr=3.,
                verbose=True, weight=True, file=file_name)
        except Exception as e:
            print(f"Error in haar_power_mod for {sim_name}: {e}")
            results = [-999] * 7  # Placeholder for results

        result_dict = {
            'Simulation': str(sim_name),
            'sigma': round(float(sigma), 3) if sigma else 0,
            'center_time': round(float(center_time), 3) if center_time else 0,
            'peak_amplitude': round(float(peak_amplitude), 3) if peak_amplitude else 0,
            'post_background_time': round(float(pre_post_background_time), 3) if pre_post_background_time else 0,
            'background_level': round(float(background_level), 3) if background_level else 0,
            "tsnr": results[0],
            "tbeta": results[1],
            "slope": results[4],
            "sigma_tsnr": results[5],
            "sigma_tmin": results[6],
            "tmin": results[2],
            "dtmin": results[3],}

        mvt_file_name = f"MVT_{sim_name}.yaml"
        mvt_path = os.path.join(output_path, mvt_file_name)
        #write_yaml(config_trigger, yaml_path, comments=[])
        yaml_content = format_par_as_yaml(result_dict, '')
        # Open the file for writing
        with open(mvt_path, 'w') as f:
            f.write(yaml_content)
        return {
                'Simulation': str(sim_name ),
                'sigma': round(float(sigma), 3) if sigma else 0,
                'mvt_ms': round(float(results[2]) * 1000, 3) if results[2] else 0,
                'mvt_error_ms': round(float(results[3]) * 1000, 3) if results[3] else 0,
                'center_time': round(float(center_time), 3) if center_time else 0,
                'peak_amplitude': round(float(peak_amplitude), 3) if peak_amplitude else 0,
                'post_background_time': round(float(pre_post_background_time), 3) if pre_post_background_time else 0,
                'background_level': round(float(background_level), 3) if background_level else 0,
            }
        
    except Exception as e:
        print(f"\nError in {sim_name}: {e}")
        traceback.print_exc()
        return {
            'Simulation': str(sim_name),
            'sigma': round(float(sigma), 3) if sigma else 0,
            'mvt_ms': -100,
            'mvt_error_ms': -100,
            'center_time': round(float(center_time), 3) if center_time else 0,
            'peak_amplitude': round(float(peak_amplitude), 3) if peak_amplitude else 0,
            'post_background_time': round(float(pre_post_background_time), 3) if pre_post_background_time else 0,
            'background_level': round(float(background_level), 3) if background_level else 0    
        }


# ========= MAIN PARALLEL LOGIC =========



# Load trigger config
with open(TRIGGER_CONFIG_FILE, 'r') as f:
    config_template = yaml.safe_load(f)

# Read GRB trigger list
#with open(GRB_LIST_FILE, 'r') as f:
#    grb_list = f.read().splitlines()

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

    values = np.arange(0.1, 2.1, 0.1)  # Sigma values from 0.1 to 2.0
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
        futures = [executor.submit(process_grb, trig) for trig in sim_list]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results_batch.append(result)

            print(f"[{i}/{len(sim_list)}] Done: {result['Simulation']}")

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


    

if __name__ == '__main__':
    main()

