# Import necessary libraries at the top
import os
import numpy as np
import pandas as pd
import yaml
import traceback
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits
import smtplib
from email.message import EmailMessage
from haar_power_mod import haar_power_mod
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm # For a nice progress bar!

# ========= (Keep all your existing functions here) =========
# send_email, print_nested_dict, e_n, generate_gaussian_light_curve_with_noise,
# generate_triangular_light_curve_with_fixed_peak_amplitude, format_par_as_yaml,
# haar_power_mod, etc.
# ...

# ========= NEW CONFIGURATION & TASK MANAGEMENT =========


# ========= USER SETTINGS =========
MAX_WORKERS = 6  # You can change this to 16 if needed
BATCH_WRITE_SIZE = 2  # Number of results to write to CSV at once
TRIGGER_CONFIG_FILE = 'config_MVT_fermi.yaml'
GMAIL_FILE = 'config_mail.yaml' 
SIM_CONFIG_FILE = 'simulations.yaml'
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
    
def plot_simulation_results(
    time_bins, observed_counts, gaussian_counts_noisy, background_noisy_counts, sim_plot
):
    """
    Plots the simulation results and saves the figure.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(time_bins, observed_counts, label='Observed Counts', color='gray', marker='o', markersize=1)
    plt.plot(time_bins, gaussian_counts_noisy, label='Gaussian Counts (Noisy)', color='lightblue', linestyle='--', alpha=0.7)
    plt.plot(time_bins, background_noisy_counts, label='Background Counts (Noisy)', color='red', linestyle=':', alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.title('Simulation Results')
    plt.legend()
    plt.grid()
    plt.savefig(sim_plot)
    plt.close()

def generate_gaussian_light_curve_with_noise(
    center_time,
    sigma,
    peak_amplitude,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None,
    sim_plot=None
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
    if sim_plot is not None:
        plot_simulation_results(time_bins, observed_counts, gaussian_counts_noisy, background_noisy_counts, sim_plot)

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

class SimulationTask:
    """An object to hold the parameters and execution logic for a single simulation."""
    def __init__(self, output_path, **params):
        self.output_path = output_path
        self.params = params
        self.sim_name = f"Gauss_s_{e_n(params['sigma'])}_pa_{e_n(params['peak_amplitude'])}"

    def run(self):
        """This method contains the logic previously in process_grb."""
        try:
            t_bins, counts, _, _ = generate_gaussian_light_curve_with_noise(
                center_time=self.params['center_time'],
                sigma=self.params['sigma'],
                peak_amplitude=self.params['peak_amplitude'],
                bin_width=self.params['bin_width'],
                background_level=self.params['background_level'],
                pre_post_background_time=self.params['pre_post_background_time'],
                random_seed=self.params.get('random_seed', 42),
                sim_plot=os.path.join(self.output_path, self.sim_name + '.png')
            )

            file_name = os.path.join(self.output_path, self.sim_name)
            logging.info(f'Analyzing and plotting results to {file_name}')
            
            # Placeholder for results in case of error
            results = [-999] * 7
            try:
                # Assuming haar_power_mod is defined elsewhere
                results = haar_power_mod(
                    counts, np.sqrt(counts), min_dt=self.params['bin_width'], max_dt=100.,
                    doplot=True, file=file_name, verbose=False # Turn off verbose for cleaner logs
                )
            except Exception as e:
                logging.error(f"Error in haar_power_mod for {self.sim_name}: {e}")

            # Create final result dictionary for the CSV
            return {
                'Simulation': self.sim_name,
                'sigma': round(self.params['sigma'], 3),
                'mvt_ms': round(float(results[2]) * 1000, 3),
                'mvt_error_ms': round(float(results[3]) * 1000, 3),
                'peak_amplitude': round(self.params['peak_amplitude'], 3),
                'background_level': round(self.params['background_level'], 3),
            }

        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            # Return a dictionary with error flags
            return {'Simulation': self.sim_name, 'mvt_ms': -100, 'mvt_error_ms': -100}

def run_simulation(task):
    """A simple top-level function to be called by the ProcessPoolExecutor."""
    return task.run()

def generate_sim_tasks_from_config(config_path, output_path):
    """
    A generator that reads the config file and yields SimulationTask objects.
    This is memory-efficient as it doesn't create a giant list upfront.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for campaign in config:
        if not campaign.get('enabled', True):
            continue

        logging.info(f"Generating tasks for campaign: {campaign['name']}")
        
        # Get parameter ranges
        params = campaign['parameters']
        sigmas = np.arange(**params['sigma'])
        peak_amps = np.arange(**params['peak_amplitude'])
        
        # Get constant values for this campaign
        constants = campaign.get('constants', {})

        for peak_amp in peak_amps:
            for sigma in sigmas:
                # Assemble all parameters for this specific run
                task_params = {
                    'sigma': sigma,
                    'peak_amplitude': float(peak_amp),
                    'bin_width': constants['bin_width'],
                    'center_time': constants['center_time'],
                    'background_level': float(peak_amp) / 10.0,
                    'pre_post_background_time': 10 * sigma + max(10, 2 * sigma),
                    'random_seed': constants.get('random_seed')
                }
                # Yield a task object
                yield SimulationTask(output_path=output_path, **task_params)

# ========= MAIN PARALLEL LOGIC (REFACTORED) =========
def main():
    # --- Basic Setup ---

    # --- Setup Output Path and Logging ---
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'SIM_vs_mvt_{now}'
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    log_file = os.path.join(output_path, 'run.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    # --- Generate Tasks ---
    tasks = list(generate_sim_tasks_from_config(SIM_CONFIG_FILE, output_path))
    if not tasks:
        logging.warning("No simulation tasks were generated. Check your config file.")
        return

    logging.info(f"Generated {len(tasks)} simulation tasks. Starting parallel processing with {MAX_WORKERS} workers.")
    
    # --- Execute Tasks in Parallel ---
    results_list = []
    output_csv_path = os.path.join(output_path, f"{output_dir}.csv")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use tqdm for a progress bar
        futures = {executor.submit(run_simulation, task) for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Simulations"):
            try:
                result = future.result()
                if result:
                    results_list.append(result)
            except Exception as e:
                logging.error(f"A task failed unexpectedly: {e}")

    # --- Save Final Results ---
    if results_list:
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(output_csv_path, index=False)
        logging.info(f"All simulations processed! Results saved to:\n{output_csv_path}")
    else:
        logging.warning("No results were generated.")
    #send_email(input=f"Simulation completed! Results saved to {output_csv_path}")

if __name__ == '__main__':
    main()