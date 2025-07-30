# ========= Import necessary libraries =========
import os
import traceback
import logging
import smtplib
from email.message import EmailMessage
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Assuming haar_power_mod is in a separate file as before
from haar_power_mod import haar_power_mod

# ========= USER SETTINGS =========
MAX_WORKERS = os.cpu_count() - 1  # Leave one core free
BATCH_WRITE_SIZE = 10             # Number of results to write to CSV at once
SIM_CONFIG_FILE = 'simulations.yaml'
GMAIL_FILE = 'config_mail.yaml'
# =================================

# ========= UTILITY FUNCTIONS =========

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

def plot_simulation_results(time_bins, observed_counts, signal_counts, background_counts, save_path):
    """Plots the generated light curve data and saves the figure."""
    plt.figure(figsize=(12, 7))
    plt.plot(time_bins, observed_counts, label='Observed (Signal + Background)', color='black', drawstyle='steps-post')
    plt.plot(time_bins, signal_counts, label='Signal Component', color='cornflowerblue', linestyle='--', alpha=0.8)
    plt.plot(time_bins, background_counts, label='Background Component', color='salmon', linestyle=':', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.title('Simulated Light Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _parse_param(param_config):
    """Helper function to parse parameter definitions from the YAML config."""
    if isinstance(param_config, dict) and 'start' in param_config:
        return np.arange(**param_config)
    if isinstance(param_config, list):
        return param_config
    if isinstance(param_config, (int, float)):
        return [param_config]
    raise TypeError(f"Unsupported parameter format in YAML: {param_config}")


# ========= DATA GENERATION FUNCTIONS =========

def generate_gaussian_light_curve(center_time, sigma, peak_amplitude, bin_width,
                                  background_level, pre_post_background_time=2.0, random_seed=None):
    """Generates a Gaussian light curve with Poisson noise."""
    rng = np.random.default_rng(seed=random_seed)
    full_start = center_time - pre_post_background_time - 4 * sigma
    full_end = center_time + pre_post_background_time + 4 * sigma
    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    gaussian_signal = peak_amplitude * np.exp(-0.5 * ((time_bins - center_time) / sigma) ** 2)
    noisy_signal = rng.poisson(gaussian_signal)
    noisy_background = rng.poisson(background_level, size=time_bins.size)
    observed_counts = noisy_signal + noisy_background

    return time_bins, observed_counts, noisy_signal, noisy_background



def generate_triangular_light_curve_with_fixed_peak_amplitude(
    width,
    start_time,
    peak_time,
    peak_amplitude,
    bin_width,
    background_level,
    peak_time_ratio,
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

def generate_triangular_light_curve_with_fixed_peak_amplitude(
    width,
    start_time,
    peak_time,
    peak_amplitude,    # new parameter
    peak_time_ratio,
    bin_width,
    background_level,
    pre_post_background_time=2.0,
    random_seed=None
):
    """
    Generates:
    - Time bins
    - Observed counts (triangle + noisy background)
    - Triangle-only counts (no noise)
    - Noisy background-only counts

    Keeps the peak amplitude of the triangle constant, regardless of width.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Time bins
    full_start = start_time - pre_post_background_time
    full_end = start_time + width + pre_post_background_time

    bin_edges = np.arange(full_start, full_end + bin_width, bin_width)
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Triangle-only (no noise)
    end_time = start_time + width
    in_rise = (time_bins >= start_time) & (time_bins < peak_time)
    in_fall = (time_bins >= peak_time) & (time_bins < end_time)

    rise_slope = peak_amplitude / (peak_time - start_time) if peak_time != start_time else 0
    fall_slope = peak_amplitude / (end_time - peak_time) if end_time != peak_time else 0

    triangle_counts = np.zeros_like(time_bins)
    triangle_counts[in_rise] = (time_bins[in_rise] - start_time) * rise_slope
    triangle_counts[in_fall] = (end_time - time_bins[in_fall]) * fall_slope

    # Noisy background (Poisson noise)
    background_noisy_counts = rng.poisson(background_level, size=time_bins.size)

    # Observed counts = triangle (noiseless) + noisy background
    observed_counts = triangle_counts + background_noisy_counts

    # Round to nearest integer (counts must be integers)
    observed_counts = np.round(observed_counts).astype(int)

    return time_bins, observed_counts, triangle_counts, background_noisy_counts


# ========= GENERIC SIMULATION FRAMEWORK =========

# --- 1. The Simulation Registry ---
# This dictionary maps the 'type' from your YAML to the correct Python classes.
SIMULATION_REGISTRY = {
    'gaussian': 'GaussianSimulationTask',
    'triangular': 'TriangularSimulationTask',
}

# --- 2. The Generic Task Classes ---
class BaseSimulationTask:
    """A base class for all simulation tasks."""
    def __init__(self, output_path, variable_params, constant_params):
        self.output_path = output_path
        self.params = {**variable_params, **constant_params}
        self.sim_name = self._create_sim_name(variable_params)

    def _create_sim_name(self, variable_params):
        """Creates a descriptive filename from the parameters that are changing."""
        class_name = self.__class__.__name__.replace("SimulationTask", "")
        name_parts = [class_name]
        for key, value in sorted(variable_params.items()):
            key_abbr = ''.join(c for c in key if c.islower() and c not in 'aeiou')[:3]
            name_parts.append(f"{key_abbr}_{e_n(value)}")
        return "_".join(name_parts)

    def run(self):
        """The main execution method, to be implemented by each subclass."""
        raise NotImplementedError("The 'run' method must be implemented by a subclass.")

class GaussianSimulationTask(BaseSimulationTask):
    """Task for Gaussian light curves."""
    def run(self):
        try:
            pre_post_time = 10 * self.params['sigma'] + max(10, 2 * self.params['sigma'])
            t_bins, counts, g_counts, b_counts = generate_gaussian_light_curve(
                pre_post_background_time=pre_post_time, **self.params
            )
            sim_plot_path = os.path.join(self.output_path, self.sim_name + '_sim.png')
            plot_simulation_results(t_bins, counts, g_counts, b_counts, sim_plot_path)

            mvt_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt.png')
            results = haar_power_mod(counts, np.sqrt(counts), min_dt=self.params['bin_width'], doplot=True, file=mvt_plot_path, verbose=False)
            
            return {'type': 'gaussian', 'mvt_ms': round(float(results[2])*1000, 3),
                    'mvt_error_ms': round(float(results[3])*1000, 3), **self.params}
        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            return {'Simulation': self.sim_name, 'type': 'gaussian', 'mvt_ms': -100}

class TriangularSimulationTask(BaseSimulationTask):
    """Task for Triangular light curves."""
    def run(self):
        try:
            peak_time = self.params['start_time'] + (self.params['width'] * self.params['peak_time_ratio'])
            t_bins, counts, t_counts, b_counts = generate_triangular_light_curve_with_fixed_peak_amplitude(
                peak_time=peak_time, **self.params
            )
            sim_plot_path = os.path.join(self.output_path, self.sim_name + '_sim.png')
            plot_simulation_results(t_bins, counts, t_counts, b_counts, sim_plot_path)

            mvt_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt.png')
            results = haar_power_mod(counts, np.sqrt(counts), min_dt=self.params['bin_width'], doplot=True, file=mvt_plot_path, verbose=False)

            return {'type': 'triangular', 'mvt_ms': round(float(results[2])*1000, 3),
                    'mvt_error_ms': round(float(results[3])*1000, 3), **self.params}
        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            return {'Simulation': self.sim_name, 'type': 'triangular', 'mvt_ms': -100}

# --- 3. The Fully Generic Task Generator ---
def generate_sim_tasks_from_config(config_path, output_path):
    """Reads the config and uses the registry and itertools to generate all parameter combinations."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for campaign in config:
        if not campaign.get('enabled', True): continue
        sim_type = campaign.get('type')
        if not sim_type or sim_type not in SIMULATION_REGISTRY:
            logging.warning(f"Campaign '{campaign['name']}' has invalid type '{sim_type}'. Skipping.")
            continue
        
        logging.info(f"Generating tasks for campaign '{campaign['name']}' of type '{sim_type}'")
        TaskClass = globals()[SIMULATION_REGISTRY[sim_type]]
        
        variable_params_config = campaign.get('parameters', {})
        constants = campaign.get('constants', {})
        
        param_names = list(variable_params_config.keys())
        param_value_lists = [_parse_param(v) for v in variable_params_config.values()]
        
        for combination in itertools.product(*param_value_lists):
            variable_params = dict(zip(param_names, combination))
            yield TaskClass(output_path, variable_params, constants)

# --- 4. The Worker Function ---
def run_simulation(task):
    """A simple top-level function to be called by the ProcessPoolExecutor."""
    return task.run()

# ========= MAIN EXECUTION BLOCK =========
def main():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'SIM_vs_mvt_{now}'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    log_file = os.path.join(output_path, 'run.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    try:
        tasks = list(generate_sim_tasks_from_config(SIM_CONFIG_FILE, output_path))
    except Exception as e:
        logging.error(f"Failed to generate simulation tasks. Check YAML format. Error: {e}", exc_info=True)
        return
        
    if not tasks:
        logging.warning("No simulation tasks were generated. Check config file.")
        return

    logging.info(f"Generated {len(tasks)} simulation tasks. Starting parallel processing with {MAX_WORKERS} workers.")
    output_csv_path = os.path.join(output_path, f"{output_dir}_results.csv")
    results_batch = []
    header_written = False

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_simulation, task) for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Simulations"):
            try:
                result = future.result()
                if result: results_batch.append(result)

                if len(results_batch) >= BATCH_WRITE_SIZE:
                    df_batch = pd.DataFrame(results_batch)
                    df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)
                    header_written = True
                    results_batch = []
            except Exception as e:
                logging.error(f"A task failed in the main loop: {e}")

    if results_batch:
        df_batch = pd.DataFrame(results_batch)
        df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)

    logging.info(f"All simulations processed! Results saved to:\n{output_csv_path}")

if __name__ == '__main__':
    main()