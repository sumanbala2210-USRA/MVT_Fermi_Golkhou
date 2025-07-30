# ========= Import necessary libraries =========
import os
import traceback
import logging
import smtplib
from email.message import EmailMessage
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Assuming haar_power_mod is in a separate file as before
from haar_power_mod import haar_power_mod


# ========= USER SETTINGS =========
MAX_WORKERS = 6  # Leave one core free for system responsiveness
BATCH_WRITE_SIZE = 10             # Number of results to write to CSV at once
SIM_CONFIG_FILE = 'simulations_back.yaml'
GMAIL_FILE = 'config_mail.yaml'


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


# ========= CORE TASK AND EXECUTION LOGIC =========

class SimulationTask:
    """An object to hold the parameters and execution logic for a single simulation."""
    def __init__(self, output_path, **params):
        self.output_path = output_path
        self.params = params
        self.sim_name = (
            f"Gauss_pa_{e_n(params['peak_amplitude'])}"
            f"_bg_{e_n(params['background_level'])}"
            f"_s_{e_n(params['sigma'])}"
        )

    def run(self):
        """Contains the full logic for running a single simulation and its analysis."""
        try:
            # 1. Generate Data
            t_bins, counts, g_counts, b_counts = generate_gaussian_light_curve(
                center_time=self.params['center_time'],
                sigma=self.params['sigma'],
                peak_amplitude=self.params['peak_amplitude'],
                bin_width=self.params['bin_width'],
                background_level=self.params['background_level'],
                pre_post_background_time=self.params['pre_post_background_time'],
                random_seed=self.params.get('random_seed')
            )
            
            # 2. Plot Generated Data
            sim_plot_path = os.path.join(self.output_path, self.sim_name + '_sim.png')
            plot_simulation_results(t_bins, counts, g_counts, b_counts, sim_plot_path)

            # 3. Run MVT Analysis
            mvt_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt.png')
            logging.info(f"Analyzing and plotting MVT results to {mvt_plot_path}")
            
            results = [-999] * 7  # Default error values
            try:
                results = haar_power_mod(
                    counts, np.sqrt(counts), min_dt=self.params['bin_width'], max_dt=100.,
                    doplot=True, file=mvt_plot_path, verbose=False
                )
            except Exception as e:
                logging.error(f"Error in haar_power_mod for {self.sim_name}: {e}")

            # 4. Return formatted results for the CSV
            return {
                'Simulation': self.sim_name,
                'sigma': round(self.params['sigma'], 3),
                'mvt_ms': round(float(results[2]) * 1000, 3),
                'mvt_error_ms': round(float(results[3]) * 1000, 3),
                'peak_amplitude': round(self.params['peak_amplitude'], 3),
                'background_level': round(self.params['background_level'], 3),
            }

        except Exception as e:
            logging.error(f"Critical error processing {self.sim_name}: {e}", exc_info=True)
            return {'Simulation': self.sim_name, 'mvt_ms': -100, 'mvt_error_ms': -100}

def generate_sim_tasks_from_config(config_path, output_path):
    """A generator that reads the config file and yields SimulationTask objects."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for campaign in config:
        if not campaign.get('enabled', True):
            continue
        logging.info(f"Generating tasks for campaign: {campaign['name']}")
        
        params = campaign['parameters']
        sigmas = _parse_param(params['sigma'])
        peak_amps = _parse_param(params['peak_amplitude'])
        background_levels = _parse_param(params.get('background_level', []))
        constants = campaign.get('constants', {})

        for peak_amp in peak_amps:
            for bg_level in background_levels:
                for sigma in sigmas:
                    task_params = {
                        'sigma': sigma, 'peak_amplitude': float(peak_amp),
                        'background_level': float(bg_level),
                        'bin_width': constants['bin_width'],
                        'center_time': constants['center_time'],
                        'pre_post_background_time': 10 * sigma + max(10, 2 * sigma),
                        'random_seed': constants.get('random_seed')
                    }
                    yield SimulationTask(output_path=output_path, **task_params)

def run_simulation(task):
    """A simple top-level function to be called by the ProcessPoolExecutor."""
    return task.run()

# ========= MAIN EXECUTION BLOCK =========
def main():
    # --- Setup Output Path and Logging ---
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'SIM_vs_mvt_{now}'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    log_file = os.path.join(output_path, 'run.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    config_file_path = os.path.abspath(SIM_CONFIG_FILE)
    logging.info(f"Attempting to load configuration from: {config_file_path}")
    # --- Generate Tasks ---
    try:
        tasks = list(generate_sim_tasks_from_config(SIM_CONFIG_FILE, output_path))
    except (FileNotFoundError, TypeError, KeyError) as e:
        logging.error(f"Failed to generate simulation tasks. Check your config file. Error: {e}")
        return
        
    if not tasks:
        logging.warning("No simulation tasks were generated. Check config file or 'enabled' flags.")
        return

    logging.info(f"Generated {len(tasks)} simulation tasks. Starting parallel processing with {MAX_WORKERS} workers.")
    
    # --- Execute Tasks in Parallel with Batch Writing---
    output_csv_path = os.path.join(output_path, f"{output_dir}_results.csv")
    results_batch = []
    header_written = False

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_simulation, task) for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Simulations"):
            try:
                result = future.result()
                if result:
                    results_batch.append(result)

                # Write to CSV when the batch is full
                if len(results_batch) >= BATCH_WRITE_SIZE:
                    df_batch = pd.DataFrame(results_batch)
                    df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)
                    header_written = True
                    results_batch = []  # Reset the batch

            except Exception as e:
                logging.error(f"A task failed unexpectedly in the main loop: {e}")

    # Write any remaining results after the loop finishes
    if results_batch:
        df_batch = pd.DataFrame(results_batch)
        df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)

    logging.info(f"All simulations processed! Results saved to:\n{output_csv_path}")
    # send_email(input=f"Simulation completed! Results saved to {output_csv_path}")

if __name__ == '__main__':
    main()