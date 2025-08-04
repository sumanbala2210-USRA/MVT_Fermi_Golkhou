# MVT_sim_v3.py
# 08.04.2025  This script simulates MVT (Mean Variance Time) data.
#             Now supports Gaussian, Triangular, Norris, FRED, and Lognormal pulses.
#             It reads 'simulations.yaml' for parameters, generates light curves,
#             applies Haar power modulation, and saves results.
# 08.04.2025  Refined code structure, added error handling, and improved logging.
#             Also added functions lognormal, fred and norris.

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

# Import the light curve generation functions
from haar_power_mod import haar_power_mod
from sim_functions import (
    generate_gaussian_light_curve,
    generate_triangular_light_curve,
    generate_norris_light_curve,
    generate_fred_light_curve,
    generate_lognormal_light_curve
)

# ========= USER SETTINGS =========
MAX_WORKERS = os.cpu_count() - 1  # Leave one core free
BATCH_WRITE_SIZE = 10             # Number of results to write to CSV at once
SIM_CONFIG_FILE = 'simulations_3.1.yaml'
GMAIL_FILE = 'config_mail.yaml'
# =================================

# ========= UTILITY FUNCTIONS (unchanged) =========

def send_email(input='!!'):
    # This function remains the same
    pass # Implementation hidden for brevity

def e_n(number):
    # This function remains the same
    if number == 0:
        return "0"
    abs_number = abs(number)
    if 1e-2 <= abs_number <= 1e3:
        if float(number).is_integer():
            return str(int(number))
        else:
            return f"{number:.2f}".replace('.', 'p') # Use 'p' for decimal point in filenames
    scientific_notation = "{:.1e}".format(abs_number)
    base, exponent = scientific_notation.split('e')
    exponent = int(exponent)
    abs_exponent = abs(exponent)
    if float(base).is_integer():
        base = str(int(float(base)))
    if exponent < 0:
        return f"{base}em{abs_exponent}"
    else:
        return f"{base}e{abs_exponent}"

def plot_simulation_results(time_bins, observed_counts, signal_counts, background_counts, save_path, **params):
    # This function remains the same
    plt.figure(figsize=(12, 7))
    plt.plot(time_bins, observed_counts, label='Observed (Signal + Background)', color='black', drawstyle='steps-post')
    plt.plot(time_bins, signal_counts, label='Signal Component', color='cornflowerblue', linestyle='--', alpha=0.8)
    plt.plot(time_bins, background_counts, label='Background Component', color='salmon', linestyle=':', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    title_params = ', '.join([f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()])
    plt.title(f'Simulated Light Curve\n{title_params}', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _parse_param(param_config):
    # This function remains the same
    if isinstance(param_config, dict) and 'start' in param_config:
        return np.arange(**param_config)
    if isinstance(param_config, list):
        return param_config
    if isinstance(param_config, (int, float)):
        return [param_config]
    raise TypeError(f"Unsupported parameter format in YAML: {param_config}")


# ========= GENERIC SIMULATION FRAMEWORK =========

# --- 1. The Simulation Registry ---
# This dictionary maps the 'type' from your YAML to the correct Python classes.
# EXTENDED to include the new models.
SIMULATION_REGISTRY = {
    'gaussian': 'GaussianSimulationTask',
    'triangular': 'TriangularSimulationTask',
    'norris': 'NorrisSimulationTask',
    'fred': 'FredSimulationTask',
    'lognormal': 'LognormalSimulationTask',
}

# --- 2. The Generic Task Classes ---
class BaseSimulationTask:
    """A base class for all simulation tasks."""
    def __init__(self, output_path, variable_params, constant_params):
        self.output_path = output_path
        self.params = {**variable_params, **constant_params}
        self.sim_name = self._create_sim_name(variable_params)
        self.sim_type = self.__class__.__name__.replace("SimulationTask", "").lower()

    def _create_sim_name(self, variable_params):
        """Creates a descriptive filename from the parameters that are changing."""
        class_name = self.__class__.__name__.replace("SimulationTask", "").lower()
        name_parts = [class_name]
        for key, value in sorted(variable_params.items()):
            key_abbr = ''.join(c for c in key if c.islower() and c not in 'aeiou')[:3]
            name_parts.append(f"{key_abbr}_{e_n(value)}")
        return "_".join(name_parts)

    def run(self):
        """The main execution method, to be implemented by each subclass."""
        raise NotImplementedError("The 'run' method must be implemented by a subclass.")

    def _execute_and_process(self, generation_func):
        """A standardized execution block for all tasks."""
        try:
            # 1. Generate light curve data
            t_bins, counts, s_counts, b_counts = generation_func(**self.params)

            # 2. Plot the simulated light curve
            sim_plot_path = os.path.join(self.output_path, self.sim_name + '_sim.png')
            plot_simulation_results(t_bins, counts, s_counts, b_counts, sim_plot_path, **self.params)

            # 3. Run the MVT analysis and plot results
            mvt_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt.png')
            results = haar_power_mod(counts, np.sqrt(counts), min_dt=self.params['bin_width'], doplot=True, afactor=-1.0, file=mvt_plot_path, verbose=False)

            # 4. Return formatted results for the CSV file
            return {'type': self.sim_type, 'mvt_ms': round(float(results[2])*1000, 3),
                    'mvt_error_ms': round(float(results[3])*1000, 3), **self.params}
        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            return {'Simulation': self.sim_name, 'type': self.sim_type, 'mvt_ms': -100}

# REFINED Gaussian Task
class GaussianSimulationTask(BaseSimulationTask):
    """Task for Gaussian light curves."""
    def run(self):
        return self._execute_and_process(generate_gaussian_light_curve)

# REFINED Triangular Task
class TriangularSimulationTask(BaseSimulationTask):
    """Task for Triangular light curves."""
    def run(self):
        return self._execute_and_process(generate_triangular_light_curve)

# NEW Norris Task
class NorrisSimulationTask(BaseSimulationTask):
    """Task for Norris light curves."""
    def run(self):
        return self._execute_and_process(generate_norris_light_curve)

# NEW FRED Task
class FredSimulationTask(BaseSimulationTask):
    """Task for FRED light curves."""
    def run(self):
        return self._execute_and_process(generate_fred_light_curve)

# NEW Lognormal Task
class LognormalSimulationTask(BaseSimulationTask):
    """Task for Lognormal light curves."""
    def run(self):
        return self._execute_and_process(generate_lognormal_light_curve)


# --- 3. The Fully Generic Task Generator (unchanged) ---
def generate_sim_tasks_from_config(config_path, output_path):
    """Reads the config and uses the registry and itertools to generate all parameter combinations."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for campaign in config:
        if not campaign.get('enabled', True):
            logging.info(f"Campaign '{campaign['name']}' is disabled. Skipping.")
            continue
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

# --- 4. The Worker Function (unchanged) ---
def run_simulation(task):
    """A simple top-level function to be called by the ProcessPoolExecutor."""
    return task.run()

# ========= MAIN EXECUTION BLOCK (unchanged) =========
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
    output_csv_path = os.path.join(output_path, f"results_{now}.csv")
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
                    if not header_written: header_written = True
                    results_batch = []
            except Exception as e:
                logging.error(f"A task failed in the main loop: {e}", exc_info=True)

    if results_batch:
        df_batch = pd.DataFrame(results_batch)
        df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)

    logging.info(f"All simulations processed! Results saved to:\n{output_csv_path}")
    # send_email(f"Successfully processed {len(tasks)} simulations.")

if __name__ == '__main__':
    main()