"""
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th July 2025: Including Fermi GBM simulation of same functions.

"""


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
from TTE_SIM import gen_GBM_pulse, gaussian2, triangular, constant, linear, quadratic

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




# ========= DATA GENERATION FUNCTIONS =========


# ========= GENERIC SIMULATION FRAMEWORK =========

# --- 1. The Simulation Registry ---
# This dictionary maps the 'type' from your YAML to the correct Python classes.
SIMULATION_REGISTRY = {
    'gbm': 'GbmSimulationTask',  # <-- ADDED THIS LINE
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
        # Use the 'pulse_shape' from params if it exists, otherwise use class name
        base_name = self.params.get('pulse_shape', self.__class__.__name__.replace("SimulationTask", ""))
        name_parts = [self.__class__.__name__.replace("SimulationTask", ""), base_name]
        for key, value in sorted(variable_params.items()):
            key_abbr = ''.join(c for c in key if c.islower() and c not in 'aeiou')[:3]
            name_parts.append(f"{key_abbr}_{e_n(value)}")
        return "_".join(name_parts)

    def run(self):
        raise NotImplementedError("The 'run' method must be implemented by a subclass.")



# NEW TASK CLASS FOR GBM
class GbmSimulationTask(BaseSimulationTask):
    """Task for Fermi GBM light curves using specified pulse shapes."""
    def run(self):
        try:
            pulse_shape = self.params.get('pulse_shape')
            if not pulse_shape:
                raise ValueError("'pulse_shape' must be defined in the YAML for gbm tasks.")

            # Determine the function and its parameters based on the pulse shape
            if pulse_shape == 'gaussian':
                func_to_use = gaussian2
                func_par = (
                    self.params['peak_amplitude'],
                    self.params['center_time'],
                    self.params['sigma']
                )
            elif pulse_shape == 'triangular':
                func_to_use = triangular
                # Translate framework params to what the 'triangular' function needs
                tstart = self.params['start_time']
                width = self.params['width']
                peak_time_ratio = self.params['peak_time_ratio']
                tpeak = tstart + (width * peak_time_ratio)
                tstop = tstart + width
                func_par = (self.params['peak_amplitude'], tstart, tpeak, tstop)
            else:
                raise ValueError(f"Unsupported pulse_shape for GBM: '{pulse_shape}'")
            
            if 'back_func' in self.params:
                back_func_name = self.params['back_func']
                if back_func_name == 'constant':
                    back_func = constant
                    back_func_par = (self.params.get('back_level', 1.0),)
                elif back_func_name == 'linear':
                    back_func = linear
                    back_func_par = (self.params.get('back_slope', 0.0),)
                elif back_func_name == 'quadratic':
                    back_func = quadratic
                    back_func_par = (self.params.get('back_level2', 1.0),)
                else:
                    raise ValueError(f"Unsupported background function: '{back_func_name}'")

            # Call the GBM pulse generation function
            # Note: We pass self.params directly; gen_GBM_pulse will pick what it needs
            # (e.g., trigger_number, det, t_start, t_stop, etc.)
            t_bins, counts = gen_GBM_pulse(
                func=func_to_use,
                func_par=func_par,
                back_func=constant, # Using the helper function defined above
                **self.params
            )

            # Process the results with haar_power_mod
            mvt_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt.png')
            min_dt = self.params.get('bin_width', 0.0001) # Ensure bin_width is available
            results = haar_power_mod(counts, np.sqrt(counts), min_dt=min_dt, doplot=True, file=mvt_plot_path, verbose=False)

            return {'type': 'gbm', 'mvt_ms': round(float(results[2])*1000, 3),
                    'mvt_error_ms': round(float(results[3])*1000, 3), **self.params}
        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            return {'Simulation': self.sim_name, 'type': 'gbm', 'mvt_ms': -100}

def _parse_param(param_config):
    """Helper function to parse parameter definitions from the YAML config."""
    if isinstance(param_config, dict) and 'start' in param_config:
        return np.arange(**param_config)
    if isinstance(param_config, list):
        return param_config
    if isinstance(param_config, (int, float)):
        return [param_config]
    raise TypeError(f"Unsupported parameter format in YAML: {param_config}")


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