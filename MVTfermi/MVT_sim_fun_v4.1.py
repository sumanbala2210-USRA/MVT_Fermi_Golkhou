"""
MVT_sim_v4.py
07.08.2025  This script simulates MVT (Mean Variance Time) data.
            It supports Gaussian, Triangular, Norris, FRED, and Lognormal pulses.
            It reads 'simulations.yaml' for parameters, generates light curves,
            applies Haar power modulation, and saves results.
            The output CSV is standardized to include all possible parameter
            columns, regardless of the pulse type, for easier analysis.
"""

# ========= Import necessary libraries =========
import os
import traceback
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import smtplib
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
    generate_lognormal_light_curve,
)

# ========= USER SETTINGS =========
MAX_WORKERS = os.cpu_count() - 1      # Leave one core free
BATCH_WRITE_SIZE = 10                 # Number of results to write to CSV at once
SIM_CONFIG_FILE = 'simulations_3.1.yaml' # The YAML configuration file

# ========= UTILITY FUNCTIONS =========

def e_n(number):
    """Formats numbers for concise filenames."""
    if number == 0:
        return "0"
    abs_number = abs(number)
    if 1e-2 <= abs_number <= 1e3:
        return f"{number:.2f}".replace('.', 'p') if not float(number).is_integer() else str(int(number))
    
    scientific_notation = f"{abs_number:.1e}"
    base, exponent = scientific_notation.split('e')
    exponent = int(exponent)
    base = str(int(float(base))) if float(base).is_integer() else base
    
    return f"{base}{'em' if exponent < 0 else 'e'}{abs(exponent)}"

def plot_simulation_results(time_bins, observed_counts, signal_counts, background_counts, save_path, **params):
    """Generates and saves a plot of the simulated light curve."""
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
    """Helper function to parse parameter definitions from the YAML config."""
    if isinstance(param_config, dict) and 'start' in param_config:
        return np.arange(**param_config)
    if isinstance(param_config, list):
        return param_config
    if isinstance(param_config, (int, float)):
        return [param_config]
    raise TypeError(f"Unsupported parameter format in YAML: {param_config}")

# ========= GENERIC SIMULATION FRAMEWORK =========

# --- 1. The Simulation Registry ---
SIMULATION_REGISTRY = {
    'gaussian': 'GaussianSimulationTask',
    'triangular': 'TriangularSimulationTask',
    'norris': 'NorrisSimulationTask',
    'fred': 'FredSimulationTask',
    'lognormal': 'LognormalSimulationTask',
}

# --- 2. The Generic Task Classes ---
class BaseSimulationTask:
    """A base class for all simulation tasks with standardized result formatting."""
    
    # Define all possible model-specific parameters across all simulation types
    ALL_MODEL_PARAMS = [ 'sigma',  'width', 'tau', 'rise_time', 'decay_time', 'start_time', 'center_time', 'peak_time', 'peak_time_ratio']


    DEFAULT_PARAM_VALUE = 999.0 # Value for non-applicable parameters

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
        """The main execution method, must be implemented by each subclass."""
        raise NotImplementedError("The 'run' method must be implemented by a subclass.")
    

    def _execute_and_process_old(self, generation_func):
        """A standardized execution block that generates data and formats results."""
        try:
            # 1. Generate light curve data
            t_bins, counts, s_counts, b_counts = generation_func(**self.params)

            # 2. Plot the simulated light curve
            sim_plot_path = os.path.join(self.output_path, self.sim_name + '_sim.png')
            plot_simulation_results(t_bins, counts, s_counts, b_counts, sim_plot_path, **self.params)

            # 3. Run the MVT analysis
            mvt_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt.png')
            results = haar_power_mod(counts, np.sqrt(counts), min_dt=self.params['bin_width'], doplot=True, afactor=-1.0, file=mvt_plot_path, verbose=False)
            plt.close('all')

            # 4. Gather results into a temporary dictionary
            result_data = {
                'type': self.sim_type,
                'mvt_ms': round(float(results[2]) * 1000, 3),
                'mvt_error_ms': round(float(results[3]) * 1000, 3),
                **self.params
            }
        
        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            result_data = {
                'type': self.sim_type,
                'mvt_ms': -100,
                'mvt_error_ms': -100,
                **self.params
            }

        # 5. Create the final, standardized dictionary for the CSV file
        final_dict = {}
        

        standard_keys = ['type', 'mvt_ms', 'mvt_error_ms', 'peak_amplitude', 'bin_width', 'background_level', 'back_factor']
        
        # Add standard, non-model keys
        for key in standard_keys:
            final_dict[key] = result_data.get(key)
        
        # Add all possible model-specific parameters
        for key in self.ALL_MODEL_PARAMS:
            final_dict[key] = result_data.get(key, self.DEFAULT_PARAM_VALUE)

        return final_dict

    def _execute_and_process(self, generation_func):
        """
        A standardized execution block that runs the simulation 100 times to get
        stable MVT statistics, then generates data and formats results.
        It now correctly modifies the random_seed for each run.
        """
        try:
            # List to store the MVT result from each of the 100 runs
            mvt_timescales_ms = []

            # --- KEY CHANGE: Handle the random seed ---
            # Store the original seed from the YAML file.
            original_seed = self.params.get('random_seed')
            
            # Define plot paths once, they will only be used in the last loop
            sim_plot_path = os.path.join(self.output_path, self.sim_name + '_sim.png')
            mvt_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt.png')

            # Loop 30 times to gather MVT statistics
            NN=100
            for i in range(NN):
                # --- KEY CHANGE: Update the seed for this specific run ---
                # If the original seed is a number, increment it. Otherwise, keep it as None.
                # This ensures each of the 30 simulations is a unique random realization.
                try:
                    if original_seed is not None:
                        self.params['random_seed'] = original_seed + i

                    # We only want to generate plots for the final iteration as an example
                    is_last_iteration = (i == NN - 1)

                    # 1. Generate a new, random light curve realization with the updated seed
                    t_bins, counts, s_counts, b_counts = generation_func(**self.params)

                    # 2. Plot the simulated light curve (only on the last run)
                    if is_last_iteration:
                        plot_simulation_results(t_bins, counts, s_counts, b_counts, sim_plot_path, **self.params)
                        plt.close('all')

                    # 3. Run the MVT analysis. doplot is controlled by the loop.
                    results = haar_power_mod(counts, np.sqrt(counts), min_dt=self.params['bin_width'], doplot=True, afactor=-1.0, file=mvt_plot_path, verbose=False)
                    plt.close('all')
                    
                    # Append the result in milliseconds
                    mvt_error = float(results[3])
                    if mvt_error != 0:
                        mvt_timescales_ms.append(float(results[2]) * 1000)
                except Exception as iter_e:
                    # If one iteration fails, log it and move to the next one
                    logging.warning(f"Run {i+1}/{NN} for {self.sim_name} failed and will be skipped. Error: {iter_e}")
                    continue
            

            # Check if there are any valid results before proceeding
            if len(mvt_timescales_ms) > 0:
                # Calculate robust statistics from the 1D Series
                p16, median_mvt, p84 = np.percentile(mvt_timescales_ms, [16, 50, 84])
                upper_error = p84 - median_mvt
                lower_error = median_mvt - p16

                # --- Create and Save the Plot ---
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot the histogram using the 1D Series
                ax.hist(mvt_timescales_ms, bins=30, density=True, color='skyblue',
                        edgecolor='black', alpha=0.8, label=f"MVT Distribution ({len(mvt_timescales_ms)}/{NN} runs)")

                # Overlay the statistics
                ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2.5,
                        label=f"Median = {median_mvt:.3f} ms")
                ax.axvspan(p16, p84, color='firebrick', alpha=0.2,
                        label=f"68% C.I. Range [{p16:.3f}, {p84:.3f}]")

                # Formatting
                ax.set_title(f"{self.sim_name}", fontsize=10)
                ax.set_xlabel("Minimum Variability Timescale (ms)", fontsize=12)
                ax.set_ylabel("Probability Density", fontsize=12)
                ax.legend(fontsize=10)
                ax.set_xlim(left=max(0, p16 - 3 * lower_error)) # This is now safe
                fig.tight_layout()

                # Save the plot
                output_plot_path = os.path.join(self.output_path, self.sim_name + '_mvt_distribution.png')
                plt.savefig(output_plot_path, dpi=300)
                plt.close(fig)

            # --- KEY CHANGE: Restore the original seed ---
            # This ensures the original parameter from the YAML is reported in the CSV.
            self.params['random_seed'] = original_seed

            # After the loop, calculate the mean and standard deviation of the results
            # --- KEY CHANGE: Safely calculate statistics ---
            # Check if the list contains any valid results before calculating
            if len(mvt_timescales_ms)>1:
                mean_mvt_ms = np.mean(mvt_timescales_ms)
                error_mvt_ms = np.std(mvt_timescales_ms)
            else:
                # If all runs had a zero error, report MVT as 0
                mean_mvt_ms = 0.0
                error_mvt_ms = 0.0
                
            # 4. Gather results into a temporary dictionary using the computed stats
            result_data = {
                'type': self.sim_type,
                'mvt_ms': round(mean_mvt_ms, 3),
                'mvt_error_ms': round(error_mvt_ms, 3),
                **self.params # Now contains the original seed
            }
        
        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            result_data = {
                'type': self.sim_type,
                'mvt_ms': -100,
                'mvt_error_ms': -100,
                **self.params
            }

        # 5. Create the final, standardized dictionary for the CSV file (this part is unchanged)
        final_dict = {}
        # Add 'random_seed' to the list of standard keys to ensure it's in the output
        standard_keys = ['type', 'mvt_ms', 'mvt_error_ms', 'peak_amplitude', 'bin_width', 'background_level', 'back_factor']
        
        # Add standard, non-model keys
        for key in standard_keys:
            final_dict[key] = result_data.get(key)
        
        # Add all possible model-specific parameters
        for key in self.ALL_MODEL_PARAMS:
            final_dict[key] = result_data.get(key, self.DEFAULT_PARAM_VALUE)
        
        return final_dict


# --- Task-specific classes ---
class GaussianSimulationTask(BaseSimulationTask):
    def run(self): return self._execute_and_process(generate_gaussian_light_curve)

class TriangularSimulationTask(BaseSimulationTask):
    def run(self): return self._execute_and_process(generate_triangular_light_curve)

class NorrisSimulationTask(BaseSimulationTask):
    def run(self): return self._execute_and_process(generate_norris_light_curve)

class FredSimulationTask(BaseSimulationTask):
    def run(self): return self._execute_and_process(generate_fred_light_curve)

class LognormalSimulationTask(BaseSimulationTask):
    def run(self): return self._execute_and_process(generate_lognormal_light_curve)

# --- 3. The Fully Generic Task Generator ---
def generate_sim_tasks_from_config(config_path, output_path):
    """Reads the config and generates all parameter combinations for simulation tasks."""
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

# --- 4. The Worker Function ---
def run_simulation(task):
    """A simple top-level function to be called by the ProcessPoolExecutor."""
    return task.run()

# ========= MAIN EXECUTION BLOCK =========
def main():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'SIM_MVT_{now}'
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    log_file = os.path.join(output_path, 'run.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    try:
        tasks = list(generate_sim_tasks_from_config(SIM_CONFIG_FILE, output_path))
    except Exception as e:
        logging.error(f"Failed to generate tasks. Check YAML format. Error: {e}", exc_info=True)
        return
        
    if not tasks:
        logging.warning("No simulation tasks generated. Check your config file.")
        return

    logging.info(f"Generated {len(tasks)} simulation tasks. Starting parallel processing...")
    output_csv_path = os.path.join(output_path, f"all_results_{now}.csv")
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

if __name__ == '__main__':
    main()