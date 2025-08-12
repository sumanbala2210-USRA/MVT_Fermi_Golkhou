"""
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th July 2025: Including Fermi GBM simulation of same functions.
1st August 2025: Refactored to support modular, function-specific parameter
                 configurations via a nested YAML structure.
2nd August 2025: Simplified YAML by using an 'enabled' flag per function.
"""

# ========= Import necessary libraries =========
import os
import yaml
import logging
import smtplib
import itertools
import numpy as np
import pandas as pd
from email.message import EmailMessage
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import matplotlib.pyplot as plt

from haar_power_mod import haar_power_mod
from TTE_SIM import gen_GBM_pulse, gaussian2, triangular, constant, linear, quadratic, norris, fred #complex_pulse_example


# ========= USER SETTINGS (Unchanged) =========
MAX_WORKERS = os.cpu_count() - 1
BATCH_WRITE_SIZE = 10
SIM_CONFIG_FILE = 'simulations_GBM_v1.yaml'
GMAIL_FILE = 'config_mail.yaml'

# ========= GENERIC SIMULATION FRAMEWORK =========
# --- 1. The Simulation Registry (Unchanged) ---
SIMULATION_REGISTRY = {
    'gbm': 'GbmSimulationTask',
}




def safe_round(val):
    return int(np.round(val)) if val is not None and not np.isnan(val) else 0

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


# --- 2. The Generic Task Classes ---
class BaseSimulationTask:
    """A base class for all simulation tasks."""
    def __init__(self, output_path, variable_params, constant_params):
        self.output_path = output_path
        self.params = {**variable_params, **constant_params}

        # The sim_name must be created *before* unpacking, as it uses the original
        # variable_params dictionary to identify what is changing for each run.
        self.sim_name = self._create_sim_name(variable_params)

        # --- UNPACKING LOGIC ---
        # Unpacks the trigger_set so its contents are available to the 'run' method.
        # This also fixes the key name mismatch between your CSV ('trigger')
        # and the code's expectation ('trigger_number').
        if 'trigger_set' in self.params:
            trigger_data = self.params.pop('trigger_set')
            if 'trigger' in trigger_data and 'trigger_number' not in trigger_data:
                trigger_data['trigger_number'] = trigger_data.pop('trigger')
            self.params.update(trigger_data)


    def _create_sim_name(self, variable_params):
        """Creates a descriptive filename from the parameters that are changing."""
        base_name = self.params.get('pulse_shape', self.__class__.__name__.replace("SimulationTask", ""))
        name_parts = [self.__class__.__name__.replace("SimulationTask", ""), base_name]
        
        for key, value in sorted(variable_params.items()):
            # If the value is a dictionary (our trigger_set), handle it specially
            if isinstance(value, dict):
                # Use the contents of the dictionary to create a descriptive name
                #trigger_val = value.get('trigger', value.get('trigger_number', ''))
                angle_val = value.get('angle', '')
                name_part = f"ang_{angle_val}"
                name_parts.append(name_part)
            else:
                # Otherwise, use the normal numeric formatting for other variables
                key_abbr = ''.join(c for c in key if c.islower() and c not in 'aeiou')[:2]  # Abbreviate the key to 2 lowercase letters
                name_parts.append(f"{key_abbr}_{e_n(value)}")
        
        return "_".join(name_parts)

    def run(self):
        raise NotImplementedError("The 'run' method must be implemented by a subclass.")


class GbmSimulationTask(BaseSimulationTask):
    """Task for Fermi GBM light curves using specified pulse shapes."""
    def run(self):
        details_path = os.path.join(self.output_path, self.sim_name)
        os.makedirs(details_path, exist_ok=True)

        # Define all possible pulse-specific parameters that can appear in the output.
        ALL_PULSE_PARAMS = ['sigma', 'center_time', 'width', 'peak_time_ratio', 'start_time', 'rise_time', 'decay_time']
        # Define a default value for parameters that don't apply to a given pulse.
        DEFAULT_PARAM_VALUE = 999
        NN= self.params.get('total_sim', 30)
        original_seed = self.params.get('random_seed')
        standard_keys = ['pulse', 'total_sim', 'failed_sim', 'peak_amplitude', 'background_level', 'trigger_number', 'det', 'angle', 'mvt_ms', 'mvt_error_ms', 'src_max', 'back_avg', 'SNR']

        try:
            # 1. Prepare pulse-specific parameters
            pulse_shape = self.params.get('pulse_shape')
            if not pulse_shape:
                raise ValueError("'pulse_shape' must be defined.")

            pulse_specific_keys = {
                'gaussian': ['sigma', 'center_time'],
                'triangular': ['width', 'center_time', 'peak_time_ratio'],
                'norris': ['rise_time', 'decay_time', 'start_time'],
                'fred': ['rise_time', 'decay_time', 'start_time']
            }
            pulse_params = {k: self.params[k] for k in pulse_specific_keys.get(pulse_shape, []) if k in self.params}

            # 2. Set up the correct function and its parameters based on pulse_shape
            # (This logic is the same as your original code)
            
            
            if pulse_shape == 'gaussian':
                func_to_use = gaussian2
                func_par = (self.params['peak_amplitude'], self.params['center_time'], self.params['sigma'])
            elif pulse_shape == 'triangular':
                func_to_use = triangular
                tpeak, width, peak_ratio = self.params['center_time'], self.params['width'], self.params['peak_time_ratio']
                tstart = tpeak - (width * peak_ratio)
                tstop = tstart + width
                func_par = (self.params['peak_amplitude'], tstart, tpeak, tstop)
            elif pulse_shape == 'norris':
                func_to_use = norris
                func_par = (
                    self.params['peak_amplitude'],
                    self.params['start_time'],
                    self.params['rise_time'],
                    self.params['decay_time']
                )
            elif pulse_shape == 'fred':
                func_to_use = fred
                func_par = (
                    self.params['peak_amplitude'],
                    self.params['start_time'],
                    self.params['rise_time'],
                    self.params['decay_time']
                )
            else:
                raise ValueError(f"Unsupported pulse_shape for GBM: '{pulse_shape}'")
            
            # These will be updated on the last successful run

            # This list will hold raw MVT values for statistical summary
            mvt_timescales_ms = np.zeros(NN, dtype=float)
            mvt_err_timescales_ms = np.zeros(NN, dtype=float)

            src_max_list = np.zeros(NN, dtype=float)
            back_avg_list = np.zeros(NN, dtype=float)
            SNR_list = np.zeros(NN, dtype=float)
            # This list will hold full dictionaries for the detailed CSV file
            iteration_details_list = []

            is_last_iteration = False
            for i in range(NN):
                iteration_seed = original_seed + i

            # 2. Set up background function and run simulation (same as your code)
                try:
                    gbm_args = {k: self.params.get(k) for k in ['trigger_number', 'det', 'angle', 't_start', 't_stop', 'bkgd_times', 'en_lo', 'en_hi', 'bin_width', 'random_seed']}
                    gbm_args['random_seed'] = iteration_seed # Ensure each run has a unique seed
                    #gbm_args['fig_name'] = sim_plot_path
                    gbm_args = {k: v for k, v in gbm_args.items() if v is not None}
                    back_func_par = (self.params['background_level'],)
                    sim_plot_path = os.path.join(details_path, self.sim_name + '.png')

                    is_last_iteration = (i == NN - 1)

                    t_bins, counts, src_max, back_avg, SNR = gen_GBM_pulse(
                        func=func_to_use, func_par=func_par, back_func=constant, back_func_par=back_func_par, simulation= is_last_iteration, 
                        fig_name=sim_plot_path, **gbm_args
                    )
                    results = haar_power_mod(counts, np.sqrt(counts), min_dt=self.params.get('bin_width', 0.0001), doplot=False, afactor=-1.0, verbose=False)
                    plt.close('all')
                    mvt_val = float(results[2]) * 1000
                    mvt_err = float(results[3]) * 1000
                
                except Exception as iter_e:
                # If one iteration fails, log it and move to the next one
                    logging.warning(f"Run {i+1}/{NN} for {self.sim_name} failed and will be skipped. Error: {iter_e}")
                    src_max, back_avg, SNR = -100, -100, -100
                    mvt_val, mvt_err = -100, -100

                src_max_list[i] = src_max
                back_avg_list[i] = back_avg
                SNR_list[i] = SNR                   
                mvt_timescales_ms[i] = mvt_val
                mvt_err_timescales_ms[i] = mvt_err


                base_params = self.params.copy()
                # Remove the key we will be adding manually
                base_params.pop('random_seed', None) 
                # Remove the nested dictionary to keep the CSV clean
                base_params.pop('pulse_configs', None)

                # --- SAVE DETAILED CSV ---
                iter_detail = {'iteration': i + 1, 'random_seed': iteration_seed, 'mvt_ms': mvt_val, 'mvt_error_ms': mvt_err, 'src_max': src_max, 'back_avg': back_avg, 'SNR': SNR, **base_params}
                iter_detail.pop('pulse_configs', None)  # Remove pulse_configs if it exists
                iteration_details_list.append(iter_detail)

                            # --- SAVE DETAILED CSV ---
            if iteration_details_list:
                detailed_df = pd.DataFrame(iteration_details_list)
                detailed_csv_path = os.path.join(details_path, f"Detailed_{self.sim_name}.csv")
                detailed_df.to_csv(detailed_csv_path, index=False)

            valid_mvt = mvt_timescales_ms[mvt_err_timescales_ms > 0]
            valid_src_max = src_max_list[src_max_list > 0]
            valid_back_avg = back_avg_list[back_avg_list > 0]
            valid_SNR = SNR_list[SNR_list > 0]
            # Check if there are any valid results before proceeding
            if len(valid_mvt) > 0:
                # Calculate robust statistics from the 1D Series
                p16, median_mvt, p84 = np.percentile(valid_mvt, [16, 50, 84])
                upper_error = p84 - median_mvt
                lower_error = median_mvt - p16

                # --- Create and Save the Plot ---
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot the histogram using the 1D Series
                ax.hist(valid_mvt, bins=30, density=True, color='skyblue',
                        edgecolor='black', alpha=0.8, label=f"MVT Distribution ({len(valid_mvt)}/{NN} runs)")

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
                output_plot_path = os.path.join(details_path, 'Distribution_mvt_' +self.sim_name + '.png')
                plt.savefig(output_plot_path, dpi=300)
                plt.close(fig)
            
            self.params['random_seed'] = original_seed
            if len(valid_mvt) == 1:
                lower_error = 0.0 # Cannot calculate std dev from one point
            else:
                # If all runs had a zero error, report MVT as 0
                median_mvt = 0.0
                lower_error = 0.0

            # 3. Run the simulation (unchanged)
            # 4. Gather results into a temporary dictionary
            result_data = {
                'pulse': pulse_shape,
                'total_sim': int(NN),
                'failed_sim': int(NN - len(valid_mvt)),
                'peak_amplitude': self.params.get('peak_amplitude'),
                'background_level': self.params.get('background_level'),
                'trigger_number': self.params.get('trigger_number'),
                'det': self.params.get('det'),
                'angle': self.params.get('angle'),
                'mvt_ms': round(float(median_mvt), 3),
                'mvt_error_ms': round(float(lower_error), 3),
                'src_max': safe_round(valid_src_max.mean() if len(valid_src_max) > 0 else -100),
                'back_avg': safe_round(valid_back_avg.mean() if len(valid_back_avg) > 0 else -100),
                'SNR': safe_round(valid_SNR.mean() if len(valid_SNR) > 0 else -100),
            }
            result_data.update(pulse_params)

        except Exception as e:
            logging.error(f"Error processing {self.sim_name}: {e}", exc_info=True)
            result_data = {
                'pulse': self.params.get('pulse_shape', 'unknown'),
                'total_sim': int(NN),
                'failed_sim': int(NN - len(valid_mvt)),
                'peak_amplitude': self.params.get('peak_amplitude'),
                'background_level': self.params.get('background_level'),
                'trigger_number': self.params.get('trigger_number'),
                'det': self.params.get('det'),
                'angle': self.params.get('angle'),
                'mvt_ms': -100, 'mvt_error_ms': -100, 'src_max': -100,
                'back_avg': -100, 'SNR': -100,
            }

        # --- FINAL STEP: Create the standardized dictionary for output ---
        final_dict = {}
        

        for key in standard_keys:
            final_dict[key] = result_data.get(key)
        
        # Add all possible pulse parameter keys, using the default value if a key is not in this run's result_data
        for key in ALL_PULSE_PARAMS:
            final_dict[key] = result_data.get(key, DEFAULT_PARAM_VALUE)

        return final_dict


def _parse_param(param_config):
    """Helper function to parse parameter definitions from the YAML config."""
    if isinstance(param_config, dict) and 'start' in param_config:
        return np.arange(**param_config)
    if isinstance(param_config, list):
        return param_config
    if isinstance(param_config, (int, float)):
        return [param_config]
    raise TypeError(f"Unsupported parameter format in YAML: {param_config}")


# --- 4. The Worker Function ---
def run_simulation(task):
    """A simple top-level function to be called by the ProcessPoolExecutor."""
    return task.run()

# Add this function to your simulation script
def create_plot_template(output_csv_path, output_dir):
    """
    Reads the header of the results CSV and creates a self-contained
    template YAML file for the plotting script.
    """
    try:
        df = pd.read_csv(output_csv_path, nrows=0) # Read only the header
        columns = df.columns.tolist()

        # Start the config by pointing to its own data file
        plot_config = {'csv_file': os.path.basename(output_csv_path)}

        for col in columns:
            plot_config[col] = 'all'

        # Pre-fill the most common roles
        if 'mvt_ms' in plot_config:
            plot_config['mvt_ms'] = 'y'
        if 'mvt_error_ms' in plot_config:
            plot_config['mvt_error_ms'] = 'yerr'
        if 'sigma' in plot_config:
            plot_config['sigma'] = 'x'
        if 'peak_amplitude' in plot_config:
            plot_config['peak_amplitude'] = 'group'

        template_path = os.path.join(output_dir, 'plot_config_template.yaml')
        with open(template_path, 'w') as f:
            yaml.dump(plot_config, f, default_flow_style=False, sort_keys=False)

        logging.info(f"Plotting template created at: {template_path}")

    except Exception as e:
        logging.warning(f"Could not create plotting template. Error: {e}")


# --- 3. The Fully Generic Task Generator ---
# MODIFIED: Rewritten to use the 'enabled' flag in the YAML
def generate_sim_tasks_from_config(config_path, output_path):
    """
    Reads the config, finds pulse functions with 'enabled: true',
    and generates all parameter combinations for simulation tasks.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for campaign in config:
        if not campaign.get('enabled', True):
            continue

        sim_type = campaign.get('type')
        if not sim_type or sim_type not in SIMULATION_REGISTRY:
            logging.warning(f"Campaign '{campaign['name']}' has invalid type '{sim_type}'. Skipping.")
            continue

        logging.info(f"Generating tasks for campaign '{campaign['name']}' of type '{sim_type}'")
        TaskClass = globals()[SIMULATION_REGISTRY[sim_type]]

        # --- Base (campaign-level) parameters and constants ---
        base_params_config = campaign.get('parameters', {})
        base_constants = campaign.get('constants', {})
        pulse_configs = base_constants.get('pulse_configs', {})

        # Handle external trigger file at the campaign level
        if 'trigger_set_file' in base_constants:
            file_path = base_constants.pop('trigger_set_file')
            df_triggers = pd.read_csv(file_path)
            base_params_config['trigger_set'] = df_triggers.to_dict('records')
        elif 'trigger_set_file' in base_params_config:
             file_path = base_params_config.pop('trigger_set_file')
             df_triggers = pd.read_csv(file_path)
             base_params_config['trigger_set'] = df_triggers.to_dict('records')

        # --- Nested Iteration Logic ---
        # 1. Iterate through base-level parameter combinations (e.g., trigger sets)
        base_param_names = list(base_params_config.keys())
        base_param_values = [_parse_param(v) for v in base_params_config.values()]
        if not base_param_values: base_param_values.append(())

        for base_combo in itertools.product(*base_param_values):
            current_base_params = dict(zip(base_param_names, base_combo))

            # 2. Iterate through all defined pulse functions in the config
            for pulse_shape, func_config in pulse_configs.items():

                # 3. Check if the function is enabled for this run
                if not func_config.get('enabled', False):
                    continue

                logging.info(f"  > Found enabled function: '{pulse_shape}'")
                func_params_config = func_config.get('parameters', {})
                func_constants = func_config.get('constants', {})

                # 4. Iterate through the function-specific parameter combinations
                func_param_names = list(func_params_config.keys())
                func_param_values = [_parse_param(v) for v in func_params_config.values()]
                if not func_param_values: func_param_values.append(())

                for func_combo in itertools.product(*func_param_values):
                    current_func_params = dict(zip(func_param_names, func_combo))

                    # 5. Merge all parameters and yield the task
                    # The pulse_shape name itself is now a constant for this task group
                    final_variable_params = {**current_base_params, **current_func_params}
                    final_constants = {**base_constants, **func_constants, 'pulse_shape': pulse_shape}

                    yield TaskClass(output_path, final_variable_params, final_constants)



# ========= MAIN EXECUTION BLOCK (Unchanged) =========
def main():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'GBM_SIM'#_{now}'
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
        logging.warning("No simulation tasks were generated. Check 'enabled' flags in config file.")
        return

    logging.info(f"Generated {len(tasks)} simulation tasks. Starting parallel processing with {MAX_WORKERS} workers.")
     # 1. Create a single list to hold all results
    """
    all_results = [] 

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_simulation, task) for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Simulations"):
            try:
                result = future.result()
                if result:
                    # 2. Append each result to the master list
                    all_results.append(result) 
            except Exception as e:
                logging.error(f"A task failed in the main loop: {e}")

    # 3. After the loop, write the entire list to a CSV file at once
    if all_results:
        output_csv_path = os.path.join(output_path, f"{output_dir}_results.csv")
        df_final = pd.DataFrame(all_results)
        df_final.to_csv(output_csv_path, index=False)
        
        # Plotting template generation remains the same
        if os.path.exists(output_csv_path):
            create_plot_template(output_csv_path, output_path)

        logging.info(f"All simulations processed! Results saved to:\n{output_csv_path}")
    else:
        logging.warning("No results were generated from the simulations.")

    """
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
    
    if os.path.exists(output_csv_path):
        create_plot_template(output_csv_path, output_path)

    logging.info(f"All simulations processed! Results saved to:\n{output_csv_path}")



if __name__ == '__main__':
    main()

