"""
# analyze_events.py
Suman Bala
17 Aug 2025: This script reads a directory of raw TTE event files,
            performs analysis (MVT, SNR) for various bin widths,
            and outputs a final summary CSV file.
"""

# ========= Import necessary libraries =========
import os
import yaml
import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List
import matplotlib.pyplot as plt

from SIM_lib import _parse_param, e_n, _create_param_directory_name, PULSE_MODEL_MAP, RELEVANT_NAMING_KEYS, SIMULATION_REGISTRY, send_email


from TTE_SIM_v2 import GBM_MVT_analysis, Function_MVT_analysis, print_nested_dict, check_param_consistency, flatten_dict
# ========= PLACEHOLDERS FOR YOUR CUSTOM MODULES =========
# In a real project, these would be imported from your library files.
try:
    from haar_power_mod import haar_power_mod
except ImportError:
    logging.warning("Could not import 'haar_power_mod'. Using a dummy function.")
    def haar_power_mod(*args, **kwargs):
        # Returns dummy data: [status, num_bins, mvt, mvt_err]
        return [0, 100, np.random.uniform(10, 50), np.random.uniform(1, 5)]

# ========= USER SETTINGS =========
MAX_WORKERS = os.cpu_count() - 2
SIM_CONFIG_FILE = 'simulations_ALL.yaml'
RESULTS_FILE_NAME = "final_summary_results.csv"


def generate_analysis_tasks(config: Dict[str, Any]) -> 'Generator':
    data_path = Path(config['project_settings']['data_path'])
    analysis_settings = config['analysis_settings']
    pulse_definitions = config.get('pulse_definitions', {})

    for campaign in config.get('simulation_campaigns', []):
        if not campaign.get('enabled', False):
            continue
        sim_type = campaign.get('type')
        for pulse_config in campaign.get('pulses_to_run', []):
            pulse_shape = pulse_config if isinstance(pulse_config, str) else list(pulse_config.keys())[0]
            base_pulse_config = pulse_definitions.get(pulse_shape, {})
            
            variable_params_config = campaign.get('parameters', {}).copy()
            variable_params_config.update(base_pulse_config.get('parameters', {}))
            
            if 'trigger_set_file' in variable_params_config:
                df_triggers = pd.read_csv(variable_params_config.pop('trigger_set_file'))
                variable_params_config['trigger_set'] = df_triggers.to_dict('records')

            #final_constants = {}
            #final_constants.update(campaign.get('constants', {}))
            total_sim_analysis = campaign['constants'].get('total_sim', 100)
            #final_constants.update(base_pulse_config.get('constants', {}))
            #final_constants.update(pulse_override_constants)
            #final_constants['pulse_shape'] = pulse_shape

            param_names = list(variable_params_config.keys())
            param_values = [_parse_param(v) for v in variable_params_config.values()]

            for combo in itertools.product(*param_values):
                current_variable_params = dict(zip(param_names, combo))
                
                param_dir_name = _create_param_directory_name(sim_type, pulse_shape, current_variable_params)
                param_dir_path = data_path / sim_type / pulse_shape / param_dir_name
                #print(f"ANALYSIS_SCRIPT --- Looking for: {param_dir_path}")

                if param_dir_path.exists():
                    yield {
                        'param_dir_path': param_dir_path,
                        'base_params': {**current_variable_params, 'pulse_shape': pulse_shape,
                                        'sim_type': sim_type, 'num_analysis': total_sim_analysis},
                        'analysis_settings': analysis_settings,
                    }





# ========= THE CORE WORKER FUNCTION =========
def analyze_one_group(task_info: Dict, data_path: Path, results_path: Path) -> List[Dict]:
    """
    Analyzes one group of NN event files, intelligently handling both 
    'function' (.npz) and 'gbm' (.fits) data products. For each group, it 
    creates a detailed per-realization CSV, a distribution plot, and 
    returns the final summary data.
    """
    # --- 1. Initial Setup ---
    param_dir = task_info['param_dir_path']
    base_params = task_info['base_params']
    analysis_settings = task_info['analysis_settings']
    sim_type = base_params['sim_type']

    # <<< Define standard keys and defaults from your original script >>>
    ALL_PULSE_PARAMS = ['sigma', 'center_time', 'width', 'peak_time_ratio', 'start_time', 'rise_time', 'decay_time']
    DEFAULT_PARAM_VALUE = -999
    STANDARD_KEYS = [
        'sim_type', 'pulse_shape', 'peak_amplitude', 'analysis_bin_width_ms',
        'total_sim', 'successful_runs', 'failed_runs',
        'median_mvt_ms', 'mvt_err_lower', 'mvt_err_upper',
        'all_median_mvt_ms', 'all_mvt_err_lower', 'all_mvt_err_upper',
        'angle', 'det', 'trigger', 
        'mean_bkgd_counts', 'mean_src_counts', 'mean_back_avg',
        'S_flu', 'S16', 'S32', 'S64', 'S128', 'S256'
    ]
    
    # Create a mirrored output directory in the results path
    relative_path = param_dir.relative_to(data_path)
    output_analysis_path = results_path / relative_path
    output_analysis_path.mkdir(parents=True, exist_ok=True)

    sim_param_file = sorted(param_dir.glob('*.yaml'))
    sim_params = yaml.safe_load(open(sim_param_file[0], 'r'))
    #print("Type:", type(sim_params))
    #print('\n')
    check_param_consistency(
            dict1=sim_params,
            dict2=flatten_dict(base_params)
    )

    #print_nested_dict(base_params)

    # --- 2. Load Data and Perform Per-Realization Analysis ---
    if sim_type == 'gbm':
        src_files = sorted(param_dir.glob('*_src.fits'))
        back_files = sorted(param_dir.glob('*_bkgd.fits'))
        #NN = len(src_files)
        #print(f"Found simulation parameter files: {sim_param_file}")
        if not src_files or not back_files:
            logging.warning(f"GBM analysis for {param_dir.name} has missing files.")
            return []

        iteration_results, NN = GBM_MVT_analysis(input_info={
            'src_event_files': src_files,
            'back_event_files': back_files,
            'sim_par_file': sim_params,
            'base_params': base_params,
            'snr_timescales': analysis_settings['snr_timescales'],
            'analysis_bin_widths_ms': analysis_settings['bin_widths_to_analyze_ms']},
            output_info = { 'file_path': output_analysis_path,
                        'file_name': param_dir.name})
        

    else: # sim_type == 'function'
        src_files = sorted(param_dir.glob('*_src.npz'))
        back_files = sorted(param_dir.glob('*_bkgd.npz'))
        if not src_files or not back_files: return []
        #NN = len(src_files)

        iteration_results, NN = Function_MVT_analysis(input_info={
            'src_event_files': src_files,
            'back_event_files': back_files,
            'sim_par_file': sim_params,
            'base_params': base_params,
            'snr_timescales': analysis_settings['snr_timescales'],
            'analysis_bin_widths_ms': analysis_settings['bin_widths_to_analyze_ms']},
            output_info={ 'file_path': output_analysis_path,
                         'file_name': param_dir.name})


    # --- 3. Aggregate Results (This logic is common to both data types) ---
    if not iteration_results: return []

    detailed_df = pd.DataFrame(iteration_results)
    detailed_df.to_csv(output_analysis_path / f"Detailed_{param_dir.name}.csv", index=False)

    # Inside analyze_one_group, after creating detailed_df
    final_summary_list = []
    # Loop through the DataFrame grouped by the analysis bin width

    
    for bin_width, group_df in detailed_df.groupby('analysis_bin_width_ms'):

        

        # The subset of those where the error was also valid
        valid_runs = group_df[group_df['mvt_err_ms'] > 0]
        # Statistics are calculated ONLY on the valid runs
        p16, median_mvt, p84 = np.percentile(valid_runs['mvt_ms'], [16, 50, 84])
        # Use the 68% confidence interval width as a robust measure of "sigma"
        ci_width = p84 - p16
        # Set plot limits to be wide enough to see the distribution, but not the extreme outliers
        data_min = max(0, p16 - 3 * ci_width)
        data_max = p84 + 10 * ci_width #
        #data_max = np.percentile(all_positive_runs['mvt_ms'], 99.5) if not all_positive_runs.empty else p84 + 3 * ci_width

        # <<< 1. Get data for BOTH sets >>>
        # All runs where MVT produced a positive timescale
        all_dist_flag = True
        try:
            all_positive_runs = group_df[(group_df['mvt_ms'] > 0) & (group_df['mvt_ms'] < 1e5)]
            all_p16, all_median_mvt, all_p84 = np.percentile(all_positive_runs['mvt_ms'], [16, 50, 84])
        except:
            all_dist_flag = False
            all_p16, all_median_mvt, all_p84 = (0, 0, 0)

        # --- Create the Enhanced MVT Distribution Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # <<< 2. Plot the background histogram of ALL non-failed runs in gray >>>
        if not all_positive_runs.empty and all_dist_flag:
            ax.hist(all_positive_runs['mvt_ms'], bins=30, density=True, 
                        label=f'All Runs w/ MVT > 0 ({len(all_positive_runs)}/{NN})',
                        color='gray', alpha=0.5, histtype='stepfilled', edgecolor='none', zorder=1)
            # Overlay the statistics from the valid runs
            ax.axvline(all_median_mvt, color='k', linestyle='-', lw=1.0,
                    label=f"Median = {all_median_mvt:.3f} ms")
            #ax.axvspan(all_p16, all_p84, color='k', alpha=0.1, hatch='///',
           #label=f"68% C.I. [{all_p16:.3f}, {all_p84:.3f}]")

            ax.axvspan(all_p16, all_p84, color='gray', alpha=0.1,
                    label=f"68% C.I. [{all_p16:.3f}, {all_p84:.3f}]")

        # <<< 3. Plot the main histogram of VALID runs (err > 0) on top >>>
        if len(valid_runs) > 2:
            ax.hist(valid_runs['mvt_ms'], bins=30, density=True, 
                    label=f'Valid Runs w/ Err > 0 ({len(valid_runs)}/{NN})',
                    color='steelblue', histtype='stepfilled', edgecolor='black', zorder=2) 

            # Overlay the statistics from the valid runs
            ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2.5,
                    label=f"Median = {median_mvt:.3f} ms")
            #ax.axvspan(p16, p84, color='darkorange', alpha=0.3,
            #        label=f"68% C.I. [{p16:.3f}, {p84:.3f}]")
            ax.axvline(p16, color='orange', linestyle='--', lw=1)
            ax.axvline(p84, color='orange', linestyle='--', lw=1)
            ax.axvspan(p16, p84, color='darkorange', alpha=0.1, hatch='///',
                    label=f"68% C.I. [{p16:.3f}, {p84:.3f}]")
            
        auto_min, auto_max = ax.get_xlim()
        final_min = max(auto_min, data_min)
        final_max = min(auto_max, data_max)

        # Formatting
        ax.set_xlim(final_min, final_max)
        ax.set_ylim(bottom=0)
        ax.set_title(f"MVT: {param_dir.name}\nBin Width: {bin_width} ms", fontsize=12)
        ax.set_xlabel("Minimum Variability Timescale (ms)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        fig.tight_layout()
        plt.savefig(output_analysis_path / f"MVT_dis_{param_dir.name}_{bin_width}ms.png", dpi=300)
        plt.close(fig)

        #exit()
        # Build the standardized final output dictionary
        result_data = {
            **base_params, 'analysis_bin_width_ms': bin_width,
            'total_sim': NN, 'successful_runs': len(valid_runs),
            'failed_runs': len(group_df) - len(valid_runs),
            'median_mvt_ms': round(median_mvt, 4),
            'mvt_err_lower': round(median_mvt - p16, 4),
            'mvt_err_upper': round(p84 - median_mvt, 4),
            'all_median_mvt_ms': round(all_median_mvt, 4),
            'all_mvt_err_lower': round(all_median_mvt - all_p16, 4),
            'all_mvt_err_upper': round(all_p84 - all_median_mvt, 4),
            'mean_src_counts': round(valid_runs['src_counts'].mean(), 2),
            'mean_bkgd_counts': round(valid_runs['bkgd_counts'].mean(), 2),
            'S_flu': round(valid_runs['S_flu'].mean(), 2),
            'S16': round(valid_runs['S16'].mean(), 2),
            'S32': round(valid_runs['S32'].mean(), 2),
            'S64': round(valid_runs['S64'].mean(), 2),
            'S128': round(valid_runs['S128'].mean(), 2),
            'S256': round(valid_runs['S256'].mean(), 2),
            'mean_back_avg': round(valid_runs['back_avg_cps'].mean(), 2),
            'trigger': sim_params['trigger_number'],
            'det': sim_params['det'],
            'angle': sim_params['angle'],
        }
        
        final_dict = {}

        """        # Populate with standard keys and any varying physical parameters
        print(f"BASE PARAMS: {base_params}")
        all_keys = STANDARD_KEYS + list(base_params.get('trigger_set', {}).keys())
        #all_keys.pop('trigger_set')
        for key in all_keys: final_dict[key] = result_data.get(key)
        # Fill in any non-applicable pulse parameters with a default
        for key in ALL_PULSE_PARAMS:
            if key not in final_dict: final_dict[key] = DEFAULT_PARAM_VALUE
        """

        for key in STANDARD_KEYS:
            final_dict[key] = result_data.get(key)
        
        # Add all possible pulse parameter keys, using the default value if a key is not in this run's result_data
        for key in ALL_PULSE_PARAMS:
            final_dict[key] = result_data.get(key, DEFAULT_PARAM_VALUE)
        
        final_summary_list.append(final_dict)
    return final_summary_list


# ========= MAIN EXECUTION BLOCK (REFACTORED FOR MEMORY STABILITY) =========
def main():
    now = datetime.now().strftime("%y_%m_%d-%H_%M")
    with open(SIM_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
        
    data_path = Path(config['project_settings']['data_path'])
    results_path = Path(config['project_settings']['results_path'])

    analysis_bin = config['analysis_settings']['bin_widths_to_analyze_ms'][0]

    # Create a new, timestamped directory for this run's results
    run_results_path = results_path / f"run_{e_n(analysis_bin)}_{now}"
    run_results_path.mkdir(parents=True, exist_ok=True)
    
    log_file = run_results_path / f'analysis_{now}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    tasks = list(generate_analysis_tasks(config))
    if not tasks:
        logging.warning("No existing simulation directories found to analyze.")
        return

    logging.info(f"Found {len(tasks)} parameter sets to analyze. Starting parallel processing.")
    
    all_results = []
    
    # <<< NEW: Define a chunk size >>>
    # A good starting point is 2-4 times the number of workers.
    CHUNK_SIZE = MAX_WORKERS * 4 

    # <<< NEW: Loop through the tasks in chunks >>>
    for i in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[i:i + CHUNK_SIZE]
        logging.info(f"--- Processing chunk {i//CHUNK_SIZE + 1}/{-(-len(tasks)//CHUNK_SIZE)} (Tasks {i+1}-{i+len(chunk)}) ---")
        
        # <<< The ProcessPoolExecutor is now INSIDE the loop >>>
        # This creates a fresh set of workers for each chunk.
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(analyze_one_group, task, data_path, run_results_path) for task in chunk}
            
            for future in tqdm(as_completed(futures), total=len(chunk), desc="Analyzing Chunk"):
                try:
                    result_list = future.result()
                    if result_list:
                        all_results.extend(result_list)
                except Exception as e:
                    logging.error(f"An analysis task failed in the pool: {e}", exc_info=True)

    # --- Save Final Summary ---
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_results_csv_path = run_results_path / RESULTS_FILE_NAME
        final_df.to_csv(final_results_csv_path, index=False)
        logging.info(f"✅ Analysis complete! Summary saved to:\n{final_results_csv_path}")
    else:
        logging.info("Analysis complete, but no results were generated.")

    #send_email(f"Analysis complete! Summary saved to:\n{final_results_csv_path}")

if __name__ == '__main__':
    main()