"""
Suman Bala
17 Aug 2025: A unified, hybrid script for simulating and analyzing GRB pulses.
            It simulates event data in-memory once per realization and
            analyzes it for multiple bin widths for high efficiency.
"""

# ========= Import necessary libraries =========
import os
import shutil
import yaml
import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, Tuple, Callable, List
import matplotlib.pyplot as plt

from SIM_lib import _parse_param, e_n, _create_param_directory_name, PULSE_MODEL_MAP, RELEVANT_NAMING_KEYS, SIMULATION_REGISTRY, calculate_adaptive_simulation_params

from TTE_SIM_v2 import generate_gbm_events, generate_function_events, gaussian2, triangular, constant, norris, fred, lognormal


# ========= USER SETTINGS =========
MAX_WORKERS = os.cpu_count() - 2
SIM_CONFIG_FILE = 'simulations_ALL.yaml'

# ========= PLACEHOLDERS FOR YOUR CUSTOM MODULES =========
# In a real project, these would be in their own files and imported.
try:
    from haar_power_mod import haar_power_mod
except ImportError:
    logging.warning("Could not import 'haar_power_mod'. Using a dummy function.")
    def haar_power_mod(*args, **kwargs):
        # Returns dummy data: [status, num_bins, mvt, mvt_err]
        return [0, 100, np.random.uniform(10, 50), np.random.uniform(1, 5)]

try:
    from TTE_SIM_v2 import gaussian2, triangular, constant, norris, fred, lognormal
    import scipy.integrate as spi
except ImportError:
    logging.warning("Could not import TTE_SIM_v2. Using dummy functions.")
    gaussian2 = triangular = constant = norris = fred = lognormal = lambda *args: 0




# ========= CORE SIMULATION LOGIC =========
def generate_events_in_memory(func: Callable, func_par: Tuple, params: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generates and returns raw source and background event times IN MEMORY.
    """
    t_start, t_stop = params['t_start'], params['t_stop']
    random_seed = params['random_seed']
    background_level = params['background_level']
    np.random.seed(random_seed)

    # --- Source Simulation ---
    def source_rate_func(t): return func(t, *func_par)
    grid_res = params.get('grid_resolution', 0.0001)
    grid_times = np.arange(t_start, t_stop, grid_res)
    rate_on_grid = source_rate_func(grid_times)
    cum_counts = np.cumsum(rate_on_grid) * grid_res
    total_expected = cum_counts[-1] if len(cum_counts) > 0 else 0
    num_events = np.random.poisson(total_expected)
    random_counts = np.random.uniform(0, total_expected, num_events)
    source_events = np.interp(random_counts, cum_counts, grid_times)

    # --- Background Generation ---
    duration = t_stop - t_start
    num_bkg = np.random.poisson(background_level * duration)
    background_events = np.random.uniform(t_start, t_stop, size=num_bkg)

    sim_info = {'t_start': t_start, 't_stop': t_stop, 'background_level_cps': background_level}
    return np.sort(source_events), np.sort(background_events), sim_info

class BaseSimulationTask:
    def __init__(self, output_dir, variable_params, constant_params, analysis_settings):
        self.output_dir = output_dir
        self.variable_params = variable_params.copy()
        self.params = {**variable_params, **constant_params, 'analysis_settings': analysis_settings}
        if 'trigger_set' in self.params:
            trigger_data = self.params.pop('trigger_set')
            self.params.update(trigger_data)
    def run(self): raise NotImplementedError

class AbstractPulseSimulationTask(BaseSimulationTask):
    def run(self) -> List[Dict]:
        # --- 1. Initial Setup ---
        sim_type = 'gbm' if self.is_gbm else 'function'
        pulse_shape = self.params.get('pulse_shape')
        NN = self.params.get('total_sim', 1)
        original_seed = self.params.get('random_seed')
        
        # <<< Define standard keys and defaults, from your original script >>>
        ALL_PULSE_PARAMS = ['sigma', 'center_time', 'width', 'peak_time_ratio', 'start_time', 'rise_time', 'decay_time']
        DEFAULT_PARAM_VALUE = -999 # Use a clear "not applicable" value
        STANDARD_KEYS = ['sim_type', 'pulse_shape', 'total_sim', 'successful_runs', 
                         'median_mvt_ms', 'mvt_err_lower', 'mvt_err_upper', 
                         'mean_src_max', 'mean_back_avg', 'mean_snr_max']

        param_dir_name = _create_param_directory_name(sim_type, pulse_shape, self.variable_params)
        details_path = self.output_dir / sim_type / pulse_shape / param_dir_name
        details_path.mkdir(parents=True, exist_ok=True)
        
        analysis_settings = self.params['analysis_settings']
        bin_widths_to_analyze_ms = analysis_settings.get('bin_widths_to_analyze_ms', [])

        func_to_use, required_params = PULSE_MODEL_MAP[pulse_shape]
        adaptive_params = calculate_adaptive_simulation_params(pulse_shape, self.params)
        self.params.update(adaptive_params)
        func_par = tuple(self.params[key] for key in required_params)
        back_func_par = (self.params['background_level'],)

        iteration_details_list = []
        for i in range(NN):
            iteration_seed = original_seed + i
            try:
                # --- Simulate events ONCE per realization (in memory) ---
                source_events, background_events, sim_info = self.simulation_function(
                    func=func_to_use, func_par=func_par, back_func=constant, back_func_par=back_func_par, params={**self.params, 'random_seed': iteration_seed}
                )
                total_events = np.sort(np.concatenate([source_events, background_events]))

                # <<< Calculate per-realization metrics (SNR, etc.) once >>>
                source_counts_fine, _ = np.histogram(source_events, bins=int(sim_info['duration']/0.001))
                snr_dict = _calculate_multi_timescale_snr(
                    source_counts=source_counts_fine, sim_bin_width=0.001,
                    back_avg_cps=sim_info['background_level_cps'],
                    search_timescales=analysis_settings['snr_analysis']['search_timescales']
                )
                
                base_iter_detail = {
                    'iteration': i + 1, 'random_seed': iteration_seed,
                    'snr_max': round(max(snr_dict.values()), 2) if snr_dict else -100,
                    'src_max_cps': round(sim_info['src_max_cps'], 2),
                    'back_avg_cps': round(sim_info['background_level_cps'], 2)
                }

                # --- Loop through ANALYSIS bin widths (fast, in-memory) ---
                for bin_width_ms in bin_widths_to_analyze_ms:
                    bin_width_s = bin_width_ms / 1000.0
                    bins = np.arange(sim_info['t_start'], sim_info['t_stop'] + bin_width_s, bin_width_s)
                    counts, _ = np.histogram(total_events, bins=bins)
                    
                    mvt_res = haar_power_mod(counts, np.sqrt(np.abs(counts)), min_dt=bin_width_s, doplot=False)
                    mvt_val = float(mvt_res[2]) * 1000
                    
                    iter_detail = {
                        **base_iter_detail,
                        'analysis_bin_width_ms': bin_width_ms,
                        'mvt_ms': round(mvt_val, 4)
                    }
                    iteration_details_list.append(iter_detail)

            except Exception as e:
                # <<< Graceful error handling for a single failed run >>>
                logging.warning(f"Run {i+1}/{NN} for {param_dir_name} failed. Error: {e}")
                for bin_width_ms in bin_widths_to_analyze_ms:
                    iteration_details_list.append({
                        'iteration': i + 1, 'random_seed': iteration_seed,
                        'analysis_bin_width_ms': bin_width_ms,
                        'mvt_ms': -100, 'snr_max': -100, 'src_max_cps': -100, 'back_avg_cps': -100
                    })
        
        if not iteration_details_list: return []
        
        detailed_df = pd.DataFrame(iteration_details_list)
        detailed_df.to_csv(details_path / f"Detailed_{param_dir_name}.csv", index=False)
        
        final_summary_list = []
        for bin_width, group_df in detailed_df.groupby('analysis_bin_width_ms'):
            valid_runs = group_df[group_df['mvt_ms'] > 0]
            if len(valid_runs) < 2: continue
            
            p16, median_mvt, p84 = np.percentile(valid_runs['mvt_ms'], [16, 50, 84])
            
            # <<< Create the MVT distribution plot for this bin width, as you liked >>>
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(valid_runs['mvt_ms'], bins=30, density=True, label=f"MVT Distribution ({len(valid_runs)}/{NN} runs)")
            ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2, label=f"Median = {median_mvt:.3f} ms")
            ax.set_title(f"MVT for {param_dir_name}\nBin Width: {bin_width} ms", fontsize=12)
            ax.set_xlabel("Minimum Variability Timescale (ms)")
            ax.legend()
            fig.tight_layout()
            plt.savefig(details_path / f"Distribution_mvt_{param_dir_name}_{bin_width}ms.png", dpi=300)
            plt.close(fig)

            # <<< Build the standardized final output dictionary, as in your original >>>
            result_data = {
                'sim_type': sim_type, 'pulse_shape': pulse_shape,
                **self.variable_params,
                'analysis_bin_width_ms': bin_width,
                'total_sim': NN, 'successful_runs': len(valid_runs),
                'median_mvt_ms': round(median_mvt, 4),
                'mvt_err_lower': round(median_mvt - p16, 4),
                'mvt_err_upper': round(p84 - median_mvt, 4),
                'mean_src_max': round(valid_runs['src_max_cps'].mean(), 2),
                'mean_back_avg': round(valid_runs['back_avg_cps'].mean(), 2),
                'mean_snr_max': round(valid_runs['snr_max'].mean(), 2)
            }
            
            final_dict = {}
            for key in STANDARD_KEYS: final_dict[key] = result_data.get(key)
            for key in ALL_PULSE_PARAMS: final_dict[key] = self.params.get(key, DEFAULT_PARAM_VALUE)
            final_summary_list.append(final_dict)
            
        return final_summary_list
    
class AbstractPulseSimulationTask(BaseSimulationTask):
    def run(self) -> List[Dict]:
        # --- 1. Initial Setup ---
        sim_type = 'gbm' if self.is_gbm else 'function'
        pulse_shape = self.params.get('pulse_shape')
        NN = self.params.get('total_sim', 1)
        original_seed = self.params.get('random_seed')
        
        # Define standard keys and defaults, from your original script
        ALL_PULSE_PARAMS = ['sigma', 'center_time', 'width', 'peak_time_ratio', 'start_time', 'rise_time', 'decay_time']
        DEFAULT_PARAM_VALUE = -999
        STANDARD_KEYS = ['sim_type', 'pulse_shape', 'total_sim', 'successful_runs', 
                         'median_mvt_ms', 'mvt_err_lower', 'mvt_err_upper', 
                         'mean_src_max', 'mean_back_avg', 'mean_snr_max']

        param_dir_name = _create_param_directory_name(sim_type, pulse_shape, self.variable_params)
        details_path = self.output_dir / sim_type / pulse_shape / param_dir_name
        details_path.mkdir(parents=True, exist_ok=True)
        
        analysis_settings = self.params.get('analysis_settings', {})
        bin_widths_to_analyze_ms = analysis_settings.get('bin_widths_to_analyze_ms', [])

        func_to_use, required_params = PULSE_MODEL_MAP[pulse_shape]
        adaptive_params = calculate_adaptive_simulation_params(pulse_shape, self.params)
        self.params.update(adaptive_params)
        func_par = tuple(self.params[key] for key in required_params)
        back_func_par = (self.params['background_level'],)

        iteration_details_list = []
        for i in range(NN):
            iteration_seed = original_seed + i
            try:
                # --- Simulate events ONCE per realization (in memory) ---
                source_events, background_events, sim_info = self.simulation_function(
                    func=func_to_use, func_par=func_par, back_func=constant, back_func_par=back_func_par, params={**self.params, 'random_seed': iteration_seed}
                )
                total_events = np.sort(np.concatenate([source_events, background_events]))

                # --- Calculate per-realization metrics (SNR, etc.) once ---
                source_counts_fine, _ = np.histogram(source_events, bins=int(sim_info['duration']/0.001))
                snr_dict = _calculate_multi_timescale_snr(
                    source_counts=source_counts_fine, sim_bin_width=0.001,
                    back_avg_cps=sim_info['background_level_cps'],
                    search_timescales=analysis_settings['snr_analysis']['search_timescales']
                )
                
                base_iter_detail = {
                    'iteration': i + 1, 'random_seed': iteration_seed,
                    'snr_max': round(max(snr_dict.values()), 2) if snr_dict else -100,
                    'src_max_cps': round(sim_info['src_max_cps'], 2),
                    'back_avg_cps': round(sim_info['background_level_cps'], 2)
                }

                # --- Loop through ANALYSIS bin widths (fast, in-memory) ---
                for bin_width_ms in bin_widths_to_analyze_ms:
                    bin_width_s = bin_width_ms / 1000.0
                    bins = np.arange(sim_info['t_start'], sim_info['t_stop'] + bin_width_s, bin_width_s)
                    counts, _ = np.histogram(total_events, bins=bins)
                    
                    mvt_res = haar_power_mod(counts, np.sqrt(np.abs(counts)), min_dt=bin_width_s, doplot=False)
                    
                    # <<< CORRECTION 1: Store both mvt_ms and mvt_err_ms >>>
                    mvt_val = float(mvt_res[2]) * 1000
                    mvt_err = float(mvt_res[3]) * 1000
                    
                    iter_detail = {
                        **base_iter_detail,
                        'analysis_bin_width_ms': bin_width_ms,
                        'mvt_ms': round(mvt_val, 4),
                        'mvt_err_ms': round(mvt_err, 4)
                    }
                    iteration_details_list.append(iter_detail)

            except Exception as e:
                logging.warning(f"Run {i+1}/{NN} for {param_dir_name} failed. Error: {e}")
                for bin_width_ms in bin_widths_to_analyze_ms:
                    # <<< CORRECTION 3: Update error placeholder >>>
                    iteration_details_list.append({
                        'iteration': i + 1, 'random_seed': iteration_seed,
                        'analysis_bin_width_ms': bin_width_ms,
                        'mvt_ms': -100, 'mvt_err_ms': -100,
                        'snr_max': -100, 'src_max_cps': -100, 'back_avg_cps': -100
                    })
        
        if not iteration_details_list: return []
        
        detailed_df = pd.DataFrame(iteration_details_list)
        detailed_df.to_csv(details_path / f"Detailed_{param_dir_name}.csv", index=False)
        
        final_summary_list = []
        for bin_width, group_df in detailed_df.groupby('analysis_bin_width_ms'):
            
            # <<< CORRECTION 2: Use mvt_err_ms > 0 to determine valid runs >>>
            valid_runs = group_df[group_df['mvt_err_ms'] > 0]
            
            if len(valid_runs) < 2: continue
            
            p16, median_mvt, p84 = np.percentile(valid_runs['mvt_ms'], [16, 50, 84])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(valid_runs['mvt_ms'], bins=30, density=True, label=f"MVT Distribution ({len(valid_runs)}/{NN} runs)")
            ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2, label=f"Median = {median_mvt:.3f} ms")
            ax.set_title(f"MVT for {param_dir_name}\nBin Width: {bin_width} ms", fontsize=12)
            ax.set_xlabel("Minimum Variability Timescale (ms)")
            ax.legend()
            fig.tight_layout()
            plt.savefig(details_path / f"Distribution_mvt_{param_dir_name}_{bin_width}ms.png", dpi=300)
            plt.close(fig)

            result_data = {
                'sim_type': sim_type, 'pulse_shape': pulse_shape,
                **self.variable_params,
                'analysis_bin_width_ms': bin_width,
                'total_sim': NN, 'successful_runs': len(valid_runs),
                'median_mvt_ms': round(median_mvt, 4),
                'mvt_err_lower': round(median_mvt - p16, 4),
                'mvt_err_upper': round(p84 - median_mvt, 4),
                'mean_src_max': round(valid_runs['src_max_cps'].mean(), 2),
                'mean_back_avg': round(valid_runs['back_avg_cps'].mean(), 2),
                'mean_snr_max': round(valid_runs['snr_max'].mean(), 2)
            }
            
            final_dict = {}
            for key in STANDARD_KEYS: final_dict[key] = result_data.get(key)
            # Add all possible pulse parameter keys, filling with a default if not used in this run
            for key in self.variable_params.keys(): final_dict[key] = result_data.get(key)
            for key in ALL_PULSE_PARAMS:
                if key not in final_dict: final_dict[key] = self.params.get(key, DEFAULT_PARAM_VALUE)
            
            final_summary_list.append(final_dict)
            
        return final_summary_list

class GbmSimulationTask(AbstractPulseSimulationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_function = generate_gbm_events # A new GBM-specific generator
        self.is_gbm = True
        
class FunSimulationTask(AbstractPulseSimulationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_function = generate_function_events
        self.is_gbm = False

# ========= YAML PARSER =========
def generate_sim_tasks_from_config(config: Dict[str, Any]) -> 'Generator':
    pulse_definitions = config.get('pulse_definitions', {})
    analysis_settings = config['analysis_settings']
    results_path = Path(config['project_settings']['results_path'])

    for campaign in config.get('simulation_campaigns', []):
        if not campaign.get('enabled', False): continue
        sim_type = campaign.get('type')
        TaskClass = globals().get(SIMULATION_REGISTRY.get(sim_type))
        if not TaskClass: continue
        
        logging.info(f"Generating tasks for campaign '{campaign['name']}'...")
        for pulse_config in campaign.get('pulses_to_run', []):
            pulse_shape = None
            pulse_override_constants = {}
            
            if isinstance(pulse_config, str):
                pulse_shape = pulse_config
            elif isinstance(pulse_config, dict):
                pulse_shape = list(pulse_config.keys())[0]
                if not pulse_config[pulse_config].get('enabled', True):
                    continue
                pulse_override_constants = pulse_config[pulse_shape].get('constants', {})

            if not pulse_shape:
                continue

            base_pulse_config = pulse_definitions.get(pulse_shape)
            if not base_pulse_config:
                logging.warning(f"Pulse shape '{pulse_shape}' not found in pulse_definitions. Skipping.")
                continue
            
            logging.info(f"  > Processing pulse: '{pulse_shape}'")

            # ▼▼▼ THE ONLY CHANGE IS HERE ▼▼▼
            variable_params_config = campaign.get('parameters', {}).copy()
            # ▲▲▲ THE ONLY CHANGE IS HERE ▲▲▲
            
            variable_params_config.update(base_pulse_config.get('parameters', {}))
            
            if 'trigger_set_file' in variable_params_config:
                file_path = variable_params_config.pop('trigger_set_file')
                df_triggers = pd.read_csv(file_path)
                variable_params_config['trigger_set'] = df_triggers.to_dict('records')

            final_constants = {}
            final_constants.update(campaign.get('constants', {}))
            final_constants.update(base_pulse_config.get('constants', {}))
            final_constants.update(pulse_override_constants)
            final_constants['pulse_shape'] = pulse_shape

            param_names = list(variable_params_config.keys())
            param_values = [_parse_param(v) for v in variable_params_config.values()]
            
            if not param_values:
                param_values.append(())
            for combo in itertools.product(*param_values):
                current_variable_params = dict(zip(param_names, combo))
                yield TaskClass(results_path, current_variable_params, final_constants, analysis_settings)

# ========= MAIN EXECUTION BLOCK =========
def main():
    now = datetime.now().strftime("%y_%m_%d-%H_%M")
    with open(SIM_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
        
    results_path = Path(config['project_settings']['results_path'])
    shutil.rmtree(results_path, ignore_errors=True)
    results_path.mkdir(parents=True, exist_ok=True)
    
    log_file = results_path / f'run_{now}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    tasks = list(generate_sim_tasks_from_config(config))
    if not tasks:
        logging.warning("No enabled simulation tasks were generated.")
        return

    logging.info(f"Generated {len(tasks)} simulation tasks. Starting processing...")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(task.run) for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Simulation Sets"):
            try:
                result_list = future.result()
                if result_list:
                    all_results.extend(result_list)
            except Exception as e:
                logging.error(f"A task failed: {e}", exc_info=True)

    if all_results:
        final_df = pd.DataFrame(all_results)
        final_csv_path = results_path / "final_summary_results.csv"
        final_df.to_csv(final_csv_path, index=False)
        logging.info(f"✅ Processing complete! Summary saved to:\n{final_csv_path}")

if __name__ == '__main__':
    main()