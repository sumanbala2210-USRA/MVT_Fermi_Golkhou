"""
Suman Bala
16 Aug 2025: This script's sole purpose is to read a YAML configuration
            and generate a cache of raw Time-Tagged Event (TTE) files.
            All analysis is deferred to a separate script.
"""

# ========= Import necessary libraries =========
import os

import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import logging
import smtplib
import itertools
import pandas as pd
from typing import Dict, Any
from email.message import EmailMessage
import subprocess
import json
import tempfile
import os
from typing import Dict, Any, Tuple, Callable, List


from sim_functions import gaussian2, triangular, constant, norris, fred, lognormal
# ========= Import necessary libraries =========


GMAIL_FILE = 'config_mail.yaml'
HAAR_ENV_PATH = "/Users/sbala/anaconda3/bin/python"
WRAPPER_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'run_haar_power_mod.py')


def run_mvt_in_subprocess(
    counts: np.ndarray, 
    bin_width_s: float,
) -> List:
    """
    Runs the haar_power_mod analysis in a separate Python environment.

    This function handles the creation of temporary files, calling the external
    script via a subprocess, and loading the results.

    Args:
        counts (np.ndarray): The binned light curve data.
        bin_width_s (float): The bin width in seconds.
        python_env_path (str): The full path to the Python executable in the other environment.
        wrapper_script_path (str): The full path to the 'run_haar_power.py' wrapper script.

    Returns:
        List: The results from the haar_power_mod function (e.g., [status, n_bins, mvt, mvt_err]).
              Returns an empty list on failure.
    """
    # Use temporary files that are automatically deleted when the block is exited
    with tempfile.NamedTemporaryFile(suffix='.npy') as tmp_input, \
         tempfile.NamedTemporaryFile(suffix='.json', mode='w+') as tmp_output:
        
        try:
            # 1. Save the input data to the temporary input file
            np.save(tmp_input.name, counts)
            
            # 2. Construct the command to run the external script
            command = [
                HAAR_ENV_PATH,
                WRAPPER_SCRIPT_PATH,
                "--input", tmp_input.name,
                "--output", tmp_output.name,
                "--min_dt", str(bin_width_s)
            ]
            
            # 3. Run the command
            # The 'check=True' will raise an error if the external script fails
            subprocess.run(command, check=True, capture_output=True, text=True)

            # 4. Load the results from the temporary output file
            tmp_output.seek(0) # Rewind file to the beginning before reading
            mvt_res = json.load(tmp_output)
            
            return mvt_res

        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess for MVT calculation failed.")
            logging.error(f"Stderr: {e.stderr}")
            return [] # Return empty list on failure
        except Exception as e:
            logging.error(f"An unexpected error occurred in run_mvt_in_subprocess: {e}")
            return []

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




# ========= USER SETTINGS =========

def write_yaml(par_dictionary, yaml_file, comments=[]):
    """ 
    Write YAML file in a more compact format than yaml.safe_dump().

    Args:
        par_dictionary (dict):      Dictionary of values to write.
        yaml_file (str):            Path of the output file to write.
        comments (list):            Optional list of comments to include in the YAML file.
    """
    # Convert the dictionary to a YAML string
    yaml_content = format_par_as_yaml(par_dictionary, '')
    # Open the file for writing
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)



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


# ========= GENERIC SIMULATION FRAMEWORK =========
# --- 1. The Simulation Registry (Unchanged) ---
SIMULATION_REGISTRY = {
    'gbm': 'GbmSimulationTask',
    'function': 'FunSimulationTask',
}

# This map defines which keys from the `variable_params` dictionary are
# relevant for creating the directory name for each pulse shape.
# This map now has separate rules for 'gbm' and 'function' types.
RELEVANT_NAMING_KEYS = {
    'gbm': {
        # For GBM runs, 'angle' from the trigger set IS a meaningful parameter.
        'gaussian':   ['peak_amplitude', 'sigma', 'angle'],
        'triangular': ['peak_amplitude', 'width', 'peak_time_ratio', 'angle'],
        'norris':     ['peak_amplitude', 'rise_time', 'decay_time', 'angle'],
        'fred':       ['peak_amplitude', 'rise_time', 'decay_time', 'angle'],
        'lognormal':  ['peak_amplitude', 'sigma', 'center_time', 'angle'],
    },
    'function': {
        # For function runs, 'angle' etc. are dummy values and should be IGNORED.
        'gaussian':   ['peak_amplitude', 'sigma'],
        'triangular': ['peak_amplitude', 'width', 'peak_time_ratio'],
        'norris':     ['peak_amplitude', 'rise_time', 'decay_time'],
        'fred':       ['peak_amplitude', 'rise_time', 'decay_time'],
        'lognormal':  ['peak_amplitude', 'sigma', 'center_time'],
    }
}

# names of the parameters it requires. This eliminates the big if/elif block.
PULSE_MODEL_MAP = {
    'gaussian':   (gaussian2, ['peak_amplitude', 'center_time', 'sigma']),
    'triangular': (triangular, ['peak_amplitude', 't_start_tri', 'center_time', 't_stop_tri']),
    'norris':     (norris, ['peak_amplitude', 'start_time', 'rise_time', 'decay_time']),
    'fred':       (fred, ['peak_amplitude', 'start_time', 'rise_time', 'decay_time']),
    'lognormal':  (lognormal, ['peak_amplitude', 'center_time', 'sigma']),
}

# ========= HELPER FUNCTIONS =========
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

# The refactored naming function
def _create_param_directory_name(sim_type: str, pulse_shape: str, variable_params: dict) -> str:
    """
    Creates a descriptive directory name using only the parameters relevant
    to the given sim_type and pulse_shape.
    """
    key_abbreviations = {
        'peak_amplitude': 'amp', 'background_level': 'bkg', 'sigma': 'sig',
        'width': 'w', 'center_time': 't0', 'rise_time': 'tr', 'decay_time': 'td',
        'peak_time_ratio': 'pr', 'angle': 'ang'
    }
    
    # Get the list of relevant keys for this specific sim_type and pulse_shape
    relevant_keys = RELEVANT_NAMING_KEYS.get(sim_type, {}).get(pulse_shape, [])
    
    name_parts = []
    
    # --- Special Handling for GBM Trigger Sets ---
    if sim_type == 'gbm' and 'trigger_set' in variable_params:
        trigger_info = variable_params['trigger_set']
        # We only want to add the parts of the trigger set that are meant to be varied
        # In this case, 'angle' is a good candidate for the name.
        if 'angle' in relevant_keys and 'angle' in trigger_info:
            abbr = key_abbreviations['angle']
            if type(trigger_info['angle']) is float:
                val_str = e_n(trigger_info['angle'])
            else:
                val_str = str(trigger_info['angle'])
            name_parts.append(f"{abbr}_{val_str}")

    # --- General Parameter Handling ---
    for key, value in sorted(variable_params.items()):
        # Skip parameters that aren't relevant for this sim type
        if key not in relevant_keys:
            continue
            
        # We already handled this dictionary above
        if key == 'trigger_set':
            continue
            
        abbr = key_abbreviations.get(key, key[:3])
        val_str = e_n(value)
        name_parts.append(f"{abbr}_{val_str}")
        
    return "-".join(name_parts)




def _parse_param(param_config: Any) -> list:
    """
    Helper function to parse parameter definitions from the YAML config.
    (This function is the same as your original).
    """
    if isinstance(param_config, dict) and 'start' in param_config:
        # Handles np.arange style definitions: {start: 0, stop: 1, step: 0.1}
        return np.arange(**param_config)
    if isinstance(param_config, list):
        # Handles simple lists: [1, 2, 3]
        return param_config
    if isinstance(param_config, (int, float, str)):
        # Handles single values, wrapping them in a list for itertools
        return [param_config]
    # You might add handling for other types if needed
    raise TypeError(f"Unsupported parameter format in YAML: {param_config}")







