import sys
import os
import yaml
import argparse

def str2bool(value):
    true_set = {"y", "yes", "true", "1", "t"}
    false_set = {"n", "no", "false", "0", "f"}

    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Invalid type for boolean value: {value}")

    val = value.strip().lower()
    if val in true_set:
        return True
    elif val in false_set:
        return False
    else:
        raise ValueError(f"Invalid boolean string: {value}")




def merge_configs(func_args=None, cli_args=None, yaml_config=None):
    func_args = func_args or {}
    cli_args = cli_args or {}
    yaml_config = yaml_config or {}
    config = {}

    # Priority: function args > CLI args > YAML config
    for key in yaml_config:
        val_func = func_args.get(key, None)
        val_cli = cli_args.get(key, None)

        if val_func is not None:
            config[key] = val_func
        elif val_cli is not None:
            config[key] = val_cli
        else:
            config[key] = yaml_config[key]

    # Include keys only in func_args or cli_args (not in YAML)
    for key, val in func_args.items():
        if key not in config and val is not None:
            config[key] = val
    for key, val in cli_args.items():
        if key not in config and val is not None:
            config[key] = val

    # Postprocess booleans (example)
    if 'limit' in config:
        config['limit'] = str2bool(str(config['limit']))
    if 'all_delta' in config:
        config['all_delta'] = str2bool(str(config['all_delta']))

    return config




def normalize_det_list(det_list):
    if isinstance(det_list, str):
        return [d.strip() for d in det_list.split(",") if d.strip()]
    elif isinstance(det_list, list):
        return det_list
    else:
        return []
    


    
def normalize_background_intervals(raw):
    """
    Normalize background_intervals input to a list of [start, end] pairs.
    Supports:
      - list of floats/ints (flat list)
      - list of strings (from nargs='+')
      - single string with comma and/or space separated values
    Raises ValueError if invalid format or odd number of values.
    """

    if raw is None:
        return None

    # Case: already list of [start, end] pairs
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
        return raw

    # Case: flat list of numbers
    if isinstance(raw, list) and all(isinstance(x, (int, float)) for x in raw):
        if len(raw) % 2 != 0:
            raise ValueError("background_intervals must contain an even number of values")
        return [[raw[i], raw[i + 1]] for i in range(0, len(raw), 2)]

    # Case: list of strings
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        joined = " ".join(raw)
    elif isinstance(raw, str):
        joined = raw
    else:
        raise ValueError("Invalid format for background_intervals")

    # Replace commas with spaces and split by whitespace
    parts = joined.replace(",", " ").split()
    
    try:
        vals = [float(x) for x in parts]
    except Exception:
        raise ValueError("All background interval values must be numeric")

    if len(vals) % 2 != 0:
        raise ValueError("background_intervals must contain an even number of values")

    return [[vals[i], vals[i + 1]] for i in range(0, len(vals), 2)]




def base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--delta", type=str)
    parser.add_argument("--limit", type=str)
    parser.add_argument("--bw", type=float)
    parser.add_argument("--delt", type=float)
    parser.add_argument("--T0", type=float)
    parser.add_argument("--T90", type=float)
    parser.add_argument("--start_padding", type=float)
    parser.add_argument("--end_padding", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--f1", type=float)
    parser.add_argument("--f2", type=float)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--output_path", type=str)
    return parser

def parse_args_fermi(args=None):
    base = base_parser()
    parser = argparse.ArgumentParser(description="MVT Fermi CLI", parents=[base])
    parser.add_argument("--trigger_number", type=str)
    parser.add_argument(
        "--background_intervals",
        type=str,
        nargs='+',
        help="Background intervals, e.g. --background_intervals -11.67 -1.04 57.5 69.58 or as a quoted comma-separated string"
    )
    parser.add_argument("--det_list", type=str)
    parser.add_argument("--en_lo", type=int)
    parser.add_argument("--en_hi", type=int)
    parser.add_argument("--data_path", type=str)
    parsed = vars(parser.parse_args(args=args or []))

    # Normalize background_intervals
    if parsed.get("background_intervals") is not None:
        parsed["background_intervals"] = normalize_background_intervals(parsed["background_intervals"])

    # Also handle comma-separated det_list
    if parsed.get("det_list") and isinstance(parsed["det_list"], str):
        parsed["det_list"] = [d.strip() for d in parsed["det_list"].split(",")]
    
    if args is None:
        if any("ipykernel_launcher" in arg for arg in sys.argv):
            return parser.parse_args([])
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)

    return parsed

'''
def parse_args_fermi(args=None):
    import sys
    base = base_parser()
    parser = argparse.ArgumentParser(description="MVT Fermi CLI", parents=[base])
    parser.add_argument("--trigger_number", type=str)
    parser.add_argument(
        "--background_intervals",
        type=str,
        nargs='+',
        help="Background intervals, e.g. --background_intervals -11.67 -1.04 57.5 69.58 or as a quoted comma-separated string"
    )
    parser.add_argument("--det_list", type=str)
    parser.add_argument("--en_lo", type=int)
    parser.add_argument("--en_hi", type=int)
    parser.add_argument("--data_path", type=str)

    # Determine how to parse: 
    # - If args provided: parse those.
    # - Else if running in Jupyter: ignore IPython args.
    # - Else normal parsing.
    if args is not None:
        parsed_args = parser.parse_args(args)
    elif any("ipykernel_launcher" in arg for arg in sys.argv):
        parsed_args = parser.parse_args([])
    else:
        parsed_args = parser.parse_args()

    parsed = vars(parsed_args)

    # Normalize background_intervals
    if parsed.get("background_intervals") is not None:
        parsed["background_intervals"] = normalize_background_intervals(parsed["background_intervals"])

    # Normalize det_list (comma-separated string to list)
    if parsed.get("det_list") and isinstance(parsed["det_list"], str):
        parsed["det_list"] = [d.strip() for d in parsed["det_list"].split(",")]

    return parsed


def parse_args_general(args=None):
    base = base_parser()
    parser = argparse.ArgumentParser(description="MVT General CLI", parents=[base])
    parser.add_argument("--file_path", type=str)

    if args is None:
        # When running in Jupyter or IPython, avoid using sys.argv directly
        import sys
        if any("ipykernel_launcher" in arg for arg in sys.argv):
            return parser.parse_args([])
        else:
            return parser.parse_args()
    else:
        return parser.parse_args(args)
'''

def parse_args_general(args=None):
    base = base_parser()
    parser = argparse.ArgumentParser(description="MVT General CLI", parents=[base])
    parser.add_argument("--file_path", type=str)

    # Determine how to parse
    if args is not None:
        parsed_args = parser.parse_args(args)
    elif any("ipykernel_launcher" in arg for arg in sys.argv):
        parsed_args = parser.parse_args([])  # skip Jupyter args
    else:
        parsed_args = parser.parse_args()

    return parsed_args




def load_and_merge_config(func_args=None, cli_args=None, default_config_file=None, parse_fn=None):
    func_args = func_args or {}

    # Always try parsing CLI if not explicitly passed
    if cli_args is None and len(sys.argv) > 1:
        #print("Parsing CLI args via parse_fn()")
        cli_args = parse_fn()            # <-- change this line to the next one
        cli_args = vars(cli_args)        # <-- convert Namespace to dict

    else:
        cli_args = cli_args or {}

    config_file = func_args.get('config') or cli_args.get('config') or default_config_file

    #print(f"Trying to load config file: {config_file}")
    if config_file:
        config_file = config_file.strip().strip('"').strip("'")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
        else:
            print("Config file path resolved but does not exist on disk.")
            yaml_config = {}
    else:
        print("No config file provided.")
        yaml_config = {}

    return merge_configs(func_args, cli_args, yaml_config)






