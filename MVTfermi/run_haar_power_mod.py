# run_haar_power.py
# This script should be placed in the other Python environment.

import numpy as np
import argparse
import json
from haar_power_mod import haar_power_mod # This environment has the tool installed

def main():
    parser = argparse.ArgumentParser(description="A wrapper to run haar_power_mod.")
    parser.add_argument("--input", required=True, help="Path to the input .npy file for counts.")
    parser.add_argument("--output", required=True, help="Path to the output .json file for results.")
    parser.add_argument("--min_dt", required=True, type=float, help="Minimum timescale (bin width).")
    args = parser.parse_args()

    # 1. Load the input data
    counts = np.load(args.input)
    errors = np.sqrt(np.abs(counts))

    # 2. Run the analysis
    results = haar_power_mod(counts, errors, min_dt=args.min_dt, doplot=False, afactor=-1.0, verbose=False)

    # 3. Save the results as a simple JSON file
    with open(args.output, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()