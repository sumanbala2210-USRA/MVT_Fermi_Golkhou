import pandas as pd
import os
from datetime import datetime

now = datetime.now().strftime("%y_%m_%d-%H_%M")

loc_dist = [
    "run_25_08_25-07_23",
    "run_25_08_25-08_34",
    "run_25_08_25-08_58",
    "run_25_08_25-10_32",
    #"run_1em3_25_08_25-10_49",
    "run_.1_25_08_25-12_14",
]

path = os.path.join(os.getcwd(), "01_ANALYSIS_RESULTS")

files = [os.path.join(path, f"{loc}/final_summary_results.csv") for loc in loc_dist]

output_file = os.path.join(path, f"gauss_combined{now}.csv")

# Collect DataFrames
dfs = []
for f in files:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))
    else:
        print(f"Warning: File not found -> {f}")

# Combine them
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to:\n{output_file}")
else:
    print("No CSVs found to combine.")
