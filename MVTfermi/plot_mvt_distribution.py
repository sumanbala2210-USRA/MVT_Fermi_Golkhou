import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def read_mvt_results(filename="mvt_distribution_results.txt"):
    mvt_values = []
    mvt_errors = []

    df = pd.read_csv(filename, sep="\t")

    # Extract columns as arrays (lists)
    iteration = df["Iteration"].to_list()
    mvt = df["MVT(ms)"].to_list()
    error = df["Error(ms)"].to_list()


    """
    
    with open(filename, "r") as f:
        for line in f:
            if "MVT(ms):" in line and "Error(ms):" in line:
                parts = line.strip().split()
                mvt = float(parts[2])
                err = float(parts[4])
                mvt_values.append(mvt)
                mvt_errors.append(err)
    
    return np.array(mvt_values), np.array(mvt_errors)
    """
    return np.array(mvt), np.array(error)

def plot_mvt_distribution(mvt_values, mvt_errors, bins=50):
    valid_mask = mvt_errors > 0
    valid_mvt = mvt_values[valid_mask]

    mean_mvt = np.mean(valid_mvt)
    std_mvt = np.std(valid_mvt)
    num_failed = np.sum(~valid_mask)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.hist(valid_mvt, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(mean_mvt, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {mean_mvt:.3f} ms")
    plt.title("High-Resolution MVT Distribution (Valid Only)")
    plt.xlabel("MVT (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mvt_distribution_fine_bins.png")
    plt.show()

    print(f"Number of valid MVT values: {len(valid_mvt)}")
    print(f"Number of failed entries (error = 0): {num_failed}")
    print(f"Mean MVT(ms): {mean_mvt:.3f}")
    print(f"Std Dev (empirical error): {std_mvt:.3f}")

# Example usage:
mvt_vals, mvt_errs = read_mvt_results("mvt_distribution_results.txt")
plot_mvt_distribution(mvt_vals, mvt_errs, bins=50)
