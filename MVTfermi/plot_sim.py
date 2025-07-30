import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sigma_vs_mvt_scatter(csv_file_path, output_filename="sigma_vs_mvt_scatter_plot.png", show_lower_limits=True):
    """
    Reads simulation results and creates a scatter plot of MVT vs. Sigma.

    Points with errors are shown with error bars. Points with zero error can
    be optionally shown as lower limits (upward arrows).

    Args:
        csv_file_path (str): The path to the input CSV file.
        output_filename (str): The name of the file to save the plot to.
        show_lower_limits (bool): If True, plots points with zero error as
                                  lower limits (upward arrows).
    """
    try:
        df = pd.read_csv(csv_file_path).fillna(0) # Fill any missing errors with 0
        print("Successfully loaded CSV file. Columns found:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 8))

    peak_amplitudes = sorted(df['peak_amplitude'].unique())
    colors = sns.color_palette("viridis", n_colors=len(peak_amplitudes))

    print(f"Found peak amplitudes: {peak_amplitudes}")

    for i, amp in enumerate(peak_amplitudes):
        subset = df[df['peak_amplitude'] == amp].sort_values('sigma')

        # --- 1. Plot points that HAVE a non-zero error ---
        with_error = subset[subset['mvt_error_ms'] > 0]
        if not with_error.empty:
            plt.errorbar(
                with_error['sigma'],
                with_error['mvt_ms'],
                yerr=with_error['mvt_error_ms'],
                label=f'Peak Amp = {int(amp)}',
                fmt='o',
                capsize=4,
                color=colors[i]
            )

        # --- 2. Plot points with ZERO error as lower limits (if enabled) ---
        if show_lower_limits:
            zero_error = subset[subset['mvt_error_ms'] == 0]
            if not zero_error.empty:
                # For visualization, we create a small upward arrow.
                arrow_length = zero_error['mvt_ms'] * 0.15
                plt.errorbar(
                    zero_error['sigma'],
                    zero_error['mvt_ms'],
                    yerr=arrow_length,
                    lolims=True,  # This creates the upward-pointing arrows
                    label='_nolegend_',
                    fmt='o',
                    color=colors[i]
                )

    # --- Add plot titles and labels ---
    plt.title("MVT vs. Signal Width (Scatter Plot)", fontsize=18, pad=20)
    plt.xlabel("Sigma (s)", fontsize=14)
    plt.ylabel("MVT (ms)", fontsize=14)
    #plt.yscale("log")
    plt.legend(title="Signal Strength")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # --- Save the plot to a file ---
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot successfully saved to '{output_filename}'")

if __name__ == '__main__':
    # --- USER: Change this to the name of your CSV file ---
    csv_file_to_plot = '/Users/sbala/work/gdt_dev2/mvt_fermi_golkhu/MVTfermi/SIM_vs_mvt_2025-07-21_14-01-38/SIM_vs_mvt_2025-07-21_14-01-38.csv'

    # --- Call the function ---
    # To hide the lower limits, change the flag to False:
    # plot_sigma_vs_mvt_scatter(csv_file_to_plot, show_lower_limits=False)
    
    plot_sigma_vs_mvt_scatter(csv_file_to_plot, show_lower_limits=False)


