import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sigma_vs_mvt_scatter(
    csv_file_path,
    output_filename="sigma_vs_mvt_scatter_plot.png",
    show_lower_limits=True,
    peak_amplitude_to_plot='all'
):
    """
    Reads simulation results and creates a scatter plot of MVT vs. Sigma.

    Args:
        csv_file_path (str): The path to the input CSV file.
        output_filename (str): The name of the file to save the plot to.
        show_lower_limits (bool): If True, plots points with zero error as
                                  lower limits (upward arrows). Defaults to True.
        peak_amplitude_to_plot (str or int): A specific peak amplitude to plot.
                                             Set to 'all' to plot every amplitude.
                                             Defaults to 'all'.
    """
    try:
        df = pd.read_csv(csv_file_path).fillna(0)
        print("Successfully loaded CSV file.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    # --- Filter for a specific Peak Amplitude if requested ---
    if peak_amplitude_to_plot != 'all':
        try:
            target_amp = float(peak_amplitude_to_plot)
            if target_amp not in df['peak_amplitude'].unique():
                print(f"Warning: Peak amplitude {target_amp} not found in the data.")
                print(f"Available amplitudes are: {sorted(df['peak_amplitude'].unique())}")
                return
            print(f"Filtering data to show only Peak Amplitude = {target_amp}")
            df = df[df['peak_amplitude'] == target_amp]
        except (ValueError, TypeError):
            print(f"Error: 'peak_amplitude_to_plot' must be a number or 'all'. You provided '{peak_amplitude_to_plot}'.")
            return

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 8))

    peak_amplitudes = sorted(df['peak_amplitude'].unique())
    if not peak_amplitudes:
        print("No data left to plot after filtering. Exiting.")
        return
        
    colors = sns.color_palette("viridis", n_colors=len(peak_amplitudes))
    print(f"Plotting for peak amplitudes: {peak_amplitudes}")

    for i, amp in enumerate(peak_amplitudes):
        subset = df[df['peak_amplitude'] == amp].sort_values('sigma')
        if subset.empty:
            continue

        # Plot points with non-zero error
        with_error = subset[subset['mvt_error_ms'] > 0]
        plt.errorbar(with_error['sigma'], with_error['mvt_ms'], yerr=with_error['mvt_error_ms'],
                     label=f'Peak Amp = {int(amp)}', fmt='o', capsize=4, color=colors[i])

        # Plot points with zero error as lower limits (if enabled)
        if show_lower_limits:
            zero_error = subset[subset['mvt_error_ms'] == 0]
            if not zero_error.empty:
                arrow_length = zero_error['mvt_ms'] * 0.15
                plt.errorbar(zero_error['sigma'], zero_error['mvt_ms'], yerr=arrow_length,
                             lolims=True, label='_nolegend_', fmt='o', color=colors[i])

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

    # Example 1: Plot ALL amplitudes (default behavior)
    plot_sigma_vs_mvt_scatter(csv_file_to_plot)

    # Example 2: Plot ONLY a single peak amplitude (e.g., 50)
    plot_sigma_vs_mvt_scatter(
         csv_file_to_plot,
         output_filename="sigma_vs_mvt_amp50_plot.png",
         peak_amplitude_to_plot=50, show_lower_limits=False
     )

    # Example 3: Plot a single amplitude and HIDE lower limits
    # plot_sigma_vs_mvt_scatter(
    #     csv_file_to_plot,
    #     output_filename="sigma_vs_mvt_amp50_no_limits.png",
    #     peak_amplitude_to_plot=50,
    #     show_lower_limits=False
    # )