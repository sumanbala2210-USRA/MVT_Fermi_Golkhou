import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sigma_vs_mvt_by_amplitude(
    csv_file_path,
    output_filename="plot_by_amplitude.png",
    show_lower_limits=True,
    peak_amplitude_to_plot='all'
):
    """
    Plots MVT vs. Sigma, with different series for each PEAK AMPLITUDE.
    """
    # This is your original function, just renamed for clarity
    # The code inside remains exactly the same.
    try:
        df = pd.read_csv(csv_file_path).fillna(0)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    if peak_amplitude_to_plot != 'all':
        try:
            target_amp = float(peak_amplitude_to_plot)
            if target_amp not in df['peak_amplitude'].unique():
                print(f"Warning: Peak amplitude {target_amp} not found.")
                return
            df = df[df['peak_amplitude'] == target_amp]
        except (ValueError, TypeError):
            print(f"Error: 'peak_amplitude_to_plot' must be a number or 'all'.")
            return

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", context="talk")
    grouping_values = sorted(df['peak_amplitude'].unique())
    colors = sns.color_palette("viridis", n_colors=len(grouping_values))

    for i, amp in enumerate(grouping_values):
        subset = df[df['peak_amplitude'] == amp].sort_values('sigma')
        with_error = subset[subset['mvt_error_ms'] > 0]
        plt.errorbar(with_error['sigma'], with_error['mvt_ms'], yerr=with_error['mvt_error_ms'],
                     label=f'Peak Amp = {int(amp)}', fmt='o', capsize=4, color=colors[i])
        if show_lower_limits:
            zero_error = subset[subset['mvt_error_ms'] == 0]
            if not zero_error.empty:
                arrow_length = zero_error['mvt_ms'] * 0.15
                plt.errorbar(zero_error['sigma'], zero_error['mvt_ms'], yerr=arrow_length,
                             lolims=True, label='_nolegend_', fmt='o', color=colors[i])

    plt.title("MVT vs. Signal Width (Grouped by Amplitude)", fontsize=18)
    plt.xlabel("Sigma (s)", fontsize=14)
    plt.ylabel("MVT (ms)", fontsize=14)
    plt.legend(title="Signal Strength")
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close() # Close figure to free memory
    print(f"Plot successfully saved to '{output_filename}'")

# ==============================================================================
# ======================== NEW PLOTTING FUNCTION ===============================
# ==============================================================================

def plot_sigma_vs_mvt_by_background(
    csv_file_path,
    output_filename="plot_by_background.png",
    show_lower_limits=True,
    background_level_to_plot='all'
):
    """
    Plots MVT vs. Sigma, with different series for each BACKGROUND LEVEL.
    """
    try:
        df = pd.read_csv(csv_file_path).fillna(0)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    # Filter for a specific background level if requested
    if background_level_to_plot != 'all':
        try:
            target_bg = float(background_level_to_plot)
            if target_bg not in df['background_level'].unique():
                print(f"Warning: Background level {target_bg} not found.")
                return
            df = df[df['background_level'] == target_bg]
        except (ValueError, TypeError):
            print(f"Error: 'background_level_to_plot' must be a number or 'all'.")
            return

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", context="talk")

    # The GROUPING variable is now 'background_level'
    grouping_values = sorted(df['background_level'].unique())
    colors = sns.color_palette("plasma", n_colors=len(grouping_values))

    for i, bg_level in enumerate(grouping_values):
        subset = df[df['background_level'] == bg_level].sort_values('sigma')
        
        # Plot points with error
        with_error = subset[subset['mvt_error_ms'] > 0]
        plt.errorbar(with_error['sigma'], with_error['mvt_ms'], yerr=with_error['mvt_error_ms'],
                     label=f'BG Level = {int(bg_level)}', fmt='.', capsize=4, color=colors[i])
        
        # Plot points with lower limits
        if show_lower_limits:
            zero_error = subset[subset['mvt_error_ms'] == 0]
            if not zero_error.empty:
                arrow_length = zero_error['mvt_ms'] * 0.15
                plt.errorbar(zero_error['sigma'], zero_error['mvt_ms'], yerr=arrow_length,
                             lolims=True, label='_nolegend_', fmt='.', color=colors[i])

    # Update titles and labels
    plt.title("MVT vs. Signal Width (Grouped by Background)", fontsize=18)
    plt.xlabel("Sigma (s)", fontsize=14)
    plt.ylabel("MVT (ms)", fontsize=14)
    plt.legend(title="Background Level")
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.ylim(0,2000)
    plt.savefig(output_filename, dpi=300)
    plt.close() # Close figure to free memory
    print(f"Plot successfully saved to '{output_filename}'")


if __name__ == '__main__':
    # --- USER: Change this to the name of your CSV file ---
    csv_file_to_plot = '/Users/sbala/work/gdt_dev2/mvt_fermi_golkhu/MVTfermi/SIM_vs_mvt_2025-07-21_16-03-39/SIM_vs_mvt_2025-07-21_16-03-39_results.csv'

    # --- Call the functions to create BOTH plots ---

    #print("--- Generating plot grouped by Amplitude ---")
    #plot_sigma_vs_mvt_by_amplitude(csv_file_to_plot

    print("\n--- Generating plot grouped by Background Level ---")
    plot_sigma_vs_mvt_by_background(csv_file_to_plot, show_lower_limits=False, background_level_to_plot=25)

    # You can also filter the new plot just like the old one:
    # print("\n--- Generating filtered plot for a single background level ---")
    # plot_sigma_vs_mvt_by_background(
    #     csv_file_to_plot,
    #     output_filename="plot_bg_10_only.png",
    #     background_level_to_plot=10
    # )