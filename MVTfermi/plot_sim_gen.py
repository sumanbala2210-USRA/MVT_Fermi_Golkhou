import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_mvt_generic(
    csv_file_path,
    x_axis_col,
    group_by_col,
    y_axis_col='mvt_ms',
    y_err_col='mvt_error_ms',
    filters=None,
    output_filename=None,
    show_lower_limits=True
):
    """
    A generalized function to plot MVT against a chosen variable,
    grouped by another variable, with optional filters.

    Args:
        csv_file_path (str): Path to the input CSV file.
        x_axis_col (str): The column name for the x-axis (e.g., 'sigma').
        group_by_col (str): The column name to group data by (e.g., 'background_level').
        y_axis_col (str): The column for the y-axis. Defaults to 'mvt_ms'.
        y_err_col (str): The column for the y-axis error. Defaults to 'mvt_error_ms'.
        filters (dict, optional): A dictionary to filter data.
                                  Example: {'peak_amplitude': 100, 'sigma': 0.5}
        output_filename (str, optional): The name for the output plot file.
                                         If None, a descriptive name is generated.
        show_lower_limits (bool): If True, plots points with zero error as lower limits.
    """
    try:
        df = pd.read_csv(csv_file_path).fillna(0)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    # --- Apply Filters ---
    if filters:
        print(f"Applying filters: {filters}")
        for key, value in filters.items():
            if key not in df.columns:
                print(f"Warning: Filter key '{key}' not found in data. Skipping.")
                continue
            # Allow filtering by a list of values or a single value
            if isinstance(value, list):
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]

    if df.empty:
        print("No data left to plot after applying filters. Exiting.")
        return

    # --- Generate a descriptive filename if not provided ---
    if output_filename is None:
        filter_str = f"_filtered" if filters else ""
        output_filename = f"plot_{y_axis_col}_vs_{x_axis_col}_by_{group_by_col}{filter_str}.png"

    # --- Plotting Setup ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", context="talk")

    grouping_values = sorted(df[group_by_col].unique())
    colors = sns.color_palette("viridis", n_colors=len(grouping_values))
    
    # --- Main Plotting Loop ---
    for i, group_val in enumerate(grouping_values):
        subset = df[df[group_by_col] == group_val].sort_values(x_axis_col)
        
        # Create a clean label for the legend
        legend_label = f"{group_by_col.replace('_', ' ').title()} = {group_val:.2f}"
        
        with_error = subset[subset[y_err_col] > 0]
        plt.errorbar(with_error[x_axis_col], with_error[y_axis_col], yerr=with_error[y_err_col],
                     label=legend_label, fmt='o', capsize=4, color=colors[i])
        
        if show_lower_limits:
            zero_error = subset[subset[y_err_col] == 0]
            if not zero_error.empty:
                arrow_length = zero_error[y_axis_col] * 0.15
                plt.errorbar(zero_error[x_axis_col], zero_error[y_axis_col], yerr=arrow_length,
                             lolims=True, label='_nolegend_', fmt='o', color=colors[i])

    # --- Dynamic Titles and Labels ---
    plt.title(f"{y_axis_col.replace('_', ' ')} vs. {x_axis_col.replace('_', ' ')}", fontsize=18)
    plt.xlabel(f"{x_axis_col.replace('_', ' ').title()}", fontsize=14)
    plt.ylabel(f"{y_axis_col.replace('_', ' ').title()}", fontsize=14)
    plt.legend(title=group_by_col.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Plot successfully saved to '{output_filename}'")


if __name__ == '__main__':
    # --- USER: Change this to the name of your CSV file ---
    csv_file_to_plot = '/Users/sbala/work/gdt_dev2/mvt_fermi_golkhu/MVTfermi/SIM_vs_mvt_2025-07-21_16-03-39/SIM_vs_mvt_2025-07-21_16-03-39_results.csv'

    # --- HOW TO USE THE GENERALIZED FUNCTION ---

    # Example 1: Recreate "MVT vs. Sigma, grouped by Peak Amplitude"
    print("\n--- Plotting MVT vs. Sigma, grouped by Amplitude ---")
    plot_mvt_generic(
        csv_file_path=csv_file_to_plot,
        x_axis_col='sigma',
        group_by_col='peak_amplitude'
    )

    # Example 2: Recreate "MVT vs. Sigma, grouped by Background Level"
    print("\n--- Plotting MVT vs. Sigma, grouped by Background ---")
    plot_mvt_generic(
        csv_file_path=csv_file_to_plot,
        x_axis_col='sigma',
        group_by_col='background_level'
    )

    # Example 3: A NEW plot - "MVT vs. Background, grouped by Sigma"
    print("\n--- Plotting MVT vs. Background, grouped by Sigma ---")
    plot_mvt_generic(
        csv_file_path=csv_file_to_plot,
        x_axis_col='background_level',
        group_by_col='sigma',
        filters={'sigma': 1.0},
        show_lower_limits=False  # Example filter
    )

    # Example 4: Using the `filters` to fix a value.
    # Plot MVT vs. Sigma (grouped by background), but ONLY for a peak_amplitude of 100.
    print("\n--- Plotting with a filter (Peak Amplitude = 100) ---")
    plot_mvt_generic(
        csv_file_path=csv_file_to_plot,
        x_axis_col='sigma',
        group_by_col='background_level',
        filters={'peak_amplitude': 100, 'background_level': 20}
    )