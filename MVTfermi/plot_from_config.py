import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# You can copy the plot_mvt_generic function from your other script
def plot_mvt_generic_old(
    df, # Now takes a DataFrame directly
    x_axis_col,
    group_by_col,
    y_axis_col='mvt_ms',
    y_err_col='mvt_error_ms',
    filters=None,
    output_filename=None,
    show_lower_limits=True,
    fact=1.0  # Example factor for the line equation
):
    """
    A generalized function to plot MVT against a chosen variable,
    grouped by another variable, with optional filters.
    (This is the same core function as before, but now accepts a DataFrame)
    """
    # --- Apply Filters ---
    if filters:
        print(f"Applying filters: {filters}")
        for key, value in filters.items():
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

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", context="talk")
    grouping_values = sorted(df[group_by_col].unique())
    colors = sns.color_palette("viridis", n_colors=len(grouping_values))
    
    for i, group_val in enumerate(grouping_values):
        subset = df[df[group_by_col] == group_val].sort_values(x_axis_col)
        legend_label = f"{group_by_col.replace('_', ' ').title()} = {group_val:.2f}"
        
        with_error = subset[subset[y_err_col] > 0]
        plt.errorbar(with_error[x_axis_col], with_error[y_axis_col], yerr=with_error[y_err_col],
                     label=legend_label, fmt='.', capsize=4, color=colors[i])
        
        if show_lower_limits:
            zero_error = subset[subset[y_err_col] == 0]
            if not zero_error.empty:
                arrow_length = zero_error[y_axis_col] * 0.15
                plt.errorbar(zero_error[x_axis_col], zero_error[y_axis_col], yerr=arrow_length,
                             lolims=True, label='_nolegend_', fmt='o', color=colors[i])
                
    line_x = np.linspace(df[x_axis_col].min(), df[x_axis_col].max(), 100)
    m = 1000
    y = fact*m*line_x
    #plt.plot(line_x, y, color='gray', linestyle='--', label=f'y = {m*fact}x')

    # --- Dynamic Titles and Labels ---
    plt.title(f"{y_axis_col.replace('_', ' ')} vs. {x_axis_col.replace('_', ' ')}", fontsize=18)
    plt.xlabel(f"{x_axis_col.replace('_', ' ').title()}", fontsize=14)
    plt.ylabel(f"{y_axis_col.replace('_', ' ').title()}", fontsize=14)
    plt.legend(title=group_by_col.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()
    plt.close()
    print(f"\nPlot successfully saved to '{output_filename}'")


def plot_mvt_generic(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='mvt_ms',
    y_err_col='mvt_error_ms',
    filters=None,
    output_filename=None,
    show_lower_limits=True,
    fact=1.0
):
    """
    A generalized function to plot MVT against a chosen variable,
    grouped by another variable, with optional filters.
    """
    # --- Apply Filters ---
    if filters:
        for key, value in filters.items():
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

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", context="talk")
    grouping_values = sorted(df[group_by_col].unique())
    colors = sns.color_palette("viridis", n_colors=len(grouping_values))
    
    for i, group_val in enumerate(grouping_values):
        subset = df[df[group_by_col] == group_val].sort_values(x_axis_col)
        
        # --- THIS IS THE CORRECTED PART ---
        # Format the legend label conditionally to handle numbers and strings
        try:
            # Try to format as a float with 2 decimal places
            legend_label = f"{group_by_col.replace('_', ' ').title()} = {group_val:.2f}"
        except (ValueError, TypeError):
            # If it fails (because it's a string), use it as is
            legend_label = f"{group_by_col.replace('_', ' ').title()} = {group_val}"
        # --- END OF CORRECTION ---
        
        with_error = subset[subset[y_err_col] > 0]
        plt.errorbar(with_error[x_axis_col], with_error[y_axis_col], yerr=with_error[y_err_col],
                     label=legend_label, fmt='.', capsize=4, color=colors[i])
        
        if show_lower_limits:
            zero_error = subset[subset[y_err_col] == 0]
            if not zero_error.empty:
                arrow_length = zero_error[y_axis_col] * 0.15
                plt.errorbar(zero_error[x_axis_col], zero_error[y_axis_col], yerr=arrow_length,
                             lolims=True, label='_nolegend_', fmt='o', color=colors[i])
                
    line_x = np.linspace(df[x_axis_col].min(), df[x_axis_col].max(), 100)
    m = 1000
    y = fact*m*line_x

    # --- Dynamic Titles and Labels ---
    plt.title(f"{y_axis_col.replace('_', ' ')} vs. {x_axis_col.replace('_', ' ')}", fontsize=18)
    plt.xlabel(f"{x_axis_col.replace('_', ' ').title()}", fontsize=14)
    plt.ylabel(f"{y_axis_col.replace('_', ' ').title()}", fontsize=14)
    plt.legend(title=group_by_col.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.show()
    #plt.savefig(output_filename, dpi=300)
    #plt.close()
    print(f"\nPlot successfully saved to '{output_filename}'")

def resolve_column_name(arg, columns):
    """Converts a user argument (name or index) to a valid column name."""
    try:
        # Try to interpret as an integer index
        col_index = int(arg)
        if 0 <= col_index < len(columns):
            return columns[col_index]
        else:
            raise IndexError
    except (ValueError, IndexError):
        # Interpret as a string name
        if arg in columns:
            return arg
        else:
            return None
        

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from MVT results using a self-contained YAML configuration file."
    )
    # The script now only needs the config file
    parser.add_argument("config_file", help="Path to the plot configuration YAML file.")
    parser.add_argument("-limits", action="store_false", dest="show_lower_limits",
                        help="Flag to disable plotting of lower limits for zero-error points.")
    args = parser.parse_args()

    # --- Load Config and Find CSV ---
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get the CSV filename from the config and build its path
        # relative to the config file's location.
        csv_filename = config.pop('csv_file')
        config_dir = os.path.dirname(os.path.abspath(args.config_file))
        csv_path = os.path.join(config_dir, csv_filename)
        
        df = pd.read_csv(csv_path)

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not find a required file or key. {e}")
        return

    # --- The rest of the script is the same ---
    # Parse the YAML Config to Get Plotting Roles
    x_col, y_col, yerr_col, group_col = None, None, None, None
    filters = {}
    
    print("--- Parsing Plot Configuration ---")
    for key, role in config.items():
        if key not in df.columns:
            print(f"Warning: Key '{key}' from YAML not found in CSV columns. Skipping.")
            continue
        
        if role == 'x':
            x_col = key
        elif role == 'y':
            y_col = key
        elif role == 'yerr':
            yerr_col = key
        elif role == 'group':
            group_col = key
        elif role in ['all', 'All', 'ALL', True, 'T', 't']:
            # This parameter will not be filtered
            pass
        else:
            # Any other value is treated as a filter
            filters[key] = role
    
    # --- Validate Roles ---
    if not all([x_col, y_col, group_col]):
        print("Error: The config file must define roles for 'x', 'y', and 'group'.")
        return
    if not yerr_col:
        print("Warning: 'yerr' not defined. Errors will not be plotted.")

    print(f"X-axis: {x_col}")
    print(f"Y-axis: {y_col} (Error: {yerr_col})")
    print(f"Group By: {group_col}")
    print(f"Filters: {filters}")

    # --- Generate Filename and Call Plotting Function ---
    filter_str = ""
    if filters:
        filter_parts = [f"{k}{v}" for k, v in filters.items()]
        filter_str = f"_filtered_by_{'_'.join(filter_parts)}".replace('[','').replace(']','').replace('.','p').replace(',','')

    output_filename = f"plot_{y_col}_vs_{x_col}_by_{group_col}{filter_str}.png"

    plot_mvt_generic(
        df=df,
        x_axis_col=x_col,
        y_axis_col=y_col,
        y_err_col=yerr_col,
        group_by_col=group_col,
        filters=filters,
        output_filename=output_filename,
        show_lower_limits=args.show_lower_limits
    )

if __name__ == '__main__':
    main()