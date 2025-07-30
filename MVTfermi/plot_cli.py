import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def plot_mvt_generic(
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
    plt.show()
    plt.savefig(output_filename, dpi=300)
    plt.close()
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
    parser = argparse.ArgumentParser(description="Generate plots from MVT simulation results.")
    parser.add_argument("-csv_file", help="Path to the input CSV file.")
    parser.add_argument("-x", "--xaxis", required=True, help="Column name or index for the X-axis (e.g., 'sigma' or 1).")
    parser.add_argument("-g", "--groupby", required=True, help="Column name or index for grouping data into series.")
    parser.add_argument("-v", "--values", nargs='+', default=['all'],
                        help="Values of the groupby column to plot. Use 'all' for every value, or provide a list (e.g., 10 50).")
    parser.add_argument("-o", "--output", default=None, help="Optional: Name of the output plot file.")
    parser.add_argument("-fact", type=float, default=1.0, help="Factor for the line equation (default is 1.0).")
    parser.add_argument("--no-limits", action="store_false", dest="show_lower_limits",
                        help="Flag to disable plotting of lower limits for zero-error points.")
    

    args = parser.parse_args()

    # Load data
    try:
        #df = pd.read_csv(args.csv_file).fillna(0)
        csv_file_path = '/Users/sbala/work/gdt_dev2/mvt_fermi_golkhu/MVTfermi/SIM_vs_mvt_2025-07-22_15-29-21/SIM_vs_mvt_2025-07-22_15-29-21_results.csv'
        csv_file_path = '/Users/sbala/work/gdt_dev2/mvt_fermi_golkhu/MVTfermi/SIM_vs_mvt_2025-07-21_16-03-39/SIM_vs_mvt_2025-07-21_16-03-39_results.csv'
        csv_file_path = '/Users/sbala/work/gdt_dev2/mvt_fermi_golkhu/MVTfermi/SIM_vs_mvt_2025-07-29_16-01-51/SIM_vs_mvt_2025-07-29_16-01-51_results.csv'

        df = pd.read_csv(csv_file_path)
        columns = df.columns.tolist()
    except FileNotFoundError:
        print(f"Error: The file '{args.csv_file}' was not found.")
        return

    # Resolve column names from user input (name or index)
    x_col = resolve_column_name(args.xaxis, columns)
    g_col = resolve_column_name(args.groupby, columns)

    if not x_col or not g_col:
        print(f"Error: Invalid column name or index provided.")
        print(f"Available columns: {list(enumerate(columns))}")
        return

    # Build filters from the --values argument
    filters = None
    if args.values != ['all']:
        try:
            # Convert values to float for filtering
            filter_values = [float(v) for v in args.values]
            filters = {g_col: filter_values}
        except ValueError:
            print(f"Error: --values must be numbers (e.g., 10 50 100). You provided: {args.values}")
            return

    # Call the plotting function
    plot_mvt_generic(
        df=df,
        x_axis_col=x_col,
        group_by_col=g_col,
        filters=filters,
        output_filename=args.output,
        show_lower_limits=args.show_lower_limits,
        fact=args.fact
    )

if __name__ == '__main__':
    main()