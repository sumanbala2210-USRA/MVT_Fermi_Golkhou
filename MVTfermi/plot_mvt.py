import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def plot_data(files, x_axis, include_limits, y_axis='mvt_ms', output_file='mvt_plot.png'):
    """
    Plots Minimum Viable Timescale (MVT) data from specified CSV files.
    The file names are hardcoded inside this function.
    """
    # Hardcoded file names
    file_map = {
        'all': 'Trigger_number_vs_mvt_all_det.csv',
        'best': 'Trigger_number_vs_mvt_best_det_2.csv',
        'one': 'Trigger_number_vs_mvt_one_det.csv'
    }

    df_list = []
    for f_key in files:
        try:
            df = pd.read_csv(file_map[f_key])
            df['source'] = f_key
            df_list.append(df)
        except FileNotFoundError:
            print(f"❌ Error: The file '{file_map[f_key]}' was not found.", file=sys.stderr)
            sys.exit(1)

    if not df_list:
        print("❌ Error: No data to plot.", file=sys.stderr)
        sys.exit(1)

    full_df = pd.concat(df_list, ignore_index=True)

    # Filter out failed analyses (mvt_error_ms == -100)
    full_df = full_df[full_df['mvt_error_ms'] != -100].copy()

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {'all': '#0072B2', 'best': '#009E73', 'one': '#D55E00'}

    # Separate data points from limiting values
    data_with_errors = full_df[full_df['mvt_error_ms'] > 0]
    limit_values = full_df[full_df['mvt_error_ms'] == 0]

    # Plot data with actual error bars
    for key, group in data_with_errors.groupby('source'):
        if key in files:
            ax.errorbar(group[x_axis], group[y_axis], yerr=group['mvt_error_ms'],
                        fmt='o', color=colors[key], capsize=5, label=f'{key} detectors')

    # Plot limiting values if requested
    if include_limits:
        for key, group in limit_values.groupby('source'):
            if key in files:
                ax.scatter(group[x_axis], group[y_axis],
                           marker='v', s=60, color='gray',
                           label=f'{key} detectors (limit)', zorder=5)

    # Finalize the plot
    ax.set_xlabel(f'{x_axis}', fontsize=12)
    ax.set_ylabel(f'{y_axis} (ms)', fontsize=12)
    ax.set_title(f'{y_axis} vs. {x_axis}', fontsize=14, fontweight='bold')

    if ax.has_data():
        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.tight_layout()
        plt.show()
        plt.savefig(output_file, dpi=300)
        print(f"✅ Plot saved to {output_file}")
    else:
        print("⚠️ Warning: No data was available to plot after filtering.", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MVT data from different detector sets using specified arguments.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d', '--datasets', nargs='+', choices=['all', 'best', 'one'], required=True,
                        help='Which dataset(s) to plot. Choose one or more from: all, best, one.')

    parser.add_argument('-x', '--x_axis', type=str, required=True,
                        help='Column for the x-axis (e.g., T90, PFLX).')

    parser.add_argument('-lim', '--limits', type=int, choices=[0, 1], default=0,
                        help='Include limiting values (1 for yes, 0 for no).')

    parser.add_argument('-o', '--output_file', type=str, default='mvt_plot.png',
                        help='Output file name for the plot (e.g., my_plot.png).')

    args = parser.parse_args()

    # Convert the integer 'limits' argument to a boolean for the function
    include_limits_bool = bool(args.limits)

    plot_data(
        files=args.datasets,
        x_axis=args.x_axis,
        include_limits=include_limits_bool,
        output_file=args.output_file
    )