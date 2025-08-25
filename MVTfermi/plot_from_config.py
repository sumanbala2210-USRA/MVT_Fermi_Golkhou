import pandas as pd
import yaml
import streamlit as st
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
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



def plot_matplotlib(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='mvt_ms',
    y_err_col='mvt_error_ms',
    filters=None,
    show_lower_limits=True,
    show_error_bars=True,
    use_log_x=False,
    use_log_y=False,
    x_range=None,
    y_range=None
):
    """
    Creates a static plot using Matplotlib/Seaborn.
    """
    df = df[df[y_err_col] >= 0].copy()

    if filters:
        for key, value in filters.items():
            df = df[df[key].isin(value) if isinstance(value, list) else df[key] == value]

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_theme(style="whitegrid", context="talk")

    grouping_values = sorted(df[group_by_col].unique())
    colors = sns.color_palette("viridis", n_colors=len(grouping_values))

    for i, group_val in enumerate(grouping_values):
        subset = df[df[group_by_col] == group_val].sort_values(x_axis_col)
        
        try:
            legend_label = f"{group_by_col.replace('_', ' ').title()} = {group_val:.2f}"
        except (ValueError, TypeError):
            legend_label = f"{group_by_col.replace('_', ' ').title()} = {group_val}"
        
        with_error = subset[subset[y_err_col] > 0]
        zero_error = subset[subset[y_err_col] == 0]

        # Plot points with error
        if not with_error.empty:
            if show_error_bars:
                ax.errorbar(with_error[x_axis_col], with_error[y_axis_col], yerr=with_error[y_err_col],
                             label=legend_label, fmt='o', capsize=4, color=colors[i], markersize=6)
            else:
                ax.plot(with_error[x_axis_col], with_error[y_axis_col],
                         label=legend_label, marker='o', color=colors[i], linestyle='None', markersize=6)
            legend_label = '_nolegend_'

        # Plot points with lower limits
        if show_lower_limits and not zero_error.empty:
            points_to_plot = zero_error[zero_error[y_axis_col] >= 0]
            if not points_to_plot.empty:
                arrow_length = points_to_plot[y_axis_col] * 0.15
                ax.errorbar(points_to_plot[x_axis_col], points_to_plot[y_axis_col], yerr=arrow_length,
                             lolims=True, label=legend_label, fmt='x', color=colors[i], markersize=6)

    if use_log_x: ax.set_xscale('log')
    if use_log_y: ax.set_yscale('log')
    if x_range: ax.set_xlim(x_range)
    if y_range: ax.set_ylim(y_range)

    ax.set_title(f"{y_axis_col.replace('_', ' ')} vs. {x_axis_col.replace('_', ' ')}", fontsize=18)
    ax.set_xlabel(f"{x_axis_col.replace('_', ' ').title()}", fontsize=14)
    ax.set_ylabel(f"{y_axis_col.replace('_', ' ').title()}", fontsize=14)
    ax.legend(title=group_by_col.replace('_', ' ').title())
    ax.grid(True, which='both', linestyle='--')
    
    plt.tight_layout()
    return fig



def plot_plotly_old(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='mvt_ms',
    y_err_col='mvt_error_ms',
    filters=None,
    show_lower_limits=True,
    show_error_bars=True,
    use_log_x=False,
    use_log_y=False,
    x_range=None,
    y_range=None,
    plot_theme="Contrast White",
    plot_height=700
):
    """
    Creates a final, advanced interactive plot using Plotly Express.
    """
    # 1. Start with clean, valid data
    #plot_df = df[df[y_err_col] >= 0].copy()

    # 2. Conditionally remove zero-error points based on the UI toggle
    if not show_lower_limits:
        plot_df = df[df[y_err_col] > 0].copy()
    else:
        plot_df = df[df[y_err_col] >= 0].copy()

    # 3. Apply user-defined filters
    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value) if isinstance(value, list) else plot_df[key] == value]

    if plot_df.empty:
        return None

    # 4. Prepare data for plotting
    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    # Define marker style and size in new columns based on error value
    plot_df['symbol'] = np.where(plot_df[y_err_col] > 0, '', 'limit ')  # Use diamond for zero-error points
    #plot_df['size'] = np.where(plot_df[y_err_col] > 0, 1, 1) # Make triangles slightly bigger

    error_y_arg = y_err_col if show_error_bars else None
    reject_list = ['t_start', 't_stop', 'det', 'trigger_number', 'peak_time_ratio', 'background_level', 'pulse']
    hover_cols = [col for col in plot_df.columns if col not in reject_list]

    # 5. Create the figure in a single, powerful call
    fig = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color=group_by_col,
        error_y=error_y_arg,
        symbol='symbol',      # Use the new symbol column
        #size='size',          # Use the new size column (hides size from legend)
        log_x=use_log_x,
        log_y=use_log_y,
        labels={ ... },
        title=f"{y_axis_col.replace('_', ' ')} vs. {x_axis_col.replace('_', ' ')}",
        template="plotly_white",
        hover_data=hover_cols
    )

    
    fig.update_layout(showlegend=True) # Ensure legend is shown

    # 6. Apply custom theme styling
    theme_styles = {
        "Contrast White": {'paper_bgcolor': "white", 'plot_bgcolor': "white", 'font_color': "black", 'gridcolor': '#D3D3D3', 'zerolinecolor': '#C0C0C0'},
        "Contrast Dark": {'paper_bgcolor': "#1E1E1E", 'plot_bgcolor': "#2E2E2E", 'font_color': "white", 'gridcolor': '#4A4A4A', 'zerolinecolor': '#7A7A7A'},
        "Seaborn-like": {'paper_bgcolor': "#F0F2F6", 'plot_bgcolor': "#F0F2F6", 'font_color': "black", 'gridcolor': 'white', 'zerolinecolor': 'white'}
    }
    if plot_theme in theme_styles:
        style = theme_styles[plot_theme]
        fig.update_layout(paper_bgcolor=style['paper_bgcolor'], plot_bgcolor=style['plot_bgcolor'], font_color=style['font_color'])
        fig.update_xaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
        fig.update_yaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
    
        # --- Final layout updates ---


    # --- END OF FIX ---

    # 7. Final layout updates
    fig.update_traces(marker=dict(size=10),error_y=dict(thickness=1))
    fig.update_layout(height=plot_height, legend_title_text=group_by_col.replace('_', ' ').title(),
                      title_x=0.5, xaxis_range=x_range, yaxis_range=y_range)
    #fig.update_layout(height=plot_height, legend_title_text=group_by_col.replace('_', ' ').title(),
    #                  title_x=0.5, xaxis_range=x_range, yaxis_range=y_range)

    if plot_theme == "simple_white":
        fig.update_layout(
            height=plot_height,
            paper_bgcolor="white",    # Outer background
            plot_bgcolor="white",     # Inner background
            font_color="black"        # All text (titles, labels, legend)
        )
        # Set axes lines and gridlines to be visible on a white background
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
    return fig



def plot_plotly_best(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='mvt_ms',
    y_err_col='mvt_error_ms',
    marker_col=None,  # --- NEW: Parameter for marker style ---
    filters=None,
    show_lower_limits=True,
    show_error_bars=True,
    use_log_x=False,
    use_log_y=False,
    x_range=None,
    y_range=None,
    plot_theme="Contrast White",
    plot_height=700
):
    """
    Creates a final, advanced interactive plot using Plotly Express,
    with support for dynamic marker styles.
    """
    # 1. Conditionally remove zero-error points based on the UI toggle
    if not show_lower_limits:
        plot_df = df[df[y_err_col] > 0].copy()
    else:
        # We need all data to properly handle the lower limit markers
        plot_df = df[df[y_err_col] >= 0].copy()

    # 2. Apply user-defined filters
    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value)]

    if plot_df.empty:
        st.warning("No data left to plot after applying filters.")
        return None

    # 3. Prepare data for plotting
    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    
    # --- NEW: Logic to handle dynamic marker symbols ---
    symbol_arg = None
    
    # This logic creates a single column for Plotly to control symbols.
    # It intelligently combines your 'marker_col' choice and the 'show_lower_limits' toggle.
    if show_lower_limits and y_err_col in plot_df.columns:
        if marker_col:
            # When a marker column is chosen, use its values for regular points
            # and a special value 'Lower Limit' for points with zero/negative error.
            legend_title = marker_col.replace('_', ' ').title()
            plot_df[marker_col] = plot_df[marker_col].astype(str)
            plot_df[legend_title] = np.where(plot_df[y_err_col] > 0, plot_df[marker_col], 'Lower Limit')
            symbol_arg = legend_title
        else:
            # If no marker column, just differentiate 'Data' from 'Lower Limit'.
            plot_df['Point Type'] = np.where(plot_df[y_err_col] > 0, 'Data', 'Lower Limit')
            symbol_arg = 'Point Type'
    elif marker_col:
        # If showing lower limits is off, just use the marker column directly.
        symbol_arg = marker_col

    # 4. Prepare other plot arguments
    error_y_arg = y_err_col if show_error_bars else None
    reject_list = ['t_start', 't_stop', 'det', 'trigger_number', 'peak_time_ratio', 'background_level', 'pulse']
    hover_cols = [col for col in plot_df.columns if col not in reject_list]

    # 5. Create the figure
    fig = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color=group_by_col,
        symbol=symbol_arg,  # --- USE THE DYNAMICALLY ASSIGNED SYMBOL ARGUMENT ---
        error_y=error_y_arg,
        log_x=use_log_x,
        log_y=use_log_y,
        labels={
            x_axis_col: x_axis_col.replace('_', ' ').title(),
            y_axis_col: y_axis_col.replace('_', ' ').title(),
            group_by_col: group_by_col.replace('_', ' ').title()
        },
        title=f"{y_axis_col.replace('_', ' ').title()} vs. {x_axis_col.replace('_', ' ').title()}",
        template="plotly_white",
        hover_data=hover_cols
    )

    # 6. Apply custom theme styling
    theme_styles = {
        "Contrast White": {'paper_bgcolor': "white", 'plot_bgcolor': "white", 'font_color': "black", 'gridcolor': '#D3D3D3', 'zerolinecolor': '#C0C0C0'},
        "Contrast Dark": {'paper_bgcolor': "#1E1E1E", 'plot_bgcolor': "#2E2E2E", 'font_color': "white", 'gridcolor': '#4A4A4A', 'zerolinecolor': '#7A7A7A'},
        "Seaborn-like": {'paper_bgcolor': "#F0F2F6", 'plot_bgcolor': "#F0F2F6", 'font_color': "black", 'gridcolor': 'white', 'zerolinecolor': 'white'}
    }
    if plot_theme in theme_styles:
        style = theme_styles[plot_theme]
        fig.update_layout(paper_bgcolor=style['paper_bgcolor'], plot_bgcolor=style['plot_bgcolor'], font_color=style['font_color'])
        fig.update_xaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
        fig.update_yaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
    elif plot_theme == "simple_white":
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')

    # 7. Final layout updates
    fig.update_traces(marker=dict(size=14), error_y=dict(thickness=1))
    fig.update_layout(
        height=plot_height, 
        legend_title_text=group_by_col.replace('_', ' ').title(),
        title_x=0.5, 
        xaxis_range=x_range, 
        yaxis_range=y_range
    )
    
    return fig




def plot_plotly(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='mvt_ms',
    y_err_col='mvt_error_ms',
    marker_col=None,
    filters=None,
    show_lower_limits=True,
    show_error_bars=True,
    use_log_x=False,
    use_log_y=False,
    x_range=None,
    y_range=None,
    plot_theme="Contrast White",
    plot_height=800
):
    """
    Creates a publication-quality, advanced interactive plot using Plotly Express.
    Features larger fonts, thicker lines, and bordered markers for clarity.
    """
    # 1. Conditionally remove zero-error points based on the UI toggle
    if not show_lower_limits:
        #plot_df = df[(df[y_err_col] > 0) & (df['failed_run'] < df['total_sim'].max()*0.9)].copy()
        plot_df = df[(df[y_err_col] > 0) & (df['failed_sim'] < df['total_sim'].max()*0.9)].copy()
        #plot_df = df[df['failed_sim'] < df['total_sim'].max()*0.1].copy()
    else:
        #plot_df = df[df[y_err_col] >= 0].copy()
        plot_df = df#[df['failed_sim'] < 10].copy()

    # 2. Apply user-defined filters
    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value)]

    if plot_df.empty:
        return None

    # 3. Prepare data for plotting
    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    
    symbol_arg = None
    color_legend_title = group_by_col.replace('_', ' ').title()

    if show_lower_limits and y_err_col in plot_df.columns:
        symbol_legend_title = 'Point Type'
        if marker_col:
            symbol_legend_title = marker_col.replace('_', ' ').title()
            plot_df[marker_col] = plot_df[marker_col].astype(str)
            plot_df[symbol_legend_title] = np.where(plot_df[y_err_col] > 0, plot_df[marker_col], 'Lower Limit')
            symbol_arg = symbol_legend_title
        else:
            plot_df[symbol_legend_title] = np.where(plot_df[y_err_col] > 0, 'Data', 'Lower Limit')
            symbol_arg = symbol_legend_title
    elif marker_col:
        symbol_arg = marker_col.replace('_', ' ').title()
        plot_df[symbol_arg] = plot_df[marker_col].astype(str)

    # 4. Prepare other plot arguments
    error_y_arg = y_err_col if show_error_bars else None
    reject_list = ['t_start', 't_stop', 'det', 'trigger_number', 'peak_time_ratio', 'background_level', 'pulse']
    hover_cols = [col for col in plot_df.columns if col not in reject_list]

    # 5. Create the figure
    fig = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color=group_by_col,
        symbol=symbol_arg,
        error_y=error_y_arg,
        log_x=use_log_x,
        log_y=use_log_y,
        labels={
            x_axis_col: x_axis_col.replace('_', ' ').title(),
            y_axis_col: y_axis_col.replace('_', ' ').title(),
            group_by_col: color_legend_title
        },
        #title=f"{y_axis_col.replace('_', ' ').title()} vs. {x_axis_col.replace('_', ' ').title()}",
        template="plotly_white",
        hover_data=hover_cols,
        symbol_map={'Lower Limit': 'diamond-open'}
    )

    # 6. Apply custom theme styling
    theme_styles = {
        "Contrast White": {'paper_bgcolor': "white", 'plot_bgcolor': "white", 'font_color': "black", 'gridcolor': '#D3D3D3', 'zerolinecolor': '#C0C0C0'},
        "Contrast Dark": {'paper_bgcolor': "#1E1E1E", 'plot_bgcolor': "#2E2E2E", 'font_color': "white", 'gridcolor': '#4A4A4A', 'zerolinecolor': '#7A7A7A'},
        "Seaborn-like": {'paper_bgcolor': "#F0F2F6", 'plot_bgcolor': "#F0F2F6", 'font_color': "black", 'gridcolor': 'white', 'zerolinecolor': 'white'}
    }
    font_color = "black"
    # --- NEW: Define border color based on theme ---
    border_color = "black" 
    if plot_theme in theme_styles:
        style = theme_styles[plot_theme]
        font_color = style['font_color']
        if plot_theme == "Contrast Dark":
            border_color = "white" # Use white border for dark theme
        fig.update_layout(paper_bgcolor=style['paper_bgcolor'], plot_bgcolor=style['plot_bgcolor'], font_color=font_color)
        fig.update_xaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
        fig.update_yaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
    elif plot_theme == "simple_white":
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')

    # 7. Final layout updates for publication quality
    fig.update_traces(
        marker=dict(
            size=12,
            # --- NEW: Add a border to all markers ---
            line=dict(
                width=1.5,
                color=border_color  # Use the theme-aware border color
            )
        ),
        error_y=dict(thickness=2.0)
    )

    fig.update_layout(
        height=plot_height,
        title_x=0.5,
        xaxis_range=x_range,
        yaxis_range=y_range,
        font=dict(
            family="Arial, sans-serif",
            size=18,
            color=font_color
        ),
        title_font_size=24,
        xaxis=dict(title_font_size=22),
        yaxis=dict(title_font_size=22),
        legend=dict(
            title_font_size=20,
            font_size=18,
            traceorder="normal"
        )
    )
    
    return fig







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