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



def plot_plotly(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='median_mvt_ms',
    # <<< MODIFIED: Replaced y_err_col with upper and lower error columns >>>
    y_err_upper_col=None,
    y_err_lower_col=None,
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
    error_col_for_filtering = y_err_lower_col or y_err_upper_col
    # 1. Conditionally remove zero-error points based on the UI toggle
    if not show_lower_limits and error_col_for_filtering:
        plot_df = df[(df[error_col_for_filtering] > 0) & (df['failed_runs'] < df['total_sim'].max()*0.9)].copy()
    else:
        plot_df = df.copy()

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

    if show_lower_limits and y_err_upper_col in plot_df.columns:
        symbol_legend_title = 'Point Type'
        if marker_col:
            symbol_legend_title = marker_col.replace('_', ' ').title()
            plot_df[marker_col] = plot_df[marker_col].astype(str)
            plot_df[symbol_legend_title] = np.where(plot_df[y_err_upper_col] > 0, plot_df[marker_col], 'Lower Limit')
            symbol_arg = symbol_legend_title
        else:
            plot_df[symbol_legend_title] = np.where(plot_df[y_err_upper_col] > 0, 'Data', 'Lower Limit')
            symbol_arg = symbol_legend_title
    elif marker_col:
        symbol_arg = marker_col.replace('_', ' ').title()
        plot_df[symbol_arg] = plot_df[marker_col].astype(str)

    # 4. Prepare other plot arguments
    error_y_arg = None
    error_y_minus_arg = None
    if show_error_bars and y_err_upper_col and y_err_lower_col:
        if y_err_upper_col in plot_df.columns and y_err_lower_col in plot_df.columns:
            error_y_arg = y_err_upper_col
            error_y_minus_arg = y_err_lower_col # This is for the lower error bar
    #error_y_arg = y_err_col if show_error_bars else None
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
        error_y_minus=error_y_minus_arg,
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


