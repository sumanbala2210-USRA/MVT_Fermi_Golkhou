import streamlit as st
import pandas as pd
from plot_from_config_v2 import plot_plotly #plot_matplotlib,
import numpy as np

st.set_page_config(layout="wide")
st.title("MVT Simulation Plotter")

uploaded_file = st.file_uploader("Choose a final summary CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    initial_rows = len(df)

    # <<< MODIFIED: A more robust way to filter for valid results >>>
    # Use the 'successful_runs' column created by the analysis script.
    df = df[~(df.drop(columns=['median_mvt_ms']) == -100).any(axis=1)].copy()

    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        st.info(f"Removed {dropped_rows} rows with no invalid simulation runs.")
    
    st.write("Data Preview (after filtering):", df.head())
    columns = df.columns.tolist()

    st.sidebar.header("Plot Configuration")
    
    plot_backend = st.sidebar.radio("Plotting Backend", ["Interactive (Plotly)", "Static (Matplotlib)"])

    # <<< MODIFIED: Updated default columns to match your new CSV format >>>
    x_col = st.sidebar.selectbox("X-Axis:", columns, index=columns.index('sigma') if 'sigma' in columns else 0)
    y_col = st.sidebar.selectbox("Y-Axis:", columns, index=columns.index('median_mvt_ms') if 'median_mvt_ms' in columns else 1)
    group_col = st.sidebar.selectbox("Group By:", columns, index=columns.index('peak_amplitude') if 'peak_amplitude' in columns else 2)
    
    marker_col_options = [None] + columns
    marker_col = st.sidebar.selectbox("Marker Style By:", marker_col_options, index=0)
    
    st.sidebar.markdown("---")
    show_error_bars = st.sidebar.checkbox("Show Error Bars", value=True)
    show_limits = st.sidebar.checkbox("Show Lower Limits", value=True)
    # The 'show_lower_limits' checkbox is removed as it's less relevant for aggregated data
    use_log_x = st.sidebar.checkbox("Use Log Scale for X-axis")
    use_log_y = st.sidebar.checkbox("Use Log Scale for Y-axis")
    
    st.sidebar.header("Axis Ranges (Optional)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x_min = st.number_input("X-axis Min", value=None, format="%g")
        y_min = st.number_input("Y-axis Min", value=None, format="%g")
    with col2:
        x_max = st.number_input("X-axis Max", value=None, format="%g")
        y_max = st.number_input("Y-axis Max", value=None, format="%g")
    
    plot_theme = st.sidebar.selectbox("Plot Theme", ["simple_white", "Contrast White", "Contrast Dark", "Seaborn-like"])
    
    st.sidebar.markdown("---") 
    plot_height = st.sidebar.number_input("Plot Height (pixels)", min_value=400, value=700, step=50)
    st.sidebar.header("Filters")
    filters = {}
    exclude_list = ['t_start', 't_stop', 'det', 'trigger_number', 'peak_time_ratio', 'background_level', 'mvt_error_ms']
    filter_cols = [c for c in columns if c not in exclude_list]
    for col in filter_cols:
        unique_vals = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"Filter by {col}:", unique_vals, default=unique_vals)
        if selected != unique_vals:
            filters[col] = selected


    if st.button("Generate Plot"):
        x_range = [x_min, x_max] if x_min is not None and x_max is not None else None
        y_range = [y_min, y_max] if y_min is not None and y_max is not None else None

        if "Interactive" in plot_backend:
            # <<< NEW: Automatically find the correct upper and lower error columns >>>
            y_err_upper = None
            y_err_lower = None
            if show_error_bars:
                # Based on our convention from analyze_events.py
                if 'mvt_err_upper' in df.columns and 'mvt_err_lower' in df.columns:
                    y_err_upper = 'mvt_err_upper'
                    y_err_lower = 'mvt_err_lower'
                    st.sidebar.success("Found asymmetric error columns.")
                else:
                    st.sidebar.warning("Could not find 'mvt_err_upper'/'mvt_err_lower' columns.")

            fig = plot_plotly(
                df=df, x_axis_col=x_col, y_axis_col=y_col, group_by_col=group_col, 
                marker_col=marker_col,
                # <<< MODIFIED: Pass the new asymmetric error bar arguments >>>
                y_err_upper_col=y_err_upper,
                y_err_lower_col=y_err_lower,
                filters=filters,
                show_lower_limits=show_limits,
                show_error_bars=show_error_bars, 
                use_log_x=use_log_x, use_log_y=use_log_y, x_range=x_range, y_range=y_range, 
                plot_theme=plot_theme, plot_height=plot_height
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, theme=None)
        else: # Static
            st.warning("Matplotlib plotting needs to be updated to support asymmetric errors.")