import streamlit as st
import pandas as pd
# Make sure this import points to your plotting function
from plot_from_config import plot_mvt_advanced
from plot_from_config import plot_with_plotly


st.set_page_config(layout="wide")
st.title("MVT Simulation Plotter")

uploaded_file = st.file_uploader("Choose a results CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- NEW: Filter out all rows containing -100 ---
    initial_rows = len(df)
    # This line finds any row where any column has the value -100 and removes it.
    df = df[~(df == -100).any(axis=1)].copy()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        st.info(f"Removed {dropped_rows} rows containing '-100' (failed simulations).")
    # --- End of new code ---

    st.write("Data Preview (after filtering):", df.head())
    
    columns = df.columns.tolist()
    
    st.sidebar.header("Plot Configuration")
    
    # --- UI Widgets to select columns ---
    x_col = st.sidebar.selectbox("X-Axis:", columns, index=columns.index('sigma') if 'sigma' in columns else 0)
    y_col = st.sidebar.selectbox("Y-Axis:", columns, index=columns.index('mvt_ms') if 'mvt_ms' in columns else 1)
    group_col = st.sidebar.selectbox("Group By:", columns, index=columns.index('peak_amplitude') if 'peak_amplitude' in columns else 2)
    show_limits = st.sidebar.checkbox("Show Lower Limits", value=True)
    
    # --- UI for Log Scales ---
    st.sidebar.markdown("---")
    use_log_x = st.sidebar.checkbox("Use Log Scale for X-axis")
    use_log_y = st.sidebar.checkbox("Use Log Scale for Y-axis")
    
    # --- UI for Filters ---
    st.sidebar.markdown("---") 
    st.sidebar.header("Filters")
    filters = {}
    filter_cols = [c for c in columns if c != y_col]
    
    for col in filter_cols:
        unique_vals = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"Filter by {col}:", unique_vals, default=unique_vals)
        if selected != unique_vals:
            filters[col] = selected

    """
    # --- Plotting ---
    if st.button("Generate Plot"):
        # Since the bad data is gone, the plotting function no longer needs a fix
        fig = plot_mvt_advanced(
            df=df,
            x_axis_col=x_col,
            y_axis_col=y_col,
            group_by_col=group_col,
            filters=filters,
            show_lower_limits=show_limits,
            use_log_x=use_log_x,
            use_log_y=use_log_y
        )
        st.pyplot(fig)
    """
    if st.button("Generate Plot"):
    # Call the new Plotly function
        fig = plot_with_plotly(
            df=df,
            x_axis_col=x_col,
            y_axis_col=y_col,
            group_by_col=group_col,
            filters=filters,
            show_lower_limits=show_limits,
            use_log_x=use_log_x,
            use_log_y=use_log_y
        )
        
        # Use st.plotly_chart to display the interactive figure
        if fig:
            st.plotly_chart(fig, use_container_width=True)