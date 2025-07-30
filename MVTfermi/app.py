import streamlit as st
import pandas as pd
# Make sure this import points to your plotting function
from plot_from_config import plot_mvt_advanced

st.set_page_config(layout="wide")
st.title("MVT Simulation Plotter")

uploaded_file = st.file_uploader("Choose a results CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    
    columns = df.columns.tolist()
    
    st.sidebar.header("Plot Configuration")
    
    # --- UI Widgets to select columns ---
    x_col = st.sidebar.selectbox("X-Axis:", columns, index=columns.index('sigma') if 'sigma' in columns else 0)
    y_col = st.sidebar.selectbox("Y-Axis:", columns, index=columns.index('mvt_ms') if 'mvt_ms' in columns else 1)
    group_col = st.sidebar.selectbox("Group By:", columns, index=columns.index('peak_amplitude') if 'peak_amplitude' in columns else 2)
    show_limits = st.sidebar.checkbox("Show Lower Limits", value=True)
    
    # --- NEW: Add checkboxes for log scales ---
    st.sidebar.markdown("---") # Visual separator
    use_log_x = st.sidebar.checkbox("Use Log Scale for X-axis")
    use_log_y = st.sidebar.checkbox("Use Log Scale for Y-axis")
    
    # --- UI for Filters ---
    st.sidebar.markdown("---") 
    st.sidebar.header("Filters")
    filters = {}
    filter_cols = [c for c in columns if c not in [x_col, y_col]]
    
    for col in filter_cols:
        unique_vals = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"Filter by {col}:", unique_vals, default=unique_vals)
        if selected != unique_vals:
            filters[col] = selected

    # --- Plotting ---
    if st.button("Generate Plot"):
        fig = plot_mvt_advanced(
            df=df,
            x_axis_col=x_col,
            y_axis_col=y_col,
            group_by_col=group_col,
            filters=filters,
            show_lower_limits=show_limits,
            use_log_x=use_log_x, # Pass the value from the checkbox
            use_log_y=use_log_y  # Pass the value from the checkbox
        )
        st.pyplot(fig)