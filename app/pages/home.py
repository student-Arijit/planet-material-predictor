import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import os
import sys

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.components.custom_widgets import (
    create_material_distribution_chart,
    create_velocity_histogram,
    create_station_analysis_chart,
    create_data_quality_metrics,
    display_data_statistics
)


def load_data():
    """Load data from DuckDB"""
    db_path = "data/processed/mars_seismic.db"

    if not os.path.exists(db_path):
        return None

    try:
        conn = duckdb.connect(db_path)
        df = conn.execute("SELECT * FROM mars_seismic_data").df()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def display_home_page():
    """Display the home page content"""

    # Header
    st.title("🪐 Mars Material Predictor")
    st.markdown("## Seismic Data Analysis and Material Prediction System")

    st.markdown("""
    Welcome to the Mars Material Predictor! This application uses machine learning to analyze 
    seismic data from Mars and predict the types of materials present beneath the surface.

    ### 🚀 Features:
    - **Data Processing**: Automated cleaning and preprocessing of seismic data
    - **Machine Learning**: Advanced ensemble models (Random Forest, XGBoost, LightGBM)
    - **Interactive Visualization**: Comprehensive charts and graphs
    - **Real-time Prediction**: Predict material types from new seismic measurements
    """)

    # Load and display data overview
    df = load_data()

    if df is not None:
        st.success(f"✅ Data loaded successfully! ({len(df)} records)")

        # Data overview section
        st.markdown("---")
        st.header("📊 Data Overview")

        # Display basic statistics
        display_data_statistics(df)

        # Data visualizations
        st.markdown("---")
        st.header("📈 Data Visualizations")

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Material Distribution", "Velocity Analysis", "Station Analysis", "Data Quality"])

        with tab1:
            if 'material_type' in df.columns:
                material_counts = df['material_type'].value_counts()
                fig_materials = create_material_distribution_chart(material_counts)
                st.plotly_chart(fig_materials, use_container_width=True)

                # Material statistics table
                st.subheader("Material Type Statistics")
                material_stats = df.groupby('material_type').agg({
                    'velocity': ['mean', 'std', 'min', 'max'],
                    'sampling': ['mean', 'std'],
                    'delta': ['mean', 'std']
                }).round(4)

                st.dataframe(material_stats, use_container_width=True)
            else:
                st.warning("Material type information not available")

        with tab2:
            if 'velocity' in df.columns:
                # Velocity histogram
                if 'material_type' in df.columns:
                    fig_velocity = create_velocity_histogram(df)
                    st.plotly_chart(fig_velocity, use_container_width=True)
                else:
                    # Simple velocity distribution
                    import plotly.express as px
                    fig_velocity = px.histogram(df, x='velocity', nbins=50,
                                                title='Velocity Distribution')
                    st.plotly_chart(fig_velocity, use_container_width=True)

                # Velocity statistics
                st.subheader("Velocity Statistics")
                velocity_stats = df['velocity'].describe()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{velocity_stats['mean']:.4f}")
                with col2:
                    st.metric("Std Dev", f"{velocity_stats['std']:.4f}")
                with col3:
                    st.metric("Min", f"{velocity_stats['min']:.4f}")
                with col4:
                    st.metric("Max", f"{velocity_stats['max']:.4f}")
            else:
                st.warning("Velocity data not available")

        with tab3:
            if 'station' in df.columns:
                fig_station = create_station_analysis_chart(df)
                st.plot