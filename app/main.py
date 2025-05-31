import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import custom components (we'll create these next)
try:
    from app.components.custom_widgets import load_data, create_sidebar, display_stats_summary
    from app.pages import home, prediction
except ImportError:
    st.error("Some modules are not yet created. Please ensure all files are in place.")

# Page configuration
st.set_page_config(
    page_title="Mars Material Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/planet-material-predictor',
        'Report a bug': "https://github.com/yourusername/planet-material-predictor/issues",
        'About': "# Mars Material Predictor\nAnalyze Mars geological data and predict material components!"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


def load_all_data():
    """Load all Excel files from the raw data directory"""
    data_dir = project_root / "data" / "raw"

    if not data_dir.exists():
        st.error(f"Data directory not found: {data_dir}")
        return None

    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))

    if not excel_files:
        st.warning("No Excel files found in the data/raw directory.")
        return None

    all_data = []
    file_info = {}

    with st.spinner(f"Loading {len(excel_files)} Excel files..."):
        for file_path in excel_files:
            try:
                # Read Excel file
                df = pd.read_excel(file_path)
                df['source_file'] = file_path.name
                all_data.append(df)
                file_info[file_path.name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size': f"{file_path.stat().st_size / 1024:.1f} KB"
                }
            except Exception as e:
                st.error(f"Error loading {file_path.name}: {str(e)}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data, file_info

    return None, {}


def main():
    """Main application function"""

    # Header
    st.markdown('<h1 class="main-header">🚀 Mars Material Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze geological data and predict material components on Mars</p>',
                unsafe_allow_html=True)

    # Load data
    data_load_state = st.empty()
    data_load_state.text("Loading data...")

    data_result = load_all_data()

    if data_result is None:
        data_load_state.error("Failed to load data. Please check your data files.")
        return

    data, file_info = data_result
    data_load_state.success(f"✅ Loaded {len(data)} records from {len(file_info)} files")

    # Store data in session state
    st.session_state['data'] = data
    st.session_state['file_info'] = file_info

    # Sidebar
    st.sidebar.markdown("## 📊 Data Overview")

    # File information
    with st.sidebar.expander("📁 Loaded Files", expanded=True):
        for filename, info in file_info.items():
            st.write(f"**{filename}**")
            st.write(f"- Rows: {info['rows']:,}")
            st.write(f"- Columns: {info['columns']}")
            st.write(f"- Size: {info['size']}")
            st.write("---")

    # Data summary
    st.sidebar.markdown("## 📈 Quick Stats")

    if data is not None and not data.empty:
        total_records = len(data)
        unique_networks = data['Network'].nunique() if 'Network' in data.columns else 0
        unique_stations = data['Station'].nunique() if 'Station' in data.columns else 0
        date_range = ""

        if 'Start Time' in data.columns:
            try:
                data['Start Time'] = pd.to_datetime(data['Start Time'])
                min_date = data['Start Time'].min().strftime('%Y-%m-%d')
                max_date = data['Start Time'].max().strftime('%Y-%m-%d')
                date_range = f"{min_date} to {max_date}"
            except:
                date_range = "Date parsing error"

        st.sidebar.metric("Total Records", f"{total_records:,}")
        st.sidebar.metric("Networks", unique_networks)
        st.sidebar.metric("Stations", unique_stations)
        if date_range:
            st.sidebar.write(f"**Date Range:** {date_range}")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📊 Data Analysis", "🔍 Predictions", "⚙️ Settings"])

    with tab1:
        display_overview_tab(data)

    with tab2:
        display_analysis_tab(data)

    with tab3:
        display_prediction_tab(data)

    with tab4:
        display_settings_tab()


def display_overview_tab(data):
    """Display the overview tab content"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🌍 About This Project")
        st.markdown("""
        This application analyzes seismic and geological data to understand Mars material composition. 
        The data includes network measurements from various stations with temporal velocity readings.

        **Key Features:**
        - 📈 Statistical analysis of geological data
        - 🗺️ Spatial analysis of measurement stations
        - ⏱️ Temporal analysis of velocity measurements
        - 🤖 Machine learning predictions for material components
        """)

        if data is not None and not data.empty:
            st.markdown("### 📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)

    with col2:
        st.markdown("### 📊 Data Quality")
        if data is not None and not data.empty:
            # Data quality metrics
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            quality_score = ((total_cells - missing_cells) / total_cells) * 100

            st.metric("Data Quality Score", f"{quality_score:.1f}%")
            st.metric("Missing Values", f"{missing_cells:,}")
            st.metric("Complete Records", f"{data.dropna().shape[0]:,}")

            # Column data types
            st.markdown("### 📝 Column Types")
            for col in data.columns[:8]:  # Show first 8 columns
                dtype = str(data[col].dtype)
                st.write(f"**{col}**: {dtype}")


def display_analysis_tab(data):
    """Display the data analysis tab content"""
    if data is None or data.empty:
        st.error("No data available for analysis.")
        return

    st.markdown("### 📊 Statistical Analysis")

    # Numerical columns analysis
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if numerical_cols:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Distribution Analysis")
            selected_col = st.selectbox("Select column for distribution:", numerical_cols)

            if selected_col:
                fig = px.histogram(data, x=selected_col, nbins=30,
                                   title=f"Distribution of {selected_col}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📋 Summary Statistics")
            st.dataframe(data[numerical_cols].describe(), use_container_width=True)

    # Velocity analysis if available
    if 'Velocity' in data.columns:
        st.markdown("### 🏃‍♂️ Velocity Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Time series plot
            if 'Time' in data.columns:
                fig = px.line(data.head(100), x='Time', y='Velocity',
                              title="Velocity Over Time (First 100 records)")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot by station
            if 'Station' in data.columns:
                unique_stations = data['Station'].unique()[:10]  # Limit to 10 stations
                filtered_data = data[data['Station'].isin(unique_stations)]
                fig = px.box(filtered_data, x='Station', y='Velocity',
                             title="Velocity Distribution by Station")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)


def display_prediction_tab(data):
    """Display the prediction tab content"""
    st.markdown("### 🔮 Material Prediction")
    st.info("🚧 Prediction functionality will be implemented in the next files (prediction.py)")

    # Placeholder for prediction interface
    if data is not None and not data.empty:
        st.markdown("#### 🎯 Prediction Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            if 'Velocity' in data.columns:
                velocity_input = st.number_input("Velocity",
                                                 min_value=float(data['Velocity'].min()),
                                                 max_value=float(data['Velocity'].max()),
                                                 value=float(data['Velocity'].mean()))

        with col2:
            if 'Delta' in data.columns:
                delta_input = st.number_input("Delta",
                                              min_value=float(data['Delta'].min()),
                                              max_value=float(data['Delta'].max()),
                                              value=float(data['Delta'].mean()))

        with col3:
            if 'Sampling' in data.columns:
                sampling_input = st.number_input("Sampling Rate",
                                                 min_value=float(data['Sampling'].min()),
                                                 max_value=float(data['Sampling'].max()),
                                                 value=float(data['Sampling'].mean()))

        if st.button("🔍 Predict Material Component", type="primary"):
            st.success("Prediction model will be implemented in upcoming files!")


def display_settings_tab():
    """Display the settings tab content"""
    st.markdown("### ⚙️ Application Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎨 Display Settings")
        theme = st.selectbox("Chart Theme", ["plotly", "seaborn", "ggplot2"])
        show_grid = st.checkbox("Show Grid Lines", value=True)
        animation = st.checkbox("Enable Animations", value=True)

    with col2:
        st.markdown("#### 📊 Analysis Settings")
        max_records = st.number_input("Max Records to Display", min_value=100, max_value=10000, value=1000)
        precision = st.selectbox("Decimal Precision", [2, 3, 4, 5], index=1)

    st.markdown("#### 🔄 Data Refresh")
    if st.button("🔄 Reload Data", type="secondary"):
        st.experimental_rerun()

    st.markdown("#### ℹ️ System Information")
    st.write(f"**Streamlit Version:** {st.__version__}")
    st.write(f"**Python Version:** {sys.version.split()[0]}")
    st.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()