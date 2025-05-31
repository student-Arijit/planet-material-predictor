import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_seismic_wave_visualization(velocity_data, time_data=None):
    """Create an interactive seismic wave visualization"""

    if time_data is None:
        time_data = np.arange(len(velocity_data))

    fig = go.Figure()

    # Add seismic wave trace
    fig.add_trace(go.Scatter(
        x=time_data,
        y=velocity_data,
        mode='lines',
        name='Seismic Wave',
        line=dict(color='#FF6B6B', width=2),
        hovertemplate='Time: %{x}<br>Velocity: %{y:.4f}<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Customize layout
    fig.update_layout(
        title='Seismic Wave Pattern',
        xaxis_title='Time',
        yaxis_title='Velocity',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_material_distribution_chart(material_counts):
    """Create a pie chart showing material distribution"""

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    fig = go.Figure(data=[go.Pie(
        labels=material_counts.index,
        values=material_counts.values,
        hole=0.4,
        marker_colors=colors[:len(material_counts)],
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Material Type Distribution',
        template='plotly_white',
        height=400
    )

    return fig


def create_velocity_histogram(df):
    """Create histogram of velocity distribution by material type"""

    fig = px.histogram(
        df,
        x='velocity',
        color='material_type',
        nbins=50,
        title='Velocity Distribution by Material Type',
        labels={'velocity': 'Velocity', 'count': 'Frequency'},
        template='plotly_white'
    )

    fig.update_layout(height=400)
    return fig


def create_station_analysis_chart(df):
    """Create analysis chart by station"""

    station_summary = df.groupby('station').agg({
        'velocity': ['mean', 'std', 'count'],
        'material_type': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
    }).round(4)

    station_summary.columns = ['velocity_mean', 'velocity_std', 'count', 'dominant_material']
    station_summary = station_summary.reset_index()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean Velocity by Station', 'Data Count by Station',
                        'Velocity Std by Station', 'Dominant Material by Station'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Mean velocity
    fig.add_trace(
        go.Bar(x=station_summary['station'], y=station_summary['velocity_mean'],
               name='Mean Velocity', marker_color='#FF6B6B'),
        row=1, col=1
    )

    # Count
    fig.add_trace(
        go.Bar(x=station_summary['station'], y=station_summary['count'],
               name='Count', marker_color='#4ECDC4'),
        row=1, col=2
    )

    # Standard deviation
    fig.add_trace(
        go.Bar(x=station_summary['station'], y=station_summary['velocity_std'],
               name='Velocity Std', marker_color='#45B7D1'),
        row=2, col=1
    )

    # Dominant material (categorical)
    material_encoded = pd.Categorical(station_summary['dominant_material']).codes
    fig.add_trace(
        go.Bar(x=station_summary['station'], y=material_encoded,
               name='Dominant Material', marker_color='#96CEB4'),
        row=2, col=2
    )

    fig.update_layout(height=600, template='plotly_white', showlegend=False)
    return fig


def create_prediction_confidence_gauge(confidence):
    """Create a gauge chart for prediction confidence"""

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence (%)"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_feature_importance_chart(importance_df):
    """Create horizontal bar chart for feature importance"""

    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker_color='#FF6B6B',
        hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white',
        height=max(400, len(importance_df) * 20)
    )

    return fig


def display_prediction_results(prediction_result):
    """Display prediction results in a formatted way"""

    st.subheader("🔍 Prediction Results")

    # Main prediction
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Material",
            value=prediction_result['predicted_material'],
            delta=f"{prediction_result['confidence']:.2%} confidence"
        )

    with col2:
        # Confidence gauge
        fig_gauge = create_prediction_confidence_gauge(prediction_result['confidence'])
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Probability breakdown
    st.subheader("📊 Probability Breakdown")

    probs_df = pd.DataFrame(
        list(prediction_result['probabilities'].items()),
        columns=['Material', 'Probability']
    ).sort_values('Probability', ascending=False)

    fig_probs = px.bar(
        probs_df,
        x='Material',
        y='Probability',
        title='Material Type Probabilities',
        color='Probability',
        color_continuous_scale='viridis'
    )

    fig_probs.update_layout(template='plotly_white')
    st.plotly_chart(fig_probs, use_container_width=True)

    # Detailed probabilities table
    st.subheader("📋 Detailed Probabilities")

    probs_df['Probability %'] = (probs_df['Probability'] * 100).round(2)
    st.dataframe(probs_df, use_container_width=True)


def create_data_quality_metrics(df):
    """Create data quality visualization"""

    # Missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100

    quality_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Percentage', ascending=False)

    fig = px.bar(
        quality_df,
        x='Column',
        y='Missing_Percentage',
        title='Data Quality: Missing Values by Column',
        labels={'Missing_Percentage': 'Missing %'},
        color='Missing_Percentage',
        color_continuous_scale='reds'
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-45,
        height=400
    )

    return fig


def display_model_performance(performance_metrics):
    """Display model performance metrics"""

    st.subheader("🎯 Model Performance")

    # Overall accuracy
    st.metric("Overall Accuracy", f"{performance_metrics['accuracy']:.2%}")

    # Classification report
    st.subheader("📊 Classification Report")

    report_df = pd.DataFrame(performance_metrics['classification_report']).transpose()
    report_df = report_df.round(3)
    st.dataframe(report_df, use_container_width=True)

    # Confusion matrix heatmap
    st.subheader("🔥 Confusion Matrix")

    fig_cm = px.imshow(
        performance_metrics['confusion_matrix'],
        title='Confusion Matrix',
        labels=dict(x="Predicted", y="Actual"),
        color_continuous_scale='Blues'
    )

    st.plotly_chart(fig_cm, use_container_width=True)


def create_input_form():
    """Create input form for manual predictions"""

    st.subheader("🎛️ Manual Input for Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            velocity = st.number_input(
                "Velocity",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.001,
                format="%.6f",
                help="Seismic velocity measurement"
            )

            sampling = st.number_input(
                "Sampling Rate",
                min_value=1,
                max_value=100,
                value=20,
                help="Sampling rate in Hz"
            )

            delta = st.number_input(
                "Delta",
                min_value=0.001,
                max_value=1.0,
                value=0.05,
                step=0.001,
                format="%.3f",
                help="Time interval between samples"
            )

        with col2:
            num_calibrations = st.number_input(
                "Number of Calibrations",
                min_value=0,
                max_value=100000,
                value=72000,
                help="Number of calibration measurements"
            )

            location = st.selectbox(
                "Location",
                options=[0, 1, 2, "Unknown"],
                index=1,
                help="Measurement location code"
            )

            station = st.selectbox(
                "Station",
                options=["ELYSE", "XB", "Other"],
                index=0,
                help="Recording station"
            )

        submitted = st.form_submit_button("🚀 Predict Material")

        if submitted:
            input_data = {
                'velocity': velocity,
                'sampling': sampling,
                'delta': delta,
                'num_calibrations': num_calibrations,
                'location': location if location != "Unknown" else 0,
                'station': station
            }

            return input_data

    return None


def display_data_statistics(df):
    """Display comprehensive data statistics"""

    st.subheader("📈 Data Statistics")

    # Basic info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Features", len(df.columns))

    with col3:
        st.metric("Stations", df['station'].nunique() if 'station' in df.columns else "N/A")

    with col4:
        st.metric("Material Types", df['material_type'].nunique() if 'material_type' in df.columns else "N/A")

    # Statistical summary
    st.subheader("📊 Statistical Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    st.dataframe(df[numeric_cols].describe().round(4), use_container_width=True)