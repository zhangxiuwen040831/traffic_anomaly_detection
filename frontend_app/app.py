"""
Traffic Anomaly Detection - Streamlit Frontend
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from deploy_bundle.inference.model_loader import TrafficAnomalyModel


@st.cache_resource
def load_model():
    """Load the anomaly detection model"""
    model_dir = project_root / "deploy_bundle" / "model"
    return TrafficAnomalyModel(str(model_dir))


def load_demo_data():
    """Load demo data from processed directory"""
    data_dir = project_root / "data" / "processed"
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    return X_test, y_test


def main():
    st.set_page_config(
        page_title="Traffic Anomaly Detection",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.title("🚦 Network Traffic Anomaly Detection")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Settings")
    
    # Model info
    st.sidebar.header("Model Information")
    try:
        model = load_model()
        st.sidebar.success(f"✅ Model loaded: {model.model_config['model_name']}")
        st.sidebar.info(f"📊 Dataset: CIC-IDS2017")
        st.sidebar.info(f"🎯 Input dim: {model.preprocessing['input_dim']}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return

    # Threshold selection
    st.sidebar.header("Threshold Strategy")
    threshold_method = st.sidebar.selectbox(
        "Select threshold method:",
        ["f1_optimal", "youden", "pr_optimal"],
        index=0
    )

    # Get threshold info
    threshold_entry = next(
        (t for t in model.threshold_config if t["threshold_method"] == threshold_method),
        model.threshold_config[0]
    )

    st.sidebar.metric("Threshold", f"{threshold_entry.get('test_threshold', threshold_entry.get('threshold', 0)):.4f}")
    st.sidebar.metric("ROC-AUC", f"{threshold_entry['roc_auc']:.4f}")
    st.sidebar.metric("PR-AUC", f"{threshold_entry['pr_auc']:.4f}")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Input Data")
        
        # Data source selection
        data_source = st.radio(
            "Select data source:",
            ["Use demo data", "Upload CSV file"],
            horizontal=True
        )

        features = None
        labels = None

        if data_source == "Use demo data":
            try:
                X_test, y_test = load_demo_data()
                features = X_test
                labels = y_test
                st.success(f"✅ Loaded {len(features)} samples from demo data")
            except Exception as e:
                st.error(f"❌ Error loading demo data: {e}")
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    # Assume all columns except last are features
                    features = df.iloc[:, :-1].values
                    if df.shape[1] > 1:
                        labels = df.iloc[:, -1].values
                    st.success(f"✅ Loaded {len(features)} samples")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")

    with col2:
        st.subheader("🔍 Inference")
        
        if features is not None:
            if st.button("Run Inference", type="primary"):
                with st.spinner("Running inference..."):
                    try:
                        # First we need to fit the scorer with some normal data
                        # For demo purposes, let's use the first portion as normal
                        normal_size = min(100, len(features) // 2)
                        model.fit_scorer(features[:normal_size])
                        
                        # Run prediction
                        result = model.predict(features, threshold_method=threshold_method)
                        
                        # Store in session state
                        st.session_state['result'] = result
                        st.session_state['features'] = features
                        st.session_state['labels'] = labels
                        
                        st.success("✅ Inference completed!")
                    except Exception as e:
                        st.error(f"❌ Error during inference: {e}")
        else:
            st.info("👈 Please select or upload data first")

    # Results section
    if 'result' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Results")
        
        result = st.session_state['result']
        features = st.session_state['features']
        labels = st.session_state['labels']
        
        # Summary stats
        total = len(features)
        normal_count = np.sum(result['predictions'] == 0)
        anomaly_count = np.sum(result['predictions'] == 1)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", total)
        col2.metric("Normal", normal_count, f"{normal_count/total*100:.1f}%")
        col3.metric("Anomaly", anomaly_count, f"{anomaly_count/total*100:.1f}%")
        col4.metric("Threshold", f"{result['threshold']:.4f}")

        # Results table
        st.subheader("📋 Detailed Results")
        
        results_df = pd.DataFrame({
            "Sample ID": range(1, total + 1),
            "Anomaly Score": result['scores'].round(4),
            "Prediction": ["Anomaly" if p == 1 else "Normal" for p in result['predictions']]
        })
        
        if labels is not None:
            results_df["True Label"] = ["Anomaly" if l == 1 else "Normal" for l in labels]
            results_df["Correct"] = results_df["Prediction"] == results_df["True Label"]
        
        st.dataframe(results_df, use_container_width=True)

        # Visualizations
        st.markdown("---")
        st.subheader("📈 Visualizations")
        
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Score distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=result['scores'], bins=50, kde=True, ax=ax)
            ax.axvline(x=result['threshold'], color='red', linestyle='--', label='Threshold')
            ax.set_title('Anomaly Score Distribution')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Count')
            ax.legend()
            st.pyplot(fig)

        with fig_col2:
            # Predictions pie chart
            fig, ax = plt.subplots(figsize=(8, 5))
            prediction_counts = [normal_count, anomaly_count]
            colors = ['#2ecc71', '#e74c3c']
            ax.pie(prediction_counts, labels=['Normal', 'Anomaly'], colors=colors, 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Prediction Distribution')
            st.pyplot(fig)

        # Show model figures
        st.markdown("---")
        st.subheader("📊 Model Performance (Training)")
        
        fig_path = project_root / "deploy_bundle" / "figures"
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            if (fig_path / "roc.png").exists():
                st.image(str(fig_path / "roc.png"), caption="ROC Curve", use_container_width=True)
        
        with perf_col2:
            if (fig_path / "pr.png").exists():
                st.image(str(fig_path / "pr.png"), caption="PR Curve", use_container_width=True)
        
        if (fig_path / "confusion.png").exists():
            st.image(str(fig_path / "confusion.png"), caption="Confusion Matrix", use_container_width=True)


if __name__ == "__main__":
    main()
