import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from home import home_page
from training import training_page
from classification import classification_page
st.set_page_config(
    page_title="Enhanced ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def convert_to_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable format"""
    if isinstance(obj, pd.DataFrame):
        df_copy = obj.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].astype(str)
            elif pd.api.types.is_numeric_dtype(df_copy[col]):
                if df_copy[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                    df_copy[col] = df_copy[col].astype(int)
                elif df_copy[col].dtype in ['float64', 'float32', 'float16']:
                    df_copy[col] = df_copy[col].astype(float)
        return df_copy
    elif isinstance(obj, pd.Series):
        series_copy = obj.copy()
        if series_copy.dtype == 'object':
            series_copy = series_copy.astype(str)
        elif pd.api.types.is_numeric_dtype(series_copy):
            if series_copy.dtype in ['int64', 'int32', 'int16', 'int8']:
                series_copy = series_copy.astype(int)
            elif series_copy.dtype in ['float64', 'float32', 'float16']:
                series_copy = series_copy.astype(float)
        return series_copy
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

def safe_session_state_set(key, value):
    """Safely set session state with serializable conversion"""
    try:
        converted_value = convert_to_serializable(value)
        st.session_state[key] = converted_value
    except Exception as e:
        st.error(f"Error storing {key} in session state: {str(e)}")
        st.session_state[key] = None

def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'working_data' not in st.session_state:
        st.session_state.working_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'labeled_data' not in st.session_state:
        st.session_state.labeled_data = None
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = []
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_evaluation' not in st.session_state:
        st.session_state.model_evaluation = None
    if 'training_step' not in st.session_state:
        st.session_state.training_step = 1
    if 'classification_model' not in st.session_state:
        st.session_state.classification_model = None
    if 'classification_model_package' not in st.session_state:
        st.session_state.classification_model_package = None
    if 'model_features' not in st.session_state:
        st.session_state.model_features = []
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}
    if 'target_label_encoder' not in st.session_state:
        st.session_state.target_label_encoder = None
    if 'target_label_mapping' not in st.session_state:
        st.session_state.target_label_mapping = None

def main():
    """Main application function"""
    initialize_session_state()
    if st.session_state.current_page == 'home':
        home_page()
    elif st.session_state.current_page == 'training':
        training_page()
    elif st.session_state.current_page == 'classification':
        classification_page()

if __name__ == "__main__":
    main()
