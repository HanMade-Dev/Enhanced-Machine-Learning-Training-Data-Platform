import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import (
    load_data, handle_missing_values, normalize_data, enhanced_data_preview,
    detect_and_handle_outliers, advanced_feature_engineering, show_preprocessing_summary
)
from feature_selection import feature_selection_page
from model_training import (
    get_available_models, train_model, enhanced_cross_validation, 
    hyperparameter_tuning, compare_models, get_model_recommendations
)
from model_evaluation import evaluate_model, create_learning_curves
from utils import save_model_for_download, generate_report, export_model_summary
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

def training_page():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("🏠 Home", key="training_back_home", type="secondary"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    st.title("📊 Advanced Training Data Platform")
    st.markdown("### *Enhanced ML Training dengan Automated Features*")
    
    if 'training_step' not in st.session_state:
        st.session_state.training_step = 1
    
    progress_steps = [
        {"name": "Upload Data", "icon": "📤", "desc": "Smart data upload & preview"},
        {"name": "Preprocessing", "icon": "⚙️", "desc": "Advanced cleaning & engineering"}, 
        {"name": "Feature Selection", "icon": "🎯", "desc": "Automated & manual selection"},
        {"name": "Model Training", "icon": "🤖", "desc": "Training & hyperparameter tuning"},
        {"name": "Evaluation", "icon": "📈", "desc": "Advanced metrics & visualization"}
    ]
    current_step = st.session_state.training_step
    
    st.markdown("### 🚀 Training Workflow Progress")
    
    progress_cols = st.columns(5)
    for i, step in enumerate(progress_steps, 1):
        with progress_cols[i-1]:
            if i < current_step:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background-color: #d4edda; border-radius: 10px; border-left: 4px solid #28a745;">
                    <div style="font-size: 24px;">{step['icon']}</div>
                    <div style="font-weight: bold; color: #155724;">✅ {step['name']}</div>
                    <div style="font-size: 12px; color: #155724;">{step['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
            elif i == current_step:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background-color: #fff3cd; border-radius: 10px; border-left: 4px solid #ffc107;">
                    <div style="font-size: 24px;">{step['icon']}</div>
                    <div style="font-weight: bold; color: #856404;">📍 {step['name']}</div>
                    <div style="font-size: 12px; color: #856404;">{step['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #6c757d;">
                    <div style="font-size: 24px; opacity: 0.5;">{step['icon']}</div>
                    <div style="font-weight: bold; color: #6c757d;">⏳ {step['name']}</div>
                    <div style="font-size: 12px; color: #6c757d;">{step['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if current_step == 1:
        enhanced_data_upload_step()
    elif current_step == 2:
        enhanced_data_preprocessing_step()
    elif current_step == 3:
        enhanced_feature_selection_step()
    elif current_step == 4:
        enhanced_model_training_step()
    elif current_step == 5:
        enhanced_model_evaluation_step()

def enhanced_data_upload_step():
    """Enhanced data upload with automatic preview and column selection"""
    st.header("1. 📤 Advanced Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📁 Select CSV or Excel files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload multiple files that will be intelligently combined"
        )
    
    with col2:
        st.markdown("### 💡 Upload Tips")
        st.info("""
        **Supported formats:**
        - CSV (auto-delimiter detection)
        - Excel (.xlsx, .xls)
        
        **Smart features:**
        - Auto data type detection
        - Column preview & selection
        - Duplicate detection
        - Format validation
        """)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        
        st.subheader("📊 File Analysis")
        
        file_info = []
        total_size = 0
        
        for file in uploaded_files:
            size_kb = file.size / 1024
            total_size += size_kb
            file_info.append({
                'File Name': file.name,
                'Size (KB)': f"{size_kb:.1f}",
                'Type': file.type if hasattr(file, 'type') else file.name.split('.')[-1].upper()
            })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(uploaded_files))
        with col2:
            st.metric("Total Size", f"{total_size:.1f} KB")
        with col3:
            st.metric("File Types", len(set([info['Type'] for info in file_info])))
        
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)
        

        if st.button("🔍 Load & Preview Data", type="primary", use_container_width=True):
            with st.spinner("Loading and analyzing data..."):
                try:

                    data = load_data(uploaded_files, header_row=0, headers_not_in_first_row=False)
                    
                    if data is not None:
                        st.session_state.raw_data = data
                        st.session_state.preview_data = data  
                        
                        st.success("✅ Data loaded successfully!")
                        

                        display_data_preview_and_analysis(data)
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    st.error("💡 Please check your file format and configuration.")
        

        if hasattr(st.session_state, 'preview_data') and st.session_state.preview_data is not None:
            st.markdown("---")
            st.subheader("⚙️ Advanced Configuration")
            st.info("Configure advanced options and apply them to update the data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_custom_header = st.checkbox(
                    "Headers not in first row",
                    help="Check if column headers are not in the first row"
                )
                
                header_row = 0
                if use_custom_header:
                    header_row = st.number_input(
                        "Header row (0-indexed):",
                        min_value=0,
                        max_value=10,
                        value=0
                    )
            
            with col2:
                use_specific_columns = st.checkbox(
                    "Gunakan Kolom Tertentu",
                    help="Pilih kolom spesifik yang ingin digunakan untuk pelatihan"
                )
            

            show_config_form = use_custom_header or use_specific_columns
            selected_columns = None
            
            if show_config_form:
                st.markdown("#### 🔧 Configuration Settings")
                
                if use_specific_columns:
                    current_data = st.session_state.preview_data
                    all_columns = current_data.columns.tolist()
                    
                    st.write("**Column Selection:**")
                    selected_columns = st.multiselect(
                        "Select columns to use:",
                        all_columns,
                        default=all_columns,
                        help="Choose which columns to include in the dataset"
                    )
                
                if st.button("🔧 Apply Configuration & Update Data", type="secondary", use_container_width=True):
                    with st.spinner("Applying configuration and reloading data..."):
                        try:

                            updated_data = load_data(
                                uploaded_files, 
                                header_row=header_row, 
                                headers_not_in_first_row=use_custom_header
                            )
                            
                            if updated_data is not None:

                                if use_specific_columns and selected_columns:
                                    updated_data = updated_data[selected_columns]
                                

                                st.session_state.raw_data = updated_data
                                st.session_state.preview_data = updated_data
                                
                                st.success("✅ Configuration applied successfully!")
                                

                                display_data_preview_and_analysis(updated_data)
                                
                        except Exception as e:
                            st.error(f"❌ Error applying configuration: {str(e)}")


    if hasattr(st.session_state, 'raw_data') and st.session_state.raw_data is not None:
        st.markdown("---")
        st.info("✅ Data successfully loaded and analyzed! Ready for preprocessing.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("♻️ Reload Data", key="reload_data"):

                if 'raw_data' in st.session_state:
                    del st.session_state.raw_data
                if 'preview_data' in st.session_state:
                    del st.session_state.preview_data
                st.rerun()
        
        with col2:
            if st.button("➡️ Continue to Preprocessing", key="proceed_to_preprocessing", type="primary"):
                st.session_state.training_step = 2
                st.rerun()

def display_data_preview_and_analysis(data):
    """Display data preview and analysis in a organized way"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success(f"**Rows:** {data.shape[0]:,}")
    with col2:
        st.success(f"**Columns:** {data.shape[1]}")
    with col3:
        st.success(f"**Memory:** {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        st.success(f"**Missing:** {missing_pct:.1f}%")

    st.subheader("📋 Data Preview & Analysis")
    enhanced_data_preview(data)

def show_data_quality_report(data):
    """Show comprehensive data quality report"""
    col1, col2 = st.columns(2)
    
    with col1:
        missing_stats = data.isnull().sum()
        if missing_stats.sum() > 0:
            st.warning(f"⚠️ Found missing values in {(missing_stats > 0).sum()} columns")
            missing_df = missing_stats[missing_stats > 0].to_frame("Missing Count")
            missing_df["Percentage"] = (missing_df["Missing Count"] / len(data) * 100).round(2)
            st.dataframe(missing_df)
        else:
            st.success("✅ No missing values detected")
    
    with col2:
        dtype_counts = data.dtypes.value_counts()
        st.write("**Data Types Distribution:**")
        for dtype, count in dtype_counts.items():
            st.write(f"- {dtype}: {count} columns")
        
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2
        if memory_usage > 100:
            st.warning(f"⚠️ Large dataset: {memory_usage:.1f} MB")
        else:
            st.success(f"✅ Memory usage: {memory_usage:.1f} MB")

def enhanced_data_preprocessing_step():
    """Enhanced preprocessing with modular optional steps"""
    st.header("2. ⚙️ Advanced Data Preprocessing")
    
    if not hasattr(st.session_state, 'raw_data') or st.session_state.raw_data is None:
        st.warning("⚠️ No data loaded. Please return to step 1.")
        if st.button("⬅️ Back to Upload", key="back_to_upload_from_preprocessing"):
            st.session_state.training_step = 1
            st.rerun()
        return
    
    data = st.session_state.raw_data

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = data.copy()

    st.subheader("📊 Original Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        string_cols = data.select_dtypes(include=['object']).shape[1]
        st.metric("String Columns", string_cols)
    with col4:
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")

    string_columns = data.select_dtypes(include=['object']).columns.tolist()
    if string_columns:
        st.info(f"🏷️ **String columns detected**: {', '.join(string_columns[:5])}{'...' if len(string_columns) > 5 else ''}")
        st.write("These columns will be automatically encoded if used as target or features.")
    
    st.markdown("---")

    st.subheader("🔧 Preprocessing Pipeline Configuration")
    

    step1_col1, step1_col2 = st.columns([1, 3])
    with step1_col1:
        enable_missing = st.checkbox("✅ Apply Missing Value Handling", key="enable_missing")
    with step1_col2:
        if enable_missing:
            st.session_state.show_missing_config = True
    
    if enable_missing and st.session_state.get('show_missing_config', False):
        with st.expander("🧹 Missing Values Configuration", expanded=True):
            missing_config = configure_missing_values_step(data)
            if st.button("▶️ Apply Missing Value Handling", key="apply_missing"):
                st.session_state.processed_data = apply_missing_values_step(st.session_state.processed_data, missing_config)
                st.session_state.show_missing_config = False
                st.success("✅ Missing value handling applied!")
                st.rerun()
    

    step2_col1, step2_col2 = st.columns([1, 3])
    with step2_col1:
        enable_outliers = st.checkbox("✅ Apply Outlier Detection", key="enable_outliers")
    with step2_col2:
        if enable_outliers:
            st.session_state.show_outlier_config = True
    
    if enable_outliers and st.session_state.get('show_outlier_config', False):
        with st.expander("📊 Outlier Detection Configuration", expanded=True):
            outlier_config = configure_outlier_detection_step(st.session_state.processed_data)
            if st.button("▶️ Apply Outlier Detection", key="apply_outliers"):
                st.session_state.processed_data, outlier_info = apply_outlier_detection_step(st.session_state.processed_data, outlier_config)
                st.session_state.outlier_info = outlier_info
                st.session_state.show_outlier_config = False
                st.success("✅ Outlier detection applied!")
                st.rerun()
    

    step3_col1, step3_col2 = st.columns([1, 3])
    with step3_col1:
        enable_feature_eng = st.checkbox("✅ Apply Feature Engineering", key="enable_feature_eng")
    with step3_col2:
        if enable_feature_eng:
            st.session_state.show_feature_eng_config = True
    
    if enable_feature_eng and st.session_state.get('show_feature_eng_config', False):
        with st.expander("🔧 Feature Engineering Configuration", expanded=True):
            feature_eng_config = configure_feature_engineering_step(st.session_state.processed_data)
            if st.button("▶️ Apply Feature Engineering", key="apply_feature_eng"):
                st.session_state.processed_data, new_features = apply_feature_engineering_step(st.session_state.processed_data, feature_eng_config)
                st.session_state.new_features = new_features
                st.session_state.show_feature_eng_config = False
                st.success("✅ Feature engineering applied!")
                st.rerun()
    

    step4_col1, step4_col2 = st.columns([1, 3])
    with step4_col1:
        enable_normalization = st.checkbox("✅ Apply Normalization", key="enable_normalization")
    with step4_col2:
        if enable_normalization:
            st.session_state.show_normalization_config = True

    if enable_normalization and st.session_state.get('show_normalization_config', False):
        with st.expander("📏 Normalization Configuration", expanded=True):
            normalization_config = configure_normalization_step(st.session_state.processed_data)
            if st.button("▶️ Apply Normalization", key="apply_normalization"):
                st.session_state.processed_data = apply_normalization_step(st.session_state.processed_data, normalization_config)
                st.session_state.show_normalization_config = False
                st.success("✅ Normalization applied!")
                st.rerun()

    if 'processed_data' in st.session_state:
        st.markdown("---")
        st.subheader("📋 Current Processed Data Preview")
        st.dataframe(st.session_state.processed_data.head(), use_container_width=True)

        original_shape = data.shape
        processed_shape = st.session_state.processed_data.shape
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", processed_shape[0], processed_shape[0] - original_shape[0])
        with col2:
            st.metric("Columns", processed_shape[1], processed_shape[1] - original_shape[1])
        with col3:
            missing_pct = (st.session_state.processed_data.isnull().sum().sum() / (processed_shape[0] * processed_shape[1])) * 100
            st.metric("Missing %", f"{missing_pct:.1f}%")

    st.markdown("---")
    st.subheader("🔬 Final Validation")
    
    if st.button("🔍 Validate Processed Data", key="validate_data"):
        validation_results = validate_processed_data(st.session_state.processed_data)
        display_validation_results(validation_results)
        
        if validation_results['is_valid']:
            st.session_state.cleaned_data = st.session_state.processed_data
            st.success("✅ Data validation passed! Ready for feature selection.")
        else:
            st.error("❌ Data validation failed. Please fix the issues above.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Upload", key="back_to_upload"):
            st.session_state.training_step = 1
            st.rerun()
    
    with col2:
        if hasattr(st.session_state, 'cleaned_data') and st.session_state.cleaned_data is not None:
            if st.button("➡️ Continue to Feature Selection", key="proceed_to_feature_selection", type="primary"):
                st.session_state.training_step = 3
                st.rerun()


def configure_missing_values_step(data):
    """Configure missing values handling"""
    st.write("**🧹 Missing Values Configuration**")
    
    missing_info = data.isnull().sum()
    
    if missing_info.sum() > 0:
        missing_cols = missing_info[missing_info > 0]
        
        fig = px.bar(
            x=missing_cols.values, 
            y=missing_cols.index,
            orientation='h',
            title="Missing Values by Column",
            labels={'x': 'Missing Count', 'y': 'Columns'}
        )
        st.plotly_chart(fig, use_container_width=True)

        strategy = st.selectbox(
            "Choose missing values strategy:",
            [
                "Remove rows with any missing values",
                "Remove rows with missing values in selected columns only",
                "Imputation (fill missing values)"
            ]
        )
        
        config = {"strategy": strategy}
        
        if strategy == "Remove rows with missing values in selected columns only":
            columns_with_missing = missing_cols.index.tolist()
            selected_columns = st.multiselect(
                "Select columns to check for missing values:",
                columns_with_missing,
                default=columns_with_missing
            )
            config["selected_columns"] = selected_columns
        
        elif "imputation" in strategy.lower():
            col1, col2 = st.columns(2)
            with col1:
                numeric_method = st.selectbox(
                    "Numeric columns method:",
                    ["Mean", "Median", "Mode", "Zero"]
                )
                config["numeric_method"] = numeric_method
            
            with col2:
                categorical_method = st.selectbox(
                    "Categorical columns method:",
                    ["Mode", "Constant value"]
                )
                config["categorical_method"] = categorical_method
                
                if categorical_method == "Constant value":
                    categorical_fill_value = st.text_input("Fill value:", "Unknown")
                    config["categorical_fill_value"] = categorical_fill_value
        
        return config
    else:
        st.success("✅ No missing values found in the dataset!")
        return {"strategy": "none"}

def apply_missing_values_step(data, config):
    """Apply missing values handling"""
    df = data.copy()
    
    if config["strategy"] == "Remove rows with any missing values":
        df = df.dropna()
    
    elif config["strategy"] == "Remove rows with missing values in selected columns only":
        selected_columns = config.get("selected_columns", [])
        if selected_columns:
            df = df.dropna(subset=selected_columns)
    
    elif "imputation" in config["strategy"].lower():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if config["numeric_method"] == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif config["numeric_method"] == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif config["numeric_method"] == "Zero":
                    df[col].fillna(0, inplace=True)
                elif config["numeric_method"] == "Mode":
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
        
        for col in categorical_cols:
            if df[col].isnull().any():
                if config["categorical_method"] == "Mode":
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                    else:
                        df[col].fillna("Unknown", inplace=True)
                elif config["categorical_method"] == "Constant value":
                    df[col].fillna(config.get("categorical_fill_value", "Unknown"), inplace=True)
    
    return df

def configure_outlier_detection_step(data):
    """Configure outlier detection"""
    st.write("**📊 Outlier Detection Configuration**")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        st.info("ℹ️ No numeric columns found for outlier detection.")
        return {"method": "none"}
    
    detection_method = st.selectbox(
        "Choose outlier detection method:",
        ["IQR", "Z-Score", "Modified Z-Score"]
    )

    treatment_action = st.selectbox(
        "Choose treatment action:",
        ["mark", "remove", "clip"]
    )
    
    config = {
        "method": detection_method,
        "action": treatment_action
    }
    
    if st.button("🔍 Preview Outlier Detection", key="preview_outliers"):
        with st.spinner("Detecting outliers..."):
            try:
                _, outlier_info = detect_and_handle_outliers(data, method=detection_method, action="mark")
                
                if outlier_info:
                    st.subheader("📈 Outlier Detection Results")
                    
                    outlier_summary = []
                    for col, info in outlier_info.items():
                        if info['count'] > 0:
                            outlier_summary.append({
                                'Column': col,
                                'Outliers': info['count'],
                                'Percentage': f"{info['percentage']:.2f}%"
                            })
                    
                    if outlier_summary:
                        st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)
                    else:
                        st.success("✅ No outliers detected!")
                
            except Exception as e:
                st.error(f"Error detecting outliers: {str(e)}")
    
    return config

def apply_outlier_detection_step(data, config):
    """Apply outlier detection"""
    if config["method"] == "none":
        return data, {}
    
    return detect_and_handle_outliers(data, method=config["method"], action=config["action"])

def configure_feature_engineering_step(data):
    """Configure feature engineering with advanced options"""
    st.write("**🔙 Feature Engineering Configuration**")
    
    config = {}
    
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_columns:
        st.write("**🏷️ Categorical Encoding**")
        
        encoding_method = st.selectbox(
            "Choose encoding method:",
            ["Auto (based on cardinality)", "One-Hot Encoding", "Label Encoding", "Target Encoding"]
        )
        config["encoding_method"] = encoding_method
        
        if encoding_method == "Auto (based on cardinality)":
            cardinality_threshold = st.slider(
                "Cardinality threshold for one-hot encoding:",
                2, 20, 10,
                help="Columns with fewer unique values will use one-hot encoding"
            )
            config["cardinality_threshold"] = cardinality_threshold
    

    st.write("**➕ Interaction Features**")
    
    create_interactions = st.checkbox(
        "Create polynomial and interaction features",
        help="Create polynomial and interaction features for numeric columns"
    )
    config["create_interactions"] = create_interactions
    
    if create_interactions:
        interaction_degree = st.slider("Polynomial degree:", 2, 3, 2)
        config["interaction_degree"] = interaction_degree
        

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_interaction_cols = st.multiselect(
                "Select columns for interaction features:",
                numeric_columns,
                default=numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns
            )
            config["interaction_columns"] = selected_interaction_cols
    
    return config

def apply_feature_engineering_step(data, config):
    """Apply feature engineering with advanced options"""
    df = data.copy()
    new_features = []
    
    encoding_method = config.get("encoding_method", "Auto (based on cardinality)")
    categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns
    
    for col in categorical_columns:
        if f'{col}_encoded' in df.columns:
            continue
            
        unique_count = df[col].nunique()
        
        if encoding_method == "Auto (based on cardinality)":
            threshold = config.get("cardinality_threshold", 10)
            if unique_count <= threshold and unique_count > 1:

                try:
                    df[col] = df[col].fillna('Unknown')
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                    df = pd.concat([df, dummies], axis=1)
                    new_features.extend(dummies.columns.tolist())
                    df = df.drop(columns=[col])
                except Exception as e:
                    st.error(f"Error in one-hot encoding for {col}: {str(e)}")
            else:

                try:
                    df[col] = df[col].fillna('Unknown')
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    new_features.append(f'{col}_encoded')
                    df = df.drop(columns=[col])
                except Exception as e:
                    st.error(f"Error in label encoding for {col}: {str(e)}")
    

    if config.get("create_interactions", False):
        from sklearn.preprocessing import PolynomialFeatures
        
        interaction_columns = config.get("interaction_columns", [])
        degree = config.get("interaction_degree", 2)
        
        if interaction_columns:
            try:

                interaction_data = df[interaction_columns]
                

                poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
                poly_features = poly.fit_transform(interaction_data)
                

                feature_names = poly.get_feature_names_out(interaction_columns)
                

                poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
                

                original_features = interaction_columns
                poly_df = poly_df.drop(columns=original_features, errors='ignore')
                

                df = pd.concat([df, poly_df], axis=1)
                new_features.extend(poly_df.columns.tolist())
                
            except Exception as e:
                st.error(f"Error creating interaction features: {str(e)}")
    

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
            except:
                df = df.drop(columns=[col])
                if col in new_features:
                    new_features.remove(col)
    
    return df, new_features

def configure_normalization_step(data):
    """Configure normalization"""
    st.write("**📏 Normalization Configuration**")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        st.info("ℹ️ No numeric columns found for normalization.")
        return {"method": "none"}
    
    normalization_method = st.selectbox(
        "Choose normalization method:",
        ["Min-Max Scaling", "Standard Scaling", "Robust Scaling", "Unit Vector Scaling"]
    )
    

    selected_numeric_cols = st.multiselect(
        "Select columns to normalize:",
        numeric_columns,
        default=numeric_columns,
        help="Choose which numeric columns to normalize"
    )
    
    config = {
        "method": normalization_method,
        "selected_columns": selected_numeric_cols
    }
    
    return config

def apply_normalization_step(data, config):
    """Apply normalization"""
    if config["method"] == "none":
        return data
    
    df = data.copy()
    selected_columns = config.get("selected_columns", [])
    
    if not selected_columns:
        return df
    
    method = config["method"]
    
    if method == "Min-Max Scaling":
        scaler = MinMaxScaler()
    elif method == "Standard Scaling":
        scaler = StandardScaler()
    elif method == "Robust Scaling":
        scaler = RobustScaler()
    else:
        return df
    
    df[selected_columns] = scaler.fit_transform(df[selected_columns])
    
    return df

def validate_processed_data(data):
    """Validate processed data"""
    validation_results = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "info": []
    }
    

    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        validation_results["warnings"].append(f"Found {missing_count} missing values")
    else:
        validation_results["info"].append("No missing values found")
    

    non_numeric_cols = []
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        validation_results["issues"].append(f"Non-numeric columns found: {non_numeric_cols}")
        validation_results["is_valid"] = False
    else:
        validation_results["info"].append("All columns are numeric")
    

    inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        validation_results["warnings"].append(f"Found {inf_count} infinite values")
    else:
        validation_results["info"].append("No infinite values found")
    

    if data.shape[0] == 0:
        validation_results["issues"].append("Dataset is empty (0 rows)")
        validation_results["is_valid"] = False
    elif data.shape[1] == 0:
        validation_results["issues"].append("Dataset has no columns")
        validation_results["is_valid"] = False
    else:
        validation_results["info"].append(f"Dataset shape: {data.shape}")
    
    return validation_results

def display_validation_results(validation_results):
    """Display validation results"""
    if validation_results["is_valid"]:
        st.success("✅ Data validation passed!")
    else:
        st.error("❌ Data validation failed!")
    

    if validation_results["issues"]:
        st.error("**Issues that need to be fixed:**")
        for issue in validation_results["issues"]:
            st.error(f"• {issue}")
    

    if validation_results["warnings"]:
        st.warning("**Warnings:**")
        for warning in validation_results["warnings"]:
            st.warning(f"• {warning}")
    

    if validation_results["info"]:
        st.info("**Information:**")
        for info in validation_results["info"]:
            st.info(f"• {info}")

def enhanced_feature_selection_step():
    """Enhanced feature selection with automated methods"""
    st.header("3. 🎯 Advanced Feature Selection & Labeling")
    
    if not hasattr(st.session_state, 'cleaned_data') or st.session_state.cleaned_data is None:
        st.warning("⚠️ No processed data available. Please complete preprocessing first.")
        if st.button("⬅️ Back to Preprocessing", key="back_to_preprocessing_from_feature"):
            st.session_state.training_step = 2
            st.rerun()
        return
    
    feature_selection_page()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Preprocessing", key="back_to_preprocessing"):
            st.session_state.training_step = 2
            st.rerun()
    
    with col2:
        if hasattr(st.session_state, 'labeled_data') and st.session_state.labeled_data is not None:
            if st.button("➡️ Continue to Model Training", key="proceed_to_model_training", type="primary"):
                st.session_state.training_step = 4
                st.rerun()

def enhanced_model_training_step():
    """Enhanced model training with advanced options"""
    st.header("4. 🤖 Advanced Model Training & Optimization")
    
    if not hasattr(st.session_state, 'labeled_data') or st.session_state.labeled_data is None:
        st.warning("⚠️ No labeled data available. Please complete feature selection first.")
        if st.button("⬅️ Back to Feature Selection", key="back_to_feature_selection"):
            st.session_state.training_step = 3
            st.rerun()
        return
    
    data = st.session_state.labeled_data
    target_column = st.session_state.target_column
    feature_columns = st.session_state.feature_columns
    

    st.subheader("📊 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Features", len(feature_columns))
    with col3:
        st.metric("Target", target_column)
    with col4:
        missing_count = data.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    

    is_classification = data[target_column].dtype == 'object' or len(data[target_column].unique()) < 20
    task_type = "Classification" if is_classification else "Regression"
    
    st.info(f"🎯 **Detected Task Type:** {task_type}")
    

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Model Selection", 
        "🔧 Hyperparameter Tuning", 
        "📊 Model Comparison", 
        "⚙️ Training Configuration"
    ])
    
    with tab1:
        model_selection_section(is_classification)
    
    with tab2:
        hyperparameter_tuning_section(data, target_column, feature_columns, is_classification)
    
    with tab3:
        model_comparison_section(data, target_column, feature_columns, is_classification)
    
    with tab4:
        training_configuration_section()
    

    st.markdown("---")
    st.subheader("🚀 Execute Training")
    
    if st.button("▶️ Start Training Process", type="primary", use_container_width=True):
        execute_training_process(data, target_column, feature_columns, is_classification)
    

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Feature Selection", key="back_to_feature_selection_from_training"):
            st.session_state.training_step = 3
            st.rerun()
    
    with col2:
        if hasattr(st.session_state, 'trained_model') and st.session_state.trained_model is not None:
            if st.button("➡️ Continue to Evaluation", key="proceed_to_evaluation", type="secondary"):
                st.session_state.training_step = 5
                st.rerun()

def model_selection_section(is_classification):
    """Model selection with intelligent suggestions"""
    st.subheader("🎯 Intelligent Model Selection")
    
    available_models = get_available_models(True)
    

    if is_classification:
        model_options = [k for k in available_models.keys() if 'Classifier' in k or k in ['Logistic Regression', 'Gaussian Naive Bayes']]
    else:
        model_options = [k for k in available_models.keys() if 'Regressor' in k or k == 'Linear Regression']
    

    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose model algorithm:",
            model_options,
            help="Select the machine learning algorithm to train"
        )
        st.session_state.selected_model = selected_model
    
    with col2:
        st.markdown("### 💡 Model Recommendations")
        if is_classification:
            st.info("""
            **For Classification:**
            - **Random Forest**: Good baseline, handles mixed data
            - **Gradient Boosting**: High performance, robust
            - **SVM**: Good for small-medium datasets
            - **Logistic Regression**: Fast, interpretable
            """)
        else:
            st.info("""
            **For Regression:**
            - **Random Forest**: Good baseline, robust
            - **Gradient Boosting**: High performance
            - **Linear Regression**: Fast, interpretable
            - **SVR**: Good for non-linear patterns
            """)
    

    if selected_model in available_models:
        st.subheader("⚙️ Model Parameters")
        params_config = available_models[selected_model]["params"]
        model_params = {}
        

        param_cols = st.columns(2)
        param_idx = 0
        
        for param_name, param_config in params_config.items():
            with param_cols[param_idx % 2]:
                if param_config["type"] == "integer":
                    model_params[param_name] = st.slider(
                        f"{param_name}:",
                        min_value=param_config["min"],
                        max_value=param_config["max"],
                        value=param_config["default"],
                        step=param_config["step"],
                        help=param_config["help"]
                    )
                elif param_config["type"] == "numeric":
                    model_params[param_name] = st.slider(
                        f"{param_name}:",
                        min_value=float(param_config["min"]),
                        max_value=float(param_config["max"]),
                        value=float(param_config["default"]),
                        step=float(param_config["step"]),
                        help=param_config["help"]
                    )
                elif param_config["type"] == "select":
                    model_params[param_name] = st.selectbox(
                        f"{param_name}:",
                        param_config["options"],
                        index=param_config["options"].index(param_config["default"]),
                        help=param_config["help"]
                    )
                elif param_config["type"] == "boolean":
                    model_params[param_name] = st.checkbox(
                        f"{param_name}:",
                        value=param_config["default"],
                        help=param_config["help"]
                    )
            
            param_idx += 1
        
        st.session_state.model_params = model_params

def hyperparameter_tuning_section(data, target_column, feature_columns, is_classification):
    """Advanced hyperparameter tuning options"""
    st.subheader("🔧 Automated Hyperparameter Tuning")
    
    enable_tuning = st.checkbox(
        "Enable hyperparameter tuning",
        value=False,
        help="Automatically find optimal parameters"
    )
    
    if enable_tuning:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_method = st.selectbox(
                "Search method:",
                ["Grid Search", "Random Search"],
                help="Method for hyperparameter optimization"
            )
        
        with col2:
            cv_folds = st.slider(
                "Cross-validation folds:",
                3, 10, 5,
                help="Number of folds for cross-validation"
            )
        
        with col3:
            if search_method == "Random Search":
                max_iter = st.slider(
                    "Max iterations:",
                    10, 100, 50,
                    help="Maximum number of parameter combinations to try"
                )
            else:
                max_iter = None
        

        st.session_state.enable_tuning = enable_tuning
        st.session_state.search_method = search_method
        st.session_state.cv_folds = cv_folds
        st.session_state.max_iter = max_iter
        

        if st.button("🔍 Preview Hyperparameter Space"):
            if hasattr(st.session_state, 'selected_model'):
                selected_model = st.session_state.selected_model
                available_models = get_available_models(True)
                
                if selected_model in available_models:
                    param_grid = available_models[selected_model].get("param_grid", {})
                    
                    if param_grid:
                        st.write("**Parameter search space:**")
                        for param, values in param_grid.items():
                            st.write(f"- **{param}**: {values}")
                        

                        total_combinations = 1
                        for values in param_grid.values():
                            total_combinations *= len(values)
                        
                        if search_method == "Grid Search":
                            st.info(f"📊 Total combinations: {total_combinations}")
                        else:
                            actual_iter = min(max_iter, total_combinations)
                            st.info(f"📊 Will test {actual_iter} random combinations")
                    else:
                        st.warning("⚠️ No parameter grid defined for this model")

def model_comparison_section(data, target_column, feature_columns, is_classification):
    """Compare multiple models automatically"""
    st.subheader("📊 Automated Model Comparison")
    
    enable_comparison = st.checkbox(
        "Enable model comparison",
        value=False,
        help="Compare multiple algorithms automatically"
    )
    
    if enable_comparison:
        available_models = get_available_models(True)
        

        if is_classification:
            all_models = [k for k in available_models.keys() if 'Classifier' in k or k in ['Logistic Regression', 'Gaussian Naive Bayes']]
        else:
            all_models = [k for k in available_models.keys() if 'Regressor' in k or k == 'Linear Regression']
        

        models_to_compare = st.multiselect(
            "Select models to compare:",
            all_models,
            default=all_models[:3] if len(all_models) >= 3 else all_models,
            help="Choose models for automatic comparison"
        )
        
        comparison_cv_folds = st.slider(
            "Cross-validation folds for comparison:",
            3, 10, 5,
            help="Number of folds for model comparison"
        )
        

        st.session_state.enable_comparison = enable_comparison
        st.session_state.models_to_compare = models_to_compare
        st.session_state.comparison_cv_folds = comparison_cv_folds
        

        if st.button("🔍 Preview Model Comparison"):
            if models_to_compare:
                st.write(f"**Will compare {len(models_to_compare)} models:**")
                for model in models_to_compare:
                    st.write(f"- {model}")
                
                estimated_time = len(models_to_compare) * comparison_cv_folds * 2  
                st.info(f"⏱️ Estimated comparison time: ~{estimated_time} seconds")

def training_configuration_section():
    """Advanced training configuration options"""
    st.subheader("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider(
            "Test set size:",
            0.1, 0.4, 0.2, 0.05,
            help="Fraction of data to use for testing"
        )
    
    with col2:
        random_state = st.number_input(
            "Random seed:",
            0, 1000, 42,
            help="Random seed for reproducibility"
        )
    
    with col3:
        use_stratify = st.checkbox(
            "Stratified sampling",
            value=True,
            help="Use stratified sampling for train/test split"
        )
    

    st.write("**🔍 Cross-Validation Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_cv = st.checkbox(
            "Enable cross-validation",
            value=True,
            help="Perform cross-validation during training"
        )
    
    with col2:
        if enable_cv:
            cv_folds_training = st.slider(
                "CV folds:",
                3, 10, 5,
                help="Number of cross-validation folds"
            )
        else:
            cv_folds_training = 5
    

    st.session_state.test_size = test_size
    st.session_state.random_state = random_state
    st.session_state.use_stratify = use_stratify
    st.session_state.enable_cv = enable_cv
    st.session_state.cv_folds_training = cv_folds_training

def execute_training_process(data, target_column, feature_columns, is_classification):
    """Execute the complete training process with proper LabelEncoder handling"""
    with st.spinner("🚀 Executing training process..."):
        try:
            features = data[feature_columns]
            target = data[target_column]
            
            test_size = getattr(st.session_state, 'test_size', 0.2)
            random_state = getattr(st.session_state, 'random_state', 42)
            use_stratify = getattr(st.session_state, 'use_stratify', True)
            
            results = {}
            
            if getattr(st.session_state, 'enable_comparison', False):
                st.write("📊 Running model comparison...")
                models_to_compare = getattr(st.session_state, 'models_to_compare', [])
                comparison_cv_folds = getattr(st.session_state, 'comparison_cv_folds', 5)
                
                if models_to_compare:
                    comparison_results = compare_models(
                        features, target, models_to_compare, comparison_cv_folds
                    )
                    results['model_comparison'] = comparison_results
                    
                    recommendations = get_model_recommendations(comparison_results, is_classification)
                    results['recommendations'] = recommendations
                    
                    st.subheader("📈 Model Comparison Results")
                    
                    comparison_df = pd.DataFrame({
                        model: {
                            'Mean Score': result.get('mean_score', 0),
                            'Std Score': result.get('std_score', 0),
                            'Min Score': result.get('min_score', 0),
                            'Max Score': result.get('max_score', 0)
                        }
                        for model, result in comparison_results.items()
                        if 'error' not in result
                    }).T
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    if len(comparison_df) > 0:
                        fig = px.bar(
                            x=comparison_df.index,
                            y=comparison_df['Mean Score'],
                            error_y=comparison_df['Std Score'],
                            title="Model Comparison - Mean Cross-Validation Score"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if 'error' not in recommendations:
                        st.success(f"🏆 **Best Model:** {recommendations['best_model']}")
                        st.info(f"📊 **Best Score:** {recommendations['best_score']:.4f}")
            
            best_params = None
            if getattr(st.session_state, 'enable_tuning', False):
                st.write("🔧 Running hyperparameter tuning...")
                
                selected_model = getattr(st.session_state, 'selected_model', None)
                search_method = getattr(st.session_state, 'search_method', 'Grid Search')
                cv_folds = getattr(st.session_state, 'cv_folds', 5)
                max_iter = getattr(st.session_state, 'max_iter', 50)
                
                if selected_model:
                    tuning_results = hyperparameter_tuning(
                        features, target, selected_model,
                        search_method.lower().replace(' ', '_'),
                        max_iter, cv_folds
                    )
                    
                    if 'error' not in tuning_results:
                        best_params = tuning_results['best_params']
                        results['hyperparameter_tuning'] = tuning_results
                        
                        st.subheader("🎯 Hyperparameter Tuning Results")
                        st.success(f"**Best Score:** {tuning_results['best_score']:.4f}")
                        st.write("**Best Parameters:**")
                        for param, value in best_params.items():
                            st.write(f"- **{param}**: {value}")
                    else:
                        st.error(f"Error in hyperparameter tuning: {tuning_results['error']}")
            
            st.write("🏗️ Training final model...")

            if best_params:
                final_params = best_params
            else:
                final_params = getattr(st.session_state, 'model_params', {})
            
            selected_model = getattr(st.session_state, 'selected_model', 'Random Forest Classifier')
            
            trained_model, X_train, X_test, y_train, y_test, label_encoder = train_model(
                features, target, selected_model, final_params,
                test_size, random_state, use_stratify
            )
            
            cv_results = None
            if getattr(st.session_state, 'enable_cv', True):
                st.write("🔍 Performing cross-validation...")
                cv_folds_training = getattr(st.session_state, 'cv_folds_training', 5)
                
                cv_results = enhanced_cross_validation(
                    features, target, selected_model, final_params, cv_folds_training
                )
                results['cv_results'] = cv_results
            
            training_info = {
                'model': trained_model,
                'model_name': selected_model,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'params': final_params,
                'test_size': test_size,
                'random_state': random_state,
                'label_encoder': label_encoder,  
                'is_classification': is_classification
            }
            
            st.session_state.label_encoder = label_encoder
            st.session_state.trained_model = training_info
            st.session_state.training_results = results
            
            st.success("✅ Training completed successfully!")
            
            display_training_summary(training_info, results, cv_results)
            
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.error("💡 Please check your configuration and try again.")

def display_training_summary(training_info, results, cv_results):
    """Display comprehensive training summary"""
    st.subheader("📋 Training Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", training_info['model_name'])
    with col2:
        st.metric("Training Samples", len(training_info['X_train']))
    with col3:
        st.metric("Test Samples", len(training_info['X_test']))
    with col4:
        st.metric("Features", len(training_info['feature_columns']))
    
    if training_info.get('label_encoder') is not None:
        st.success("🏷️ **Label Encoder Active**: Predictions will show original string labels")
        
        try:
            label_encoder = training_info['label_encoder']
            original_classes = label_encoder.classes_
            
            st.write("**Target Label Mapping:**")
            mapping_data = []
            for i, cls in enumerate(original_classes):
                mapping_data.append({"Original Label": cls, "Numeric Code": i})
            
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
            
        except Exception as e:
            st.warning(f"⚠️ Could not display label mapping: {e}")
    
    if cv_results:
        st.subheader("🔍 Cross-Validation Results")
        
        cv_summary = []
        for metric, result in cv_results.items():
            if result['mean'] is not None:
                cv_summary.append({
                    'Metric': metric,
                    'Mean': f"{result['mean']:.4f}",
                    'Std Dev': f"{result['std']:.4f}",
                    'Min': f"{result['min']:.4f}",
                    'Max': f"{result['max']:.4f}"
                })
        
        if cv_summary:
            st.dataframe(pd.DataFrame(cv_summary), use_container_width=True)
    
    st.subheader("⚙️ Final Model Parameters")
    params_df = pd.DataFrame(list(training_info['params'].items()), columns=['Parameter', 'Value'])
    st.dataframe(params_df, use_container_width=True)

def enhanced_model_evaluation_step():
    """Enhanced model evaluation with comprehensive analysis"""
    st.header("5. 📈 Advanced Model Evaluation & Analysis")
    
    if not hasattr(st.session_state, 'trained_model') or st.session_state.trained_model is None:
        st.warning("⚠️ No trained model available. Please complete model training first.")
        if st.button("⬅️ Back to Training", key="back_to_training"):
            st.session_state.training_step = 4
            st.rerun()
        return
    
    training_info = st.session_state.trained_model
    model = training_info['model']
    X_test = training_info['X_test']
    y_test = training_info['y_test']
    label_encoder = training_info.get('label_encoder')
    

    with st.spinner("🔍 Evaluating model performance..."):
        evaluation_results = evaluate_model(model, X_test, y_test, True)
        
        if label_encoder is not None and 'accuracy' in evaluation_results:
            try:
                y_test_original = label_encoder.inverse_transform(y_test)
                evaluation_results['y_test_original'] = y_test_original
                evaluation_results['label_encoder'] = label_encoder
            except:
                pass
        

        if hasattr(st.session_state, 'training_results'):
            training_results = st.session_state.training_results
            if 'cv_results' in training_results:
                evaluation_results['cv_results'] = training_results['cv_results']
        
        st.session_state.model_evaluation = evaluation_results
    

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Metrics",
        "📈 Visualizations", 
        "🔍 Feature Analysis",
        "📋 Detailed Reports",
        "💾 Export & Deploy"
    ])
    
    with tab1:
        display_performance_metrics(evaluation_results)
    
    with tab2:
        display_evaluation_visualizations(evaluation_results)
    
    with tab3:
        display_feature_analysis(evaluation_results, training_info)
    
    with tab4:
        display_detailed_reports(evaluation_results, training_info)
    
    with tab5:
        export_and_deployment_section(training_info, evaluation_results)
    

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Training", key="back_to_training_from_eval"):
            st.session_state.training_step = 4
            st.rerun()
    
    with col2:
        if st.button("🏠 Return to Home", key="return_to_home"):
            st.session_state.current_page = 'home'
            st.rerun()

def display_performance_metrics(evaluation_results):
    """Display comprehensive performance metrics"""
    st.subheader("📊 Performance Metrics Overview")
    

    is_classification = 'accuracy' in evaluation_results
    
    if is_classification:

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = evaluation_results.get('accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
        
        with col2:
            precision = evaluation_results.get('precision', 0)
            st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
        
        with col3:
            recall = evaluation_results.get('recall', 0)
            st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
        
        with col4:
            f1 = evaluation_results.get('f1', 0)
            st.metric("F1-Score", f"{f1:.4f}", f"{f1*100:.2f}%")
        

        if 'roc_auc' in evaluation_results:
            col1, col2 = st.columns(2)
            with col1:
                roc_auc = evaluation_results['roc_auc']
                st.metric("ROC AUC", f"{roc_auc:.4f}")
            
            if 'average_precision' in evaluation_results:
                with col2:
                    avg_precision = evaluation_results['average_precision']
                    st.metric("Average Precision", f"{avg_precision:.4f}")
    
    else:

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            r2 = evaluation_results.get('r2', 0)
            st.metric("R² Score", f"{r2:.4f}", f"{r2*100:.2f}%")
        
        with col2:
            rmse = evaluation_results.get('rmse', 0)
            st.metric("RMSE", f"{rmse:.4f}")
        
        with col3:
            mae = evaluation_results.get('mae', 0)
            st.metric("MAE", f"{mae:.4f}")
        
        with col4:
            mape = evaluation_results.get('mape', 0)
            if not np.isnan(mape):
                st.metric("MAPE", f"{mape:.2f}%")
            else:
                st.metric("MAPE", "N/A")
    

    if 'cv_results' in evaluation_results:
        st.subheader("🟢 Cross-Validation Results")
        cv_results = evaluation_results['cv_results']
        
        cv_data = []
        for metric, result in cv_results.items():
            if result.get('mean') is not None:
                cv_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Mean': f"{result['mean']:.4f}",
                    'Std Dev': f"{result['std']:.4f}",
                    'Range': f"{result['min']:.4f} - {result['max']:.4f}"
                })
        
        if cv_data:
            st.dataframe(pd.DataFrame(cv_data), use_container_width=True)

def display_evaluation_visualizations(evaluation_results):
    """Display evaluation visualizations"""
    st.subheader("📈 Performance Visualizations")
    

    is_classification = 'accuracy' in evaluation_results
    
    if is_classification:

        

        if 'interactive_confusion_matrix' in evaluation_results:
            st.subheader("🔥 Interactive Confusion Matrix")
            st.plotly_chart(evaluation_results['interactive_confusion_matrix'], use_container_width=True)
        

        if 'roc_curve' in evaluation_results:
            st.subheader("📈 ROC Curve")
            st.plotly_chart(evaluation_results['roc_curve'], use_container_width=True)
        

        if 'pr_curve' in evaluation_results:
            st.subheader("📊 Precision-Recall Curve")
            st.plotly_chart(evaluation_results['pr_curve'], use_container_width=True)
        

        if 'class_distribution_plot' in evaluation_results:
            st.subheader("📊 Class Distribution Analysis")
            st.plotly_chart(evaluation_results['class_distribution_plot'], use_container_width=True)
    
    else:

        

        if 'actual_vs_predicted_plot' in evaluation_results:
            st.subheader("📈 Actual vs Predicted Values")
            st.plotly_chart(evaluation_results['actual_vs_predicted_plot'], use_container_width=True)
        

        if 'residual_analysis' in evaluation_results:
            st.subheader("🔍 Residual Analysis")
            st.plotly_chart(evaluation_results['residual_analysis'], use_container_width=True)
        

        if 'prediction_intervals_plot' in evaluation_results:
            st.subheader("📊 Prediction Intervals")
            st.plotly_chart(evaluation_results['prediction_intervals_plot'], use_container_width=True)
            
            coverage = evaluation_results.get('prediction_interval_coverage', 0)
            st.info(f"📊 **Prediction Interval Coverage:** {coverage:.1%}")

def display_feature_analysis(evaluation_results, training_info):
    """Display feature importance and analysis"""
    st.subheader("🔍 Feature Importance Analysis")
    
    if 'feature_importance_plot_interactive' in evaluation_results:
        st.plotly_chart(evaluation_results['feature_importance_plot_interactive'], use_container_width=True)
        

        if 'feature_importance_stats' in evaluation_results:
            stats = evaluation_results['feature_importance_stats']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Top 5 Most Important Features:**")
                for i, (feature, importance) in enumerate(zip(stats['top_features'][:5], stats['top_importances'][:5])):
                    st.write(f"{i+1}. **{feature}**: {importance:.4f}")
            
            with col2:
                features_80_pct = stats.get('features_for_80_percent', 0)
                total_features = len(training_info['feature_columns'])
                
                st.metric("Features for 80% Importance", features_80_pct)
                st.metric("Total Features", total_features)
                
                if features_80_pct > 0:
                    efficiency = (features_80_pct / total_features) * 100
                    st.metric("Feature Efficiency", f"{efficiency:.1f}%")
    
    else:
        st.info("ℹ️ Feature importance not available for this model type.")

def display_detailed_reports(evaluation_results, training_info):
    """Display detailed classification/regression reports"""
    st.subheader("📋 Detailed Performance Reports")
    

    if 'classification_report' in evaluation_results:
        st.subheader("📊 Classification Report")
        report_df = evaluation_results['classification_report']
        

        styled_report = report_df.style.format("{:.4f}").background_gradient(
            subset=['precision', 'recall', 'f1-score'], cmap='RdYlGn'
        )
        st.dataframe(styled_report, use_container_width=True)
    

    if 'residual_stats' in evaluation_results:
        st.subheader("📊 Residual Statistics")
        residual_stats = evaluation_results['residual_stats']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Residual", f"{residual_stats['mean']:.6f}")
        with col2:
            st.metric("Std Residual", f"{residual_stats['std']:.4f}")
        with col3:
            st.metric("Skewness", f"{residual_stats['skewness']:.4f}")
        with col4:
            st.metric("Kurtosis", f"{residual_stats['kurtosis']:.4f}")
        

        if abs(residual_stats['mean']) < 0.01:
            st.success("✅ Residuals are well-centered around zero")
        else:
            st.warning("⚠️ Residuals show some bias")
        
        if abs(residual_stats['skewness']) < 0.5:
            st.success("✅ Residuals are approximately symmetric")
        else:
            st.warning("⚠️ Residuals show skewness")

def export_and_deployment_section(training_info, evaluation_results):
    """Export model and generate deployment materials"""
    st.subheader("💾 Export & Deployment")
    

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📦 Model Export**")
        
        export_format = st.selectbox(
            "Export format:",
            ["joblib", "pickle"],
            help="Choose format for saving the model"
        )
        
        if 'model_filename' not in st.session_state:
            default_name = f"{training_info['model_name'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.model_filename = default_name
        
        model_name = st.text_input(
            "Model filename:",
            value=st.session_state.model_filename,
            help="Filename for the saved model (without extension)",
            key="model_filename_input"
        )
        
        if model_name != st.session_state.model_filename:
            st.session_state.model_filename = model_name
        

        if st.button("💾 Download Model", type="primary", use_container_width=True):
            try:
                with st.spinner("Preparing model for download..."):
                    label_encoder = training_info.get('label_encoder')
                    
                    model_data = save_model_for_download(
                        training_info['model'], 
                        label_encoder, 
                        st.session_state.model_filename, 
                        export_format
                    )
                    
                    filename = f"{st.session_state.model_filename}.{export_format}"
                    
                    st.download_button(
                        label="⬇️ Download Now",
                        data=model_data,
                        file_name=filename,
                        mime="application/octet-stream",
                        use_container_width=True,
                        key="download_model_button"
                    )
                    
                    st.success(f"✅ Model ready for download!")
                    st.info(f"📁 **Filename:** {filename}")
                    st.info(f"📊 **Size:** {len(model_data) / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"❌ Error preparing model for download: {str(e)}")
        
        st.markdown("---")
        st.write("**ℹ️ Model Package Contents:**")
        st.write("• Trained model object")
        st.write("• Label encoder (for classification)")
        st.write("• Timestamp and metadata")
        st.write("• Model parameters")
    
    with col2:
        st.write("**📊 Generate Reports**")
        
        if st.button("📄 Generate HTML Report", type="secondary"):
            try:
                with st.spinner("Generating comprehensive report..."):
                    model_info = {
                        "model_object": training_info['model'],
                        "model_name": training_info['model_name'],
                        "X_train": training_info['X_train'],
                        "X_test": training_info['X_test'],
                        "y_train": training_info['y_train'],
                        "y_test": training_info['y_test'],
                        "params": training_info['params'],
                        "is_supervised": True
                    }
                    
                    model_path = f"{st.session_state.model_filename}.{export_format}"
                    
                    html_report = generate_report(model_info, evaluation_results, model_path)
                    
                    report_filename = f"ml_report_{st.session_state.model_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=html_report,
                        file_name=report_filename,
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    st.success("✅ HTML report generated successfully!")
                    st.info(f"📁 **Report:** {report_filename}")
                    st.info(f"📊 **Size:** {len(html_report.encode()) / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
        
        if st.button("📊 Export Model Summary (JSON)"):
            try:
                with st.spinner("Exporting model summary..."):
                    model_info = {
                        "model_object": training_info['model'],
                        "model_name": training_info['model_name'],
                        "X_train": training_info['X_train'],
                        "X_test": training_info['X_test'],
                        "y_train": training_info['y_train'],
                        "y_test": training_info['y_test'],
                        "params": training_info['params'],
                        "is_supervised": True
                    }
                    
                    summary = export_model_summary(model_info, evaluation_results)
                    
                    import json
                    summary_json = json.dumps(summary, indent=2, default=str)
                    
                    summary_filename = f"model_summary_{st.session_state.model_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    st.download_button(
                        label="⬇️ Download Model Summary",
                        data=summary_json,
                        file_name=summary_filename,
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    st.success("✅ Model summary exported successfully!")
                    st.info(f"📁 **Summary:** {summary_filename}")
                    st.info(f"📊 **Size:** {len(summary_json.encode()) / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"❌ Error exporting summary: {str(e)}")
    
    st.markdown("---")
    st.subheader("🚀 Deployment Instructions")
    
    deployment_tab1, deployment_tab2, deployment_tab3 = st.tabs([
        "🐍 Python Usage", 
        "🌐 Web Deployment",
        "☁️ Cloud Deployment"
    ])
    
    with deployment_tab1:
        st.write("**📝 Python Code Example:**")
        
        feature_list = "', '".join(training_info['feature_columns'])
        
        code_example = f"""
# Load and use your trained model
import joblib
import pandas as pd
import numpy as np

# Load the model package
model_package = joblib.load('{st.session_state.model_filename}.{export_format}')
model = model_package['model']
label_encoder = model_package.get('label_encoder')  # May be None for regression

# Prepare your data (ensure same features and order)
required_features = ['{feature_list}']

# Example data (replace with your actual data)
new_data = pd.DataFrame({{
    # Add your data here with the same column names
    # Example:
    # '{training_info['feature_columns'][0]}': [value1, value2, ...],
    # Continue for all features...
}})

# Make predictions
predictions = model.predict(new_data)

# For classification models, convert back to original labels
if label_encoder is not None:
    predictions = label_encoder.inverse_transform(predictions)

# For classification models with probability estimates:
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(new_data)
    
print("Predictions:", predictions)
"""
        
        st.code(code_example, language='python')
    
    with deployment_tab2:
        st.write("**🌐 Streamlit Web App Template:**")
        
        streamlit_code = f"""
import streamlit as st
import joblib
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    model_package = joblib.load('{st.session_state.model_filename}.{export_format}')
    return model_package['model'], model_package.get('label_encoder')

model, label_encoder = load_model()

st.title('ML Model Prediction App')

# Create input fields for each feature
feature_inputs = {{}}
"""
        
        for feature in training_info['feature_columns'][:5]: 
            streamlit_code += f"""
feature_inputs['{feature}'] = st.number_input('{feature}', value=0.0)
"""
        
        streamlit_code += """
# Make prediction
if st.button('Predict'):
    input_data = pd.DataFrame([feature_inputs])
    prediction = model.predict(input_data)
    
    # Convert back to original labels if needed
    if label_encoder is not None:
        prediction = label_encoder.inverse_transform(prediction)
    
    st.success(f'Prediction: {prediction[0]}')
"""
        
        st.code(streamlit_code, language='python')
    
    with deployment_tab3:
        st.write("**☁️ Cloud Deployment Options:**")
        
        st.markdown("""
        **Recommended Cloud Platforms:**
        
        1. **Streamlit Cloud** (Free)
           - Upload your code to GitHub
           - Connect to Streamlit Cloud
           - Automatic deployment
        
        2. **Heroku** 
           - Create Procfile and requirements.txt
           - Deploy via Git or GitHub
           - Add-ons available for databases
        
        3. **AWS/Google Cloud/Azure**
           - Use container services (ECS, Cloud Run, Container Instances)
           - Serverless functions (Lambda, Cloud Functions, Azure Functions)
           - ML-specific services (SageMaker, AI Platform, ML Studio)
        
        4. **Docker Deployment**
           - Create Dockerfile for your app
           - Deploy to any container platform
           - Kubernetes for scaling
        """)
        
        st.write("**🐳 Docker Example:**")
        docker_code = f"""
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Build and run:
# docker build -t ml-model-app .
# docker run -p 8501:8501 ml-model-app
"""
        st.code(docker_code, language='dockerfile')

def clean_numbers(df):
    """
    Cleans Indonesian number formatting in a Pandas DataFrame.
    This function replaces dots ('.') used as thousand separators with empty strings,
    and commas (',') used as decimal points with dots ('.').

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            try:

                cleaned = df[col].str.replace('\.', '', regex=True).str.replace(',', '.', regex=False).astype(float)
                df[col] = cleaned
            except:

                pass
    return df
