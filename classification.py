import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_model_from_file, validate_model_package, create_sample_data_template, 
    export_predictions_to_csv, load_external_model_flexible, 
    detect_model_format_and_requirements
)
from datetime import datetime
import io

def detect_csv_delimiter_advanced(file_content):
    """
    Advanced CSV delimiter detection
    """
    import csv
    

    delimiters = [';', ',', '\t', '|']
    delimiter_scores = {}
    
    for delimiter in delimiters:
        try:

            lines = file_content.split('\n')[:5]
            total_count = 0
            consistent_count = True
            first_line_count = None
            
            for line in lines:
                if line.strip():
                    count = line.count(delimiter)
                    if first_line_count is None:
                        first_line_count = count
                    elif count != first_line_count and count > 0:
                        consistent_count = False
                    total_count += count
            

            score = total_count
            if consistent_count and first_line_count and first_line_count > 0:
                score *= 2
            
            delimiter_scores[delimiter] = score
            
        except:
            delimiter_scores[delimiter] = 0
    

    best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
    return best_delimiter if delimiter_scores[best_delimiter] > 0 else ','

def classification_page():
    """
    Enhanced classification page with flexible model loading capabilities
    """
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("⬅️ Home", key="classification_back_home", type="secondary"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    st.title("🎯 Enhanced Model Classification")
    st.markdown("### *Advanced Prediction Platform dengan Flexible Model Support*")
    

    if 'classification_model' not in st.session_state:
        st.session_state.classification_model = None
    if 'classification_model_package' not in st.session_state:
        st.session_state.classification_model_package = None
    if 'model_features' not in st.session_state:
        st.session_state.model_features = []
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}
    

    st.header("1. 📤 Load Trained Model")
    
    tab1, tab2 = st.tabs(["📁 Upload Model File", "🔄 Use Current Session Model"])
    
    with tab1:
        st.markdown("**Supported Model Formats:**")
        st.markdown("""
        - 🔹 **Internal Format**: Models trained with this platform (.joblib, .pkl)
        - 🔹 **Scikit-learn Models**: Raw sklearn models or pickled models
        - 🔹 **Custom Packages**: Dictionary-based model packages
        - 🔹 **External Models**: Models from other ML platforms
        """)
        
        uploaded_model = st.file_uploader(
            "Upload trained model file",
            type=['joblib', 'pkl', 'pickle'],
            help="Upload any model file - the system will automatically detect the format"
        )
        
        if uploaded_model is not None:
            with st.spinner("Analyzing and loading model..."):

                model_package = load_external_model_flexible(uploaded_model)
                
                if model_package is not None:

                    analysis = detect_model_format_and_requirements(model_package)
                    
                    st.success("✅ Model loaded successfully!")
                    

                    display_model_analysis(model_package, analysis)
                    

                    if analysis['requires_manual_setup']:
                        model_package = handle_manual_model_setup(model_package, analysis)
                    
                    if model_package and model_package.get('model') is not None:

                        st.session_state.classification_model_package = model_package
                        st.session_state.classification_model = model_package['model']
                        st.session_state.model_features = model_package.get('feature_columns', [])
                        st.session_state.model_info = {
                            'model_type': model_package.get('model_type', 'Unknown'),
                            'timestamp': model_package.get('timestamp', 'Unknown'),
                            'target_column': model_package.get('target_column', 'Unknown'),
                            'label_encoder': model_package.get('label_encoder'),
                            'target_label_mapping': model_package.get('target_label_mapping'),
                            'format': model_package.get('format', 'external')
                        }
                        
                        st.success("🎉 Model is ready for predictions!")
                        display_model_info(model_package, {'is_valid': True, 'issues': [], 'warnings': [], 'info': []})
                else:
                    st.error("❌ Could not load the model file")
                    st.markdown("""
                    **Troubleshooting Tips:**
                    - Ensure the file contains a trained ML model
                    - Check if the model was saved properly
                    - Try re-saving the model using joblib or pickle
                    - Contact support if the issue persists
                    """)
    
    with tab2:
        if hasattr(st.session_state, 'trained_model') and st.session_state.trained_model is not None:
            if st.button("🔄 Use Current Session Model", type="primary"):
                training_info = st.session_state.trained_model
                

                model_package = {
                    'model': training_info['model'],
                    'label_encoder': training_info.get('label_encoder'),
                    'feature_columns': training_info['feature_columns'],
                    'target_column': training_info['target_column'],
                    'model_type': training_info['model_name'],
                    'timestamp': datetime.now().isoformat(),
                    'target_label_mapping': getattr(st.session_state, 'target_label_mapping', None),
                    'format': 'internal'
                }
                
                st.session_state.classification_model_package = model_package
                st.session_state.classification_model = model_package['model']
                st.session_state.model_features = model_package['feature_columns']
                st.session_state.model_info = {
                    'model_type': model_package['model_type'],
                    'timestamp': model_package['timestamp'],
                    'target_column': model_package['target_column'],
                    'label_encoder': model_package.get('label_encoder'),
                    'target_label_mapping': model_package.get('target_label_mapping'),
                    'format': 'internal'
                }
                
                st.success("✅ Current session model loaded!")
                validation = validate_model_package(model_package)
                display_model_info(model_package, validation)
        else:
            st.info("ℹ️ No trained model available in current session. Please train a model first or upload a model file.")
    

    if st.session_state.classification_model is not None:
        st.markdown("---")
        st.header("2. 🎯 Make Predictions")
        
        prediction_tab1, prediction_tab2, prediction_tab3 = st.tabs([
            "✏️ Manual Input", 
            "📁 Batch File Upload", 
            "🧪 Sample Data Testing"
        ])
        
        with prediction_tab1:
            manual_prediction_interface()
        
        with prediction_tab2:
            batch_prediction_interface()
        
        with prediction_tab3:
            sample_data_testing_interface()

def display_model_analysis(model_package, analysis):
    """
    Display analysis results of the loaded model
    """
    st.subheader("🔍 Model Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        format_detected = analysis.get('format_detected', 'unknown')
        st.metric("Format Detected", format_detected.title())
    
    with col2:
        feature_count = analysis.get('feature_count', 0)
        st.metric("Features Required", feature_count if feature_count > 0 else "Unknown")
    
    with col3:
        has_names = "✅ Yes" if analysis.get('has_feature_names', False) else "❌ No"
        st.metric("Feature Names Available", has_names)
    

    if analysis.get('suggestions'):
        st.write("**📋 Analysis Notes:**")
        for suggestion in analysis['suggestions']:
            st.info(f"ℹ️ {suggestion}")

def handle_manual_model_setup(model_package, analysis):
    """
    Handle manual setup for models that require additional configuration
    """
    st.subheader("⚙️ Manual Model Configuration")
    st.write("This model requires some manual configuration to work properly.")
    

    if not model_package.get('feature_columns'):
        st.write("**🏷️ Feature Column Names:**")
        
        feature_count = analysis.get('feature_count', 0)
        
        if feature_count > 0:
            st.write(f"Your model expects {feature_count} features. Please provide names for each feature:")
            
            feature_names = []
            cols = st.columns(min(3, feature_count))
            
            for i in range(feature_count):
                with cols[i % len(cols)]:
                    feature_name = st.text_input(
                        f"Feature {i+1}:",
                        value=f"feature_{i}",
                        key=f"feature_name_{i}"
                    )
                    feature_names.append(feature_name)
            
            if st.button("✅ Apply Feature Names", type="primary"):
                model_package['feature_columns'] = feature_names
                st.success("Feature names applied successfully!")
                st.rerun()
        else:
            st.warning("⚠️ Could not determine the number of features required by this model.")
            
            manual_count = st.number_input(
                "How many features does your model expect?",
                min_value=1,
                max_value=1000,
                value=5,
                step=1
            )
            
            if st.button("🔧 Setup Features Manually"):
                feature_names = [f"feature_{i}" for i in range(manual_count)]
                model_package['feature_columns'] = feature_names
                st.success(f"Created {manual_count} generic feature names!")
                st.rerun()
    

    if not model_package.get('target_column') or model_package.get('target_column') == 'target':
        target_name = st.text_input(
            "Target/Label Column Name:",
            value="target",
            help="What is the name of the column you want to predict?"
        )
        
        if st.button("✅ Set Target Column"):
            model_package['target_column'] = target_name
            st.success("Target column name set!")
    
    return model_package

def display_model_info(model_package, validation):
    """
    Display comprehensive model information
    """
    st.subheader("📋 Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", model_package.get('model_type', 'Unknown'))
    with col2:
        st.metric("Features", len(model_package.get('feature_columns', [])))
    with col3:
        format_type = model_package.get('format', 'unknown')
        st.metric("Format", format_type.title())
    with col4:
        timestamp = model_package.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                st.metric("Created", formatted_time)
            except:
                st.metric("Created", timestamp)
        else:
            st.metric("Created", "Unknown")
    

    model = model_package['model']
    capabilities = []
    
    if hasattr(model, 'predict'):
        capabilities.append("✅ Predictions")
    if hasattr(model, 'predict_proba'):
        capabilities.append("✅ Probability Estimates")
    if hasattr(model, 'feature_importances_'):
        capabilities.append("✅ Feature Importance")
    if hasattr(model, 'decision_function'):
        capabilities.append("✅ Decision Function")
    
    st.write("**Model Capabilities:**")
    for capability in capabilities:
        st.write(f"  {capability}")
    

    label_encoder = model_package.get('label_encoder')
    if label_encoder is not None:
        st.success("🏷️ **String Label Support**: This model can predict original string labels")
        
        try:
            classes = label_encoder.classes_
            st.write(f"**Target Classes**: {', '.join(map(str, classes))}")
        except:
            st.write("**Target Classes**: Available but could not display")
    else:
        st.info("🔢 **Numeric Predictions**: This model outputs numeric predictions")
    

    feature_columns = model_package.get('feature_columns', [])
    if feature_columns:
        st.write("**Required Features:**")
        

        if len(feature_columns) <= 10:
            feature_cols = st.columns(min(3, len(feature_columns)))
            for i, feature in enumerate(feature_columns):
                with feature_cols[i % len(feature_cols)]:
                    st.write(f"• {feature}")
        else:

            with st.expander(f"View all {len(feature_columns)} features"):
                feature_text = ", ".join(feature_columns)
                st.write(feature_text)
    

    if validation.get('warnings'):
        st.warning("⚠️ **Warnings:**")
        for warning in validation['warnings']:
            st.warning(f"• {warning}")
    
    if validation.get('info'):
        with st.expander("ℹ️ Additional Information"):
            for info in validation['info']:
                st.info(f"• {info}")

def manual_prediction_interface():
    """
    Manual input interface for single predictions
    """
    st.subheader("✏️ Manual Feature Input")
    
    model_features = st.session_state.model_features
    
    if not model_features:
        st.error("❌ No feature information available. Please ensure the model package includes feature columns.")
        return
    
    st.write(f"**Enter values for {len(model_features)} features:**")
    

    feature_values = {}
    

    num_cols = min(3, len(model_features))
    feature_cols = st.columns(num_cols)
    
    for i, feature in enumerate(model_features):
        with feature_cols[i % num_cols]:

            if any(keyword in feature.lower() for keyword in ['age', 'year', 'count', 'number']):
                feature_values[feature] = st.number_input(
                    f"{feature}:", 
                    value=0, 
                    step=1,
                    key=f"manual_{feature}"
                )
            elif any(keyword in feature.lower() for keyword in ['price', 'cost', 'amount', 'salary']):
                feature_values[feature] = st.number_input(
                    f"{feature}:", 
                    value=0.0, 
                    step=0.01,
                    key=f"manual_{feature}"
                )
            else:
                feature_values[feature] = st.number_input(
                    f"{feature}:", 
                    value=0.0, 
                    step=0.1,
                    key=f"manual_{feature}"
                )
    

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎯 Make Prediction", type="primary", use_container_width=True):
            make_single_prediction(feature_values)
    
    with col2:
        if st.button("🔄 Reset Values", use_container_width=True):
            st.rerun()

def batch_prediction_interface():
    """
    Enhanced batch prediction interface with automatic column detection
    """
    st.subheader("📁 Batch Prediction from File")
    
    model_features = st.session_state.model_features
    
    if not model_features:
        st.error("❌ No feature information available.")
        return
    
    st.write("**Upload a CSV or Excel file containing the required data:**")
    

    required_cols_df = pd.DataFrame({
        'Required Column': model_features,
        'Data Type': ['Numeric'] * len(model_features),
        'Description': [f"Feature used by the model" for _ in model_features]
    })
    st.dataframe(required_cols_df, use_container_width=True)
    

    uploaded_file = st.file_uploader(
        "Choose file for batch prediction",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file. The system will automatically detect and extract required columns."
    )
    
    if uploaded_file is not None:
        try:

            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            with st.spinner("Loading and analyzing file..."):
                if file_extension == 'csv':

                    try:

                        uploaded_file.seek(0)
                        sample_content = uploaded_file.read(1024).decode('utf-8', errors='ignore')
                        uploaded_file.seek(0)
                        

                        delimiter_counts = {
                            ';': sample_content.count(';'),
                            ',': sample_content.count(','),
                            '\t': sample_content.count('\t'),
                            '|': sample_content.count('|')
                        }
                        

                        detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                        

                        if delimiter_counts[detected_delimiter] == 0:
                            detected_delimiter = ','
                        
                        st.info(f"🔍 Detected delimiter: '{detected_delimiter}'")
                        

                        try:
                            batch_data = pd.read_csv(uploaded_file, encoding='utf-8', sep=detected_delimiter)
                        except UnicodeDecodeError:
                            try:
                                uploaded_file.seek(0)
                                batch_data = pd.read_csv(uploaded_file, encoding='latin-1', sep=detected_delimiter)
                            except:
                                uploaded_file.seek(0)
                                batch_data = pd.read_csv(uploaded_file, encoding='cp1252', sep=detected_delimiter)
                        

                        batch_data.columns = batch_data.columns.str.strip()
                        

                        for col in batch_data.columns:
                            if batch_data[col].dtype == 'object':

                                sample_vals = batch_data[col].dropna().astype(str).head(10)
                                numeric_pattern_count = 0
                                
                                for val in sample_vals:

                                    val_clean = val.strip().replace(',', '.')
                                    try:
                                        float(val_clean)
                                        numeric_pattern_count += 1
                                    except:
                                        pass
                                

                                if len(sample_vals) > 0 and numeric_pattern_count / len(sample_vals) > 0.7:
                                    try:
                                        batch_data[col] = batch_data[col].astype(str).str.replace(',', '.').astype(float)
                                    except:
                                        pass
                    
                    except Exception as e:
                        st.error(f"Error loading CSV: {str(e)}")
                        return
                
                elif file_extension in ['xlsx', 'xls']:

                    excel_file = pd.ExcelFile(uploaded_file)
                    

                    if len(excel_file.sheet_names) > 1:
                        selected_sheet = st.selectbox(
                            "Select Excel sheet:",
                            excel_file.sheet_names,
                            help="Choose which sheet contains your data"
                        )
                        batch_data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    else:
                        batch_data = pd.read_excel(uploaded_file, sheet_name=0)
                
                else:
                    st.error("❌ Unsupported file format")
                    return
            
            st.success(f"✅ File loaded successfully! Found {len(batch_data)} rows and {len(batch_data.columns)} columns.")
            

            if file_extension == 'csv':
                st.info(f"📋 **CSV Format Details:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"• Delimiter detected: `{detected_delimiter}`")
                    st.write(f"• Encoding: UTF-8")
                with col2:
                    st.write(f"• Columns found: {len(batch_data.columns)}")
                    st.write(f"• Data rows: {len(batch_data)}")
            

            st.write("**Original Data Preview:**")
            st.dataframe(batch_data.head(10), use_container_width=True)
            

            st.write("**🔍 Automatic Column Detection:**")
            
            column_mapping, missing_columns, matched_data = detect_and_map_columns(
                batch_data, model_features
            )
            

            if column_mapping:
                st.success("✅ **Column Mapping Results:**")
                
                mapping_df = pd.DataFrame([
                    {"Required Column": req_col, "Found in File": file_col, "Match Type": match_type}
                    for req_col, (file_col, match_type) in column_mapping.items()
                ])
                st.dataframe(mapping_df, use_container_width=True)
                

                if matched_data is not None and not matched_data.empty:
                    st.write("**📋 Extracted Data Preview (Ready for Prediction):**")
                    st.dataframe(matched_data.head(10), use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Ready", len(matched_data))
                    with col2:
                        st.metric("Features Matched", len(column_mapping))
                    with col3:
                        st.metric("Missing Features", len(missing_columns))
                    

                    if missing_columns:
                        st.warning(f"⚠️ **Missing Required Columns**: {', '.join(missing_columns)}")
                        

                        st.write("**Provide default values for missing columns:**")
                        default_values = {}
                        
                        cols = st.columns(min(3, len(missing_columns)))
                        for i, missing_col in enumerate(missing_columns):
                            with cols[i % len(cols)]:
                                default_values[missing_col] = st.number_input(
                                    f"Default for {missing_col}:",
                                    value=0.0,
                                    key=f"default_{missing_col}"
                                )
                        
                        if st.button("➕ Add Default Values", type="secondary"):
                            for col, default_val in default_values.items():
                                matched_data[col] = default_val
                            

                            matched_data = matched_data[model_features]
                            st.success("✅ Default values added successfully!")
                            st.dataframe(matched_data.head(), use_container_width=True)
                    

                    if len(missing_columns) == 0 or all(col in matched_data.columns for col in model_features):

                        final_data = matched_data[model_features]
                        
                        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                            make_batch_predictions(final_data, batch_data)
                    else:
                        st.info("ℹ️ Please provide default values for missing columns to proceed with prediction.")
                
            else:
                st.error("❌ **No matching columns found!**")
                st.write("**Available columns in your file:**")
                available_cols = list(batch_data.columns)
                st.write(", ".join(available_cols))
                
                st.write("**Manual column mapping:**")
                manual_mapping = {}
                
                cols = st.columns(2)
                for i, req_col in enumerate(model_features):
                    with cols[i % 2]:
                        manual_mapping[req_col] = st.selectbox(
                            f"Map '{req_col}' to:",
                            ["-- Select Column --"] + available_cols,
                            key=f"manual_map_{req_col}"
                        )
                
                if st.button("🔗 Apply Manual Mapping"):

                    valid_mapping = {k: v for k, v in manual_mapping.items() if v != "-- Select Column --"}
                    
                    if len(valid_mapping) == len(model_features):
                        try:
                            mapped_data = batch_data[list(valid_mapping.values())].copy()
                            mapped_data.columns = list(valid_mapping.keys())
                            
                            st.success("✅ Manual mapping applied successfully!")
                            st.dataframe(mapped_data.head(), use_container_width=True)
                            
                            if st.button("🚀 Run Prediction with Manual Mapping", type="primary"):
                                make_batch_predictions(mapped_data, batch_data)
                        except Exception as e:
                            st.error(f"❌ Error applying manual mapping: {str(e)}")
                    else:
                        st.warning("⚠️ Please map all required columns before proceeding.")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.error("Please check your file format and try again.")

def detect_and_map_columns(data, required_columns):
    """
    Automatically detect and map columns from uploaded data to required model features
    
    Args:
        data: pandas DataFrame with uploaded data
        required_columns: list of required column names for the model
        
    Returns:
        tuple: (column_mapping, missing_columns, matched_data)
    """
    import difflib
    
    available_columns = list(data.columns)
    column_mapping = {}
    missing_columns = []
    

    available_lower = [col.lower().strip() for col in available_columns]
    

    available_normalized = []
    required_normalized = []

    for col in available_columns:

        normalized = col.lower().strip()

        import re
        normalized = re.sub(r'[^a-z0-9]', '_', normalized)

        normalized = re.sub(r'_+', '_', normalized)

        normalized = normalized.strip('_')
        available_normalized.append(normalized)

    for col in required_columns:
        normalized = col.lower().strip()
        normalized = re.sub(r'[^a-z0-9]', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')
        required_normalized.append(normalized)
    
    required_lower = [col.lower().strip() for col in required_columns]
    
    for req_col in required_columns:
        req_lower = req_col.lower().strip()
        best_match = None
        match_type = None
        

        if req_lower in available_lower:
            idx = available_lower.index(req_lower)
            best_match = available_columns[idx]
            match_type = "Exact Match"
        

        elif not best_match:
            for i, avail_col in enumerate(available_lower):
                if req_lower in avail_col or avail_col in req_lower:
                    best_match = available_columns[i]
                    match_type = "Partial Match"
                    break
        

        elif not best_match:
            matches = difflib.get_close_matches(req_lower, available_lower, n=1, cutoff=0.6)
            if matches:
                idx = available_lower.index(matches[0])
                best_match = available_columns[idx]
                match_type = "Fuzzy Match"
        

        elif not best_match:
            req_keywords = req_lower.replace('_', ' ').split()
            for i, avail_col in enumerate(available_lower):
                avail_keywords = avail_col.replace('_', ' ').split()
                if any(keyword in avail_keywords for keyword in req_keywords):
                    best_match = available_columns[i]
                    match_type = "Keyword Match"
                    break
        
        if best_match:
            column_mapping[req_col] = (best_match, match_type)
        else:
            missing_columns.append(req_col)
    

    matched_data = None
    if column_mapping:
        try:

            file_columns = [mapping[0] for mapping in column_mapping.values()]
            model_columns = list(column_mapping.keys())
            
            matched_data = data[file_columns].copy()
            matched_data.columns = model_columns
            

            for missing_col in missing_columns:
                matched_data[missing_col] = np.nan
                
        except Exception as e:
            st.error(f"Error creating matched data: {str(e)}")
            matched_data = None
    
    return column_mapping, missing_columns, matched_data

def sample_data_testing_interface():
    """
    Sample data testing interface
    """
    st.subheader("🧪 Test with Sample Data")
    
    model_features = st.session_state.model_features
    
    if not model_features:
        st.error("❌ No feature information available.")
        return
    
    st.write("**Generate sample data for testing the model:**")
    

    if 'sample_data_generated' not in st.session_state:
        st.session_state.sample_data_generated = False
    if 'current_sample_data' not in st.session_state:
        st.session_state.current_sample_data = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.slider("Number of sample rows:", 1, 100, 5)
    
    with col2:
        random_seed = st.number_input("Random seed:", value=42, step=1)
    
    if st.button("🎲 Generate Sample Data", use_container_width=True, key="generate_sample_btn"):

        np.random.seed(random_seed)
        
        sample_data = {}
        for feature in model_features:

            if any(keyword in feature.lower() for keyword in ['age', 'year']):
                sample_data[feature] = np.random.randint(18, 80, num_samples)
            elif any(keyword in feature.lower() for keyword in ['price', 'cost', 'amount', 'salary']):
                sample_data[feature] = np.random.uniform(1000, 10000, num_samples).round(2)
            elif any(keyword in feature.lower() for keyword in ['rate', 'ratio', 'percent']):
                sample_data[feature] = np.random.uniform(0, 1, num_samples).round(3)
            elif any(keyword in feature.lower() for keyword in ['score', 'rating']):
                sample_data[feature] = np.random.uniform(1, 5, num_samples).round(1)
            else:
                sample_data[feature] = np.random.uniform(-2, 2, num_samples).round(3)
        
        sample_df = pd.DataFrame(sample_data)
        st.session_state.current_sample_data = sample_df
        st.session_state.sample_data_generated = True
        
        st.write("**Generated Sample Data:**")
        st.dataframe(sample_df, use_container_width=True)


    if st.session_state.sample_data_generated and st.session_state.current_sample_data is not None:
        if st.button("🎯 Predict on Sample Data", type="primary", key="predict_sample_data_btn", use_container_width=True):
            make_batch_predictions(st.session_state.current_sample_data, st.session_state.current_sample_data)

def make_single_prediction(feature_values):
    """
    Make a single prediction with detailed output
    """
    try:
        model = st.session_state.classification_model
        model_package = st.session_state.classification_model_package
        

        input_df = pd.DataFrame([feature_values])
        

        prediction = model.predict(input_df)[0]
        

        label_encoder = model_package.get('label_encoder')
        original_prediction = prediction
        
        if label_encoder is not None:
            try:
                original_prediction = label_encoder.inverse_transform([int(prediction)])[0]
            except:
                pass
        

        st.success("✅ Prediction completed!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", str(original_prediction))
        
        with col2:
            if label_encoder is not None:
                st.metric("Numeric Code", int(prediction))
        

        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_df)[0]
                
                st.subheader("📊 Prediction Probabilities")
                

                if label_encoder is not None:
                    try:
                        class_labels = [str(label_encoder.inverse_transform([i])[0]) for i in range(len(probabilities))]
                    except:
                        class_labels = [f"Class {i}" for i in range(len(probabilities))]
                else:
                    class_labels = [f"Class {i}" for i in range(len(probabilities))]
                
                prob_df = pd.DataFrame({
                    'Class': class_labels,
                    'Probability': probabilities,
                    'Percentage': (probabilities * 100).round(2)
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(prob_df, use_container_width=True)
                

                fig = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    title='Prediction Probabilities by Class',
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not compute probabilities: {e}")
        

        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Contribution Analysis")
            
            feature_importance = dict(zip(st.session_state.model_features, model.feature_importances_))
            importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

def make_batch_predictions(prediction_data, original_data):
    """
    Make batch predictions with comprehensive results
    """
    try:
        model = st.session_state.classification_model
        model_package = st.session_state.classification_model_package
        
        with st.spinner("Making batch predictions..."):

            predictions = model.predict(prediction_data)
            

            label_encoder = model_package.get('label_encoder')
            original_predictions = predictions.copy()
            
            if label_encoder is not None:
                try:
                    original_predictions = label_encoder.inverse_transform(predictions.astype(int))
                except:
                    pass
            

            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(prediction_data)
                except:
                    pass
            

            results_df = original_data.copy()
            results_df['Prediction'] = original_predictions
            
            if label_encoder is not None:
                results_df['Numeric_Code'] = predictions
            

            if probabilities is not None:
                if label_encoder is not None:
                    try:
                        class_labels = [str(label_encoder.inverse_transform([i])[0]) for i in range(probabilities.shape[1])]
                    except:
                        class_labels = [f"Class_{i}" for i in range(probabilities.shape[1])]
                else:
                    class_labels = [f"Class_{i}" for i in range(probabilities.shape[1])]
                
                for i, class_label in enumerate(class_labels):
                    results_df[f'Prob_{class_label}'] = probabilities[:, i]
            

            results_df['Prediction_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            st.success("✅ Batch predictions completed!")
            

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", len(predictions))
            
            with col2:
                unique_predictions = len(np.unique(original_predictions))
                st.metric("Unique Predictions", unique_predictions)
            
            with col3:
                if probabilities is not None:
                    avg_confidence = np.mean(np.max(probabilities, axis=1))
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            

            st.subheader("📋 Prediction Results")
            st.dataframe(results_df, use_container_width=True)
            

            st.subheader("📊 Prediction Distribution")
            
            pred_counts = pd.Series(original_predictions).value_counts()
            fig = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Distribution of Predictions"
            )
            st.plotly_chart(fig, use_container_width=True)
            

            st.subheader("💾 Download Results")
            
            csv_data = export_predictions_to_csv(
                original_predictions, 
                probabilities, 
                original_data
            )
            
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
            

            if probabilities is not None:
                st.subheader("🎯 Confidence Analysis")
                
                max_probs = np.max(probabilities, axis=1)
                confidence_df = pd.DataFrame({
                    'Prediction': original_predictions,
                    'Confidence': max_probs
                })
                

                fig = px.histogram(
                    confidence_df,
                    x='Confidence',
                    title='Distribution of Prediction Confidence',
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
                

                low_confidence_threshold = st.slider(
                    "Low confidence threshold:",
                    0.0, 1.0, 0.7, 0.05
                )
                
                low_confidence_mask = max_probs < low_confidence_threshold
                low_confidence_count = np.sum(low_confidence_mask)
                
                if low_confidence_count > 0:
                    st.warning(f"⚠️ {low_confidence_count} predictions have confidence below {low_confidence_threshold}")
                    
                    if st.checkbox("Show low confidence predictions"):
                        low_conf_results = results_df[low_confidence_mask]
                        st.dataframe(low_conf_results, use_container_width=True)
                else:
                    st.success(f"✅ All predictions have confidence above {low_confidence_threshold}")
            
    except Exception as e:
        st.error(f"❌ Error making batch predictions: {str(e)}")
        st.error("Please check that your data format matches the model requirements.")
