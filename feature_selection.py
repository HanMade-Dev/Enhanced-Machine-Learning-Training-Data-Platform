import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def safe_session_state_set(key, value):
    """Safely set session state with serializable conversion"""
    try:
        if isinstance(value, pd.DataFrame):
            df_copy = value.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].astype(str)
                elif pd.api.types.is_numeric_dtype(df_copy[col]):
                    if df_copy[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                        df_copy[col] = df_copy[col].astype('int64')
                    elif df_copy[col].dtype in ['float64', 'float32', 'float16']:
                        df_copy[col] = df_copy[col].astype('float64')
            st.session_state[key] = df_copy
        else:
            st.session_state[key] = value
    except Exception as e:
        st.error(f"Error storing {key} in session state: {str(e)}")
        st.session_state[key] = None

def feature_selection_page():
    """
    Enhanced feature selection page with manual labeling and optional automated methods
    """
    st.subheader("🎯 Advanced Feature Selection & Labeling")
    
    if st.session_state.cleaned_data is None:
        st.warning("⚠️ No processed data available. Please complete preprocessing first.")
        return
    
    data = st.session_state.cleaned_data
    st.header("1. Target Definition Method")
    labeling_method = st.radio(
        "Choose how to define your target variable:",
        ["Use existing column", "Create manual rules", "Create range-based rules"],
        help="Select whether to use an existing column or create custom labeling rules"
    )
    
    if labeling_method == "Use existing column":
        st.subheader("📊 Target Column Selection")
        target_column = st.selectbox(
            "Select target column:",
            data.columns.tolist(),
            help="Choose the column you want to predict"
        )
        if target_column and data[target_column].dtype == 'object':
            st.info("🏷️ **String target detected** - Will create label encoder for training while preserving original labels for display")
            original_target_data = data[target_column].copy()
            unique_labels = original_target_data.dropna().unique()
            label_encoder = LabelEncoder()
            target_for_encoding = original_target_data.fillna('Unknown').astype(str)
            encoded_values = label_encoder.fit_transform(target_for_encoding)
            st.session_state.target_label_encoder = label_encoder
            st.session_state.original_target_data = original_target_data
            st.session_state.target_label_mapping = {
                'original_column': target_column,
                'encoder': label_encoder,
                'classes': label_encoder.classes_.tolist(),
                'numeric_to_string': {i: cls for i, cls in enumerate(label_encoder.classes_)},
                'string_to_numeric': {cls: i for i, cls in enumerate(label_encoder.classes_)}
            }
            st.write("**Label Mapping:**")
            mapping_df = pd.DataFrame({
                'Original Label': label_encoder.classes_,
                'Numeric Code': range(len(label_encoder.classes_))
            })
            st.dataframe(mapping_df, use_container_width=True)
            st.session_state.target_column_original = target_column
            st.session_state.target_column_encoded = target_column  # Same name
            data[target_column] = encoded_values
            
        else:
            st.session_state.target_label_encoder = None
            st.session_state.original_target_data = None
            st.session_state.target_label_mapping = None
            st.session_state.target_column_original = target_column
            st.session_state.target_column_encoded = target_column
    
    elif labeling_method == "Create manual rules":
        st.subheader("📝 Enhanced Manual Rule-Based Labeling")
        target_column = enhanced_manual_rule_labeling(data)
    
    else:  # Range-based rules
        st.subheader("📏 Range-Based Labeling")
        target_column = range_based_labeling(data)
    
    if target_column:
        st.session_state.target_column = target_column
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Values", data[target_column].nunique())
        with col2:
            st.metric("Missing Values", data[target_column].isnull().sum())
        with col3:
            is_classification = data[target_column].nunique() < 20 or data[target_column].dtype in ['object', 'int64']
            task_type = "Classification" if is_classification else "Regression"
            st.metric("Task Type", task_type)
        if target_column and target_column in data.columns:
            st.subheader("Target Distribution")
            if is_classification:
                if hasattr(st.session_state, 'original_target_data') and st.session_state.original_target_data is not None:
                    original_target = st.session_state.original_target_data
                    target_counts = original_target.value_counts()
                    fig = px.bar(
                        x=target_counts.index.astype(str), 
                        y=target_counts.values, 
                        title=f"Distribution of {target_column} (Original Labels)",
                        labels={'x': 'Target Classes', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True, key="target_distribution_original")
                    st.success("✅ Displaying original string labels")
                    
                elif hasattr(st.session_state, 'target_label_encoder') and st.session_state.target_label_encoder is not None:
                    label_encoder = st.session_state.target_label_encoder
                    try:
                        original_labels = label_encoder.inverse_transform(data[target_column].astype(int))
                        target_counts = pd.Series(original_labels).value_counts()
                        
                        fig = px.bar(
                            x=target_counts.index.astype(str), 
                            y=target_counts.values, 
                            title=f"Distribution of {target_column} (Original Labels)",
                            labels={'x': 'Target Classes', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True, key="target_distribution_converted")
                        st.success("✅ Converted back to original string labels")
                    except Exception as e:
                        target_counts = data[target_column].value_counts()
                        fig = px.bar(x=target_counts.index.astype(str), y=target_counts.values, 
                                    title=f"Distribution of {target_column} (Numeric)")
                        st.plotly_chart(fig, use_container_width=True, key="target_distribution_numeric")
                        st.warning(f"⚠️ Could not convert to original labels: {e}")
                else:
                    target_counts = data[target_column].value_counts()
                    fig = px.bar(x=target_counts.index.astype(str), y=target_counts.values, 
                                title=f"Distribution of {target_column}")
                    st.plotly_chart(fig, use_container_width=True, key="target_distribution_bar")
            else:
                fig = px.histogram(data, x=target_column, title=f"Distribution of {target_column}")
                st.plotly_chart(fig, use_container_width=True, key="target_distribution_hist")
        st.header("2. Feature Selection Options")
        available_features = [col for col in data.columns if col != target_column]
        numeric_features = data[available_features].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = data[available_features].select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.write(f"**Available features:** {len(available_features)}")
        st.write(f"- Numeric: {len(numeric_features)}")
        st.write(f"- Categorical: {len(categorical_features)}")
        use_feature_selection = st.checkbox(
            "Apply automated feature selection", 
            value=False,
            help="Uncheck to use all available features"
        )
        
        if use_feature_selection:
            tab1, tab2, tab3, tab4 = st.tabs(["🔥 Automated Selection", "🔍 Correlation Analysis", "🎯 Manual Selection", "📊 Final Review"])
            
            with tab1:
                if len(numeric_features) > 0:
                    automated_feature_selection(data, target_column, numeric_features, is_classification)
                else:
                    st.warning("No numeric features available for automated selection.")
            
            with tab2:
                if len(numeric_features) > 0:
                    correlation_analysis(data, numeric_features, target_column)
                else:
                    st.warning("No numeric features available for correlation analysis.")
            
            with tab3:
                manual_feature_selection(data, available_features, target_column)
            
            with tab4:
                final_feature_review(data, target_column)
        else:
            st.session_state.feature_columns = available_features
            safe_session_state_set('labeled_data', data)
            st.success(f"Using all {len(available_features)} available features")

def enhanced_manual_rule_labeling(data):
    """Enhanced manual rule-based labeling with multiple columns and range support"""
    st.write("Create custom labels based on conditions from multiple columns")
    if 'manual_rules' not in st.session_state:
        st.session_state.manual_rules = []
    
    if 'column_rules' not in st.session_state:
        st.session_state.column_rules = {}
    st.subheader("📋 Column Rules Configuration")
    
    available_columns = data.columns.tolist()
    col1, col2 = st.columns([3, 1])
    with col1:
        new_column = st.selectbox(
            "Select column to add rules:",
            [""] + available_columns,
            key="new_column_selector"
        )
    
    with col2:
        if st.button("➕ Add Column Rule", disabled=not new_column):
            if new_column and new_column not in st.session_state.column_rules:
                st.session_state.column_rules[new_column] = {
                    'type': 'condition',  # or 'range'
                    'rules': []
                }
                st.rerun()
    if st.session_state.column_rules:
        st.subheader("🔧 Configure Rules for Each Column")
        
        for column_name in list(st.session_state.column_rules.keys()):
            with st.expander(f"📊 Rules for Column: {column_name}", expanded=True):
                configure_column_rules(data, column_name)
    st.subheader("🎯 Target Column Configuration")
    target_col_name = st.text_input(
        "Target column name:",
        value="target_label",
        help="Name for the new target column"
    )
    
    default_label = st.text_input(
        "Default label (for unmatched cases):",
        value="Other",
        help="Label for rows that don't match any rules"
    )
    if st.button("🚀 Apply All Rules", type="primary"):
        if st.session_state.column_rules:
            try:
                new_target = apply_all_column_rules(data, st.session_state.column_rules, default_label)
                data[target_col_name] = new_target
                unique_labels = new_target.unique()
                if len(unique_labels) > 1 and any(isinstance(label, str) for label in unique_labels):
                    label_encoder = LabelEncoder()
                    encoded_target = label_encoder.fit_transform(new_target.fillna('Unknown').astype(str))
                    st.session_state.target_label_encoder = label_encoder
                    st.session_state.original_target_data = new_target.copy()
                    st.session_state.target_label_mapping = {
                        'original_column': target_col_name,
                        'encoder': label_encoder,
                        'classes': label_encoder.classes_.tolist(),
                        'numeric_to_string': {i: cls for i, cls in enumerate(label_encoder.classes_)},
                        'string_to_numeric': {cls: i for i, cls in enumerate(label_encoder.classes_)}
                    }
                    data[target_col_name] = encoded_target
                    
                    st.info(f"🏷️ Label encoder created for target column with classes: {', '.join(label_encoder.classes_)}")
                safe_session_state_set('cleaned_data', data)
                
                st.success(f"✅ Created new target column: {target_col_name}")
                label_counts = new_target.value_counts()
                fig = px.bar(x=label_counts.index.astype(str), y=label_counts.values, 
                           title=f"Distribution of {target_col_name} (Original Labels)")
                st.plotly_chart(fig, use_container_width=True, key="manual_rule_distribution")
                
                return target_col_name
                
            except Exception as e:
                st.error(f"Error applying rules: {str(e)}")
                return None
        else:
            st.warning("⚠️ Please add at least one column rule before applying.")
            return None
    
    return None

def configure_column_rules(data, column_name):
    """Configure rules for a specific column"""
    column_data = data[column_name]
    column_rules = st.session_state.column_rules[column_name]
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Column Information:**")
        if pd.api.types.is_numeric_dtype(column_data):
            st.write(f"- Type: Numeric")
            st.write(f"- Min: {column_data.min():.2f}")
            st.write(f"- Max: {column_data.max():.2f}")
            st.write(f"- Mean: {column_data.mean():.2f}")
        else:
            st.write(f"- Type: Categorical")
            st.write(f"- Unique values: {column_data.nunique()}")
            if column_data.nunique() <= 10:
                st.write(f"- Values: {', '.join(map(str, column_data.unique()[:10]))}")
    
    with col2:
        rule_type = st.radio(
            f"Rule type for {column_name}:",
            ["Condition", "Range"],
            key=f"rule_type_{column_name}",
            horizontal=True
        )
        column_rules['type'] = rule_type.lower()
    if rule_type == "Condition":
        configure_condition_rules(data, column_name, column_data)
    else:
        configure_range_rules(data, column_name, column_data)
    if st.button(f"🗑️ Remove {column_name} Rules", key=f"remove_{column_name}"):
        del st.session_state.column_rules[column_name]
        st.rerun()

def configure_condition_rules(data, column_name, column_data):
    """Configure condition-based rules for a column"""
    column_rules = st.session_state.column_rules[column_name]
    st.write("**Add Condition Rule:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if pd.api.types.is_numeric_dtype(column_data):
            condition_type = st.selectbox(
                "Condition:",
                [">=", "<=", "==", "!=", ">", "<"],
                key=f"new_cond_type_{column_name}"
            )
        else:
            condition_type = st.selectbox(
                "Condition:",
                ["==", "!=", "contains", "startswith", "endswith"],
                key=f"new_cond_type_{column_name}"
            )
    
    with col2:
        if pd.api.types.is_numeric_dtype(column_data):
            condition_value = st.number_input(
                "Value:",
                value=float(column_data.mean()),
                key=f"new_cond_val_{column_name}"
            )
        else:
            condition_value = st.text_input(
                "Value:",
                key=f"new_cond_val_{column_name}"
            )
    
    with col3:
        label = st.text_input(
            "Label:",
            value=f"Label_{len(column_rules['rules'])+1}",
            key=f"new_label_{column_name}"
        )
    
    with col4:
        if st.button("➕ Add", key=f"add_cond_{column_name}"):
            if label:
                new_rule = {
                    'type': 'condition',
                    'condition_type': condition_type,
                    'condition_value': condition_value,
                    'label': label
                }
                column_rules['rules'].append(new_rule)
                st.rerun()
    if column_rules['rules']:
        st.write("**Current Rules:**")
        for i, rule in enumerate(column_rules['rules']):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"{rule.get('condition_type', 'N/A')}")
            with col2:
                st.write(f"{rule.get('condition_value', 'N/A')}")
            with col3:
                st.write(f"{rule.get('label', 'N/A')}")
            with col4:
                if st.button("🗑️", key=f"del_cond_{column_name}_{i}"):
                    column_rules['rules'].pop(i)
                    st.rerun()

def configure_range_rules(data, column_name, column_data):
    """Configure range-based rules for a column - PERBAIKAN: Remove Include MIN/MAX checkboxes"""
    column_rules = st.session_state.column_rules[column_name]
    
    if not pd.api.types.is_numeric_dtype(column_data):
        st.warning("⚠️ Range rules are only available for numeric columns")
        return
    
    st.write("**Value ranges for column {}:**".format(column_name))
    st.write("**Add Range Rule:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_val = st.number_input(
            "Min:",
            value=float(column_data.min()),
            key=f"new_range_min_{column_name}"
        )
    
    with col2:
        max_val = st.number_input(
            "Max:",
            value=float(column_data.max()),
            key=f"new_range_max_{column_name}"
        )
    
    with col3:
        label = st.text_input(
            "Label:",
            value=f"Category_{len(column_rules['rules'])+1}",
            key=f"new_range_label_{column_name}"
        )
    
    with col4:
        if st.button("➕ Add Range", key=f"add_range_{column_name}"):
            if label and min_val < max_val:
                new_rule = {
                    'type': 'range',
                    'min_value': min_val,
                    'max_value': max_val,
                    'label': label,
                    'include_min': True,  # Default values
                    'include_max': False
                }
                column_rules['rules'].append(new_rule)
                st.rerun()
            elif min_val >= max_val:
                st.error("❌ Min value must be less than Max value")
    if column_rules['rules']:
        st.write("**Current Ranges:**")
        for i, rule in enumerate(column_rules['rules']):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("Min")
                st.write(f"{rule['min_value']:.2f}")
            with col2:
                st.write("Max")
                st.write(f"{rule['max_value']:.2f}")
            with col3:
                st.write("Label")
                st.write(f"{rule['label']}")
            with col4:
                if st.button("🗑️", key=f"del_range_{column_name}_{i}"):
                    column_rules['rules'].pop(i)
                    st.rerun()

def apply_all_column_rules(data, column_rules, default_label):
    """Apply all column rules to create target labels"""
    target = pd.Series([default_label] * len(data), index=data.index)
    for column_name, column_config in column_rules.items():
        column_data = data[column_name]
        
        for rule in column_config['rules']:
            if rule['type'] == 'condition':
                mask = apply_condition_rule(column_data, rule)
            elif rule['type'] == 'range':
                mask = apply_range_rule(column_data, rule)
            else:
                continue
            target[mask] = rule['label']
    
    return target

def apply_condition_rule(column_data, rule):
    """Apply a condition rule to column data"""
    condition_type = rule['condition_type']
    condition_value = rule['condition_value']
    
    try:
        if pd.api.types.is_numeric_dtype(column_data):
            if condition_type == ">=":
                return column_data >= condition_value
            elif condition_type == "<=":
                return column_data <= condition_value
            elif condition_type == "==":
                return column_data == condition_value
            elif condition_type == "!=":
                return column_data != condition_value
            elif condition_type == ">":
                return column_data > condition_value
            elif condition_type == "<":
                return column_data < condition_value
        else:
            if condition_type == "==":
                return column_data == condition_value
            elif condition_type == "!=":
                return column_data != condition_value
            elif condition_type == "contains":
                return column_data.str.contains(str(condition_value), na=False)
            elif condition_type == "startswith":
                return column_data.str.startswith(str(condition_value), na=False)
            elif condition_type == "endswith":
                return column_data.str.endswith(str(condition_value), na=False)
    except Exception as e:
        st.warning(f"Error applying condition rule: {e}")
        return pd.Series([False] * len(column_data), index=column_data.index)
    
    return pd.Series([False] * len(column_data), index=column_data.index)

def apply_range_rule(column_data, rule):
    """Apply a range rule to column data"""
    try:
        min_val = rule['min_value']
        max_val = rule['max_value']
        include_min = rule.get('include_min', True)
        include_max = rule.get('include_max', False)
        
        if include_min and include_max:
            return (column_data >= min_val) & (column_data <= max_val)
        elif include_min and not include_max:
            return (column_data >= min_val) & (column_data < max_val)
        elif not include_min and include_max:
            return (column_data > min_val) & (column_data <= max_val)
        else:
            return (column_data > min_val) & (column_data < max_val)
    except Exception as e:
        st.warning(f"Error applying range rule: {e}")
        return pd.Series([False] * len(column_data), index=column_data.index)

def range_based_labeling(data):
    """Create range-based labeling"""
    st.write("Create labels based on value ranges")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.warning("No numeric columns available for range-based labeling")
        return None
    
    base_column = st.selectbox(
        "Select numeric column:",
        numeric_columns,
        help="Choose the numeric column to create range-based labels"
    )
    
    if base_column:
        col_stats = data[base_column].describe()
        st.write("**Column Statistics:**")
        st.write(col_stats)
        st.write("**Create Value Ranges:**")
        
        method = st.radio(
            "Range creation method:",
            ["Equal intervals", "Quantiles", "Custom ranges"]
        )
        
        if method == "Equal intervals":
            num_bins = st.slider("Number of intervals:", 2, 10, 3)
            bin_labels = []
            for i in range(num_bins):
                label = st.text_input(f"Label for interval {i+1}:", value=f"Range_{i+1}", key=f"eq_label_{i}")
                bin_labels.append(label)
            
            if st.button("Create Equal Intervals"):
                try:
                    new_target = pd.cut(data[base_column], bins=num_bins, labels=bin_labels)
                    target_col_name = f"{base_column}_ranges"
                    st.session_state.original_target_data = new_target.copy()
                    if any(isinstance(label, str) for label in bin_labels):
                        label_encoder = LabelEncoder()
                        encoded_target = label_encoder.fit_transform(new_target.fillna('Unknown').astype(str))
                        
                        st.session_state.target_label_encoder = label_encoder
                        st.session_state.target_label_mapping = {
                            'original_column': target_col_name,
                            'encoder': label_encoder,
                            'classes': label_encoder.classes_.tolist(),
                            'numeric_to_string': {i: cls for i, cls in enumerate(label_encoder.classes_)},
                            'string_to_numeric': {cls: i for i, cls in enumerate(label_encoder.classes_)}
                        }
                        data[target_col_name] = encoded_target
                    else:
                        data[target_col_name] = new_target
                    
                    safe_session_state_set('cleaned_data', data)
                    st.success(f"✅ Created range-based target: {target_col_name}")
                    range_counts = new_target.value_counts()
                    fig = px.bar(x=range_counts.index.astype(str), y=range_counts.values, 
                               title=f"Distribution of {target_col_name} (Original Labels)")
                    st.plotly_chart(fig, use_container_width=True, key="equal_intervals_distribution")
                    
                    return target_col_name
                    
                except Exception as e:
                    st.error(f"Error creating ranges: {str(e)}")
                    return None
        
        elif method == "Quantiles":
            num_quantiles = st.slider("Number of quantiles:", 2, 10, 4)
            quantile_labels = []
            for i in range(num_quantiles):
                label = st.text_input(f"Label for quantile {i+1}:", value=f"Q{i+1}", key=f"q_label_{i}")
                quantile_labels.append(label)
            
            if st.button("Create Quantiles"):
                try:
                    new_target = pd.qcut(data[base_column], q=num_quantiles, labels=quantile_labels)
                    target_col_name = f"{base_column}_quantiles"
                    st.session_state.original_target_data = new_target.copy()
                    if any(isinstance(label, str) for label in quantile_labels):
                        label_encoder = LabelEncoder()
                        encoded_target = label_encoder.fit_transform(new_target.fillna('Unknown').astype(str))
                        
                        st.session_state.target_label_encoder = label_encoder
                        st.session_state.target_label_mapping = {
                            'original_column': target_col_name,
                            'encoder': label_encoder,
                            'classes': label_encoder.classes_.tolist(),
                            'numeric_to_string': {i: cls for i, cls in enumerate(label_encoder.classes_)},
                            'string_to_numeric': {cls: i for i, cls in enumerate(label_encoder.classes_)}
                        }
                        data[target_col_name] = encoded_target
                    else:
                        data[target_col_name] = new_target
                    
                    safe_session_state_set('cleaned_data', data)
                    st.success(f"✅ Created quantile-based target: {target_col_name}")
                    quantile_counts = new_target.value_counts()
                    fig = px.bar(x=quantile_counts.index.astype(str), y=quantile_counts.values, 
                               title=f"Distribution of {target_col_name} (Original Labels)")
                    st.plotly_chart(fig, use_container_width=True, key="quantiles_distribution")
                    
                    return target_col_name
                    
                except Exception as e:
                    st.error(f"Error creating quantiles: {str(e)}")
                    return None
        
        else:  # Custom ranges
            st.write("Define custom ranges:")
            min_val = float(data[base_column].min())
            max_val = float(data[base_column].max())
            
            st.write(f"Column range: {min_val:.2f} to {max_val:.2f}")
            
            num_ranges = st.number_input("Number of custom ranges:", min_value=2, max_value=10, value=3)
            
            ranges = []
            labels = []
            
            for i in range(num_ranges):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if i == 0:
                        start = st.number_input(f"Range {i+1} start:", value=min_val, key=f"start_{i}")
                    else:
                        start = ranges[i-1][1]  # End of previous range
                        st.write(f"Range {i+1} start: {start}")
                
                with col2:
                    if i == num_ranges - 1:
                        end = st.number_input(f"Range {i+1} end:", value=max_val, key=f"end_{i}")
                    else:
                        end = st.number_input(f"Range {i+1} end:", value=min_val + (max_val - min_val) * (i+1) / num_ranges, key=f"end_{i}")
                
                with col3:
                    label = st.text_input(f"Label {i+1}:", value=f"Range_{i+1}", key=f"custom_label_{i}")
                
                ranges.append((start, end))
                labels.append(label)
            
            if st.button("Create Custom Ranges"):
                try:
                    bins = [ranges[0][0]] + [r[1] for r in ranges]
                    new_target = pd.cut(data[base_column], bins=bins, labels=labels, include_lowest=True)
                    target_col_name = f"{base_column}_custom_ranges"
                    st.session_state.original_target_data = new_target.copy()
                    if any(isinstance(label, str) for label in labels):
                        label_encoder = LabelEncoder()
                        encoded_target = label_encoder.fit_transform(new_target.fillna('Unknown').astype(str))
                        
                        st.session_state.target_label_encoder = label_encoder
                        st.session_state.target_label_mapping = {
                            'original_column': target_col_name,
                            'encoder': label_encoder,
                            'classes': label_encoder.classes_.tolist(),
                            'numeric_to_string': {i: cls for i, cls in enumerate(label_encoder.classes_)},
                            'string_to_numeric': {cls: i for i, cls in enumerate(label_encoder.classes_)}
                        }
                        data[target_col_name] = encoded_target
                    else:
                        data[target_col_name] = new_target
                    
                    safe_session_state_set('cleaned_data', data)
                    st.success(f"✅ Created custom range-based target: {target_col_name}")
                    range_counts = new_target.value_counts()
                    fig = px.bar(x=range_counts.index.astype(str), y=range_counts.values, 
                               title=f"Distribution of {target_col_name} (Original Labels)")
                    st.plotly_chart(fig, use_container_width=True, key="custom_ranges_distribution")
                    
                    return target_col_name
                    
                except Exception as e:
                    st.error(f"Error creating custom ranges: {str(e)}")
                    return None
    
    return None

def automated_feature_selection(data, target_column, numeric_features, is_classification):
    """
    Automated feature selection using various methods
    """
    st.subheader("🤖 Automated Feature Selection Methods")
    
    if len(numeric_features) < 2:
        st.warning("⚠️ Need at least 2 numeric features for automated selection.")
        return
    X = data[numeric_features].fillna(0)  # Fill NaN values for feature selection
    y = data[target_column]
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        st.warning("⚠️ No valid data for feature selection after removing missing targets.")
        return
    selection_method = st.selectbox(
        "Choose automated selection method:",
        [
            "Statistical Tests (SelectKBest)",
            "Mutual Information",
            "Recursive Feature Elimination (RFE)",
            "Random Forest Importance",
            "All Methods Combined"
        ]
    )
    max_features = min(20, len(numeric_features))
    n_features = st.slider(
        "Number of features to select:",
        min_value=1,
        max_value=max_features,
        value=min(10, max_features),
        help="Maximum number of features to select"
    )
    
    if st.button("🚀 Run Automated Selection", type="primary"):
        with st.spinner("Running feature selection..."):
            
            results = {}
            if selection_method in ["Statistical Tests (SelectKBest)", "All Methods Combined"]:
                if is_classification:
                    selector = SelectKBest(score_func=f_classif, k=n_features)
                else:
                    selector = SelectKBest(score_func=f_regression, k=n_features)
                
                try:
                    X_selected = selector.fit_transform(X, y)
                    selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
                    scores = selector.scores_
                    
                    results['SelectKBest'] = {
                        'features': selected_features,
                        'scores': dict(zip(numeric_features, scores.tolist())),
                        'method': 'Statistical Tests'
                    }
                except Exception as e:
                    st.error(f"Error in SelectKBest: {str(e)}")
            if selection_method in ["Mutual Information", "All Methods Combined"]:
                try:
                    if is_classification:
                        mi_scores = mutual_info_classif(X, y, random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X, y, random_state=42)
                    feature_scores = list(zip(numeric_features, mi_scores.tolist()))
                    feature_scores.sort(key=lambda x: x[1], reverse=True)
                    selected_features = [f[0] for f in feature_scores[:n_features]]
                    
                    results['Mutual_Information'] = {
                        'features': selected_features,
                        'scores': dict(feature_scores),
                        'method': 'Mutual Information'
                    }
                except Exception as e:
                    st.error(f"Error in Mutual Information: {str(e)}")
            if selection_method in ["Recursive Feature Elimination (RFE)", "All Methods Combined"]:
                try:
                    if is_classification:
                        estimator = LogisticRegression(random_state=42, max_iter=1000)
                    else:
                        estimator = LinearRegression()
                    
                    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
                    rfe.fit(X, y)
                    
                    selected_features = [numeric_features[i] for i in rfe.get_support(indices=True)]
                    rankings = dict(zip(numeric_features, rfe.ranking_.tolist()))
                    
                    results['RFE'] = {
                        'features': selected_features,
                        'scores': {f: 1/r for f, r in rankings.items()},  # Convert ranking to score
                        'method': 'Recursive Feature Elimination'
                    }
                except Exception as e:
                    st.error(f"Error in RFE: {str(e)}")
            if selection_method in ["Random Forest Importance", "All Methods Combined"]:
                try:
                    if is_classification:
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    rf.fit(X, y)
                    importances = rf.feature_importances_
                    feature_importance = list(zip(numeric_features, importances.tolist()))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    selected_features = [f[0] for f in feature_importance[:n_features]]
                    
                    results['Random_Forest'] = {
                        'features': selected_features,
                        'scores': dict(feature_importance),
                        'method': 'Random Forest Importance'
                    }
                except Exception as e:
                    st.error(f"Error in Random Forest: {str(e)}")
            if results:
                display_feature_selection_results(results, selection_method)
                st.session_state.feature_selection_results = results

def correlation_analysis(data, numeric_features, target_column):
    """
    Correlation analysis and multicollinearity detection
    """
    st.subheader("🔗 Correlation Analysis & Multicollinearity Detection")
    
    if len(numeric_features) < 2:
        st.warning("⚠️ Need at least 2 numeric features for correlation analysis.")
        return
    st.write("**Correlation with Target Variable**")
    
    target_correlations = {}
    for feature in numeric_features:
        try:
            corr, p_value = pearsonr(data[feature].fillna(0), data[target_column].fillna(0))
            target_correlations[feature] = {'correlation': float(corr), 'p_value': float(p_value)}
        except:
            target_correlations[feature] = {'correlation': 0.0, 'p_value': 1.0}
    corr_df = pd.DataFrame(target_correlations).T
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    fig = px.bar(
        x=corr_df.index, 
        y=corr_df['correlation'].abs(), 
        title="Absolute Correlation with Target",
        labels={'x': 'Features', 'y': 'Absolute Correlation'}
    )
    st.plotly_chart(fig, use_container_width=True, key="target_correlation_bar")
    st.dataframe(corr_df.round(4), use_container_width=True)
    st.write("**Feature-to-Feature Correlation Matrix**")
    
    correlation_threshold = st.slider(
        "Correlation threshold for multicollinearity:",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="Features with correlation above this threshold are considered highly correlated"
    )
    feature_corr_matrix = data[numeric_features].corr()
    fig = px.imshow(
        feature_corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
    highly_correlated_pairs = []
    for i in range(len(feature_corr_matrix.columns)):
        for j in range(i+1, len(feature_corr_matrix.columns)):
            corr_value = abs(feature_corr_matrix.iloc[i, j])
            if corr_value > correlation_threshold:
                highly_correlated_pairs.append({
                    'Feature 1': feature_corr_matrix.columns[i],
                    'Feature 2': feature_corr_matrix.columns[j],
                    'Correlation': float(corr_value)
                })
    
    if highly_correlated_pairs:
        st.warning(f"⚠️ Found {len(highly_correlated_pairs)} highly correlated feature pairs (>{correlation_threshold}):")
        hc_df = pd.DataFrame(highly_correlated_pairs)
        st.dataframe(hc_df.round(4), use_container_width=True)
        features_to_remove = set()
        for pair in highly_correlated_pairs:
            feat1_target_corr = abs(target_correlations[pair['Feature 1']]['correlation'])
            feat2_target_corr = abs(target_correlations[pair['Feature 2']]['correlation'])
            
            if feat1_target_corr < feat2_target_corr:
                features_to_remove.add(pair['Feature 1'])
            else:
                features_to_remove.add(pair['Feature 2'])
        
        if features_to_remove:
            st.info(f"💡 Suggested features to remove due to multicollinearity: {', '.join(features_to_remove)}")
            
            if st.button("Apply Multicollinearity Filtering"):
                remaining_features = [f for f in numeric_features if f not in features_to_remove]
                st.session_state.correlation_filtered_features = remaining_features
                st.success(f"✅ Filtered features. Remaining: {len(remaining_features)} features")
    else:
        st.success("✅ No significant multicollinearity detected!")

def manual_feature_selection(data, available_features, target_column):
    """
    Manual feature selection interface
    """
    st.subheader("✋ Manual Feature Selection")
    st.write("**Available Features Information:**")
    
    feature_info = []
    for feature in available_features:
        info = {
            'Feature': feature,
            'Type': str(data[feature].dtype),
            'Missing': int(data[feature].isnull().sum()),
            'Unique': int(data[feature].nunique()),
            'Example': str(data[feature].iloc[0]) if len(data) > 0 else 'N/A'
        }
        feature_info.append(info)
    
    feature_df = pd.DataFrame(feature_info)
    st.dataframe(feature_df, use_container_width=True)
    st.write("**Select Features for Training:**")
    default_features = available_features
    if hasattr(st.session_state, 'feature_selection_results'):
        results = st.session_state.feature_selection_results
        if results:
            first_method = list(results.keys())[0]
            default_features = results[first_method]['features']
    elif hasattr(st.session_state, 'correlation_filtered_features'):
        default_features = st.session_state.correlation_filtered_features
    
    selected_features = st.multiselect(
        "Choose features to include in the model:",
        available_features,
        default=default_features,
        help="Select the features you want to use for training the model"
    )
    
    if selected_features:
        st.success(f"✅ Selected {len(selected_features)} features")
        if len(selected_features) > 0:
            numeric_selected = [f for f in selected_features if f in data.select_dtypes(include=[np.number]).columns]
            
            if len(numeric_selected) > 0:
                st.write("**Correlation of Selected Features with Target:**")
                correlations = []
                for feature in numeric_selected:
                    try:
                        corr, _ = pearsonr(data[feature].fillna(0), data[target_column].fillna(0))
                        correlations.append({'Feature': feature, 'Correlation': float(corr)})
                    except:
                        correlations.append({'Feature': feature, 'Correlation': 0.0})
                
                corr_df = pd.DataFrame(correlations)
                fig = px.bar(corr_df, x='Feature', y='Correlation', 
                           title="Selected Features Correlation with Target")
                st.plotly_chart(fig, use_container_width=True, key="selected_features_correlation")
        st.session_state.feature_columns = selected_features
        final_columns = selected_features + [target_column]
        labeled_data = data[final_columns].copy()
        labeled_data = labeled_data.dropna(subset=[target_column])
        
        safe_session_state_set('labeled_data', labeled_data)
        
        st.info(f"📊 Final dataset shape: {labeled_data.shape}")

def final_feature_review(data, target_column):
    """
    Final review of selected features
    """
    st.subheader("📋 Final Feature Review")
    
    if st.session_state.labeled_data is None:
        st.warning("⚠️ No features selected yet. Please complete feature selection first.")
        return
    
    labeled_data = st.session_state.labeled_data
    feature_columns = st.session_state.feature_columns
    
    st.success(f"✅ Dataset ready for training!")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(labeled_data))
    with col2:
        st.metric("Features", len(feature_columns))
    with col3:
        st.metric("Target", target_column)
    with col4:
        missing_count = labeled_data.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    st.write("**Final Dataset Preview:**")
    st.dataframe(labeled_data.head(10), use_container_width=True)
    st.write("**Selected Features Summary:**")
    feature_summary = []
    for feature in feature_columns:
        summary = {
            'Feature': feature,
            'Type': str(labeled_data[feature].dtype),
            'Missing': int(labeled_data[feature].isnull().sum()),
            'Unique Values': int(labeled_data[feature].nunique()),
            'Min': labeled_data[feature].min() if pd.api.types.is_numeric_dtype(labeled_data[feature]) else 'N/A',
            'Max': labeled_data[feature].max() if pd.api.types.is_numeric_dtype(labeled_data[feature]) else 'N/A',
        }
        feature_summary.append(summary)
    
    summary_df = pd.DataFrame(feature_summary)
    st.dataframe(summary_df, use_container_width=True)
    st.write("**Target Distribution in Final Dataset:**")
    is_classification = labeled_data[target_column].nunique() < 20 or labeled_data[target_column].dtype in ['object', 'int64']

    if is_classification:
        if hasattr(st.session_state, 'original_target_data') and st.session_state.original_target_data is not None:
            original_target = st.session_state.original_target_data
            target_counts = original_target.value_counts()
            st.success("✅ Displaying original string labels")
            title_suffix = " (Original Labels)"
        elif hasattr(st.session_state, 'target_label_encoder') and st.session_state.target_label_encoder is not None:
            try:
                label_encoder = st.session_state.target_label_encoder
                original_labels = label_encoder.inverse_transform(labeled_data[target_column].astype(int))
                target_counts = pd.Series(original_labels).value_counts()
                st.success("✅ Converted back to original string labels")
                title_suffix = " (Original Labels)"
            except Exception as e:
                target_counts = labeled_data[target_column].value_counts()
                title_suffix = " (Numeric)"
                st.warning(f"⚠️ Could not convert to original labels: {e}")
        else:
            target_counts = labeled_data[target_column].value_counts()
            title_suffix = ""
    
        fig = px.pie(values=target_counts.values, names=target_counts.index.astype(str), 
                    title=f"Final Target Distribution: {target_column}{title_suffix}")
        st.plotly_chart(fig, use_container_width=True, key="final_target_pie")
        if len(target_counts) > 1:
            imbalance_ratio = target_counts.max() / target_counts.min()
            if imbalance_ratio > 3:
                st.warning(f"⚠️ Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
                st.info("💡 Consider using stratified sampling or class balancing techniques.")
    else:
        fig = px.histogram(labeled_data, x=target_column, 
                          title=f"Final Target Distribution: {target_column}")
        st.plotly_chart(fig, use_container_width=True, key="final_target_hist")

def display_feature_selection_results(results, selection_method):
    """
    Display feature selection results in a comprehensive way
    """
    st.subheader("🎯 Feature Selection Results")
    
    if selection_method == "All Methods Combined":
        st.write("**Comparison of Feature Selection Methods:**")
        
        method_comparison = {}
        all_features = set()
        
        for method_name, result in results.items():
            method_comparison[result['method']] = result['features']
            all_features.update(result['features'])
        comparison_data = []
        for feature in all_features:
            row = {'Feature': feature}
            for method, features in method_comparison.items():
                row[method] = '✓' if feature in features else ''
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        st.write("**Method Agreement Analysis:**")
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for method, features in method_comparison.items() if feature in features)
            feature_votes[feature] = votes
        
        votes_df = pd.DataFrame(list(feature_votes.items()), columns=['Feature', 'Votes'])
        votes_df = votes_df.sort_values('Votes', ascending=False)
        
        fig = px.bar(votes_df, x='Feature', y='Votes', 
                    title="Feature Selection Agreement Across Methods")
        st.plotly_chart(fig, use_container_width=True, key="method_agreement_bar")
        consensus_threshold = len(results) // 2 + 1
        consensus_features = [f for f, v in feature_votes.items() if v >= consensus_threshold]
        
        if consensus_features:
            st.success(f"✅ Consensus features (selected by ≥{consensus_threshold} methods): {consensus_features}")
            
            if st.button("Use Consensus Features"):
                st.session_state.feature_columns = consensus_features
                st.success("Consensus features selected!")
    
    else:
        method_name = list(results.keys())[0]
        result = results[method_name]
        
        st.write(f"**{result['method']} Results:**")
        st.write(f"Selected features: {result['features']}")
        scores_df = pd.DataFrame(list(result['scores'].items()), columns=['Feature', 'Score'])
        scores_df = scores_df.sort_values('Score', ascending=False)
        
        fig = px.bar(scores_df.head(20), x='Feature', y='Score', 
                    title=f"Feature Scores - {result['method']}")
        st.plotly_chart(fig, use_container_width=True, key="single_method_scores")
        
        if st.button(f"Use {result['method']} Features"):
            st.session_state.feature_columns = result['features']
            st.success(f"{result['method']} features selected!")
