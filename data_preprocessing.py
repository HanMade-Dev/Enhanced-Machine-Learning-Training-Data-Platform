import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

def convert_dtypes_safely(df):
    """Convert DataFrame dtypes to avoid JSON serialization issues"""
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            try:
                numeric_series = pd.to_numeric(df_copy[col], errors='coerce')
                if not numeric_series.isna().all():
                    non_null_ratio = numeric_series.notna().sum() / len(numeric_series)
                    if non_null_ratio > 0.7:  # If more than 70% can be converted
                        df_copy[col] = numeric_series
                        continue
            except:
                pass
            df_copy[col] = df_copy[col].astype(str).replace('nan', np.nan)
        elif pd.api.types.is_numeric_dtype(df_copy[col]):
            if df_copy[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                df_copy[col] = df_copy[col].astype('int64')
            elif df_copy[col].dtype in ['float64', 'float32', 'float16']:
                df_copy[col] = df_copy[col].astype('float64')
        elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    
    return df_copy

def load_data(uploaded_files, header_row=0, headers_not_in_first_row=False):
    """
    Enhanced load data function with better preview and validation
    """
    if not uploaded_files:
        return None
    
    all_dataframes = []
    file_info = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                delimiter = detect_csv_delimiter(uploaded_file)
                
                if headers_not_in_first_row:
                    df = pd.read_csv(uploaded_file, header=header_row, sep=delimiter)
                else:
                    df = pd.read_csv(uploaded_file, sep=delimiter)
                    
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                if headers_not_in_first_row:
                    df = pd.read_excel(uploaded_file, header=header_row)
                else:
                    df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue
            df.columns = df.columns.astype(str)
            df = convert_dtypes_safely(df)
            
            all_dataframes.append(df)
            file_info.append({
                'name': uploaded_file.name,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': df.isnull().sum().to_dict()
            })
            
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
            continue
    
    if not all_dataframes:
        return None
    try:
        if len(all_dataframes) == 1:
            combined_df = all_dataframes[0]
        else:
            combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        combined_df = convert_dtypes_safely(combined_df)
        
        return combined_df
        
    except Exception as e:
        st.error(f"❌ Error combining dataframes: {str(e)}")
        return None

def detect_csv_delimiter(uploaded_file):
    """
    Detect CSV delimiter automatically
    """
    uploaded_file.seek(0)
    sample = uploaded_file.read(1024).decode('utf-8')
    uploaded_file.seek(0)
    
    delimiters = [',', ';', '\t', '|']
    delimiter_counts = {}
    
    for delimiter in delimiters:
        count = sample.count(delimiter)
        delimiter_counts[delimiter] = count
    best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
    if delimiter_counts[best_delimiter] == 0:
        return ','
    
    return best_delimiter

def enhanced_data_preview(data):
    """
    Enhanced data preview with detailed statistics and visualizations
    """
    st.subheader("📊 Data Analysis Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📋 Data Types", "🔍 Missing Values", "📊 Statistics"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", data.shape[0])
        with col2:
            st.metric("Columns", data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Missing Values", data.isnull().sum().sum())
        st.subheader("Data Preview")
        st.dataframe(data.head(20), use_container_width=True)
    
    with tab2:
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': [str(dtype) for dtype in data.dtypes],
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum(),
            'Null Percentage': (data.isnull().sum() / len(data) * 100).round(2)
        })
        
        st.dataframe(dtype_df, use_container_width=True)
        type_counts = data.dtypes.astype(str).value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                    title="Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if data.isnull().sum().sum() > 0:
            missing_data = data.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                fig = px.bar(x=missing_data.values, y=missing_data.index, 
                           orientation='h', title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
                if len(missing_data) <= 20:  # Only for manageable number of columns
                    missing_matrix = data[missing_data.index].isnull().astype(int)
                    fig = px.imshow(missing_matrix.T, aspect="auto", 
                                  title="Missing Values Pattern")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    with tab4:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            st.subheader("Numeric Columns Statistics")
            st.dataframe(data[numeric_columns].describe(), use_container_width=True)
            if len(numeric_columns) > 0:
                cols_to_plot = numeric_columns[:6]
                for i in range(0, len(cols_to_plot), 2):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if i < len(cols_to_plot):
                            fig = px.histogram(data, x=cols_to_plot[i], 
                                             title=f"Distribution of {cols_to_plot[i]}")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if i + 1 < len(cols_to_plot):
                            fig = px.histogram(data, x=cols_to_plot[i + 1], 
                                             title=f"Distribution of {cols_to_plot[i + 1]}")
                            st.plotly_chart(fig, use_container_width=True)
        categorical_columns = data.select_dtypes(include=['object', 'string', 'category']).columns
        if len(categorical_columns) > 0:
            st.subheader("Categorical Columns Analysis")
            for col in categorical_columns[:5]:  
                unique_count = data[col].nunique()
                st.write(f"**{col}**: {unique_count} unique values")
                
                if unique_count <= 10:  # Show value counts for columns with few unique values
                    value_counts = data[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.values, y=value_counts.index, 
                               orientation='h', title=f"Value Counts for {col}")
                    st.plotly_chart(fig, use_container_width=True)

def handle_missing_values(data, strategy, selected_columns=None, numeric_method="Mean", 
                         categorical_method="Mode", categorical_fill_value="Unknown"):
    """
    Enhanced missing values handling with more options
    """
    df = data.copy()
    
    if strategy == "Remove rows with any missing values":
        df = df.dropna()
    
    elif strategy == "Remove rows with missing values in selected columns only":
        if selected_columns:
            df = df.dropna(subset=selected_columns)
    
    elif strategy == "Fill missing values (Imputation)":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if numeric_method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_method == "Zero":
                    df[col].fillna(0, inplace=True)
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
        
        for col in categorical_cols:
            if df[col].isnull().any():
                if categorical_method == "Mode":
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                    else:
                        df[col].fillna("Unknown", inplace=True)
                elif categorical_method == "A constant value":
                    df[col].fillna(categorical_fill_value, inplace=True)
    df = convert_dtypes_safely(df)
    
    return df

def detect_and_handle_outliers(data, method="IQR", action="mark"):
    """
    Enhanced outlier detection and handling
    """
    df = data.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_columns:
        outliers_mask = None
        
        if method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == "Z-Score":
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = 3
            outliers_mask = pd.Series(False, index=df.index)
            outliers_mask.iloc[df[col].dropna().index] = z_scores > threshold
            
        elif method == "Modified Z-Score":
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            if mad != 0:
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outliers_mask = np.abs(modified_z_scores) > 3.5
            else:
                outliers_mask = pd.Series(False, index=df.index)
        
        if outliers_mask is not None:
            outlier_count = outliers_mask.sum()
            outlier_info[col] = {
                'count': int(outlier_count),
                'percentage': float((outlier_count / len(df)) * 100),
                'values': df.loc[outliers_mask, col].tolist()[:10]  # Limit to first 10 values
            }
            
            if action == "remove" and outlier_count > 0:
                df = df[~outliers_mask]
            elif action == "clip" and outlier_count > 0:
                if method == "IQR":
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                elif method in ["Z-Score", "Modified Z-Score"]:
                    upper_clip = df[col].quantile(0.99)
                    lower_clip = df[col].quantile(0.01)
                    df[col] = df[col].clip(lower=lower_clip, upper=upper_clip)
            elif action == "mark" and outlier_count > 0:
                df[f'{col}_outlier'] = outliers_mask
    df = convert_dtypes_safely(df)
    
    return df, outlier_info

def advanced_feature_engineering(data):
    """
    Advanced feature engineering for datetime and categorical columns
    """
    df = data.copy()
    new_features = []
    datetime_columns = df.select_dtypes(include=['datetime64']).columns
    
    for col in datetime_columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        df[f'{col}_quarter'] = df[col].dt.quarter
        df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        new_features.extend([
            f'{col}_year', f'{col}_month', f'{col}_day', 
            f'{col}_dayofweek', f'{col}_quarter', f'{col}_is_weekend'
        ])
    categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns
    
    for col in categorical_columns:
        if f'{col}_encoded' in df.columns:
            continue
            
        unique_count = df[col].nunique()
        if unique_count <= 10 and unique_count > 1:
            try:
                df[col] = df[col].fillna('Unknown')
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                new_features.extend(dummies.columns.tolist())
                df = df.drop(columns=[col])
                
            except Exception as e:
                st.warning(f"Error in one-hot encoding for {col}: {str(e)}")
                try:
                    df[col] = df[col].fillna('Unknown')
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    new_features.append(f'{col}_encoded')
                    df = df.drop(columns=[col])
                except Exception as e2:
                    st.warning(f"Error in label encoding for {col}: {str(e2)}")
        else:
            try:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                new_features.append(f'{col}_encoded')
                df = df.drop(columns=[col])
            except Exception as e:
                st.warning(f"Error in label encoding for {col}: {str(e)}")
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
            except:
                df = df.drop(columns=[col])
                if col in new_features:
                    new_features.remove(col)
    df = convert_dtypes_safely(df)
    
    return df, new_features

def normalize_data(data, method="Min-Max Scaling"):
    """
    Enhanced data normalization with multiple methods
    """
    df = data.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        return df
    
    if method == "Min-Max Scaling":
        scaler = MinMaxScaler()
    elif method == "Standard Scaling (Z-score)":
        scaler = StandardScaler()
    elif method == "Robust Scaling":
        scaler = RobustScaler()
    else:
        return df
    
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    df = convert_dtypes_safely(df)
    
    return df

def show_preprocessing_summary(original_data, processed_data, outlier_info=None, new_features=None):
    """
    Show comprehensive preprocessing summary
    """
    st.subheader("📊 Preprocessing Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Preprocessing:**")
        st.write(f"- Rows: {original_data.shape[0]}")
        st.write(f"- Columns: {original_data.shape[1]}")
        st.write(f"- Missing values: {original_data.isnull().sum().sum()}")
        st.write(f"- Memory usage: {original_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.write("**After Preprocessing:**")
        st.write(f"- Rows: {processed_data.shape[0]}")
        st.write(f"- Columns: {processed_data.shape[1]}")
        st.write(f"- Missing values: {processed_data.isnull().sum().sum()}")
        st.write(f"- Memory usage: {processed_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    if outlier_info:
        st.write("**Outliers Detected:**")
        outlier_df = pd.DataFrame(outlier_info).T
        if not outlier_df.empty:
            outlier_df.columns = ['Count', 'Percentage', 'Values']
            outlier_df['Percentage'] = outlier_df['Percentage'].round(2)
            st.dataframe(outlier_df[['Count', 'Percentage']], use_container_width=True)
    if new_features:
        st.write(f"**New Features Created:** {len(new_features)}")
        with st.expander("View new features"):
            st.write(new_features)

def clean_indonesian_numbers(df):
    """
    Clean Indonesian number format (with dots as thousand separators)
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            sample_values = df_clean[col].dropna().astype(str).head(10)
            numeric_pattern_count = 0
            for val in sample_values:
                cleaned_val = val.replace('.', '').replace(',', '')
                if cleaned_val.isdigit():
                    numeric_pattern_count += 1
            if numeric_pattern_count / len(sample_values) > 0.7:
                try:
                    df_clean[col] = df_clean[col].astype(str).str.replace('.', '').str.replace(',', '.')
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    pass
    
    return df_clean
