import pandas as pd
import numpy as np
import streamlit as st
import joblib
import pickle
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import base64

def save_model_for_download(model, label_encoder, filename, format_type="joblib"):
    """
    Save model with label encoder for download
    
    Args:
        model: Trained model
        label_encoder: Label encoder (can be None)
        filename: Filename for the model
        format_type: "joblib" or "pickle"
        
    Returns:
        bytes: Model data for download
    """
    # Create model package
    model_package = {
        'model': model,
        'label_encoder': label_encoder,
        'timestamp': datetime.now().isoformat(),
        'format': format_type,
        'filename': filename
    }
    
    # Add model metadata if available
    if hasattr(model, '__class__'):
        model_package['model_type'] = model.__class__.__name__
    
    # Add feature names if available
    if hasattr(st.session_state, 'feature_columns'):
        model_package['feature_columns'] = st.session_state.feature_columns
    
    # Add target column info if available
    if hasattr(st.session_state, 'target_column'):
        model_package['target_column'] = st.session_state.target_column
    
    # Add label mapping if available
    if hasattr(st.session_state, 'target_label_mapping'):
        model_package['target_label_mapping'] = st.session_state.target_label_mapping
    
    # Serialize the model package
    buffer = io.BytesIO()
    
    if format_type == "joblib":
        joblib.dump(model_package, buffer)
    else:  # pickle
        pickle.dump(model_package, buffer)
    
    buffer.seek(0)
    return buffer.getvalue()

def generate_report(model_info, evaluation_results, model_path):
    """
    Generate comprehensive HTML report
    
    Args:
        model_info: Dictionary containing model information
        evaluation_results: Dictionary containing evaluation results
        model_path: Path to the saved model
        
    Returns:
        str: HTML report content
    """
    # Determine if classification or regression
    is_classification = 'accuracy' in evaluation_results
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
            .table {{ border-collapse: collapse; width: 100%; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #f2f2f2; }}
            .good {{ color: #28a745; }}
            .warning {{ color: #ffc107; }}
            .danger {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Machine Learning Model Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Model Type:</strong> {model_info.get('model_name', 'Unknown')}</p>
            <p><strong>Task Type:</strong> {'Classification' if is_classification else 'Regression'}</p>
        </div>
        
        <div class="section">
            <h2>📊 Model Performance</h2>
    """
    
    # Add performance metrics
    if is_classification:
        accuracy = evaluation_results.get('accuracy', 0)
        precision = evaluation_results.get('precision', 0)
        recall = evaluation_results.get('recall', 0)
        f1 = evaluation_results.get('f1', 0)
        
        html_content += f"""
            <div class="metric">
                <strong>Accuracy:</strong> <span class="{'good' if accuracy > 0.8 else 'warning' if accuracy > 0.6 else 'danger'}">{accuracy:.4f}</span>
            </div>
            <div class="metric">
                <strong>Precision:</strong> <span class="{'good' if precision > 0.8 else 'warning' if precision > 0.6 else 'danger'}">{precision:.4f}</span>
            </div>
            <div class="metric">
                <strong>Recall:</strong> <span class="{'good' if recall > 0.8 else 'warning' if recall > 0.6 else 'danger'}">{recall:.4f}</span>
            </div>
            <div class="metric">
                <strong>F1-Score:</strong> <span class="{'good' if f1 > 0.8 else 'warning' if f1 > 0.6 else 'danger'}">{f1:.4f}</span>
            </div>
        """
    else:
        r2 = evaluation_results.get('r2', 0)
        rmse = evaluation_results.get('rmse', 0)
        mae = evaluation_results.get('mae', 0)
        
        html_content += f"""
            <div class="metric">
                <strong>R² Score:</strong> <span class="{'good' if r2 > 0.8 else 'warning' if r2 > 0.6 else 'danger'}">{r2:.4f}</span>
            </div>
            <div class="metric">
                <strong>RMSE:</strong> <span>{rmse:.4f}</span>
            </div>
            <div class="metric">
                <strong>MAE:</strong> <span>{mae:.4f}</span>
            </div>
        """
    
    # Add model parameters
    html_content += f"""
        </div>
        
        <div class="section">
            <h2>⚙️ Model Configuration</h2>
            <table class="table">
                <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    for param, value in model_info.get('params', {}).items():
        html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    # Add dataset information
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Dataset Information</h2>
            <table class="table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Training Samples</td><td>{len(model_info.get('X_train', []))}</td></tr>
                <tr><td>Test Samples</td><td>{len(model_info.get('X_test', []))}</td></tr>
                <tr><td>Features</td><td>{model_info.get('X_train', pd.DataFrame()).shape[1] if hasattr(model_info.get('X_train', None), 'shape') else 'N/A'}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>💾 Model Deployment</h2>
            <p><strong>Model File:</strong> {model_path}</p>
            <p><strong>Usage Instructions:</strong></p>
            <pre>
import joblib
import pandas as pd

# Load the model
model_package = joblib.load('{model_path}')
model = model_package['model']
label_encoder = model_package.get('label_encoder')

# Make predictions
# predictions = model.predict(your_data)

# For classification with string labels:
# if label_encoder is not None:
#     predictions = label_encoder.inverse_transform(predictions)
            </pre>
        </div>
        
        <div class="section">
            <h2>📝 Notes</h2>
            <ul>
                <li>This model was trained using automated ML pipeline</li>
                <li>Performance metrics are based on test set evaluation</li>
                <li>For production use, consider additional validation</li>
                <li>Model package includes label encoder for string target conversion</li>
            </ul>
        </div>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
            <p>Generated by Enhanced ML Training Platform</p>
        </footer>
    </body>
    </html>
    """
    
    return html_content

def export_model_summary(model_info, evaluation_results):
    """
    Export model summary as JSON
    
    Args:
        model_info: Dictionary containing model information
        evaluation_results: Dictionary containing evaluation results
        
    Returns:
        dict: Model summary
    """
    # Determine if classification or regression
    is_classification = 'accuracy' in evaluation_results
    
    summary = {
        'model_info': {
            'model_name': model_info.get('model_name', 'Unknown'),
            'model_type': model_info.get('model_object', type(None)).__class__.__name__,
            'task_type': 'Classification' if is_classification else 'Regression',
            'parameters': model_info.get('params', {}),
            'is_supervised': model_info.get('is_supervised', True)
        },
        'dataset_info': {
            'training_samples': len(model_info.get('X_train', [])),
            'test_samples': len(model_info.get('X_test', [])),
            'features': model_info.get('X_train', pd.DataFrame()).shape[1] if hasattr(model_info.get('X_train', None), 'shape') else 0,
            'feature_names': getattr(model_info.get('X_train', pd.DataFrame()), 'columns', []).tolist()
        },
        'performance_metrics': {},
        'timestamp': datetime.now().isoformat(),
        'platform': 'Enhanced ML Training Platform'
    }
    
    # Add performance metrics
    if is_classification:
        summary['performance_metrics'] = {
            'accuracy': evaluation_results.get('accuracy', 0),
            'precision': evaluation_results.get('precision', 0),
            'recall': evaluation_results.get('recall', 0),
            'f1_score': evaluation_results.get('f1', 0),
            'roc_auc': evaluation_results.get('roc_auc', None)
        }
    else:
        summary['performance_metrics'] = {
            'r2_score': evaluation_results.get('r2', 0),
            'rmse': evaluation_results.get('rmse', 0),
            'mae': evaluation_results.get('mae', 0),
            'mape': evaluation_results.get('mape', None)
        }
    
    # Add feature importance if available
    if 'feature_importance_stats' in evaluation_results:
        summary['feature_importance'] = evaluation_results['feature_importance_stats']
    
    # Add cross-validation results if available
    if 'cv_results' in evaluation_results:
        summary['cross_validation'] = evaluation_results['cv_results']
    
    return summary

def create_deployment_code(model_filename, feature_columns, target_column, is_classification=True):
    """
    Generate deployment code template
    
    Args:
        model_filename: Name of the model file
        feature_columns: List of feature column names
        target_column: Name of target column
        is_classification: Whether the model is for classification
        
    Returns:
        str: Python code template for deployment
    """
    feature_list = "', '".join(feature_columns)
    
    code_template = f"""
# Model Deployment Code
# Generated by Enhanced ML Training Platform

import joblib
import pandas as pd
import numpy as np

class ModelPredictor:
    def __init__(self, model_path='{model_filename}'):
        \"\"\"
        Initialize the model predictor
        
        Args:
            model_path: Path to the saved model file
        \"\"\"
        self.model_package = joblib.load(model_path)
        self.model = self.model_package['model']
        self.label_encoder = self.model_package.get('label_encoder')
        self.feature_columns = ['{feature_list}']
        self.target_column = '{target_column}'
        
    def predict(self, data):
        \"\"\"
        Make predictions on new data
        
        Args:
            data: pandas DataFrame or dict with feature values
            
        Returns:
            predictions: numpy array or list of predictions
        \"\"\"
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure correct column order
        data = data[self.feature_columns]
        
        # Make predictions
        predictions = self.model.predict(data)
        
        # Convert back to original labels for classification
        if self.label_encoder is not None and hasattr(self.model, 'predict_proba'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, data):
        \"\"\"
        Get prediction probabilities (classification only)
        
        Args:
            data: pandas DataFrame or dict with feature values
            
        Returns:
            probabilities: numpy array of prediction probabilities
        \"\"\"
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure correct column order
        data = data[self.feature_columns]
        
        return self.model.predict_proba(data)
    
    def get_feature_importance(self):
        \"\"\"
        Get feature importance (if available)
        
        Returns:
            dict: Feature importance scores
        \"\"\"
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Example prediction
    sample_data = {{
        # Add your feature values here
        # '{feature_columns[0] if feature_columns else 'feature1'}': 0.5,
        # '{feature_columns[1] if len(feature_columns) > 1 else 'feature2'}': 1.2,
        # ... continue for all features
    }}
    
    # Make prediction
    # prediction = predictor.predict(sample_data)
    # print(f"Prediction: {{prediction}}")
    
    # For classification models, get probabilities
    # if hasattr(predictor.model, 'predict_proba'):
    #     probabilities = predictor.predict_proba(sample_data)
    #     print(f"Probabilities: {{probabilities}}")
    
    # Get feature importance
    # importance = predictor.get_feature_importance()
    # if importance:
    #     print("Feature Importance:")
    #     for feature, score in importance.items():
    #         print(f"  {{feature}}: {{score:.4f}}")
"""
    
    return code_template

def load_model_from_file(uploaded_file):
    """
    Load model from uploaded file (internal format)
    """
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Try loading with joblib first
        try:
            model_package = joblib.load(uploaded_file)
            return model_package
        except:
            pass
        
        # Try loading with pickle
        try:
            uploaded_file.seek(0)
            model_package = pickle.load(uploaded_file)
            return model_package
        except:
            pass
        
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_external_model_flexible(uploaded_file):
    """
    Flexible model loading that can handle various formats
    """
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Method 1: Try joblib loading
        try:
            model_data = joblib.load(uploaded_file)
            return process_loaded_model_data(model_data, 'joblib')
        except Exception as e:
            st.write(f"Joblib loading failed: {str(e)}")
        
        # Method 2: Try pickle loading
        try:
            uploaded_file.seek(0)
            model_data = pickle.load(uploaded_file)
            return process_loaded_model_data(model_data, 'pickle')
        except Exception as e:
            st.write(f"Pickle loading failed: {str(e)}")
        
        # Method 3: Try loading as bytes and then unpickling
        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            model_data = pickle.loads(file_bytes)
            return process_loaded_model_data(model_data, 'bytes_pickle')
        except Exception as e:
            st.write(f"Bytes pickle loading failed: {str(e)}")
        
        # Method 4: Try joblib from bytes
        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            model_data = joblib.load(io.BytesIO(file_bytes))
            return process_loaded_model_data(model_data, 'bytes_joblib')
        except Exception as e:
            st.write(f"Bytes joblib loading failed: {str(e)}")
        
        return None
        
    except Exception as e:
        st.error(f"Error in flexible model loading: {str(e)}")
        return None

def process_loaded_model_data(model_data, load_method):
    """
    Process loaded model data and standardize format
    """
    try:
        # Case 1: Already a proper model package (dictionary with expected keys)
        if isinstance(model_data, dict):
            if 'model' in model_data:
                # Internal format - return as is
                return model_data
            else:
                # Dictionary but not our format - check if it contains a model object
                for key, value in model_data.items():
                    if hasattr(value, 'predict'):
                        return {
                            'model': value,
                            'format': 'external_dict',
                            'original_key': key,
                            'load_method': load_method,
                            'timestamp': datetime.now().isoformat()
                        }
        
        # Case 2: Direct model object
        elif hasattr(model_data, 'predict'):
            return {
                'model': model_data,
                'format': 'raw_model',
                'load_method': load_method,
                'timestamp': datetime.now().isoformat()
            }
        
        # Case 3: Tuple or list (might contain model and other data)
        elif isinstance(model_data, (tuple, list)):
            for item in model_data:
                if hasattr(item, 'predict'):
                    return {
                        'model': item,
                        'format': 'tuple_model',
                        'load_method': load_method,
                        'timestamp': datetime.now().isoformat()
                    }
        
        # Case 4: Other formats - try to find model-like objects
        else:
            if hasattr(model_data, 'predict'):
                return {
                    'model': model_data,
                    'format': 'unknown_model',
                    'load_method': load_method,
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
        
    except Exception as e:
        st.error(f"Error processing model data: {str(e)}")
        return None

def detect_model_format_and_requirements(model_package):
    """
    Analyze loaded model and detect its format and requirements
    """
    analysis = {
        'format_detected': 'unknown',
        'feature_count': 0,
        'has_feature_names': False,
        'model_type': 'unknown',
        'requires_manual_setup': False,
        'suggestions': []
    }
    
    try:
        model = model_package.get('model')
        if model is None:
            analysis['suggestions'].append("No model object found")
            return analysis
        
        # Detect model type
        model_class = model.__class__.__name__
        analysis['model_type'] = model_class
        
        # Detect format
        format_type = model_package.get('format', 'internal')
        analysis['format_detected'] = format_type
        
        # Try to get feature count
        feature_count = 0
        
        # Method 1: Check if model has n_features_in_ (sklearn 0.24+)
        if hasattr(model, 'n_features_in_'):
            feature_count = model.n_features_in_
        
        # Method 2: Check coef_ shape for linear models
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                feature_count = len(model.coef_)
            else:
                feature_count = model.coef_.shape[1]
        
        # Method 3: Check feature_importances_ for tree-based models
        elif hasattr(model, 'feature_importances_'):
            feature_count = len(model.feature_importances_)
        
        # Method 4: Check support vectors for SVM
        elif hasattr(model, 'support_vectors_'):
            feature_count = model.support_vectors_.shape[1]
        
        analysis['feature_count'] = feature_count
        
        # Check if feature names are available
        feature_columns = model_package.get('feature_columns', [])
        if feature_columns:
            analysis['has_feature_names'] = True
        else:
            # Check if model has feature_names_in_ (sklearn 0.24+)
            if hasattr(model, 'feature_names_in_'):
                analysis['has_feature_names'] = True
                model_package['feature_columns'] = list(model.feature_names_in_)
        
        # Determine if manual setup is required
        if not analysis['has_feature_names'] and feature_count > 0:
            analysis['requires_manual_setup'] = True
            analysis['suggestions'].append(f"Model expects {feature_count} features but no feature names provided")
        
        if not model_package.get('target_column'):
            analysis['requires_manual_setup'] = True
            analysis['suggestions'].append("No target column name specified")
        
        # Add capability suggestions
        capabilities = []
        if hasattr(model, 'predict'):
            capabilities.append("predict")
        if hasattr(model, 'predict_proba'):
            capabilities.append("predict_proba")
        if hasattr(model, 'decision_function'):
            capabilities.append("decision_function")
        
        if capabilities:
            analysis['suggestions'].append(f"Model capabilities: {', '.join(capabilities)}")
        
    except Exception as e:
        analysis['suggestions'].append(f"Error analyzing model: {str(e)}")
    
    return analysis

def validate_model_package(model_package):
    """
    Validate model package structure
    """
    validation = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'info': []
    }
    
    try:
        # Check if model exists
        if 'model' not in model_package or model_package['model'] is None:
            validation['is_valid'] = False
            validation['issues'].append("Model object not found in package")
            return validation
        
        model = model_package['model']
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            validation['is_valid'] = False
            validation['issues'].append("Model does not have predict method")
        
        # Check feature columns
        if not model_package.get('feature_columns'):
            validation['warnings'].append("No feature column names specified")
        
        # Check target column
        if not model_package.get('target_column'):
            validation['warnings'].append("No target column name specified")
        
        # Check label encoder
        if model_package.get('label_encoder'):
            validation['info'].append("Label encoder available for string predictions")
        
        # Check model type
        model_type = model_package.get('model_type', 'Unknown')
        validation['info'].append(f"Model type: {model_type}")
        
    except Exception as e:
        validation['is_valid'] = False
        validation['issues'].append(f"Error validating model package: {str(e)}")
    
    return validation

def create_sample_data_template(feature_columns):
    """
    Create a sample data template for the given features
    """
    try:
        sample_data = {}
        for feature in feature_columns:
            # Generate sample values based on feature name patterns
            if any(keyword in feature.lower() for keyword in ['age', 'year']):
                sample_data[feature] = [25, 30, 35, 40, 45]
            elif any(keyword in feature.lower() for keyword in ['price', 'cost', 'amount', 'salary']):
                sample_data[feature] = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
            elif any(keyword in feature.lower() for keyword in ['rate', 'ratio', 'percent']):
                sample_data[feature] = [0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                sample_data[feature] = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error creating sample data: {str(e)}")
        return None

def export_predictions_to_csv(predictions, probabilities, original_data):
    """
    Export predictions to CSV format
    """
    try:
        # Create results dataframe
        results_df = original_data.copy()
        results_df['Prediction'] = predictions
        
        # Add probabilities if available
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'Probability_Class_{i}'] = probabilities[:, i]
        
        # Add timestamp
        results_df['Prediction_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error exporting predictions: {str(e)}")
        return None

def safe_numeric_conversion(series, column_name):
    """
    Safely convert series to numeric, handling various formats
    """
    try:
        # First, try direct conversion
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Check for Indonesian number format (comma as decimal separator)
        if numeric_series.isna().sum() > len(series) * 0.1:  # If more than 10% are NaN
            # Try converting comma to dot
            string_series = series.astype(str)
            converted_series = string_series.str.replace(',', '.', regex=False)
            numeric_series = pd.to_numeric(converted_series, errors='coerce')
        
        # Report conversion results
        na_count = numeric_series.isna().sum()
        if na_count > 0:
            st.warning(f"Column '{column_name}': {na_count} values could not be converted to numeric")
        
        return numeric_series
    except Exception as e:
        st.error(f"Error converting column '{column_name}' to numeric: {str(e)}")
        return series

def detect_delimiter(file_content):
    """
    Detect CSV delimiter
    """
    import csv
    
    try:
        # Sample first few lines
        sample = file_content[:1024]
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        return delimiter
    except:
        # Default to comma if detection fails
        return ','

def validate_data_quality(df):
    """
    Validate data quality and provide recommendations
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': 0,
        'data_types': {},
        'recommendations': []
    }
    
    try:
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                quality_report['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        # Check duplicate rows
        quality_report['duplicate_rows'] = df.duplicated().sum()
        
        # Check data types
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
        
        # Generate recommendations
        if quality_report['missing_values']:
            quality_report['recommendations'].append("Consider handling missing values before training")
        
        if quality_report['duplicate_rows'] > 0:
            quality_report['recommendations'].append("Consider removing duplicate rows")
        
        # Check for potential categorical columns
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    quality_report['recommendations'].append(f"Column '{col}' might be categorical")
        
    except Exception as e:
        quality_report['recommendations'].append(f"Error in quality validation: {str(e)}")
    
    return quality_report

def create_download_link(data, filename, link_text):
    """
    Create a download link for data
    """
    try:
        if isinstance(data, pd.DataFrame):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
        else:
            b64 = base64.b64encode(data).decode()
        
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

def format_model_info(model_package):
    """
    Format model information for display
    """
    try:
        info = {}
        
        # Basic info
        info['Model Type'] = model_package.get('model_type', 'Unknown')
        info['Features'] = len(model_package.get('feature_columns', []))
        info['Target'] = model_package.get('target_column', 'Unknown')
        
        # Timestamp
        timestamp = model_package.get('timestamp')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                info['Created'] = dt.strftime('%Y-%m-%d %H:%M')
            except:
                info['Created'] = timestamp
        else:
            info['Created'] = 'Unknown'
        
        # Model capabilities
        model = model_package.get('model')
        if model:
            capabilities = []
            if hasattr(model, 'predict'):
                capabilities.append('Predict')
            if hasattr(model, 'predict_proba'):
                capabilities.append('Probabilities')
            if hasattr(model, 'feature_importances_'):
                capabilities.append('Feature Importance')
            
            info['Capabilities'] = ', '.join(capabilities) if capabilities else 'Basic Prediction'
        
        return info
    except Exception as e:
        return {'Error': f"Could not format model info: {str(e)}"}
