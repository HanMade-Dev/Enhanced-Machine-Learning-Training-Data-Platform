import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test, is_supervised=True):
    """
    Comprehensive model evaluation with visualizations
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        is_supervised: Whether the model is supervised
        
    Returns:
        dict: Comprehensive evaluation results
    """
    results = {}
    
    if not is_supervised:
        return evaluate_unsupervised_model(model, X_test)
    y_pred = model.predict(X_test)
    is_classification = hasattr(model, 'predict_proba') or len(np.unique(y_test)) < 20
    
    if is_classification:
        results.update(evaluate_classification_model(model, X_test, y_test, y_pred))
    else:
        results.update(evaluate_regression_model(model, X_test, y_test, y_pred))
    if hasattr(model, 'feature_importances_'):
        results.update(evaluate_feature_importance(model, X_test.columns))
    
    return results

def evaluate_classification_model(model, X_test, y_test, y_pred):
    """
    Evaluate classification model with comprehensive metrics and visualizations
    """
    results = {}
    results['accuracy'] = float(accuracy_score(y_test, y_pred))
    results['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    results['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    results['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    results['classification_report'] = report_df
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    import streamlit as st
    if hasattr(st.session_state, 'target_label_encoder') and st.session_state.target_label_encoder is not None:
        try:
            label_encoder = st.session_state.target_label_encoder
            original_labels = [str(label_encoder.inverse_transform([int(label)])[0]) for label in unique_labels]
        except:
            original_labels = [str(label) for label in unique_labels]
    else:
        original_labels = [str(label) for label in unique_labels]
    
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=original_labels,
        y=original_labels,
        annotation_text=cm,
        colorscale='Blues',
        showscale=True
    )
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    results['interactive_confusion_matrix'] = fig_cm
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        
        if len(unique_labels) == 2:  # Binary classification
            try:
                auc_score = roc_auc_score(y_test, y_proba[:, 1])
                results['roc_auc'] = float(auc_score)
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {auc_score:.3f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                fig_roc.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                results['roc_curve'] = fig_roc
                precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
                avg_precision = average_precision_score(y_test, y_proba[:, 1])
                results['average_precision'] = float(avg_precision)
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, name=f'PR Curve (AP = {avg_precision:.3f})'))
                fig_pr.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision'
                )
                results['pr_curve'] = fig_pr
                
            except Exception as e:
                st.warning(f"Could not compute ROC/PR curves: {e}")
        
        else:  # Multiclass
            try:
                auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                results['roc_auc'] = float(auc_score)
            except Exception as e:
                st.warning(f"Could not compute multiclass AUC: {e}")
    test_distribution = pd.Series(y_test).value_counts()
    pred_distribution = pd.Series(y_pred).value_counts()
    if hasattr(st.session_state, 'target_label_encoder') and st.session_state.target_label_encoder is not None:
        try:
            label_encoder = st.session_state.target_label_encoder
            test_dist_original = {}
            pred_dist_original = {}
            
            for label, count in test_distribution.items():
                original_label = label_encoder.inverse_transform([int(label)])[0]
                test_dist_original[str(original_label)] = count
            
            for label, count in pred_distribution.items():
                original_label = label_encoder.inverse_transform([int(label)])[0]
                pred_dist_original[str(original_label)] = count
            
            test_distribution = pd.Series(test_dist_original)
            pred_distribution = pd.Series(pred_dist_original)
        except:
            pass
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(x=test_distribution.index, y=test_distribution.values, name='Actual'))
    fig_dist.add_trace(go.Bar(x=pred_distribution.index, y=pred_distribution.values, name='Predicted'))
    fig_dist.update_layout(
        title='Class Distribution: Actual vs Predicted',
        xaxis_title='Classes',
        yaxis_title='Count',
        barmode='group'
    )
    results['class_distribution_plot'] = fig_dist
    
    return results

def evaluate_regression_model(model, X_test, y_test, y_pred):
    """
    Evaluate regression model with comprehensive metrics and visualizations
    """
    results = {}
    results['r2'] = float(r2_score(y_test, y_pred))
    results['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    results['mae'] = float(mean_absolute_error(y_test, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
        results['mape'] = float(mape * 100)  # Convert to percentage
    except:
        results['mape'] = float('nan')
    residuals = y_test - y_pred
    results['residual_stats'] = {
        'mean': float(residuals.mean()),
        'std': float(residuals.std()),
        'skewness': float(stats.skew(residuals)),
        'kurtosis': float(stats.kurtosis(residuals))
    }
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        name='Predictions', opacity=0.6
    ))
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    fig_scatter.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values'
    )
    results['actual_vs_predicted_plot'] = fig_scatter
    fig_residual = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Residuals vs Predicted', 'Residual Distribution', 
                       'Q-Q Plot', 'Residuals vs Index'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    fig_residual.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
        row=1, col=1
    )
    fig_residual.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig_residual.add_trace(
        go.Histogram(x=residuals, name='Residual Distribution', nbinsx=30),
        row=1, col=2
    )
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    fig_residual.add_trace(
        go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot'),
        row=2, col=1
    )
    fig_residual.add_trace(
        go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', name='Normal Line'),
        row=2, col=1
    )
    fig_residual.add_trace(
        go.Scatter(x=list(range(len(residuals))), y=residuals, mode='markers', name='Residuals vs Index'),
        row=2, col=2
    )
    fig_residual.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig_residual.update_layout(title='Residual Analysis', showlegend=False)
    results['residual_analysis'] = fig_residual
    residual_std = residuals.std()
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    prediction_intervals = {
        'lower': y_pred - z_score * residual_std,
        'upper': y_pred + z_score * residual_std
    }
    coverage = np.mean((y_test >= prediction_intervals['lower']) & 
                      (y_test <= prediction_intervals['upper']))
    results['prediction_interval_coverage'] = float(coverage)
    sorted_indices = np.argsort(y_pred)
    fig_intervals = go.Figure()
    
    fig_intervals.add_trace(go.Scatter(
        x=y_pred[sorted_indices], y=prediction_intervals['upper'][sorted_indices],
        mode='lines', name='Upper Bound', line=dict(color='lightblue')
    ))
    fig_intervals.add_trace(go.Scatter(
        x=y_pred[sorted_indices], y=prediction_intervals['lower'][sorted_indices],
        mode='lines', name='Lower Bound', line=dict(color='lightblue'),
        fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)'
    ))
    fig_intervals.add_trace(go.Scatter(
        x=y_pred, y=y_test, mode='markers', name='Actual Values',
        marker=dict(color='red', size=4)
    ))
    fig_intervals.add_trace(go.Scatter(
        x=y_pred[sorted_indices], y=y_pred[sorted_indices],
        mode='lines', name='Predicted', line=dict(color='blue')
    ))
    
    fig_intervals.update_layout(
        title=f'Prediction Intervals ({confidence_level*100:.0f}% Confidence)',
        xaxis_title='Predicted Values',
        yaxis_title='Values'
    )
    results['prediction_intervals_plot'] = fig_intervals
    
    return results

def evaluate_feature_importance(model, feature_names):
    """
    Evaluate and visualize feature importance
    """
    results = {}
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        results['feature_importance_df'] = feature_importance_df
        fig_importance = px.bar(
            feature_importance_df.head(20),  # Top 20 features
            x='importance', y='feature',
            orientation='h',
            title='Feature Importance (Top 20)',
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        results['feature_importance_plot_interactive'] = fig_importance
        total_importance = importances.sum()
        cumulative_importance = np.cumsum(np.sort(importances)[::-1])
        features_for_80_percent = np.argmax(cumulative_importance >= 0.8 * total_importance) + 1
        
        results['feature_importance_stats'] = {
            'total_features': len(feature_names),
            'features_for_80_percent': int(features_for_80_percent),
            'top_features': feature_importance_df['feature'].head(10).tolist(),
            'top_importances': feature_importance_df['importance'].head(10).tolist()
        }
    
    return results

def evaluate_unsupervised_model(model, X):
    """
    Evaluate unsupervised learning model
    """
    results = {}
    
    if hasattr(model, 'labels_'):
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        results['n_clusters'] = n_clusters
        results['labels'] = labels.tolist()
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_dist = dict(zip(unique_labels.astype(str), counts.tolist()))
        results['cluster_distribution'] = cluster_dist
        if X.shape[1] >= 2:
            fig_clusters = px.scatter(
                x=X.iloc[:, 0], y=X.iloc[:, 1],
                color=labels.astype(str),
                title='Cluster Visualization',
                labels={'x': X.columns[0], 'y': X.columns[1]}
            )
            results['cluster_plot'] = fig_clusters
    
    elif hasattr(model, 'components_'):
        explained_variance_ratio = model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        results['explained_variance_ratio'] = explained_variance_ratio.tolist()
        results['cumulative_variance'] = cumulative_variance.tolist()
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(
            x=list(range(1, len(explained_variance_ratio) + 1)),
            y=explained_variance_ratio,
            mode='lines+markers',
            name='Individual'
        ))
        fig_scree.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative'
        ))
        fig_scree.update_layout(
            title='Explained Variance Ratio',
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio'
        )
        results['scree_plot'] = fig_scree
    
    return results

def create_learning_curves(model, X, y, cv=5):
    """
    Create learning curves to analyze model performance vs training size
    """
    from sklearn.model_selection import learning_curve
    
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            error_y=dict(type='data', array=train_std, visible=True)
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red'),
            error_y=dict(type='data', array=val_std, visible=True)
        ))
        
        fig.update_layout(
            title='Learning Curves',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create learning curves: {e}")
        return None
