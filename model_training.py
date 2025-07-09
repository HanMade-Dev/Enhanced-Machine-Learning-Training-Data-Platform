import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

def get_available_models(is_supervised=True):
    """
    Get available machine learning models based on whether the task is supervised or not.
    
    Args:
        is_supervised: Boolean indicating if the task is supervised learning
        
    Returns:
        dict: Dictionary of available models and their default parameters
    """
    if is_supervised:
        return {
            "Random Forest Classifier": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": {"type": "integer", "min": 10, "max": 1000, "default": 100, "step": 10, "help": "The number of trees in the forest."},
                    "max_depth": {"type": "integer", "min": 1, "max": 100, "default": 10, "step": 1, "help": "The maximum depth of each tree."},
                    "min_samples_split": {"type": "integer", "min": 2, "max": 20, "default": 2, "step": 1, "help": "The minimum number of samples required to split an internal node."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Random Forest Regressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": {"type": "integer", "min": 10, "max": 1000, "default": 100, "step": 10, "help": "The number of trees in the forest."},
                    "max_depth": {"type": "integer", "min": 1, "max": 100, "default": 10, "step": 1, "help": "The maximum depth of each tree."},
                    "min_samples_split": {"type": "integer", "min": 2, "max": 20, "default": 2, "step": 1, "help": "The minimum number of samples required to split an internal node."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Gradient Boosting Classifier": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": {"type": "integer", "min": 10, "max": 500, "default": 100, "step": 10, "help": "The number of boosting stages."},
                    "learning_rate": {"type": "numeric", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01, "help": "Learning rate shrinks the contribution of each tree."},
                    "max_depth": {"type": "integer", "min": 1, "max": 20, "default": 3, "step": 1, "help": "Maximum depth of the individual regression estimators."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            },
            "Gradient Boosting Regressor": {
                "model": GradientBoostingRegressor,
                "params": {
                    "n_estimators": {"type": "integer", "min": 10, "max": 500, "default": 100, "step": 10, "help": "The number of boosting stages."},
                    "learning_rate": {"type": "numeric", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01, "help": "Learning rate shrinks the contribution of each tree."},
                    "max_depth": {"type": "integer", "min": 1, "max": 20, "default": 3, "step": 1, "help": "Maximum depth of the individual regression estimators."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            },
            "Decision Tree Classifier": {
                "model": DecisionTreeClassifier,
                "params": {
                    "max_depth": {"type": "integer", "min": 1, "max": 50, "default": 10, "step": 1, "help": "The maximum depth of the tree."},
                    "min_samples_split": {"type": "integer", "min": 2, "max": 20, "default": 2, "step": 1, "help": "The minimum number of samples required to split an internal node."},
                    "min_samples_leaf": {"type": "integer", "min": 1, "max": 20, "default": 1, "step": 1, "help": "The minimum number of samples required to be at a leaf node."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "max_depth": [5, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Decision Tree Regressor": {
                "model": DecisionTreeRegressor,
                "params": {
                    "max_depth": {"type": "integer", "min": 1, "max": 50, "default": 10, "step": 1, "help": "The maximum depth of the tree."},
                    "min_samples_split": {"type": "integer", "min": 2, "max": 20, "default": 2, "step": 1, "help": "The minimum number of samples required to split an internal node."},
                    "min_samples_leaf": {"type": "integer", "min": 1, "max": 20, "default": 1, "step": 1, "help": "The minimum number of samples required to be at a leaf node."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "max_depth": [5, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Logistic Regression": {
                "model": LogisticRegression,
                "params": {
                    "C": {"type": "numeric", "min": 0.01, "max": 100.0, "default": 1.0, "step": 0.01, "help": "Inverse of regularization strength."},
                    "max_iter": {"type": "integer", "min": 100, "max": 5000, "default": 1000, "step": 100, "help": "Maximum number of iterations."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "C": [0.1, 1.0, 10.0, 100.0],
                    "solver": ['liblinear', 'lbfgs']
                }
            },
            "Linear Regression": {
                "model": LinearRegression,
                "params": {},
                "param_grid": {}
            },
            "SVM Classifier": {
                "model": SVC,
                "params": {
                    "C": {"type": "numeric", "min": 0.01, "max": 100.0, "default": 1.0, "step": 0.01, "help": "Regularization parameter."},
                    "kernel": {"type": "select", "options": ["linear", "poly", "rbf", "sigmoid"], "default": "rbf", "help": "Specifies the kernel type."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                },
                "param_grid": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ['linear', 'rbf'],
                    "gamma": ['scale', 'auto']
                }
            },
            "SVM Regressor": {
                "model": SVR,
                "params": {
                    "C": {"type": "numeric", "min": 0.01, "max": 100.0, "default": 1.0, "step": 0.01, "help": "Regularization parameter."},
                    "kernel": {"type": "select", "options": ["linear", "poly", "rbf", "sigmoid"], "default": "rbf", "help": "Specifies the kernel type."},
                    "epsilon": {"type": "numeric", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01, "help": "Epsilon in the epsilon-SVR model."}
                },
                "param_grid": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ['linear', 'rbf'],
                    "epsilon": [0.01, 0.1, 0.2]
                }
            },
            "K-Nearest Neighbors Classifier": {
                "model": KNeighborsClassifier,
                "params": {
                    "n_neighbors": {"type": "integer", "min": 1, "max": 50, "default": 5, "step": 1, "help": "Number of neighbors to use."},
                    "weights": {"type": "select", "options": ["uniform", "distance"], "default": "uniform", "help": "Weight function used in prediction."}
                },
                "param_grid": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ['uniform', 'distance']
                }
            },
            "K-Nearest Neighbors Regressor": {
                "model": KNeighborsRegressor,
                "params": {
                    "n_neighbors": {"type": "integer", "min": 1, "max": 50, "default": 5, "step": 1, "help": "Number of neighbors to use."},
                    "weights": {"type": "select", "options": ["uniform", "distance"], "default": "uniform", "help": "Weight function used in prediction."}
                },
                "param_grid": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ['uniform', 'distance']
                }
            },
            "Gaussian Naive Bayes": {
                "model": GaussianNB,
                "params": {},
                "param_grid": {}
            }
        }
    else:
        return {
            "K-Means Clustering": {
                "model": KMeans,
                "params": {
                    "n_clusters": {"type": "integer", "min": 2, "max": 20, "default": 3, "step": 1, "help": "The number of clusters to form."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."},
                    "max_iter": {"type": "integer", "min": 100, "max": 1000, "default": 300, "step": 50, "help": "Maximum number of iterations."}
                }
            },
            "DBSCAN Clustering": {
                "model": DBSCAN,
                "params": {
                    "eps": {"type": "numeric", "min": 0.1, "max": 5.0, "default": 0.5, "step": 0.1, "help": "The maximum distance between two samples."},
                    "min_samples": {"type": "integer", "min": 1, "max": 20, "default": 5, "step": 1, "help": "The number of samples in a neighborhood."}
                }
            },
            "Hierarchical Clustering": {
                "model": AgglomerativeClustering,
                "params": {
                    "n_clusters": {"type": "integer", "min": 2, "max": 20, "default": 3, "step": 1, "help": "The number of clusters to find."},
                    "linkage": {"type": "select", "options": ["ward", "complete", "average", "single"], "default": "ward", "help": "Which linkage criterion to use."}
                }
            },
            "PCA": {
                "model": PCA,
                "params": {
                    "n_components": {"type": "integer", "min": 1, "max": 50, "default": 2, "step": 1, "help": "Number of components to keep."},
                    "random_state": {"type": "integer", "min": 0, "max": 100, "default": 42, "step": 1, "help": "Random state for reproducibility."}
                }
            }
        }

def train_model(X, y, model_name, params, test_size=0.2, random_state=42, use_stratify=True):
    """
    Train a machine learning model with proper label encoder handling
    
    Args:
        X: Feature matrix
        y: Target vector
        model_name: Name of the model to train
        params: Model parameters
        test_size: Size of test set
        random_state: Random state for reproducibility
        use_stratify: Whether to use stratified sampling
        
    Returns:
        tuple: (trained_model, X_train, X_test, y_train, y_test, label_encoder)
    """
    label_encoder = None
    if y.dtype == 'object' or (y.dtype == 'int64' and y.min() >= 0 and y.max() < 100):
        if hasattr(y, 'name') and hasattr(y.name, 'endswith') and y.name.endswith('_encoded'):
            import streamlit as st
            if hasattr(st.session_state, 'target_label_encoder'):
                label_encoder = st.session_state.target_label_encoder
        else:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y.fillna('Unknown').astype(str))
            y = pd.Series(y_encoded, index=y.index)
    is_classification = len(np.unique(y)) < 20 and y.dtype in ['int64', 'int32', 'object']
    if use_stratify and is_classification:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    available_models = get_available_models(True)
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not available")
    
    model_class = available_models[model_name]["model"]
    model = model_class(**params)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, label_encoder

def enhanced_cross_validation(X, y, model_name, params, cv_folds=5):
    """
    Perform enhanced cross-validation with multiple metrics
    
    Args:
        X: Feature matrix
        y: Target vector
        model_name: Name of the model
        params: Model parameters
        cv_folds: Number of cross-validation folds
        
    Returns:
        dict: Cross-validation results with multiple metrics
    """
    available_models = get_available_models(True)
    if model_name not in available_models:
        return {"error": f"Model {model_name} not available"}
    
    model_class = available_models[model_name]["model"]
    model = model_class(**params)
    is_classification = len(np.unique(y)) < 20 and y.dtype in ['int64', 'int32', 'object']
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    
    results = {}
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            if metric.startswith('neg_'):
                scores = -scores
                metric_name = metric.replace('neg_', '')
            else:
                metric_name = metric
            
            results[metric_name] = {
                'scores': scores.tolist(),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max())
            }
        except Exception as e:
            results[metric] = {
                'error': str(e),
                'mean': None,
                'std': None,
                'min': None,
                'max': None
            }
    
    return results

def hyperparameter_tuning(X, y, model_name, search_method="grid_search", max_iter=50, cv_folds=5):
    """
    Perform hyperparameter tuning using Grid Search or Random Search
    
    Args:
        X: Feature matrix
        y: Target vector
        model_name: Name of the model
        search_method: "grid_search" or "random_search"
        max_iter: Maximum iterations for random search
        cv_folds: Number of cross-validation folds
        
    Returns:
        dict: Tuning results with best parameters and score
    """
    available_models = get_available_models(True)
    if model_name not in available_models:
        return {"error": f"Model {model_name} not available"}
    
    model_info = available_models[model_name]
    model_class = model_info["model"]
    param_grid = model_info.get("param_grid", {})
    
    if not param_grid:
        return {"error": f"No parameter grid defined for {model_name}"}
    model = model_class()
    is_classification = len(np.unique(y)) < 20 and y.dtype in ['int64', 'int32', 'object']
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = 'accuracy'
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = 'neg_mean_squared_error'
    
    try:
        if search_method == "grid_search":
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, 
                n_jobs=-1, verbose=0
            )
        else:  # random_search
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, scoring=scoring,
                n_iter=max_iter, n_jobs=-1, verbose=0, random_state=42
            )
        
        search.fit(X, y)
        best_score = search.best_score_
        if scoring.startswith('neg_'):
            best_score = -best_score
        
        return {
            'best_params': search.best_params_,
            'best_score': float(best_score),
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'params': search.cv_results_['params']
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

def compare_models(X, y, model_names, cv_folds=5):
    """
    Compare multiple models using cross-validation
    
    Args:
        X: Feature matrix
        y: Target vector
        model_names: List of model names to compare
        cv_folds: Number of cross-validation folds
        
    Returns:
        dict: Comparison results for each model
    """
    available_models = get_available_models(True)
    results = {}
    is_classification = len(np.unique(y)) < 20 and y.dtype in ['int64', 'int32', 'object']
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = 'accuracy'
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = 'neg_mean_squared_error'
    
    for model_name in model_names:
        if model_name not in available_models:
            results[model_name] = {"error": f"Model {model_name} not available"}
            continue
        
        try:
            model_class = available_models[model_name]["model"]
            default_params = {k: v["default"] for k, v in available_models[model_name]["params"].items()}
            
            model = model_class(**default_params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            if scoring.startswith('neg_'):
                scores = -scores
            
            results[model_name] = {
                'scores': scores.tolist(),
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'min_score': float(scores.min()),
                'max_score': float(scores.max())
            }
            
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return results

def get_model_recommendations(comparison_results, is_classification=True):
    """
    Get model recommendations based on comparison results
    
    Args:
        comparison_results: Results from compare_models function
        is_classification: Whether the task is classification
        
    Returns:
        dict: Recommendations with best model and insights
    """
    valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
    
    if not valid_results:
        return {"error": "No valid model results to compare"}
    best_model = max(valid_results.keys(), key=lambda k: valid_results[k]['mean_score'])
    best_score = valid_results[best_model]['mean_score']
    insights = []
    if is_classification:
        if best_score > 0.9:
            insights.append("Excellent performance achieved!")
        elif best_score > 0.8:
            insights.append("Good performance, consider hyperparameter tuning for improvement.")
        else:
            insights.append("Performance could be improved. Consider feature engineering or different models.")
    else:
        insights.append(f"Best model achieved error of {best_score:.4f}")
    best_std = valid_results[best_model]['std_score']
    if best_std < 0.05:
        insights.append("Model shows consistent performance across folds.")
    else:
        insights.append("Model performance varies across folds. Consider more data or regularization.")
    
    return {
        'best_model': best_model,
        'best_score': best_score,
        'insights': insights,
        'all_results': valid_results
    }
