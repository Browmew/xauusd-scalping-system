"""
LightGBM Model Implementation

Primary model for the XAUUSD scalping system based on research synthesis.
LightGBM chosen for speed, explainability, and strong baseline performance.
Configured for binary classification with early stopping and cross-validation.

Research Configuration:
- Primary model in ensemble (40% weight)
- Fast training and inference
- Built-in feature importance
- SHAP integration for explainability
- Early stopping to prevent overfitting

Performance Targets:
- Training time: <5 minutes for monthly retrain
- Inference time: <10ms per prediction
- Hit rate: 70%+ baseline
- Model size: <50MB

Time Complexity: O(n * m * log(m)) for training
Space Complexity: O(n * m) for training data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from .base import BaseModel, ModelConfig, ModelMetrics
from src.core.config import get_config_manager
from src.core.utils import Timer

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM implementation for binary scalp prediction
    
    Optimized for speed and explainability with research-validated
    hyperparameters for financial time series prediction.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                name="LightGBM_Scalping",
                model_type="lightgbm"
            )
            
        super().__init__(config)
        
        # LightGBM specific parameters from research synthesis
        self.lgb_params = self._get_default_params()
        
        # Update with any custom parameters
        if 'lightgbm' in self.config.training_params:
            self.lgb_params.update(self.config.training_params['lightgbm'])
            
        # Training artifacts
        self.lgb_train_data = None
        self.lgb_val_data = None
        self.training_log = []
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters from research synthesis"""
        return {
            # Core parameters
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 100,
            
            # Regularization (from research)
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_split_gain': 0.1,
            
            # Training control
            'num_boost_round': 1000,
            'early_stopping_rounds': 100,
            'verbose': -1,
            
            # Performance
            'n_jobs': -1,
            'device_type': 'cpu',  # Can be 'gpu' if available
            
            # Additional parameters for financial data
            'is_unbalance': True,
            'boost_from_average': True,
            'force_row_wise': True  # Better for small datasets
        }
        
    def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Train LightGBM model with validation"""
        
        with Timer("lightgbm_training"):
            try:
                # Prepare datasets
                self.lgb_train_data = lgb.Dataset(
                    X.values,
                    label=y.values,
                    feature_name=X.columns.tolist()
                )
                
                valid_sets = [self.lgb_train_data]
                valid_names = ['train']
                
                if X_val is not None and y_val is not None:
                    self.lgb_val_data = lgb.Dataset(
                        X_val.values,
                        label=y_val.values,
                        feature_name=X.columns.tolist(),
                        reference=self.lgb_train_data
                    )
                    valid_sets.append(self.lgb_val_data)
                    valid_names.append('valid')
                    
                # Callbacks for logging
                callbacks = [
                    lgb.log_evaluation(period=0),  # No output during training
                    lgb.record_evaluation(self.training_log)
                ]
                
                if len(valid_sets) > 1:
                    callbacks.append(lgb.early_stopping(
                        stopping_rounds=self.lgb_params['early_stopping_rounds'],
                        verbose=False
                    ))
                    
                # Train model
                logger.info(f"Training LightGBM with {len(X)} samples, {len(X.columns)} features")
                
                self.model = lgb.train(
                    params=self.lgb_params,
                    train_set=self.lgb_train_data,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks
                )
                
                # Log training results
                best_iteration = self.model.best_iteration
                if best_iteration > 0:
                    logger.info(f"Training completed at iteration {best_iteration}")
                    
                    if 'valid' in self.training_log:
                        best_score = self.training_log['valid']['binary_logloss'][best_iteration]
                        logger.info(f"Best validation score: {best_score:.6f}")
                        
            except Exception as e:
                logger.error(f"LightGBM training failed: {e}")
                raise
                
    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Ensure feature order matches training
        X_ordered = X[self.feature_names]
        
        # Get probabilities
        probabilities = self.model.predict(X_ordered.values, num_iteration=self.model.best_iteration)
        
        # LightGBM returns probabilities directly for binary classification
        return probabilities
        
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            return {}
            
        try:
            # Get importance scores
            importance_gain = self.model.feature_importance(importance_type='gain')
            importance_split = self.model.feature_importance(importance_type='split')
            
            # Combine gain and split importance (weighted average)
            combined_importance = 0.7 * importance_gain + 0.3 * importance_split
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_names, combined_importance))
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return {}
            
    def _save_model(self, filepath: str) -> None:
        """Save LightGBM model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            # Save LightGBM model in native format
            self.model.save_model(filepath + '_lgb.txt')
            
            # Also save training log
            import pickle
            with open(filepath + '_log.pkl', 'wb') as f:
                pickle.dump(self.training_log, f)
                
        except Exception as e:
            logger.error(f"Failed to save LightGBM model: {e}")
            raise
            
    def _load_model(self, filepath: str) -> None:
        """Load LightGBM model"""
        try:
            # Load LightGBM model
            self.model = lgb.Booster(model_file=filepath + '_lgb.txt')
            
            # Load training log if available
            import pickle
            try:
                with open(filepath + '_log.pkl', 'rb') as f:
                    self.training_log = pickle.load(f)
            except FileNotFoundError:
                self.training_log = []
                
        except Exception as e:
            logger.error(f"Failed to load LightGBM model: {e}")
            raise
            
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5, shuffle: bool = False) -> Dict[str, Any]:
        """
        Perform time-series cross-validation
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            shuffle: Whether to shuffle data (not recommended for time series)
            
        Returns:
            Cross-validation results
        """
        
        with Timer("lightgbm_cv"):
            try:
                if shuffle:
                    from sklearn.model_selection import StratifiedKFold
                    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                else:
                    # Time series split (recommended)
                    cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
                    
                cv_scores = []
                cv_predictions = np.zeros(len(y))
                feature_importance_cv = {}
                
                for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                    logger.info(f"Training fold {fold + 1}/{cv_folds}")
                    
                    # Split data
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    # Train fold model
                    fold_model = LightGBMModel(self.config)
                    fold_model.lgb_params = self.lgb_params.copy()
                    fold_model._train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                    
                    # Predict on validation set
                    val_pred = fold_model._predict_proba(X_val_fold)
                    cv_predictions[val_idx] = val_pred
                    
                    # Calculate fold score
                    fold_score = log_loss(y_val_fold, val_pred)
                    cv_scores.append(fold_score)
                    
                    # Accumulate feature importance
                    fold_importance = fold_model._get_feature_importance()
                    for feature, importance in fold_importance.items():
                        if feature not in feature_importance_cv:
                            feature_importance_cv[feature] = []
                        feature_importance_cv[feature].append(importance)
                        
                    logger.info(f"Fold {fold + 1} score: {fold_score:.6f}")
                    
                # Average feature importance across folds
                avg_feature_importance = {}
                for feature, importances in feature_importance_cv.items():
                    avg_feature_importance[feature] = np.mean(importances)
                    
                # Calculate final CV score
                cv_score_mean = np.mean(cv_scores)
                cv_score_std = np.std(cv_scores)
                
                # Calculate out-of-fold predictions score
                oof_score = log_loss(y, cv_predictions)
                
                results = {
                    'cv_scores': cv_scores,
                    'cv_score_mean': cv_score_mean,
                    'cv_score_std': cv_score_std,
                    'oof_score': oof_score,
                    'oof_predictions': cv_predictions,
                    'feature_importance': avg_feature_importance
                }
                
                logger.info(
                    f"CV completed: {cv_score_mean:.6f} ± {cv_score_std:.6f}, "
                    f"OOF: {oof_score:.6f}"
                )
                
                return results
                
            except Exception as e:
                logger.error(f"Cross-validation failed: {e}")
                raise
                
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                           param_space: Dict[str, Any] = None,
                           n_trials: int = 100) -> Dict[str, Any]:
        """
        Hyperparameter tuning using Optuna
        
        Args:
            X: Training features
            y: Training target
            param_space: Custom parameter space (optional)
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters and optimization results
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Default parameter space
            if param_space is None:
                param_space = {
                    'num_leaves': (31, 255),
                    'learning_rate': (0.01, 0.3),
                    'feature_fraction': (0.6, 1.0),
                    'bagging_fraction': (0.6, 1.0),
                    'min_child_samples': (10, 200),
                    'reg_alpha': (0.0, 1.0),
                    'reg_lambda': (0.0, 1.0)
                }
                
            def objective(trial):
                # Sample hyperparameters
                params = self.lgb_params.copy()
                
                for param_name, param_range in param_space.items():
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        if isinstance(param_range[0], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                        
                # Perform cross-validation with these parameters
                temp_model = LightGBMModel(self.config)
                temp_model.lgb_params = params
                
                cv_results = temp_model.cross_validate(X, y, cv_folds=3, shuffle=False)
                
                return cv_results['cv_score_mean']
                
            # Run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            # Update model parameters
            self.lgb_params.update(best_params)
            
            logger.info(f"Hyperparameter tuning completed: best score = {best_score:.6f}")
            logger.info(f"Best parameters: {best_params}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study,
                'n_trials': n_trials
            }
            
        except ImportError:
            logger.error("Optuna not available for hyperparameter tuning")
            return {}
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            return {}
            
    def get_training_curves(self) -> Dict[str, List[float]]:
        """Get training and validation curves"""
        return self.training_log
        
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None) -> None:
        """Plot feature importance"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.feature_importance:
                logger.warning("No feature importance available")
                return
                
            # Get top N features
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            features, importances = zip(*sorted_features)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance - {self.config.name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")
            
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_info()
        
        # Add LightGBM specific information
        if self.model is not None:
            summary.update({
                'num_trees': self.model.num_trees(),
                'best_iteration': getattr(self.model, 'best_iteration', 0),
                'lgb_params': self.lgb_params,
                'training_log_length': len(self.training_log)
            })
            
        return summary
        
    def explain_prediction_detailed(self, X: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        """
        Detailed prediction explanation for LightGBM
        
        Args:
            X: Single sample (1 row)
            top_n: Number of top contributing features to show
            
        Returns:
            Detailed explanation with feature contributions
        """
        if len(X) != 1:
            raise ValueError("Can only explain single predictions")
            
        try:
            # Get base prediction explanation
            base_explanation = self.explain_prediction(X)
            
            if 'error' in base_explanation:
                return base_explanation
                
            # Add LightGBM specific details
            prediction_result = self.predict(X, return_proba=True)
            
            # Get leaf indices for this prediction
            if hasattr(self.model, 'predict'):
                leaf_indices = self.model.predict(
                    X[self.feature_names].values,
                    pred_leaf=True,
                    num_iteration=self.model.best_iteration
                )
                
            # Sort feature contributions by absolute value
            shap_values = base_explanation['shap_values']
            sorted_contributions = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]
            
            detailed_explanation = {
                **base_explanation,
                'model_type': 'lightgbm',
                'num_trees_used': self.model.best_iteration or self.model.num_trees(),
                'top_contributing_features': sorted_contributions,
                'prediction_probability': prediction_result.probability,
                'prediction_confidence': prediction_result.confidence,
                'leaf_indices': leaf_indices.tolist() if 'leaf_indices' in locals() else None
            }
            
            return detailed_explanation
            
        except Exception as e:
            logger.error(f"Detailed explanation failed: {e}")
            return {'error': str(e)}