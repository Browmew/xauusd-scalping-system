"""
XGBoost Model Implementation

Secondary ensemble model for XAUUSD scalping system. XGBoost provides
complementary strengths to LightGBM with different regularization and
feature handling characteristics.

Research Configuration:
- Secondary model in ensemble (35% weight)
- Robust to overfitting with strong regularization
- Excellent for complex feature interactions
- GPU acceleration support
- SHAP native integration

Performance Targets:
- Training time: <10 minutes for monthly retrain
- Inference time: <15ms per prediction
- Hit rate: 70%+ baseline
- Model size: <100MB

Time Complexity: O(n * m * log(m)) for training
Space Complexity: O(n * m) for training data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from .base import BaseModel, ModelConfig, ModelMetrics
from src.core.config import get_config_manager
from src.core.utils import Timer

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost implementation for binary scalp prediction
    
    Configured for financial time series with emphasis on regularization
    and robustness to overfitting.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                name="XGBoost_Scalping",
                model_type="xgboost"
            )
            
        super().__init__(config)
        
        # XGBoost specific parameters from research synthesis
        self.xgb_params = self._get_default_params()
        
        # Update with any custom parameters
        if 'xgboost' in self.config.training_params:
            self.xgb_params.update(self.config.training_params['xgboost'])
            
        # Training artifacts
        self.xgb_train_data = None
        self.xgb_val_data = None
        self.eval_results = {}
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters from research synthesis"""
        return {
            # Core parameters
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            
            # Regularization (from research)
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0.1,
            
            # Training control
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            
            # Performance
            'n_jobs': -1,
            'tree_method': 'auto',  # 'gpu_hist' if GPU available
            'gpu_id': 0,
            
            # Random state for reproducibility
            'random_state': 42,
            
            # Additional parameters for financial data
            'scale_pos_weight': 1,  # Will be adjusted based on class balance
            'max_delta_step': 1     # Conservative updates
        }
        
    def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Train XGBoost model with validation"""
        
        with Timer("xgboost_training"):
            try:
                # Adjust scale_pos_weight based on class balance
                pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0
                self.xgb_params['scale_pos_weight'] = pos_weight
                
                logger.info(f"Adjusted scale_pos_weight to {pos_weight:.3f}")
                
                # Prepare datasets
                self.xgb_train_data = xgb.DMatrix(
                    X.values,
                    label=y.values,
                    feature_names=X.columns.tolist()
                )
                
                eval_sets = [(self.xgb_train_data, 'train')]
                
                if X_val is not None and y_val is not None:
                    self.xgb_val_data = xgb.DMatrix(
                        X_val.values,
                        label=y_val.values,
                        feature_names=X.columns.tolist()
                    )
                    eval_sets.append((self.xgb_val_data, 'valid'))
                    
                # Train model
                logger.info(f"Training XGBoost with {len(X)} samples, {len(X.columns)} features")
                
                self.model = xgb.train(
                    params=self.xgb_params,
                    dtrain=self.xgb_train_data,
                    num_boost_round=self.xgb_params['n_estimators'],
                    evals=eval_sets,
                    evals_result=self.eval_results,
                    early_stopping_rounds=self.xgb_params.get('early_stopping_rounds', 100),
                    verbose_eval=False
                )
                
                # Log training results
                best_iteration = self.model.best_iteration
                logger.info(f"Training completed at iteration {best_iteration}")
                
                if 'valid' in self.eval_results and 'logloss' in self.eval_results['valid']:
                    best_score = self.eval_results['valid']['logloss'][best_iteration]
                    logger.info(f"Best validation score: {best_score:.6f}")
                    
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")
                raise
                
    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Ensure feature order matches training
        X_ordered = X[self.feature_names]
        
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(X_ordered.values, feature_names=self.feature_names)
        
        # Get probabilities
        probabilities = self.model.predict(dmatrix, iteration_range=(0, self.model.best_iteration))
        
        return probabilities
        
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            return {}
            
        try:
            # Get different types of importance
            importance_gain = self.model.get_score(importance_type='gain')
            importance_weight = self.model.get_score(importance_type='weight')
            importance_cover = self.model.get_score(importance_type='cover')
            
            # Combine different importance types (weighted average)
            all_features = set(importance_gain.keys()) | set(importance_weight.keys()) | set(importance_cover.keys())
            
            combined_importance = {}
            for feature in all_features:
                gain = importance_gain.get(feature, 0)
                weight = importance_weight.get(feature, 0)
                cover = importance_cover.get(feature, 0)
                
                # Weighted combination
                combined_importance[feature] = 0.5 * gain + 0.3 * weight + 0.2 * cover
                
            return combined_importance
            
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return {}
            
    def _save_model(self, filepath: str) -> None:
        """Save XGBoost model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            # Save XGBoost model in native format
            self.model.save_model(filepath + '_xgb.json')
            
            # Also save evaluation results
            import pickle
            with open(filepath + '_eval.pkl', 'wb') as f:
                pickle.dump(self.eval_results, f)
                
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")
            raise
            
    def _load_model(self, filepath: str) -> None:
        """Load XGBoost model"""
        try:
            # Load XGBoost model
            self.model = xgb.Booster()
            self.model.load_model(filepath + '_xgb.json')
            
            # Load evaluation results if available
            import pickle
            try:
                with open(filepath + '_eval.pkl', 'rb') as f:
                    self.eval_results = pickle.load(f)
            except FileNotFoundError:
                self.eval_results = {}
                
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
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
        
        with Timer("xgboost_cv"):
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
                    fold_model = XGBoostModel(self.config)
                    fold_model.xgb_params = self.xgb_params.copy()
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
            
            # Default parameter space for XGBoost
            if param_space is None:
                param_space = {
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'min_child_weight': (1, 20),
                    'reg_alpha': (0.0, 1.0),
                    'reg_lambda': (0.0, 1.0),
                    'gamma': (0.0, 1.0)
                }
                
            def objective(trial):
                # Sample hyperparameters
                params = self.xgb_params.copy()
                
                for param_name, param_range in param_space.items():
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        if isinstance(param_range[0], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                        
                # Perform cross-validation with these parameters
                temp_model = XGBoostModel(self.config)
                temp_model.xgb_params = params
                
                cv_results = temp_model.cross_validate(X, y, cv_folds=3, shuffle=False)
                
                return cv_results['cv_score_mean']
                
            # Run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            # Update model parameters
            self.xgb_params.update(best_params)
            
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
            
    def get_training_curves(self) -> Dict[str, Any]:
        """Get training and validation curves"""
        return self.eval_results
        
    def plot_feature_importance(self, importance_type: str = 'gain', top_n: int = 20, save_path: str = None) -> None:
        """
        Plot feature importance
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.model is None:
                logger.warning("No trained model available")
                return
                
            # Get feature importance
            importance = self.model.get_score(importance_type=importance_type)
            
            if not importance:
                logger.warning("No feature importance available")
                return
                
            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel(f'Feature Importance ({importance_type})')
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
            
    def plot_training_curves(self, save_path: str = None) -> None:
        """Plot training and validation curves"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.eval_results:
                logger.warning("No training curves available")
                return
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot training curve
            if 'train' in self.eval_results and 'logloss' in self.eval_results['train']:
                train_scores = self.eval_results['train']['logloss']
                ax.plot(train_scores, label='Training', alpha=0.8)
                
            # Plot validation curve
            if 'valid' in self.eval_results and 'logloss' in self.eval_results['valid']:
                val_scores = self.eval_results['valid']['logloss']
                ax.plot(val_scores, label='Validation', alpha=0.8)
                
                # Mark best iteration
                if hasattr(self.model, 'best_iteration'):
                    ax.axvline(x=self.model.best_iteration, color='red', 
                              linestyle='--', alpha=0.7, label='Best Iteration')
                              
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Log Loss')
            ax.set_title(f'Training Curves - {self.config.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training curves plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot training curves: {e}")
            
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_info()
        
        # Add XGBoost specific information
        if self.model is not None:
            summary.update({
                'num_trees': self.model.num_boosted_rounds(),
                'best_iteration': getattr(self.model, 'best_iteration', 0),
                'xgb_params': self.xgb_params,
                'eval_results_keys': list(self.eval_results.keys())
            })
            
        return summary
        
    def get_tree_info(self, tree_idx: int = 0) -> Dict[str, Any]:
        """
        Get information about a specific tree
        
        Args:
            tree_idx: Index of the tree to analyze
            
        Returns:
            Tree information dictionary
        """
        if self.model is None:
            return {}
            
        try:
            # Get tree dump
            tree_dump = self.model.get_dump(dump_format='json')
            
            if tree_idx < len(tree_dump):
                import json
                tree_info = json.loads(tree_dump[tree_idx])
                
                # Calculate tree statistics
                def count_nodes(node):
                    if 'children' not in node:
                        return 1  # Leaf node
                    return 1 + sum(count_nodes(child) for child in node['children'])
                    
                def max_depth(node, current_depth=0):
                    if 'children' not in node:
                        return current_depth
                    return max(max_depth(child, current_depth + 1) for child in node['children'])
                    
                return {
                    'tree_index': tree_idx,
                    'num_nodes': count_nodes(tree_info),
                    'max_depth': max_depth(tree_info),
                    'tree_structure': tree_info
                }
            else:
                return {'error': f'Tree index {tree_idx} out of range'}
                
        except Exception as e:
            logger.error(f"Failed to get tree info: {e}")
            return {'error': str(e)}
            
    def explain_prediction_detailed(self, X: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        """
        Detailed prediction explanation for XGBoost
        
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
                
            # Add XGBoost specific details
            prediction_result = self.predict(X, return_proba=True)
            
            # Get leaf indices for this prediction
            dmatrix = xgb.DMatrix(X[self.feature_names].values, feature_names=self.feature_names)
            leaf_indices = self.model.predict(dmatrix, pred_leaf=True)
            
            # Sort feature contributions by absolute value
            shap_values = base_explanation['shap_values']
            sorted_contributions = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]
            
            detailed_explanation = {
                **base_explanation,
                'model_type': 'xgboost',
                'num_trees_used': self.model.best_iteration or self.model.num_boosted_rounds(),
                'top_contributing_features': sorted_contributions,
                'prediction_probability': prediction_result.probability,
                'prediction_confidence': prediction_result.confidence,
                'leaf_indices': leaf_indices[0].tolist(),
                'trees_per_class': 1  # Binary classification
            }
            
            return detailed_explanation
            
        except Exception as e:
            logger.error(f"Detailed explanation failed: {e}")
            return {'error': str(e)}