"""
Base Model Framework

Abstract base classes and common functionality for all trading models.
Provides standardized interfaces for training, prediction, evaluation,
and model management with performance monitoring and explainability.

Key Features:
- Standardized model interface
- Performance tracking and monitoring
- SHAP-based explainability
- Model persistence and versioning
- Cost-aware evaluation metrics
- Real-time inference optimization

Time Complexity: Varies by model implementation
Space Complexity: O(n * m) for training data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import joblib
import pickle
from pathlib import Path
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
import time

from src.core.config import get_config_manager
from src.core.utils import Timer, safe_divide
from src.core.logging import get_perf_logger

logger = logging.getLogger(__name__)
perf_logger = get_perf_logger()


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    name: str
    model_type: str  # 'lightgbm', 'xgboost', 'lstm', 'ensemble'
    target_type: str = 'binary_classification'
    
    # Training parameters
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_method: str = 'walk_forward'
    cv_folds: int = 5
    
    # Performance targets
    min_hit_rate: float = 0.70
    min_sharpe: float = 1.8
    min_profit_factor: float = 1.5
    max_inference_latency_ms: float = 30.0
    
    # Retraining
    retrain_frequency: str = 'monthly'
    retrain_threshold: float = 0.05  # Performance degradation threshold
    
    # Model persistence
    save_path: str = './data/models'
    versioning: bool = True
    
    # Explainability
    enable_shap: bool = True
    shap_sample_size: int = 1000


@dataclass
class ModelMetrics:
    """Comprehensive model evaluation metrics"""
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    
    # Financial metrics
    hit_rate: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Model performance
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0
    
    # Validation info
    validation_method: str = ""
    cv_score_mean: float = 0.0
    cv_score_std: float = 0.0
    
    # Feature importance
    top_features: List[str] = field(default_factory=list)
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc,
            'hit_rate': self.hit_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'training_time_seconds': self.training_time_seconds,
            'inference_time_ms': self.inference_time_ms,
            'model_size_mb': self.model_size_mb,
            'validation_method': self.validation_method,
            'cv_score_mean': self.cv_score_mean,
            'cv_score_std': self.cv_score_std,
            'top_features': self.top_features,
            'feature_importance_scores': self.feature_importance_scores
        }


@dataclass 
class PredictionResult:
    """Model prediction result with metadata"""
    prediction: Union[float, int, np.ndarray]
    probability: Optional[float] = None
    confidence: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    inference_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseModel(ABC):
    """
    Abstract base class for all trading models
    
    Provides common functionality and standardized interface for
    model training, prediction, evaluation, and management.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_data_hash = None
        self.version = None
        
        # Performance tracking
        self.metrics = ModelMetrics()
        self.training_history: List[Dict[str, Any]] = []
        
        # Feature information
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        # SHAP explainer
        self.shap_explainer = None
        
        # Ensure save directory exists
        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Train the underlying model implementation"""
        pass
        
    @abstractmethod
    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from the model"""
        pass
        
    @abstractmethod
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        pass
        
    @abstractmethod
    def _save_model(self, filepath: str) -> None:
        """Save model to file"""
        pass
        
    @abstractmethod
    def _load_model(self, filepath: str) -> None:
        """Load model from file"""
        pass
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              sample_weights: np.ndarray = None) -> ModelMetrics:
        """
        Train the model with comprehensive validation and evaluation
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            sample_weights: Sample weights (optional)
            
        Returns:
            ModelMetrics with evaluation results
        """
        start_time = time.time()
        
        with Timer(f"model_training_{self.config.name}"):
            try:
                # Validate inputs
                if X.empty or y.empty:
                    raise ValueError("Training data cannot be empty")
                    
                if len(X) != len(y):
                    raise ValueError("Feature and target lengths must match")
                    
                # Store feature names
                self.feature_names = X.columns.tolist()
                
                # Calculate data hash for model versioning
                self.training_data_hash = self._calculate_data_hash(X, y)
                
                # Train the model
                logger.info(f"Training {self.config.name} with {len(X)} samples, {len(X.columns)} features")
                
                self._train_model(X, y, X_val, y_val)
                
                # Mark as trained
                self.is_trained = True
                
                # Extract feature importance
                self.feature_importance = self._get_feature_importance()
                
                # Setup SHAP explainer
                if self.config.enable_shap:
                    self._setup_shap_explainer(X)
                    
                # Evaluate model
                if X_val is not None and y_val is not None:
                    self.metrics = self._evaluate_model(X_val, y_val)
                else:
                    # Use training data for evaluation (not ideal but better than nothing)
                    self.metrics = self._evaluate_model(X, y)
                    
                # Record training time
                training_time = time.time() - start_time
                self.metrics.training_time_seconds = training_time
                
                # Update version
                self.version = self._generate_version()
                
                # Record training history
                self.training_history.append({
                    'timestamp': datetime.now(timezone.utc),
                    'version': self.version,
                    'metrics': self.metrics.to_dict(),
                    'feature_count': len(X.columns),
                    'sample_count': len(X)
                })
                
                logger.info(
                    f"Training completed: Hit rate: {self.metrics.hit_rate:.3f}, "
                    f"Sharpe: {self.metrics.sharpe_ratio:.3f}, "
                    f"Time: {training_time:.2f}s"
                )
                
                # Log performance
                perf_logger.log_latency(f"model_training_{self.config.name}", training_time * 1000)
                
                return self.metrics
                
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                raise
                
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> PredictionResult:
        """
        Make predictions with performance monitoring
        
        Args:
            X: Features for prediction
            return_proba: Whether to return probabilities
            
        Returns:
            PredictionResult with predictions and metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        start_time = time.perf_counter()
        
        try:
            with Timer(f"model_inference_{self.config.name}"):
                # Validate features
                if not all(col in X.columns for col in self.feature_names):
                    missing_features = set(self.feature_names) - set(X.columns)
                    raise ValueError(f"Missing features: {missing_features}")
                    
                # Ensure feature order matches training
                X_ordered = X[self.feature_names]
                
                # Get predictions
                if return_proba or self.config.target_type == 'binary_classification':
                    probabilities = self._predict_proba(X_ordered)
                    
                    if self.config.target_type == 'binary_classification':
                        # Binary classification: use class 1 probability
                        if probabilities.ndim == 2:
                            prediction_proba = probabilities[:, 1]
                        else:
                            prediction_proba = probabilities
                            
                        # Convert to binary prediction (0.5 threshold)
                        prediction = (prediction_proba > 0.5).astype(int)
                        
                        # Calculate confidence (distance from 0.5)
                        confidence = np.abs(prediction_proba - 0.5) * 2
                        
                    else:
                        prediction = probabilities
                        prediction_proba = None
                        confidence = None
                        
                else:
                    prediction = self._predict_proba(X_ordered)
                    prediction_proba = None
                    confidence = None
                    
                # Calculate inference time
                inference_time = (time.perf_counter() - start_time) * 1000
                
                # Check latency target
                if inference_time > self.config.max_inference_latency_ms:
                    logger.warning(
                        f"Inference latency exceeded target: {inference_time:.2f}ms > "
                        f"{self.config.max_inference_latency_ms}ms"
                    )
                    
                # Feature importance for this prediction (if single sample)
                feature_importance = None
                if len(X) == 1 and self.shap_explainer is not None:
                    try:
                        shap_values = self.shap_explainer.shap_values(X_ordered.values)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # Class 1 for binary classification
                        feature_importance = dict(zip(self.feature_names, shap_values[0]))
                    except Exception as e:
                        logger.warning(f"SHAP explanation failed: {e}")
                        
                # Log performance
                perf_logger.log_latency(f"model_inference_{self.config.name}", inference_time)
                
                return PredictionResult(
                    prediction=prediction,
                    probability=prediction_proba,
                    confidence=confidence,
                    feature_importance=feature_importance,
                    inference_time_ms=inference_time,
                    model_version=self.version
                )
                
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise
            
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Comprehensive model evaluation"""
        try:
            # Get predictions
            pred_result = self.predict(X, return_proba=True)
            
            if isinstance(pred_result.prediction, np.ndarray):
                y_pred = pred_result.prediction
                y_proba = pred_result.probability
            else:
                y_pred = np.array([pred_result.prediction])
                y_proba = np.array([pred_result.probability]) if pred_result.probability is not None else None
                
            # Classification metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='binary', zero_division=0)
            recall = recall_score(y, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y, y_pred, average='binary', zero_division=0)
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else 0.0
            except:
                roc_auc = 0.0
                
            # Precision-Recall AUC
            try:
                if y_proba is not None:
                    precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba)
                    pr_auc = auc(recall_curve, precision_curve)
                else:
                    pr_auc = 0.0
            except:
                pr_auc = 0.0
                
            # Financial metrics
            hit_rate = accuracy  # Same as accuracy for binary classification
            
            # Calculate financial metrics (simplified)
            returns = self._calculate_strategy_returns(y, y_pred, y_proba)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            profit_factor = self._calculate_profit_factor(returns)
            expectancy = np.mean(returns) if len(returns) > 0 else 0.0
            
            # Feature importance
            top_features = []
            if self.feature_importance:
                sorted_features = sorted(
                    self.feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                top_features = [name for name, _ in sorted_features[:10]]
                
            # Model size
            model_size_mb = self._estimate_model_size()
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                hit_rate=hit_rate,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                profit_factor=profit_factor,
                expectancy=expectancy,
                inference_time_ms=pred_result.inference_time_ms or 0.0,
                model_size_mb=model_size_mb,
                top_features=top_features,
                feature_importance_scores=self.feature_importance
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return ModelMetrics()
            
    def _calculate_strategy_returns(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray = None) -> np.ndarray:
        """Calculate strategy returns based on predictions"""
        # Simplified return calculation
        # In practice, this should incorporate actual price movements and transaction costs
        
        # Assume 1% average move, with direction based on prediction correctness
        base_return = 0.01
        
        returns = []
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                # Correct prediction
                ret = base_return
                if y_proba is not None:
                    # Scale by confidence
                    confidence = abs(y_proba[i] - 0.5) * 2
                    ret *= confidence
                returns.append(ret)
            else:
                # Incorrect prediction
                ret = -base_return
                if y_proba is not None:
                    confidence = abs(y_proba[i] - 0.5) * 2
                    ret *= confidence
                returns.append(ret)
                
        return np.array(returns)
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return np.inf if np.mean(returns) > 0 else 0.0
        downside_std = np.std(negative_returns)
        if downside_std == 0:
            return 0.0
        return np.mean(returns) / downside_std * np.sqrt(252)
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown))
        
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_gains = np.sum(gains) if len(gains) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
        
        return safe_divide(total_gains, total_losses)
        
    def _setup_shap_explainer(self, X: pd.DataFrame) -> None:
        """Setup SHAP explainer for model interpretability"""
        try:
            if hasattr(self.model, 'predict'):
                # Use a sample for performance
                sample_size = min(self.config.shap_sample_size, len(X))
                X_sample = X.sample(n=sample_size, random_state=42)
                
                # Different explainers for different model types
                if hasattr(self.model, 'predict_proba'):
                    # Tree-based models
                    if hasattr(self.model, 'booster'):  # LightGBM/XGBoost
                        self.shap_explainer = shap.TreeExplainer(self.model)
                    else:
                        # Other models
                        self.shap_explainer = shap.Explainer(self.model.predict_proba, X_sample)
                else:
                    self.shap_explainer = shap.Explainer(self.model.predict, X_sample)
                    
                logger.info("SHAP explainer initialized successfully")
                
        except Exception as e:
            logger.warning(f"Failed to setup SHAP explainer: {e}")
            self.shap_explainer = None
            
    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate hash of training data for versioning"""
        import hashlib
        
        # Combine feature hash and target hash
        X_hash = hashlib.md5(pd.util.hash_pandas_object(X).values).hexdigest()
        y_hash = hashlib.md5(pd.util.hash_pandas_object(y).values).hexdigest()
        
        combined = f"{X_hash}_{y_hash}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
        
    def _generate_version(self) -> str:
        """Generate model version string"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{self.config.name}_{timestamp}_{self.training_data_hash[:8]}"
        
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB"""
        try:
            if self.model is not None:
                # Serialize model to estimate size
                import pickle
                serialized = pickle.dumps(self.model)
                return len(serialized) / (1024 * 1024)  # Convert to MB
        except:
            pass
        return 0.0
        
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save model with metadata
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        if filepath is None:
            filename = f"{self.version}.joblib"
            filepath = Path(self.config.save_path) / filename
        else:
            filepath = Path(filepath)
            
        try:
            # Prepare model data
            model_data = {
                'model': self.model,
                'config': self.config,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'version': self.version,
                'training_data_hash': self.training_data_hash,
                'training_history': self.training_history,
                'saved_at': datetime.now(timezone.utc)
            }
            
            # Save using joblib for sklearn compatibility
            joblib.dump(model_data, filepath)
            
            logger.info(f"Model saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
            
    def load(self, filepath: str) -> None:
        """
        Load model from file
        
        Args:
            filepath: Path to saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            # Restore model state
            self.model = model_data['model']
            self.config = model_data.get('config', self.config)
            self.metrics = model_data.get('metrics', ModelMetrics())
            self.feature_names = model_data.get('feature_names', [])
            self.feature_importance = model_data.get('feature_importance', {})
            self.version = model_data.get('version', 'unknown')
            self.training_data_hash = model_data.get('training_data_hash', None)
            self.training_history = model_data.get('training_history', [])
            
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath} (version: {self.version})")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features"""
        if not self.feature_importance:
            return {}
            
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
        
    def explain_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP
        
        Args:
            X: Single sample features (1 row)
            
        Returns:
            Dictionary with SHAP explanation
        """
        if len(X) != 1:
            raise ValueError("Can only explain single predictions")
            
        if self.shap_explainer is None:
            return {'error': 'SHAP explainer not available'}
            
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X[self.feature_names].values)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 for binary classification
                
            # Create explanation
            explanation = {
                'base_value': self.shap_explainer.expected_value,
                'shap_values': dict(zip(self.feature_names, shap_values[0])),
                'feature_values': dict(zip(self.feature_names, X[self.feature_names].iloc[0])),
                'prediction': self.predict(X).prediction
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}
            
    def should_retrain(self, performance_metric: float = None) -> bool:
        """
        Determine if model should be retrained based on performance degradation
        
        Args:
            performance_metric: Current performance metric (e.g., hit rate)
            
        Returns:
            True if retraining is recommended
        """
        if not self.is_trained or not self.training_history:
            return False
            
        # Get baseline performance from training
        baseline_metric = self.metrics.hit_rate
        
        if performance_metric is not None:
            # Check if performance has degraded significantly
            degradation = (baseline_metric - performance_metric) / baseline_metric
            
            if degradation > self.config.retrain_threshold:
                logger.info(
                    f"Performance degradation detected: {degradation:.3f} > "
                    f"{self.config.retrain_threshold:.3f}"
                )
                return True
                
        # Check if it's time for scheduled retraining
        last_training = self.training_history[-1]['timestamp']
        time_since_training = datetime.now(timezone.utc) - last_training
        
        if self.config.retrain_frequency == 'daily' and time_since_training.days >= 1:
            return True
        elif self.config.retrain_frequency == 'weekly' and time_since_training.days >= 7:
            return True
        elif self.config.retrain_frequency == 'monthly' and time_since_training.days >= 30:
            return True
            
        return False
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'name': self.config.name,
            'type': self.config.model_type,
            'version': self.version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'metrics': self.metrics.to_dict(),
            'config': self.config.__dict__,
            'training_history_count': len(self.training_history),
            'last_trained': self.training_history[-1]['timestamp'] if self.training_history else None
        }