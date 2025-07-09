"""
Model System for XAUUSD Scalping

Ensemble ML approach with LightGBM primary, XGBoost + LSTM secondary,
and stacking meta-learner. Optimized for binary scalp target prediction
with 70-78% hit rate targets based on research synthesis.

Architecture:
- Primary: LightGBM (fast, explainable baseline)
- Secondary: XGBoost + LSTM ensemble
- Meta-learner: Stacking with logistic regression
- Ensemble weights: LightGBM 40%, XGBoost 35%, LSTM 25%

Research Targets:
- Hit rate: 70-78% (research achieved)
- Micro-Sharpe: ≥1.8
- Profit factor: ≥1.5
- Inference latency: <30ms
- Monthly retraining with walk-forward validation

Key Features:
- Walk-forward cross-validation
- Optuna hyperparameter optimization
- SHAP explainability
- Cost-aware evaluation
- Real-time inference optimization
"""

from .base import (
    BaseModel,
    ModelConfig,
    ModelMetrics,
    PredictionResult
)

from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .ensemble import EnsembleModel, StackingEnsemble
from .optuna_tuner import OptunaHyperparameterTuner

__all__ = [
    "BaseModel",
    "ModelConfig",
    "ModelMetrics", 
    "PredictionResult",
    "LightGBMModel",
    "XGBoostModel",
    "LSTMModel",
    "EnsembleModel",
    "StackingEnsemble",
    "OptunaHyperparameterTuner"
]