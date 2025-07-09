"""
XAUUSD Scalping System

A production-grade, low-latency intraday scalping system for XAUUSD trading.
Built on extensive research synthesis focusing on order-flow microstructure,
hybrid ML approaches, and institutional-level risk management.

Key Features:
- Sub-50ms latency execution
- Ensemble ML models (LightGBM + XGBoost + LSTM)
- Advanced order flow features
- Comprehensive risk management
- Full backtesting engine
- Live trading capabilities

Research Foundation:
Based on synthesis of 7 research studies emphasizing:
- Order-flow superiority over traditional TA
- Hybrid ML + classical indicator stacks
- Session/regime filtering requirements
- Cost realism and latency budgets
- Explainability via SHAP analysis
"""

__version__ = "1.0.0"
__author__ = "XAUUSD Scalping System"
__email__ = "system@xauusd-scalping.com"

# Core imports for easy access
from src.core.config import ConfigManager
from src.core.logging import setup_logging
from src.core.utils import Timer, PerformanceMonitor

__all__ = [
    "ConfigManager",
    "setup_logging", 
    "Timer",
    "PerformanceMonitor",
]