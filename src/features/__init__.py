"""
Feature Engineering System for XAUUSD Scalping

Comprehensive feature engineering pipeline implementing 200+ engineered features
based on research synthesis emphasizing order-flow superiority over traditional TA.

Feature Categories:
- Order Flow & Microstructure (40 features) - PRIMARY FOCUS
- Technical Indicators (30 features) - Selected survivors only
- Market Structure (30 features) - Support/resistance, patterns
- Volume Profile (20 features) - VPVR, POC analysis
- Smart Money Concepts (25 features) - Liquidity sweeps, order blocks
- Cross-Asset Correlations (15 features) - DXY, TIPS, SPY
- Regime & Session Filters (15 features) - Critical per research
- Volatility Estimators (25 features) - GKYZ, Parkinson, etc.

Performance Targets:
- Minimal subset: 6 features, ~85% Sharpe retention, <25ms latency
- Extended set: 200+ features, full analysis, <50ms latency
- Real-time computation with vectorized operations
- SHAP-based feature importance and pruning

Research Findings:
- Order-flow > legacy TA consistently
- Hybrid stacks (ML + classical) win by 15-25%
- Session/regime filtering mandatory
- Cost realism critical (spread + commission)
"""

from .base import (
    FeatureEngine,
    BaseFeatureExtractor,
    FeatureConfig,
    FeatureImportance
)

from .technical import (
    TechnicalIndicators,
    MovingAverageFeatures,
    MomentumFeatures,
    VolatilityFeatures
)

from .orderbook import (
    OrderBookFeatures,
    ImbalanceFeatures,
    DepthFeatures,
    MicropriceFatures
)

from .microstructure import (
    MicrostructureFeatures,
    VolumeProfileFeatures,
    SmartMoneyFeatures,
    CrossAssetFeatures
)

from .ensemble import (
    FeatureEnsemble,
    FeatureSelector,
    FeaturePruner
)

__all__ = [
    "FeatureEngine",
    "BaseFeatureExtractor", 
    "FeatureConfig",
    "FeatureImportance",
    "TechnicalIndicators",
    "MovingAverageFeatures",
    "MomentumFeatures", 
    "VolatilityFeatures",
    "OrderBookFeatures",
    "ImbalanceFeatures",
    "DepthFeatures",
    "MicropriceFatures",
    "MicrostructureFeatures",
    "VolumeProfileFeatures",
    "SmartMoneyFeatures", 
    "CrossAssetFeatures",
    "FeatureEnsemble",
    "FeatureSelector",
    "FeaturePruner"
]