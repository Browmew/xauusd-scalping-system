"""
Feature Ensemble and Selection System

Orchestrates all feature extractors and implements intelligent feature selection
based on SHAP importance, correlation analysis, and performance constraints.
Manages the transition between minimal (6 features) and extended (200+) feature sets.

Key Components:
- FeatureEnsemble: Main coordinator for all feature extraction
- FeatureSelector: SHAP-based feature selection and ranking
- FeaturePruner: Correlation-based pruning and performance optimization
- Real-time feature importance monitoring

Research Implementation:
- Minimal subset: HMA-5, GKYZ, 5-level OBI, VPVR breakout, liquidity sweep, ATR-14
- Extended set: Full 200+ feature catalog with automatic pruning
- Performance targets: <25ms minimal, <50ms extended
- SHAP analysis for explainability and feature ranking

Time Complexity: O(n * m) where n=samples, m=features
Space Complexity: O(n * m) with memory optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timezone
import shap
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from .base import BaseFeatureExtractor, FeatureEngine, FeatureConfig, FeatureImportance
from .technical import TechnicalIndicators
from .orderbook import OrderBookFeatures, ImbalanceFeatures, DepthFeatures, MicropriceFatures
from .microstructure import MicrostructureFeatures
from src.core.config import get_config_manager
from src.core.utils import Timer, safe_divide, get_performance_monitor
from src.core.logging import get_perf_logger

logger = logging.getLogger(__name__)
perf_logger = get_perf_logger()


class FeatureEnsemble:
    """
    Main feature ensemble coordinator
    
    Orchestrates all feature extractors to produce either minimal or extended
    feature sets with performance monitoring and intelligent caching.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Initialize feature engine
        self.feature_engine = FeatureEngine(config_manager)
        
        # Initialize all feature extractors
        self._initialize_extractors()
        
        # Feature selection and pruning
        self.feature_selector = FeatureSelector(config_manager)
        self.feature_pruner = FeaturePruner(config_manager)
        
        # Performance tracking
        self.performance_monitor = get_performance_monitor()
        self.extraction_stats = {
            'total_extractions': 0,
            'avg_latency_ms': 0.0,
            'feature_counts': {'minimal': 0, 'extended': 0},
            'last_extraction': None
        }
        
    def _initialize_extractors(self) -> None:
        """Initialize and register all feature extractors"""
        
        # Technical Analysis Features
        tech_indicators = TechnicalIndicators(self.config)
        for name, extractor in tech_indicators.extractors.items():
            self.feature_engine.register_extractor(extractor)
            
        # Order Book Features (Primary Focus)
        orderbook_config = FeatureConfig(
            name="OrderBookFeatures",
            category="order_flow",
            priority=1,
            max_latency_ms=15.0
        )
        orderbook_extractor = OrderBookFeatures(orderbook_config)
        self.feature_engine.register_extractor(orderbook_extractor)
        
        # Imbalance Features (Research Core)
        imbalance_config = FeatureConfig(
            name="ImbalanceFeatures", 
            category="order_flow",
            priority=1,
            max_latency_ms=20.0
        )
        imbalance_extractor = ImbalanceFeatures(imbalance_config)
        self.feature_engine.register_extractor(imbalance_extractor)
        
        # Depth Features
        depth_config = FeatureConfig(
            name="DepthFeatures",
            category="order_flow", 
            priority=2,
            max_latency_ms=25.0
        )
        depth_extractor = DepthFeatures(depth_config)
        self.feature_engine.register_extractor(depth_extractor)
        
        # Microprice Features
        microprice_config = FeatureConfig(
            name="MicropriceFatures",
            category="order_flow",
            priority=1,
            max_latency_ms=15.0
        )
        microprice_extractor = MicropriceFatures(microprice_config)
        self.feature_engine.register_extractor(microprice_extractor)
        
        # Microstructure Features
        microstructure = MicrostructureFeatures(self.config)
        for name, extractor in microstructure.extractors.items():
            self.feature_engine.register_extractor(extractor)
            
        logger.info(f"Initialized {len(self.feature_engine.extractors)} feature extractors")
        
    def extract_features(
        self,
        tick_data: pd.DataFrame,
        orderbook_data: pd.DataFrame = None,
        cross_asset_data: Dict[str, pd.DataFrame] = None,
        feature_set: str = "extended",
        force_minimal: bool = False
    ) -> pd.DataFrame:
        """
        Extract complete feature set
        
        Args:
            tick_data: OHLCV tick/bar data
            orderbook_data: Level 2 order book data
            cross_asset_data: Cross-asset price data
            feature_set: "minimal" or "extended"
            force_minimal: Force minimal set for latency constraints
            
        Returns:
            DataFrame with computed features
        """
        start_time = time.time()
        
        with Timer("feature_ensemble_extraction"):
            try:
                # Determine feature set based on performance constraints
                if force_minimal or self._should_use_minimal_set():
                    feature_set = "minimal"
                    
                # Extract base features from tick data
                if feature_set == "minimal":
                    features = self._extract_minimal_features(tick_data, orderbook_data)
                else:
                    features = self._extract_extended_features(
                        tick_data, orderbook_data, cross_asset_data
                    )
                    
                # Post-processing
                features = self._post_process_features(features, feature_set)
                
                # Update statistics
                extraction_time = (time.time() - start_time) * 1000
                self._update_extraction_stats(extraction_time, len(features.columns), feature_set)
                
                logger.info(
                    f"Extracted {len(features.columns)} features ({feature_set}) "
                    f"in {extraction_time:.2f}ms"
                )
                
                return features
                
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                return pd.DataFrame(index=tick_data.index)
                
    def _should_use_minimal_set(self) -> bool:
        """Determine if minimal feature set should be used based on performance"""
        # Check current system performance
        perf_snapshot = self.performance_monitor.get_snapshot()
        
        # Use minimal set if system under stress
        if perf_snapshot.cpu_percent > 80 or perf_snapshot.memory.rss_mb > 1000:
            return True
            
        # Check recent extraction performance
        if self.extraction_stats['avg_latency_ms'] > 45:  # Close to 50ms limit
            return True
            
        return False
        
    def _extract_minimal_features(
        self, 
        tick_data: pd.DataFrame, 
        orderbook_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Extract minimal feature set for low-latency operation
        
        Research minimal set: HMA-5, GKYZ, 5-level OBI, VPVR breakout, 
        liquidity sweep, ATR-14 (~85% Sharpe retention, <25ms latency)
        """
        features = pd.DataFrame(index=tick_data.index)
        
        with Timer("minimal_feature_extraction"):
            try:
                # 1. HMA-5 (Hull Moving Average - research favorite for low lag)
                if 'close' in tick_data.columns:
                    hma_5 = self._calculate_hma_fast(tick_data['close'].values, 5)
                    features['hma_5'] = hma_5
                    
                # 2. ATR-14 (Essential for risk management)
                if all(col in tick_data.columns for col in ['high', 'low', 'close']):
                    atr_14 = self._calculate_atr_fast(
                        tick_data['high'].values,
                        tick_data['low'].values, 
                        tick_data['close'].values,
                        14
                    )
                    features['atr_14'] = atr_14
                    
                # 3. GKYZ Volatility (Research favorite for regime detection)
                if all(col in tick_data.columns for col in ['open', 'high', 'low', 'close']):
                    gkyz_vol = self._calculate_gkyz_fast(
                        tick_data['open'].values,
                        tick_data['high'].values,
                        tick_data['low'].values,
                        tick_data['close'].values,
                        20
                    )
                    features['gkyz_volatility'] = gkyz_vol
                    
                # 4. 5-level OBI (Order Book Imbalance - core research feature)
                if orderbook_data is not None and not orderbook_data.empty:
                    obi_5 = self._calculate_obi_fast(orderbook_data, 5)
                    features['obi_5_level'] = obi_5
                else:
                    # Fallback: use volume-based imbalance proxy
                    if 'volume' in tick_data.columns:
                        features['obi_5_level'] = self._volume_imbalance_proxy(tick_data)
                        
                # 5. VPVR Breakout (Volume Profile breakout detection)
                if all(col in tick_data.columns for col in ['high', 'low', 'close', 'volume']):
                    vpvr_breakout = self._calculate_vpvr_breakout_fast(tick_data, 100)
                    features['vpvr_breakout'] = vpvr_breakout
                    
                # 6. Liquidity Sweep (Smart Money Concepts)
                if all(col in tick_data.columns for col in ['high', 'low', 'close']):
                    liquidity_sweep = self._calculate_liquidity_sweep_fast(tick_data, 20)
                    features['liquidity_sweep'] = liquidity_sweep
                    
                # Ensure no NaN values in minimal set
                features = features.fillna(method='ffill').fillna(0)
                
                logger.info(f"Extracted {len(features.columns)} minimal features")
                
            except Exception as e:
                logger.error(f"Minimal feature extraction failed: {e}")
                
        return features
        
    def _extract_extended_features(
        self,
        tick_data: pd.DataFrame,
        orderbook_data: pd.DataFrame = None,
        cross_asset_data: Dict[str, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Extract full extended feature set (200+ features)"""
        
        with Timer("extended_feature_extraction"):
            # Use the feature engine to extract all features
            features = self.feature_engine.extract_features(
                data=tick_data,
                feature_set="extended"
            )
            
            # Add order book features if available
            if orderbook_data is not None and not orderbook_data.empty:
                try:
                    # Extract order book specific features
                    ob_features = self._extract_orderbook_features(orderbook_data)
                    if not ob_features.empty:
                        features = pd.concat([features, ob_features], axis=1)
                except Exception as e:
                    logger.warning(f"Order book feature extraction failed: {e}")
                    
            # Add cross-asset features if available
            if cross_asset_data:
                try:
                    cross_features = self._extract_cross_asset_features(tick_data, cross_asset_data)
                    if not cross_features.empty:
                        features = pd.concat([features, cross_features], axis=1)
                except Exception as e:
                    logger.warning(f"Cross-asset feature extraction failed: {e}")
                    
        return features
        
    def _post_process_features(self, features: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        """Post-process features with cleaning and selection"""
        
        if features.empty:
            return features
            
        with Timer("feature_post_processing"):
            # Remove duplicate columns
            features = features.loc[:, ~features.columns.duplicated()]
            
            # Handle missing values
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove constant columns
            constant_columns = features.columns[features.nunique() <= 1]
            if len(constant_columns) > 0:
                features = features.drop(columns=constant_columns)
                logger.info(f"Removed {len(constant_columns)} constant columns")
                
            # Feature selection for extended set
            if feature_set == "extended" and len(features.columns) > 50:
                features = self.feature_selector.select_top_features(
                    features, 
                    max_features=200,
                    method="variance_threshold"
                )
                
            # Correlation-based pruning
            if len(features.columns) > 20:
                features = self.feature_pruner.prune_correlated_features(
                    features,
                    correlation_threshold=0.95
                )
                
        return features
        
    # Fast calculation methods for minimal features
    
    def _calculate_hma_fast(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Fast Hull Moving Average calculation"""
        try:
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            # Weighted moving averages
            wma_half = self._wma_fast(prices, half_period)
            wma_full = self._wma_fast(prices, period)
            
            # Hull calculation
            hull_raw = 2 * wma_half - wma_full
            hma = self._wma_fast(hull_raw, sqrt_period)
            
            return hma
        except:
            # Fallback to EMA
            return self._ema_fast(prices, period)
            
    def _wma_fast(self, values: np.ndarray, period: int) -> np.ndarray:
        """Fast weighted moving average"""
        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()
        
        result = np.full_like(values, np.nan)
        
        for i in range(period - 1, len(values)):
            window = values[i - period + 1:i + 1]
            result[i] = np.dot(window, weights)
            
        return result
        
    def _ema_fast(self, values: np.ndarray, period: int) -> np.ndarray:
        """Fast EMA calculation"""
        alpha = 2.0 / (period + 1.0)
        ema = np.empty_like(values)
        ema[0] = values[0]
        
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
            
        return ema
        
    def _calculate_atr_fast(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, period: int) -> np.ndarray:
        """Fast ATR calculation"""
        n = len(close)
        true_range = np.empty(n)
        
        # First true range
        true_range[0] = high[0] - low[0]
        
        # Calculate true range for remaining periods
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_range[i] = max(tr1, tr2, tr3)
            
        # Calculate ATR using EMA smoothing
        return self._ema_fast(true_range, period)
        
    def _calculate_gkyz_fast(self, open_prices: np.ndarray, high: np.ndarray,
                            low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Fast GKYZ volatility calculation"""
        gkyz_vol = np.full_like(close, np.nan)
        
        for i in range(period, len(close)):
            # Get window data
            o = open_prices[i - period + 1:i + 1]
            h = high[i - period + 1:i + 1]
            l = low[i - period + 1:i + 1]
            c = close[i - period + 1:i + 1]
            c_prev = np.concatenate([[close[i - period]], c[:-1]])
            
            # GKYZ components (simplified)
            try:
                ln_ho = np.log(h / o)
                ln_lo = np.log(l / o)
                ln_co = np.log(c / o)
                ln_cc = np.log(c / c_prev)
                
                gk = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)
                rs = ln_ho * ln_lo
                
                gkyz_daily = gk - rs + 0.5 * ln_cc**2
                gkyz_vol[i] = np.sqrt(np.mean(gkyz_daily[np.isfinite(gkyz_daily)])) * np.sqrt(252)
            except:
                gkyz_vol[i] = 0.0
                
        return gkyz_vol
        
    def _calculate_obi_fast(self, orderbook_data: pd.DataFrame, levels: int) -> pd.Series:
        """Fast Order Book Imbalance calculation"""
        try:
            bid_volume = 0
            ask_volume = 0
            
            for i in range(1, levels + 1):
                bid_col = f'bid_size_{i}'
                ask_col = f'ask_size_{i}'
                
                if bid_col in orderbook_data.columns and ask_col in orderbook_data.columns:
                    bid_volume += orderbook_data[bid_col].fillna(0)
                    ask_volume += orderbook_data[ask_col].fillna(0)
                    
            total_volume = bid_volume + ask_volume
            obi = safe_divide(bid_volume - ask_volume, total_volume)
            
            return obi
            
        except Exception:
            return pd.Series(0, index=orderbook_data.index)
            
    def _volume_imbalance_proxy(self, tick_data: pd.DataFrame) -> pd.Series:
        """Volume-based imbalance proxy when order book data not available"""
        try:
            # Use price change direction as proxy for buy/sell pressure
            price_change = tick_data['close'].diff()
            volume = tick_data['volume']
            
            # Positive price change = buy pressure, negative = sell pressure
            buy_volume = np.where(price_change > 0, volume, 0)
            sell_volume = np.where(price_change < 0, volume, 0)
            
            # Rolling imbalance
            window = 10
            buy_vol_sum = pd.Series(buy_volume).rolling(window).sum()
            sell_vol_sum = pd.Series(sell_volume).rolling(window).sum()
            
            total_vol = buy_vol_sum + sell_vol_sum
            imbalance = safe_divide(buy_vol_sum - sell_vol_sum, total_vol)
            
            return imbalance.fillna(0)
            
        except Exception:
            return pd.Series(0, index=tick_data.index)
            
    def _calculate_vpvr_breakout_fast(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Fast VPVR breakout detection"""
        breakout_signals = []
        
        for i in range(len(data)):
            start_idx = max(0, i - lookback + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < lookback // 2:
                breakout_signals.append(0)
                continue
                
            window_data = data.iloc[start_idx:end_idx]
            
            # Simple volume profile approximation
            try:
                high_vol_price = window_data.loc[window_data['volume'].idxmax(), 'close']
                current_price = data.iloc[i]['close']
                
                # Calculate price distance from high volume area
                distance_pct = abs(current_price - high_vol_price) / high_vol_price
                
                # Breakout signal if price moves significantly from high volume area
                if distance_pct > 0.02:  # 2% breakout threshold
                    breakout_signals.append(1 if current_price > high_vol_price else -1)
                else:
                    breakout_signals.append(0)
                    
            except:
                breakout_signals.append(0)
                
        return pd.Series(breakout_signals, index=data.index)
        
    def _calculate_liquidity_sweep_fast(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Fast liquidity sweep detection"""
        sweep_signals = []
        
        for i in range(len(data)):
            start_idx = max(0, i - lookback + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < 5:
                sweep_signals.append(0)
                continue
                
            window_data = data.iloc[start_idx:end_idx]
            current_bar = data.iloc[i]
            
            # Find recent highs and lows
            recent_high = window_data['high'].max()
            recent_low = window_data['low'].min()
            
            # Sweep detection (simplified)
            tolerance = 0.01  # 1 pip tolerance
            
            # High sweep: break above recent high and close below
            if (current_bar['high'] > recent_high + tolerance and 
                current_bar['close'] < recent_high):
                sweep_signals.append(1)
            # Low sweep: break below recent low and close above
            elif (current_bar['low'] < recent_low - tolerance and 
                  current_bar['close'] > recent_low):
                sweep_signals.append(-1)
            else:
                sweep_signals.append(0)
                
        return pd.Series(sweep_signals, index=data.index)
        
    def _extract_orderbook_features(self, orderbook_data: pd.DataFrame) -> pd.DataFrame:
        """Extract order book specific features"""
        try:
            # Use registered order book extractors
            ob_extractor = self.feature_engine.extractors.get('OrderBookFeatures')
            imbalance_extractor = self.feature_engine.extractors.get('ImbalanceFeatures')
            
            features = pd.DataFrame(index=orderbook_data.index)
            
            if ob_extractor:
                ob_features = ob_extractor.extract_features(orderbook_data)
                if not ob_features.empty:
                    features = pd.concat([features, ob_features], axis=1)
                    
            if imbalance_extractor:
                imbalance_features = imbalance_extractor.extract_features(orderbook_data)
                if not imbalance_features.empty:
                    features = pd.concat([features, imbalance_features], axis=1)
                    
            return features
            
        except Exception as e:
            logger.warning(f"Order book feature extraction failed: {e}")
            return pd.DataFrame()
            
    def _extract_cross_asset_features(self, tick_data: pd.DataFrame, 
                                    cross_asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract cross-asset correlation features"""
        try:
            cross_asset_extractor = self.feature_engine.extractors.get('CrossAssetFeatures')
            
            if cross_asset_extractor:
                return cross_asset_extractor.extract_features(tick_data, cross_asset_data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"Cross-asset feature extraction failed: {e}")
            return pd.DataFrame()
            
    def _update_extraction_stats(self, latency_ms: float, feature_count: int, feature_set: str) -> None:
        """Update extraction performance statistics"""
        self.extraction_stats['total_extractions'] += 1
        
        # Update rolling average latency
        alpha = 0.1  # Smoothing factor
        if self.extraction_stats['avg_latency_ms'] == 0:
            self.extraction_stats['avg_latency_ms'] = latency_ms
        else:
            self.extraction_stats['avg_latency_ms'] = (
                alpha * latency_ms + (1 - alpha) * self.extraction_stats['avg_latency_ms']
            )
            
        self.extraction_stats['feature_counts'][feature_set] = feature_count
        self.extraction_stats['last_extraction'] = datetime.now(timezone.utc)
        
        # Log performance
        perf_logger.log_latency("feature_ensemble_extraction", latency_ms)
        
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance analysis"""
        return self.feature_engine.feature_importance
        
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get feature extraction performance statistics"""
        return self.extraction_stats.copy()


class FeatureSelector:
    """
    Intelligent feature selection using multiple methods
    
    Implements SHAP-based importance ranking, statistical tests,
    and mutual information for feature selection optimization.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        self.selection_methods = {
            'shap': self._select_by_shap,
            'mutual_info': self._select_by_mutual_info,
            'f_test': self._select_by_f_test,
            'variance_threshold': self._select_by_variance,
            'correlation_threshold': self._select_by_correlation
        }
        
    def select_top_features(
        self,
        features: pd.DataFrame,
        target: pd.Series = None,
        max_features: int = 50,
        method: str = "variance_threshold"
    ) -> pd.DataFrame:
        """
        Select top features using specified method
        
        Args:
            features: Feature DataFrame
            target: Target variable (required for supervised methods)
            max_features: Maximum number of features to select
            method: Selection method to use
            
        Returns:
            DataFrame with selected features
        """
        if features.empty or len(features.columns) <= max_features:
            return features
            
        with Timer(f"feature_selection_{method}"):
            try:
                if method in self.selection_methods:
                    selected_features = self.selection_methods[method](
                        features, target, max_features
                    )
                    
                    logger.info(
                        f"Selected {len(selected_features)} features using {method} "
                        f"from {len(features.columns)} original features"
                    )
                    
                    return features[selected_features]
                else:
                    logger.warning(f"Unknown selection method: {method}")
                    return features.iloc[:, :max_features]  # Fallback
                    
            except Exception as e:
                logger.error(f"Feature selection failed: {e}")
                return features.iloc[:, :max_features]  # Fallback
                
    def _select_by_shap(self, features: pd.DataFrame, target: pd.Series, 
                       max_features: int) -> List[str]:
        """Select features by SHAP importance"""
        if target is None:
            raise ValueError("Target required for SHAP-based selection")
            
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Train simple model for SHAP analysis
            X = features.fillna(0)
            y = target.fillna(0)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # SHAP analysis
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[:1000])  # Sample for speed
            
            # Calculate mean absolute SHAP values
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Get top features
            feature_importance = pd.Series(shap_importance, index=features.columns)
            top_features = feature_importance.nlargest(max_features).index.tolist()
            
            return top_features
            
        except Exception as e:
            logger.warning(f"SHAP selection failed: {e}")
            return features.columns[:max_features].tolist()
            
    def _select_by_mutual_info(self, features: pd.DataFrame, target: pd.Series,
                              max_features: int) -> List[str]:
        """Select features by mutual information"""
        if target is None:
            raise ValueError("Target required for mutual information selection")
            
        try:
            X = features.fillna(0)
            y = target.fillna(0)
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Get top features
            feature_importance = pd.Series(mi_scores, index=features.columns)
            top_features = feature_importance.nlargest(max_features).index.tolist()
            
            return top_features
            
        except Exception as e:
            logger.warning(f"Mutual information selection failed: {e}")
            return features.columns[:max_features].tolist()
            
    def _select_by_f_test(self, features: pd.DataFrame, target: pd.Series,
                         max_features: int) -> List[str]:
        """Select features by F-test"""
        if target is None:
            raise ValueError("Target required for F-test selection")
            
        try:
            X = features.fillna(0)
            y = target.fillna(0)
            
            # F-test selection
            selector = SelectKBest(score_func=f_regression, k=max_features)
            selector.fit(X, y)
            
            selected_features = features.columns[selector.get_support()].tolist()
            return selected_features
            
        except Exception as e:
            logger.warning(f"F-test selection failed: {e}")
            return features.columns[:max_features].tolist()
            
    def _select_by_variance(self, features: pd.DataFrame, target: pd.Series,
                           max_features: int) -> List[str]:
        """Select features by variance threshold"""
        try:
            # Calculate variance for each feature
            variances = features.var()
            
            # Remove low-variance features
            variance_threshold = variances.quantile(0.1)  # Bottom 10%
            high_variance_features = variances[variances > variance_threshold]
            
            # Select top features by variance
            top_features = high_variance_features.nlargest(max_features).index.tolist()
            
            return top_features
            
        except Exception as e:
            logger.warning(f"Variance selection failed: {e}")
            return features.columns[:max_features].tolist()
            
    def _select_by_correlation(self, features: pd.DataFrame, target: pd.Series,
                              max_features: int) -> List[str]:
        """Select features by correlation with target"""
        if target is None:
            # Use auto-correlation if no target
            correlations = features.corrwith(features.iloc[:, 0]).abs()
        else:
            correlations = features.corrwith(target).abs()
            
        top_features = correlations.nlargest(max_features).index.tolist()
        return top_features


class FeaturePruner:
    """
    Feature pruning based on correlation and redundancy analysis
    
    Removes highly correlated features to reduce overfitting and
    improve model performance while maintaining predictive power.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
    def prune_correlated_features(
        self,
        features: pd.DataFrame,
        correlation_threshold: float = 0.95,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            features: Feature DataFrame
            correlation_threshold: Correlation threshold for removal
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame with pruned features
        """
        if features.empty or len(features.columns) <= 2:
            return features
            
        with Timer("feature_pruning"):
            try:
                # Calculate correlation matrix
                corr_matrix = features.corr(method=method).abs()
                
                # Find highly correlated pairs
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Identify features to drop
                to_drop = []
                for column in upper_triangle.columns:
                    correlated_features = upper_triangle.index[
                        upper_triangle[column] > correlation_threshold
                    ].tolist()
                    
                    if correlated_features:
                        # Keep the feature with higher variance
                        variances = features[correlated_features + [column]].var()
                        feature_to_keep = variances.idxmax()
                        
                        for feature in correlated_features:
                            if feature != feature_to_keep:
                                to_drop.append(feature)
                                
                # Remove duplicates
                to_drop = list(set(to_drop))
                
                # Drop correlated features
                pruned_features = features.drop(columns=to_drop, errors='ignore')
                
                logger.info(
                    f"Pruned {len(to_drop)} highly correlated features "
                    f"(threshold: {correlation_threshold})"
                )
                
                return pruned_features
                
            except Exception as e:
                logger.error(f"Feature pruning failed: {e}")
                return features
                
    def remove_constant_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove features with constant values"""
        try:
            constant_features = features.columns[features.nunique() <= 1]
            
            if len(constant_features) > 0:
                features = features.drop(columns=constant_features)
                logger.info(f"Removed {len(constant_features)} constant features")
                
            return features
            
        except Exception as e:
            logger.error(f"Constant feature removal failed: {e}")
            return features
            
    def remove_low_variance_features(
        self, 
        features: pd.DataFrame, 
        variance_threshold: float = 0.01
    ) -> pd.DataFrame:
        """Remove features with low variance"""
        try:
            variances = features.var()
            low_variance_features = variances[variances < variance_threshold].index
            
            if len(low_variance_features) > 0:
                features = features.drop(columns=low_variance_features)
                logger.info(f"Removed {len(low_variance_features)} low variance features")
                
            return features
            
        except Exception as e:
            logger.error(f"Low variance feature removal failed: {e}")
            return features