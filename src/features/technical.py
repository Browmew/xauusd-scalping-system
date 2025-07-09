"""
Technical Analysis Features

Implementation of technical indicators based on research synthesis.
Focus on "survivors" - indicators that provide value after costs and in
modern market conditions. Traditional oscillators are de-emphasized
in favor of trend-following and volatility measures.

Research Findings:
- Williams %R outperforms peers (81% win-rate in studies)
- HMA-5 provides trend clarity with low lag
- ATR essential for position sizing and stops
- Most retail indicators ≈ noise after costs
- EMAs better than SMAs for momentum
- Bollinger Bands useful for mean reversion

Categories:
- Moving Averages: EMA, HMA, TEMA (trend following)
- Momentum: Williams %R, RSI (selected oscillators)
- Volatility: ATR, Bollinger Bands, GKYZ estimators
- Trend: MACD, Slope measures
- Volume: VWAP, Volume Rate of Change

Time Complexity: O(n) for most indicators
Space Complexity: O(n) for output series
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Optional, Any, Tuple
from numba import njit
import logging

from .base import BaseFeatureExtractor, FeatureConfig, cached_feature
from src.core.utils import fast_ema, fast_sma, fast_rsi, fast_atr, safe_divide

logger = logging.getLogger(__name__)


class MovingAverageFeatures(BaseFeatureExtractor):
    """
    Moving average based features
    
    Implements various moving averages with focus on trend detection
    and momentum analysis. Emphasizes faster, adaptive averages over
    simple moving averages based on research findings.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="MovingAverageFeatures",
            category="trend_momentum",
            priority=2,  # Medium priority
            parameters={
                'ema_periods': [3, 5, 8, 13, 21, 34, 55],
                'sma_periods': [5, 10, 20, 50],
                'hma_periods': [5, 10, 20],  # Hull MA - low lag
                'tema_periods': [8, 21],     # Triple EMA
                'crossover_periods': [(5, 13), (8, 21), (13, 34)]
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        """Required input columns"""
        return ['close', 'high', 'low', 'volume']
        
    def get_min_periods(self) -> int:
        """Minimum periods required"""
        return max(self.config.parameters.get('ema_periods', [21]))
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces"""
        names = []
        
        # EMA features
        for period in self.config.parameters.get('ema_periods', []):
            names.extend([
                f'ema_{period}',
                f'ema_{period}_slope',
                f'price_above_ema_{period}'
            ])
            
        # SMA features  
        for period in self.config.parameters.get('sma_periods', []):
            names.extend([
                f'sma_{period}',
                f'ema_sma_ratio_{period}'
            ])
            
        # HMA features (Hull Moving Average - research favorite)
        for period in self.config.parameters.get('hma_periods', []):
            names.extend([
                f'hma_{period}',
                f'hma_{period}_slope',
                f'hma_{period}_trend'
            ])
            
        # TEMA features
        for period in self.config.parameters.get('tema_periods', []):
            names.append(f'tema_{period}')
            
        # Crossover features
        for fast, slow in self.config.parameters.get('crossover_periods', []):
            names.extend([
                f'ema_cross_{fast}_{slow}',
                f'ema_cross_momentum_{fast}_{slow}'
            ])
            
        # VWAP features
        names.extend([
            'vwap',
            'vwap_distance',
            'vwap_slope'
        ])
        
        return names
        
    @cached_feature(ttl=1800)  # 30 minute cache
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract moving average features"""
        features = pd.DataFrame(index=data.index)
        
        # Extract price arrays for performance
        close = data['close'].values
        high = data['high'].values  
        low = data['low'].values
        volume = data['volume'].values
        
        try:
            # EMA features
            self._compute_ema_features(features, close, data.index)
            
            # SMA features
            self._compute_sma_features(features, close, data.index)
            
            # HMA features (research emphasis)
            self._compute_hma_features(features, close, high, low, data.index)
            
            # TEMA features
            self._compute_tema_features(features, close, data.index)
            
            # Crossover features
            self._compute_crossover_features(features, close, data.index)
            
            # VWAP features
            self._compute_vwap_features(features, close, high, low, volume, data.index)
            
        except Exception as e:
            logger.error(f"Error computing moving average features: {e}")
            
        return features
        
    def _compute_ema_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute EMA-based features"""
        for period in self.config.parameters.get('ema_periods', []):
            try:
                # EMA calculation
                ema = fast_ema(close, period)
                col_name = f'ema_{period}'
                features[col_name] = ema
                
                # EMA slope (trend strength)
                ema_slope = np.gradient(ema)
                features[f'ema_{period}_slope'] = ema_slope
                
                # Price position relative to EMA
                features[f'price_above_ema_{period}'] = (close > ema).astype(int)
                
                # Add descriptions
                self.add_feature_description(col_name, f"Exponential Moving Average ({period} periods)")
                self.add_feature_description(f'ema_{period}_slope', f"EMA {period} slope (trend strength)")
                
            except Exception as e:
                logger.warning(f"Failed to compute EMA {period}: {e}")
                
    def _compute_sma_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute SMA-based features"""
        for period in self.config.parameters.get('sma_periods', []):
            try:
                # SMA calculation
                sma = fast_sma(close, period)
                features[f'sma_{period}'] = sma
                
                # EMA/SMA ratio (momentum vs trend)
                if f'ema_{period}' in features.columns:
                    ema_sma_ratio = safe_divide(features[f'ema_{period}'], sma)
                    features[f'ema_sma_ratio_{period}'] = ema_sma_ratio
                    
                self.add_feature_description(f'sma_{period}', f"Simple Moving Average ({period} periods)")
                
            except Exception as e:
                logger.warning(f"Failed to compute SMA {period}: {e}")
                
    def _compute_hma_features(self, features: pd.DataFrame, close: np.ndarray, 
                            high: np.ndarray, low: np.ndarray, index: pd.Index) -> None:
        """Compute Hull Moving Average features (research favorite for low lag)"""
        for period in self.config.parameters.get('hma_periods', []):
            try:
                # Hull Moving Average calculation
                hma = self._calculate_hma(close, period)
                col_name = f'hma_{period}'
                features[col_name] = hma
                
                # HMA slope
                hma_slope = np.gradient(hma)
                features[f'hma_{period}_slope'] = hma_slope
                
                # HMA trend classification
                hma_trend = np.where(hma_slope > 0.01, 1, 
                                   np.where(hma_slope < -0.01, -1, 0))
                features[f'hma_{period}_trend'] = hma_trend
                
                self.add_feature_description(col_name, f"Hull Moving Average ({period} periods) - Low lag trend indicator")
                
            except Exception as e:
                logger.warning(f"Failed to compute HMA {period}: {e}")
                
    def _calculate_hma(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Hull Moving Average"""
        try:
            # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            # Weighted moving averages
            wma_half = self._weighted_ma(close, half_period)
            wma_full = self._weighted_ma(close, period)
            
            # Hull calculation
            hull_raw = 2 * wma_half - wma_full
            hma = self._weighted_ma(hull_raw, sqrt_period)
            
            return hma
            
        except Exception:
            # Fallback to EMA if HMA calculation fails
            return fast_ema(close, period)
            
    def _weighted_ma(self, values: np.ndarray, period: int) -> np.ndarray:
        """Calculate weighted moving average"""
        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()
        
        result = np.full_like(values, np.nan)
        
        for i in range(period - 1, len(values)):
            window = values[i - period + 1:i + 1]
            result[i] = np.dot(window, weights)
            
        return result
        
    def _compute_tema_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute Triple Exponential Moving Average features"""
        for period in self.config.parameters.get('tema_periods', []):
            try:
                # TEMA calculation: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
                ema1 = fast_ema(close, period)
                ema2 = fast_ema(ema1, period)
                ema3 = fast_ema(ema2, period)
                
                tema = 3 * ema1 - 3 * ema2 + ema3
                features[f'tema_{period}'] = tema
                
                self.add_feature_description(f'tema_{period}', f"Triple EMA ({period} periods) - Reduced lag")
                
            except Exception as e:
                logger.warning(f"Failed to compute TEMA {period}: {e}")
                
    def _compute_crossover_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute EMA crossover features"""
        for fast_period, slow_period in self.config.parameters.get('crossover_periods', []):
            try:
                if f'ema_{fast_period}' in features.columns and f'ema_{slow_period}' in features.columns:
                    fast_ema = features[f'ema_{fast_period}'].values
                    slow_ema = features[f'ema_{slow_period}'].values
                    
                    # Crossover signal
                    crossover = np.where(fast_ema > slow_ema, 1, -1)
                    features[f'ema_cross_{fast_period}_{slow_period}'] = crossover
                    
                    # Crossover momentum (distance between EMAs)
                    cross_momentum = safe_divide(fast_ema - slow_ema, slow_ema) * 100
                    features[f'ema_cross_momentum_{fast_period}_{slow_period}'] = cross_momentum
                    
                    self.add_feature_description(
                        f'ema_cross_{fast_period}_{slow_period}',
                        f"EMA crossover signal ({fast_period}/{slow_period})"
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to compute crossover {fast_period}/{slow_period}: {e}")
                
    def _compute_vwap_features(self, features: pd.DataFrame, close: np.ndarray, 
                             high: np.ndarray, low: np.ndarray, volume: np.ndarray, index: pd.Index) -> None:
        """Compute VWAP (Volume Weighted Average Price) features"""
        try:
            # Typical price
            typical_price = (high + low + close) / 3
            
            # Cumulative volume and price*volume
            cum_volume = np.cumsum(volume)
            cum_price_volume = np.cumsum(typical_price * volume)
            
            # VWAP calculation
            vwap = safe_divide(cum_price_volume, cum_volume)
            features['vwap'] = vwap
            
            # Distance from VWAP (as percentage)
            vwap_distance = safe_divide(close - vwap, vwap) * 100
            features['vwap_distance'] = vwap_distance
            
            # VWAP slope
            vwap_slope = np.gradient(vwap)
            features['vwap_slope'] = vwap_slope
            
            self.add_feature_description('vwap', "Volume Weighted Average Price")
            self.add_feature_description('vwap_distance', "Distance from VWAP (%)")
            
        except Exception as e:
            logger.warning(f"Failed to compute VWAP features: {e}")


class MomentumFeatures(BaseFeatureExtractor):
    """
    Momentum oscillator features
    
    Focus on momentum indicators that provide value according to research.
    Williams %R is emphasized due to superior performance in studies.
    Traditional oscillators are included but with lower priority.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="MomentumFeatures",
            category="trend_momentum", 
            priority=3,  # Lower priority than trend features
            parameters={
                'williams_r_periods': [14, 21],  # Research favorite
                'rsi_periods': [7, 14, 21],
                'stoch_params': [(14, 3, 3)],
                'macd_params': [(12, 26, 9)],
                'roc_periods': [10, 20],
                'overbought_levels': {'rsi': 70, 'williams_r': -20, 'stoch': 80},
                'oversold_levels': {'rsi': 30, 'williams_r': -80, 'stoch': 20}
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        return ['close', 'high', 'low', 'volume']
        
    def get_min_periods(self) -> int:
        return 26  # For MACD
        
    def get_feature_names(self) -> List[str]:
        names = []
        
        # Williams %R (research favorite)
        for period in self.config.parameters.get('williams_r_periods', []):
            names.extend([
                f'williams_r_{period}',
                f'williams_r_{period}_overbought',
                f'williams_r_{period}_oversold',
                f'williams_r_{period}_momentum'
            ])
            
        # RSI features
        for period in self.config.parameters.get('rsi_periods', []):
            names.extend([
                f'rsi_{period}',
                f'rsi_{period}_overbought',
                f'rsi_{period}_oversold',
                f'rsi_{period}_divergence'
            ])
            
        # Stochastic features
        for k, d, smooth in self.config.parameters.get('stoch_params', []):
            names.extend([
                f'stoch_k_{k}',
                f'stoch_d_{d}',
                f'stoch_cross_{k}_{d}'
            ])
            
        # MACD features
        for fast, slow, signal in self.config.parameters.get('macd_params', []):
            names.extend([
                f'macd_{fast}_{slow}',
                f'macd_signal_{fast}_{slow}_{signal}',
                f'macd_histogram_{fast}_{slow}_{signal}',
                f'macd_cross_{fast}_{slow}_{signal}'
            ])
            
        # Rate of Change
        for period in self.config.parameters.get('roc_periods', []):
            names.append(f'roc_{period}')
            
        return names
        
    @cached_feature(ttl=1800)
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum features"""
        features = pd.DataFrame(index=data.index)
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        try:
            # Williams %R (research emphasis)
            self._compute_williams_r_features(features, close, high, low, data.index)
            
            # RSI features
            self._compute_rsi_features(features, close, data.index)
            
            # Stochastic features
            self._compute_stochastic_features(features, close, high, low, data.index)
            
            # MACD features
            self._compute_macd_features(features, close, data.index)
            
            # Rate of Change features
            self._compute_roc_features(features, close, data.index)
            
        except Exception as e:
            logger.error(f"Error computing momentum features: {e}")
            
        return features
        
    def _compute_williams_r_features(self, features: pd.DataFrame, close: np.ndarray,
                                   high: np.ndarray, low: np.ndarray, index: pd.Index) -> None:
        """Compute Williams %R features (research favorite - 81% win rate)"""
        for period in self.config.parameters.get('williams_r_periods', []):
            try:
                # Williams %R calculation
                williams_r = self._calculate_williams_r(close, high, low, period)
                col_name = f'williams_r_{period}'
                features[col_name] = williams_r
                
                # Overbought/oversold levels
                overbought_level = self.config.parameters['overbought_levels']['williams_r']
                oversold_level = self.config.parameters['oversold_levels']['williams_r']
                
                features[f'williams_r_{period}_overbought'] = (williams_r > overbought_level).astype(int)
                features[f'williams_r_{period}_oversold'] = (williams_r < oversold_level).astype(int)
                
                # Williams %R momentum (rate of change)
                williams_r_momentum = np.diff(williams_r, prepend=williams_r[0])
                features[f'williams_r_{period}_momentum'] = williams_r_momentum
                
                self.add_feature_description(
                    col_name, 
                    f"Williams %R ({period} periods) - Research shows 81% win rate"
                )
                
            except Exception as e:
                logger.warning(f"Failed to compute Williams %R {period}: {e}")
                
    def _calculate_williams_r(self, close: np.ndarray, high: np.ndarray, 
                            low: np.ndarray, period: int) -> np.ndarray:
        """Calculate Williams %R"""
        williams_r = np.full_like(close, np.nan)
        
        for i in range(period - 1, len(close)):
            period_high = np.max(high[i - period + 1:i + 1])
            period_low = np.min(low[i - period + 1:i + 1])
            
            if period_high != period_low:
                williams_r[i] = ((period_high - close[i]) / (period_high - period_low)) * -100
            else:
                williams_r[i] = -50  # Neutral when no range
                
        return williams_r
        
    def _compute_rsi_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute RSI features"""
        for period in self.config.parameters.get('rsi_periods', []):
            try:
                # RSI calculation
                rsi = fast_rsi(close, period)
                col_name = f'rsi_{period}'
                features[col_name] = rsi
                
                # Overbought/oversold levels
                overbought_level = self.config.parameters['overbought_levels']['rsi']
                oversold_level = self.config.parameters['oversold_levels']['rsi']
                
                features[f'rsi_{period}_overbought'] = (rsi > overbought_level).astype(int)
                features[f'rsi_{period}_oversold'] = (rsi < oversold_level).astype(int)
                
                # RSI divergence (simplified)
                price_momentum = np.gradient(close)
                rsi_momentum = np.gradient(rsi)
                rsi_divergence = np.where(
                    (price_momentum > 0) & (rsi_momentum < 0), 1,
                    np.where((price_momentum < 0) & (rsi_momentum > 0), -1, 0)
                )
                features[f'rsi_{period}_divergence'] = rsi_divergence
                
                self.add_feature_description(col_name, f"RSI ({period} periods)")
                
            except Exception as e:
                logger.warning(f"Failed to compute RSI {period}: {e}")
                
    def _compute_stochastic_features(self, features: pd.DataFrame, close: np.ndarray,
                                   high: np.ndarray, low: np.ndarray, index: pd.Index) -> None:
        """Compute Stochastic oscillator features"""
        for k_period, d_period, smooth in self.config.parameters.get('stoch_params', []):
            try:
                # Stochastic %K calculation
                stoch_k = self._calculate_stoch_k(close, high, low, k_period)
                features[f'stoch_k_{k_period}'] = stoch_k
                
                # Stochastic %D (moving average of %K)
                stoch_d = fast_sma(stoch_k, d_period)
                features[f'stoch_d_{d_period}'] = stoch_d
                
                # Stochastic crossover
                stoch_cross = np.where(stoch_k > stoch_d, 1, -1)
                features[f'stoch_cross_{k_period}_{d_period}'] = stoch_cross
                
                self.add_feature_description(f'stoch_k_{k_period}', f"Stochastic %K ({k_period} periods)")
                
            except Exception as e:
                logger.warning(f"Failed to compute Stochastic {k_period}: {e}")
                
    def _calculate_stoch_k(self, close: np.ndarray, high: np.ndarray, 
                          low: np.ndarray, period: int) -> np.ndarray:
        """Calculate Stochastic %K"""
        stoch_k = np.full_like(close, np.nan)
        
        for i in range(period - 1, len(close)):
            period_high = np.max(high[i - period + 1:i + 1])
            period_low = np.min(low[i - period + 1:i + 1])
            
            if period_high != period_low:
                stoch_k[i] = ((close[i] - period_low) / (period_high - period_low)) * 100
            else:
                stoch_k[i] = 50  # Neutral when no range
                
        return stoch_k
        
    def _compute_macd_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute MACD features"""
        for fast, slow, signal in self.config.parameters.get('macd_params', []):
            try:
                # MACD calculation
                ema_fast = fast_ema(close, fast)
                ema_slow = fast_ema(close, slow)
                macd_line = ema_fast - ema_slow
                features[f'macd_{fast}_{slow}'] = macd_line
                
                # Signal line
                signal_line = fast_ema(macd_line, signal)
                features[f'macd_signal_{fast}_{slow}_{signal}'] = signal_line
                
                # MACD histogram
                macd_histogram = macd_line - signal_line
                features[f'macd_histogram_{fast}_{slow}_{signal}'] = macd_histogram
                
                # MACD crossover
                macd_cross = np.where(macd_line > signal_line, 1, -1)
                features[f'macd_cross_{fast}_{slow}_{signal}'] = macd_cross
                
                self.add_feature_description(f'macd_{fast}_{slow}', f"MACD ({fast}/{slow})")
                
            except Exception as e:
                logger.warning(f"Failed to compute MACD {fast}/{slow}/{signal}: {e}")
                
    def _compute_roc_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute Rate of Change features"""
        for period in self.config.parameters.get('roc_periods', []):
            try:
                # Rate of Change calculation
                roc = np.full_like(close, np.nan)
                for i in range(period, len(close)):
                    if close[i - period] != 0:
                        roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
                        
                features[f'roc_{period}'] = roc
                
                self.add_feature_description(f'roc_{period}', f"Rate of Change ({period} periods)")
                
            except Exception as e:
                logger.warning(f"Failed to compute ROC {period}: {e}")


class VolatilityFeatures(BaseFeatureExtractor):
    """
    Volatility and range-based features
    
    Implements volatility estimators essential for risk management and
    position sizing. Emphasizes GKYZ volatility (research favorite) and
    ATR for practical trading applications.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="VolatilityFeatures",
            category="volatility",
            priority=1,  # High priority - essential for risk management
            parameters={
                'atr_periods': [7, 14, 21],  # Essential for stops/targets
                'bb_params': [(20, 2.0)],    # Bollinger Bands
                'gkyz_periods': [20, 60],    # GKYZ volatility (research favorite)
                'parkinson_periods': [20, 60],
                'rogers_satchell_periods': [20, 60],
                'realized_vol_periods': [20, 60]
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
        
    def get_min_periods(self) -> int:
        return 60  # For longer volatility estimates
        
    def get_feature_names(self) -> List[str]:
        names = []
        
        # ATR features (essential)
        for period in self.config.parameters.get('atr_periods', []):
            names.extend([
                f'atr_{period}',
                f'atr_{period}_pct',
                f'atr_{period}_normalized'
            ])
            
        # Bollinger Bands
        for period, std_dev in self.config.parameters.get('bb_params', []):
            names.extend([
                f'bb_upper_{period}_{std_dev}',
                f'bb_lower_{period}_{std_dev}',
                f'bb_percent_b_{period}_{std_dev}',
                f'bb_bandwidth_{period}_{std_dev}',
                f'bb_squeeze_{period}_{std_dev}'
            ])
            
        # GKYZ volatility (research favorite)
        for period in self.config.parameters.get('gkyz_periods', []):
            names.extend([
                f'gkyz_vol_{period}',
                f'gkyz_vol_{period}_regime'
            ])
            
        # Other volatility estimators
        for period in self.config.parameters.get('parkinson_periods', []):
            names.append(f'parkinson_vol_{period}')
            
        for period in self.config.parameters.get('rogers_satchell_periods', []):
            names.append(f'rogers_satchell_vol_{period}')
            
        for period in self.config.parameters.get('realized_vol_periods', []):
            names.append(f'realized_vol_{period}')
            
        # Additional range features
        names.extend([
            'true_range',
            'high_low_ratio',
            'close_range_position'
        ])
        
        return names
        
    @cached_feature(ttl=1800)
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility features"""
        features = pd.DataFrame(index=data.index)
        
        open_prices = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        try:
            # ATR features (essential for risk management)
            self._compute_atr_features(features, high, low, close, data.index)
            
            # Bollinger Bands
            self._compute_bollinger_features(features, close, data.index)
            
            # GKYZ volatility (research favorite)
            self._compute_gkyz_features(features, open_prices, high, low, close, data.index)
            
            # Other volatility estimators
            self._compute_other_volatility_features(features, open_prices, high, low, close, data.index)
            
            # Basic range features
            self._compute_range_features(features, high, low, close, data.index)
            
        except Exception as e:
            logger.error(f"Error computing volatility features: {e}")
            
        return features
        
    def _compute_atr_features(self, features: pd.DataFrame, high: np.ndarray,
                            low: np.ndarray, close: np.ndarray, index: pd.Index) -> None:
        """Compute ATR features (essential for position sizing)"""
        for period in self.config.parameters.get('atr_periods', []):
            try:
                # ATR calculation
                atr = fast_atr(high, low, close, period)
                col_name = f'atr_{period}'
                features[col_name] = atr
                
                # ATR as percentage of price
                atr_pct = safe_divide(atr, close) * 100
                features[f'atr_{period}_pct'] = atr_pct
                
                # Normalized ATR (vs 60-period average)
                atr_60 = fast_atr(high, low, close, 60)
                atr_normalized = safe_divide(atr, atr_60)
                features[f'atr_{period}_normalized'] = atr_normalized
                
                self.add_feature_description(
                    col_name,
                    f"Average True Range ({period} periods) - Essential for risk management"
                )
                
            except Exception as e:
                logger.warning(f"Failed to compute ATR {period}: {e}")
                
    def _compute_bollinger_features(self, features: pd.DataFrame, close: np.ndarray, index: pd.Index) -> None:
        """Compute Bollinger Bands features"""
        for period, std_dev in self.config.parameters.get('bb_params', []):
            try:
                # Moving average and standard deviation
                sma = fast_sma(close, period)
                rolling_std = self._rolling_std(close, period)
                
                # Bollinger Bands
                bb_upper = sma + (std_dev * rolling_std)
                bb_lower = sma - (std_dev * rolling_std)
                
                features[f'bb_upper_{period}_{std_dev}'] = bb_upper
                features[f'bb_lower_{period}_{std_dev}'] = bb_lower
                
                # %B (position within bands)
                bb_percent_b = safe_divide(close - bb_lower, bb_upper - bb_lower)
                features[f'bb_percent_b_{period}_{std_dev}'] = bb_percent_b
                
                # Bandwidth
                bb_bandwidth = safe_divide(bb_upper - bb_lower, sma) * 100
                features[f'bb_bandwidth_{period}_{std_dev}'] = bb_bandwidth
                
                # Squeeze detection (low volatility)
                bb_squeeze = (bb_bandwidth < np.percentile(bb_bandwidth[np.isfinite(bb_bandwidth)], 20)).astype(int)
                features[f'bb_squeeze_{period}_{std_dev}'] = bb_squeeze
                
                self.add_feature_description(
                    f'bb_percent_b_{period}_{std_dev}',
                    f"Bollinger %B ({period} periods) - Position within bands"
                )
                
            except Exception as e:
                logger.warning(f"Failed to compute Bollinger Bands {period}/{std_dev}: {e}")
                
    def _rolling_std(self, values: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        std_values = np.full_like(values, np.nan)
        
        for i in range(period - 1, len(values)):
            window = values[i - period + 1:i + 1]
            std_values[i] = np.std(window, ddof=1)
            
        return std_values
        
    def _compute_gkyz_features(self, features: pd.DataFrame, open_prices: np.ndarray,
                             high: np.ndarray, low: np.ndarray, close: np.ndarray, index: pd.Index) -> None:
        """Compute GKYZ volatility (research favorite for regime detection)"""
        for period in self.config.parameters.get('gkyz_periods', []):
            try:
                # GKYZ volatility calculation
                gkyz_vol = self._calculate_gkyz_volatility(open_prices, high, low, close, period)
                col_name = f'gkyz_vol_{period}'
                features[col_name] = gkyz_vol
                
                # Volatility regime (high/low relative to historical)
                vol_percentile = self._rolling_percentile(gkyz_vol, period * 2, 75)
                vol_regime = (gkyz_vol > vol_percentile).astype(int)
                features[f'gkyz_vol_{period}_regime'] = vol_regime
                
                self.add_feature_description(
                    col_name,
                    f"GKYZ Volatility ({period} periods) - Research favorite for regime detection"
                )
                
            except Exception as e:
                logger.warning(f"Failed to compute GKYZ volatility {period}: {e}")
                
    def _calculate_gkyz_volatility(self, open_prices: np.ndarray, high: np.ndarray,
                                 low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Garman-Klass-Yang-Zhang volatility estimator"""
        gkyz_vol = np.full_like(close, np.nan)
        
        for i in range(period, len(close)):
            # Get window data
            o = open_prices[i - period + 1:i + 1]
            h = high[i - period + 1:i + 1]
            l = low[i - period + 1:i + 1]
            c = close[i - period + 1:i + 1]
            c_prev = np.concatenate([[close[i - period]], c[:-1]])
            
            # GKYZ components
            ln_ho = np.log(h / o)
            ln_lo = np.log(l / o)
            ln_co = np.log(c / o)
            ln_cc = np.log(c / c_prev)
            
            # GKYZ volatility
            gk = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)
            rs = ln_ho * ln_lo
            
            gkyz_daily = gk - rs + 0.5 * ln_cc**2
            gkyz_vol[i] = np.sqrt(np.mean(gkyz_daily)) * np.sqrt(252)  # Annualized
            
        return gkyz_vol
        
    def _rolling_percentile(self, values: np.ndarray, period: int, percentile: float) -> np.ndarray:
        """Calculate rolling percentile"""
        result = np.full_like(values, np.nan)
        
        for i in range(period - 1, len(values)):
            window = values[i - period + 1:i + 1]
            finite_window = window[np.isfinite(window)]
            if len(finite_window) > 0:
                result[i] = np.percentile(finite_window, percentile)
                
        return result
        
    def _compute_other_volatility_features(self, features: pd.DataFrame, open_prices: np.ndarray,
                                         high: np.ndarray, low: np.ndarray, close: np.ndarray, index: pd.Index) -> None:
        """Compute other volatility estimators"""
        # Parkinson volatility
        for period in self.config.parameters.get('parkinson_periods', []):
            try:
                parkinson_vol = self._calculate_parkinson_volatility(high, low, period)
                features[f'parkinson_vol_{period}'] = parkinson_vol
                
                self.add_feature_description(f'parkinson_vol_{period}', f"Parkinson volatility ({period} periods)")
                
            except Exception as e:
                logger.warning(f"Failed to compute Parkinson volatility {period}: {e}")
                
        # Rogers-Satchell volatility
        for period in self.config.parameters.get('rogers_satchell_periods', []):
            try:
                rs_vol = self._calculate_rogers_satchell_volatility(open_prices, high, low, close, period)
                features[f'rogers_satchell_vol_{period}'] = rs_vol
                
            except Exception as e:
                logger.warning(f"Failed to compute Rogers-Satchell volatility {period}: {e}")
                
        # Realized volatility
        for period in self.config.parameters.get('realized_vol_periods', []):
            try:
                realized_vol = self._calculate_realized_volatility(close, period)
                features[f'realized_vol_{period}'] = realized_vol
                
            except Exception as e:
                logger.warning(f"Failed to compute realized volatility {period}: {e}")
                
    def _calculate_parkinson_volatility(self, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """Calculate Parkinson volatility estimator"""
        parkinson_vol = np.full_like(high, np.nan)
        
        for i in range(period - 1, len(high)):
            h = high[i - period + 1:i + 1]
            l = low[i - period + 1:i + 1]
            
            ln_hl = np.log(h / l)
            parkinson_daily = 0.25 * ln_hl**2
            parkinson_vol[i] = np.sqrt(np.mean(parkinson_daily) * 252)  # Annualized
            
        return parkinson_vol
        
    def _calculate_rogers_satchell_volatility(self, open_prices: np.ndarray, high: np.ndarray,
                                            low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Rogers-Satchell volatility estimator"""
        rs_vol = np.full_like(close, np.nan)
        
        for i in range(period - 1, len(close)):
            o = open_prices[i - period + 1:i + 1]
            h = high[i - period + 1:i + 1]
            l = low[i - period + 1:i + 1]
            c = close[i - period + 1:i + 1]
            
            ln_ho = np.log(h / o)
            ln_co = np.log(c / o)
            ln_lo = np.log(l / o)
            
            rs_daily = ln_ho * ln_co + ln_lo * ln_co
            rs_vol[i] = np.sqrt(np.mean(rs_daily) * 252)  # Annualized
            
        return rs_vol
        
    def _calculate_realized_volatility(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate realized volatility"""
        realized_vol = np.full_like(close, np.nan)
        
        # Calculate returns
        returns = np.diff(np.log(close))
        
        for i in range(period, len(close)):
            window_returns = returns[i - period:i]
            realized_vol[i] = np.sqrt(np.sum(window_returns**2) * 252)  # Annualized
            
        return realized_vol
        
    def _compute_range_features(self, features: pd.DataFrame, high: np.ndarray,
                              low: np.ndarray, close: np.ndarray, index: pd.Index) -> None:
        """Compute basic range features"""
        try:
            # True Range
            true_range = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            true_range[0] = high[0] - low[0]  # First period
            features['true_range'] = true_range
            
            # High/Low ratio
            high_low_ratio = safe_divide(high, low)
            features['high_low_ratio'] = high_low_ratio
            
            # Close position within range
            close_range_position = safe_divide(close - low, high - low)
            features['close_range_position'] = close_range_position
            
            self.add_feature_description('true_range', "True Range - Maximum of daily ranges")
            self.add_feature_description('close_range_position', "Close position within daily range (0-1)")
            
        except Exception as e:
            logger.warning(f"Failed to compute range features: {e}")


class TechnicalIndicators:
    """
    Consolidated technical indicators class
    
    Provides unified interface to all technical analysis features
    with emphasis on research-validated indicators.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Initialize feature extractors
        self.moving_average_features = MovingAverageFeatures()
        self.momentum_features = MomentumFeatures()
        self.volatility_features = VolatilityFeatures()
        
        # Register with feature engine
        self.extractors = {
            'moving_averages': self.moving_average_features,
            'momentum': self.momentum_features, 
            'volatility': self.volatility_features
        }
        
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all technical analysis features"""
        all_features = pd.DataFrame(index=data.index)
        
        for name, extractor in self.extractors.items():
            try:
                features = extractor.compute_with_performance_tracking(data)
                if not features.empty:
                    all_features = pd.concat([all_features, features], axis=1)
            except Exception as e:
                logger.error(f"Failed to extract {name} features: {e}")
                
        return all_features
        
    def get_minimal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract minimal feature set for low-latency operation"""
        minimal_features = pd.DataFrame(index=data.index)
        
        # Focus on research-validated features
        try:
            # HMA-5 (research favorite)
            hma_5 = self.moving_average_features._calculate_hma(data['close'].values, 5)
            minimal_features['hma_5'] = hma_5
            
            # ATR-14 (essential for risk)
            atr_14 = fast_atr(data['high'].values, data['low'].values, data['close'].values, 14)
            minimal_features['atr_14'] = atr_14
            
            # Williams %R (research winner)
            williams_r = self.momentum_features._calculate_williams_r(
                data['close'].values, data['high'].values, data['low'].values, 14
            )
            minimal_features['williams_r_14'] = williams_r
            
        except Exception as e:
            logger.error(f"Failed to extract minimal features: {e}")
            
        return minimal_features