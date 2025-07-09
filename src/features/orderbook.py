"""
Order Book Features - PRIMARY FOCUS

Implementation of order flow and microstructure features based on Level 2 data.
Research synthesis shows order-flow consistently outperforms traditional TA.
These features form the core of the XAUUSD scalping edge.

Research Findings:
- Order-flow > legacy TA consistently
- Bid/ask imbalance, CVD rank above RSI/MACD
- 5-level OBI core feature in minimal set
- Microprice superior to mid-price for predictions
- Volume at price critical for support/resistance

Key Features:
- Order Book Imbalance (OBI) - 1, 3, 5, 10 levels
- Microprice calculation
- Depth analysis and slope features  
- Volume profile at price levels
- Cumulative Volume Delta (CVD)
- Trade intensity and size analysis

Time Complexity: O(k) where k is number of levels (typically 5-10)
Space Complexity: O(n) for time series output
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from numba import njit
import logging

from .base import BaseFeatureExtractor, FeatureConfig, cached_feature
from src.core.utils import safe_divide
from src.data.ingestion import OrderBookSnapshot, OrderBookLevel

logger = logging.getLogger(__name__)


class OrderBookFeatures(BaseFeatureExtractor):
    """
    Core order book microstructure features
    
    Extracts fundamental order book metrics including spreads, depths,
    and basic imbalance measures that form the foundation for
    higher-level order flow analysis.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="OrderBookFeatures",
            category="order_flow",
            priority=1,  # Highest priority - core to strategy
            parameters={
                'levels': 5,  # Number of order book levels to analyze
                'spread_features': True,
                'depth_features': True,
                'basic_imbalance': True
            },
            max_latency_ms=15.0  # Critical path - must be fast
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        """Required columns from order book data"""
        return ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1']
        
    def get_min_periods(self) -> int:
        return 1  # Can work with single snapshot
        
    def get_feature_names(self) -> List[str]:
        names = []
        levels = self.config.parameters.get('levels', 5)
        
        # Basic price features
        names.extend([
            'bid_price', 'ask_price', 'mid_price', 'spread_abs', 'spread_pct'
        ])
        
        # Depth features
        if self.config.parameters.get('depth_features', True):
            for i in range(1, levels + 1):
                names.extend([
                    f'bid_size_{i}', f'ask_size_{i}',
                    f'bid_price_{i}', f'ask_price_{i}'
                ])
                
            names.extend([
                'total_bid_volume', 'total_ask_volume',
                'bid_volume_ratio', 'ask_volume_ratio'
            ])
            
        # Basic imbalance
        if self.config.parameters.get('basic_imbalance', True):
            names.extend([
                'volume_imbalance_1', 'volume_imbalance_3', 'volume_imbalance_5'
            ])
            
        return names
        
    @cached_feature(ttl=300)  # 5 minute cache - order book changes frequently
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract order book features from L2 data"""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Basic price features
            self._compute_price_features(features, data)
            
            # Depth analysis
            if self.config.parameters.get('depth_features', True):
                self._compute_depth_features(features, data)
                
            # Basic imbalance measures
            if self.config.parameters.get('basic_imbalance', True):
                self._compute_basic_imbalance(features, data)
                
        except Exception as e:
            logger.error(f"Error computing order book features: {e}")
            
        return features
        
    def _compute_price_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute basic price and spread features"""
        try:
            # Best bid/ask prices
            features['bid_price'] = data['bid_price_1']
            features['ask_price'] = data['ask_price_1']
            
            # Mid price
            features['mid_price'] = (features['bid_price'] + features['ask_price']) / 2
            
            # Spread measures
            features['spread_abs'] = features['ask_price'] - features['bid_price']
            features['spread_pct'] = safe_divide(features['spread_abs'], features['mid_price']) * 100
            
            # Add descriptions
            self.add_feature_description('spread_abs', "Absolute bid-ask spread")
            self.add_feature_description('spread_pct', "Percentage bid-ask spread")
            self.add_feature_description('mid_price', "Mid-price (bid+ask)/2")
            
        except Exception as e:
            logger.warning(f"Failed to compute price features: {e}")
            
    def _compute_depth_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute market depth features"""
        try:
            levels = self.config.parameters.get('levels', 5)
            
            # Individual level volumes and prices
            total_bid_volume = 0
            total_ask_volume = 0
            
            for i in range(1, levels + 1):
                bid_col = f'bid_size_{i}'
                ask_col = f'ask_size_{i}'
                bid_price_col = f'bid_price_{i}'
                ask_price_col = f'ask_price_{i}'
                
                if bid_col in data.columns:
                    features[bid_col] = data[bid_col].fillna(0)
                    total_bid_volume += features[bid_col]
                    
                if ask_col in data.columns:
                    features[ask_col] = data[ask_col].fillna(0)
                    total_ask_volume += features[ask_col]
                    
                if bid_price_col in data.columns:
                    features[bid_price_col] = data[bid_price_col]
                    
                if ask_price_col in data.columns:
                    features[ask_price_col] = data[ask_price_col]
                    
            # Total volumes
            features['total_bid_volume'] = total_bid_volume
            features['total_ask_volume'] = total_ask_volume
            
            # Volume ratios
            total_volume = total_bid_volume + total_ask_volume
            features['bid_volume_ratio'] = safe_divide(total_bid_volume, total_volume)
            features['ask_volume_ratio'] = safe_divide(total_ask_volume, total_volume)
            
            self.add_feature_description('total_bid_volume', f"Total bid volume ({levels} levels)")
            self.add_feature_description('bid_volume_ratio', "Bid volume ratio vs total volume")
            
        except Exception as e:
            logger.warning(f"Failed to compute depth features: {e}")
            
    def _compute_basic_imbalance(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute basic volume imbalance measures"""
        try:
            # Volume imbalance at different levels
            for levels in [1, 3, 5]:
                bid_volume = 0
                ask_volume = 0
                
                # Sum volumes up to specified level
                for i in range(1, levels + 1):
                    bid_col = f'bid_size_{i}'
                    ask_col = f'ask_size_{i}'
                    
                    if bid_col in data.columns:
                        bid_volume += data[bid_col].fillna(0)
                    if ask_col in data.columns:
                        ask_volume += data[ask_col].fillna(0)
                        
                # Calculate imbalance
                total_volume = bid_volume + ask_volume
                imbalance = safe_divide(bid_volume - ask_volume, total_volume)
                features[f'volume_imbalance_{levels}'] = imbalance
                
                self.add_feature_description(
                    f'volume_imbalance_{levels}',
                    f"Volume imbalance ({levels} levels): (bid-ask)/(bid+ask)"
                )
                
        except Exception as e:
            logger.warning(f"Failed to compute basic imbalance: {e}")


class ImbalanceFeatures(BaseFeatureExtractor):
    """
    Advanced order book imbalance features
    
    Implements sophisticated imbalance measures including weighted imbalances,
    momentum indicators, and regime detection based on order flow patterns.
    Research shows these are among the strongest predictive features.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="ImbalanceFeatures", 
            category="order_flow",
            priority=1,  # Highest priority
            parameters={
                'levels': [1, 3, 5, 10],
                'weighted_imbalance': True,
                'imbalance_momentum': True,
                'imbalance_volatility': True,
                'rolling_windows': [5, 10, 20]
            },
            max_latency_ms=20.0
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        return ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1']
        
    def get_min_periods(self) -> int:
        return max(self.config.parameters.get('rolling_windows', [20]))
        
    def get_feature_names(self) -> List[str]:
        names = []
        levels = self.config.parameters.get('levels', [1, 3, 5, 10])
        windows = self.config.parameters.get('rolling_windows', [5, 10, 20])
        
        # OBI (Order Book Imbalance) - research emphasis
        for level in levels:
            names.append(f'obi_{level}')
            
        # Weighted imbalances
        if self.config.parameters.get('weighted_imbalance', True):
            for level in levels:
                names.extend([
                    f'obi_weighted_{level}',
                    f'obi_price_weighted_{level}',
                    f'obi_log_weighted_{level}'
                ])
                
        # Imbalance momentum
        if self.config.parameters.get('imbalance_momentum', True):
            for level in levels:
                for window in windows:
                    names.extend([
                        f'obi_{level}_momentum_{window}',
                        f'obi_{level}_acceleration_{window}'
                    ])
                    
        # Imbalance volatility and regimes
        if self.config.parameters.get('imbalance_volatility', True):
            for level in levels:
                for window in windows:
                    names.extend([
                        f'obi_{level}_volatility_{window}',
                        f'obi_{level}_regime_{window}'
                    ])
                    
        return names
        
    @cached_feature(ttl=300)
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced imbalance features"""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Core OBI calculation
            self._compute_obi_features(features, data)
            
            # Weighted imbalances
            if self.config.parameters.get('weighted_imbalance', True):
                self._compute_weighted_imbalances(features, data)
                
            # Imbalance momentum
            if self.config.parameters.get('imbalance_momentum', True):
                self._compute_imbalance_momentum(features, data)
                
            # Imbalance volatility and regimes
            if self.config.parameters.get('imbalance_volatility', True):
                self._compute_imbalance_volatility(features, data)
                
        except Exception as e:
            logger.error(f"Error computing imbalance features: {e}")
            
        return features
        
    def _compute_obi_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute Order Book Imbalance (OBI) - core research feature"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
            
            for levels in levels_list:
                bid_volume = 0
                ask_volume = 0
                
                # Aggregate volumes up to specified level
                for i in range(1, min(levels + 1, 11)):  # Cap at 10 levels
                    bid_col = f'bid_size_{i}'
                    ask_col = f'ask_size_{i}'
                    
                    if bid_col in data.columns and ask_col in data.columns:
                        bid_volume += data[bid_col].fillna(0)
                        ask_volume += data[ask_col].fillna(0)
                        
                # OBI calculation: (bid_volume - ask_volume) / (bid_volume + ask_volume)
                total_volume = bid_volume + ask_volume
                obi = safe_divide(bid_volume - ask_volume, total_volume)
                features[f'obi_{levels}'] = obi
                
                self.add_feature_description(
                    f'obi_{levels}',
                    f"Order Book Imbalance ({levels} levels) - Research core feature"
                )
                
        except Exception as e:
            logger.warning(f"Failed to compute OBI features: {e}")
            
    def _compute_weighted_imbalances(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute volume and price weighted imbalances"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
            
            for levels in levels_list:
                # Volume weighted imbalance
                weighted_bid = 0
                weighted_ask = 0
                total_weighted = 0
                
                # Price weighted imbalance
                price_weighted_bid = 0
                price_weighted_ask = 0
                
                # Log weighted imbalance (emphasizes deeper levels less)
                log_weighted_bid = 0
                log_weighted_ask = 0
                
                for i in range(1, min(levels + 1, 11)):
                    bid_size_col = f'bid_size_{i}'
                    ask_size_col = f'ask_size_{i}'
                    bid_price_col = f'bid_price_{i}'
                    ask_price_col = f'ask_price_{i}'
                    
                    if all(col in data.columns for col in [bid_size_col, ask_size_col, bid_price_col, ask_price_col]):
                        bid_size = data[bid_size_col].fillna(0)
                        ask_size = data[ask_size_col].fillna(0)
                        bid_price = data[bid_price_col].fillna(0)
                        ask_price = data[ask_price_col].fillna(0)
                        
                        # Volume weights (larger sizes get more weight)
                        volume_weight = bid_size + ask_size
                        weighted_bid += bid_size * volume_weight
                        weighted_ask += ask_size * volume_weight
                        total_weighted += volume_weight
                        
                        # Price weights (closer to mid gets more weight)
                        mid_price = (bid_price + ask_price) / 2
                        price_distance_bid = np.abs(bid_price - mid_price)
                        price_distance_ask = np.abs(ask_price - mid_price)
                        
                        price_weight_bid = 1 / (1 + price_distance_bid)
                        price_weight_ask = 1 / (1 + price_distance_ask)
                        
                        price_weighted_bid += bid_size * price_weight_bid
                        price_weighted_ask += ask_size * price_weight_ask
                        
                        # Log weights (diminishing importance by level)
                        log_weight = 1 / np.log(i + 1)
                        log_weighted_bid += bid_size * log_weight
                        log_weighted_ask += ask_size * log_weight
                        
                # Calculate weighted imbalances
                total_volume_weighted = weighted_bid + weighted_ask
                obi_weighted = safe_divide(weighted_bid - weighted_ask, total_volume_weighted)
                features[f'obi_weighted_{levels}'] = obi_weighted
                
                total_price_weighted = price_weighted_bid + price_weighted_ask
                obi_price_weighted = safe_divide(price_weighted_bid - price_weighted_ask, total_price_weighted)
                features[f'obi_price_weighted_{levels}'] = obi_price_weighted
                
                total_log_weighted = log_weighted_bid + log_weighted_ask
                obi_log_weighted = safe_divide(log_weighted_bid - log_weighted_ask, total_log_weighted)
                features[f'obi_log_weighted_{levels}'] = obi_log_weighted
                
        except Exception as e:
            logger.warning(f"Failed to compute weighted imbalances: {e}")
            
    def _compute_imbalance_momentum(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute imbalance momentum and acceleration features"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
            windows = self.config.parameters.get('rolling_windows', [5, 10, 20])
            
            for levels in levels_list:
                obi_col = f'obi_{levels}'
                
                if obi_col in features.columns:
                    obi_values = features[obi_col]
                    
                    for window in windows:
                        # Momentum (rate of change)
                        obi_momentum = obi_values.diff(window)
                        features[f'obi_{levels}_momentum_{window}'] = obi_momentum
                        
                        # Acceleration (second derivative)
                        obi_acceleration = obi_momentum.diff(window)
                        features[f'obi_{levels}_acceleration_{window}'] = obi_acceleration
                        
                        self.add_feature_description(
                            f'obi_{levels}_momentum_{window}',
                            f"OBI momentum ({levels} levels, {window} periods)"
                        )
                        
        except Exception as e:
            logger.warning(f"Failed to compute imbalance momentum: {e}")
            
    def _compute_imbalance_volatility(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute imbalance volatility and regime features"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
            windows = self.config.parameters.get('rolling_windows', [5, 10, 20])
            
            for levels in levels_list:
                obi_col = f'obi_{levels}'
                
                if obi_col in features.columns:
                    obi_values = features[obi_col]
                    
                    for window in windows:
                        # Rolling volatility of imbalance
                        obi_volatility = obi_values.rolling(window).std()
                        features[f'obi_{levels}_volatility_{window}'] = obi_volatility
                        
                        # Regime detection (high vs low imbalance periods)
                        obi_mean = obi_values.rolling(window * 2).mean()
                        obi_std = obi_values.rolling(window * 2).std()
                        
                        # Regime: 1 = high imbalance, 0 = normal, -1 = reverse imbalance
                        obi_regime = np.where(
                            obi_values > obi_mean + obi_std, 1,
                            np.where(obi_values < obi_mean - obi_std, -1, 0)
                        )
                        features[f'obi_{levels}_regime_{window}'] = obi_regime
                        
        except Exception as e:
            logger.warning(f"Failed to compute imbalance volatility: {e}")


class DepthFeatures(BaseFeatureExtractor):
    """
    Order book depth and slope analysis
    
    Analyzes the shape and characteristics of the order book to detect
    hidden liquidity, support/resistance levels, and market maker behavior.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="DepthFeatures",
            category="order_flow",
            priority=2,  # Medium-high priority
            parameters={
                'levels': 5,
                'slope_analysis': True,
                'depth_ratios': True,
                'liquidity_analysis': True,
                'depth_momentum': True
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        levels = self.config.parameters.get('levels', 5)
        cols = []
        for i in range(1, levels + 1):
            cols.extend([f'bid_price_{i}', f'ask_price_{i}', f'bid_size_{i}', f'ask_size_{i}'])
        return cols
        
    def get_min_periods(self) -> int:
        return 10
        
    def get_feature_names(self) -> List[str]:
        names = []
        
        if self.config.parameters.get('slope_analysis', True):
            names.extend([
                'bid_slope', 'ask_slope', 'bid_curve', 'ask_curve'
            ])
            
        if self.config.parameters.get('depth_ratios', True):
            names.extend([
                'depth_ratio_1_2', 'depth_ratio_1_3', 'depth_ratio_1_5',
                'bid_concentration', 'ask_concentration'
            ])
            
        if self.config.parameters.get('liquidity_analysis', True):
            names.extend([
                'effective_spread_1', 'effective_spread_2', 'effective_spread_5',
                'market_impact_bid', 'market_impact_ask'
            ])
            
        if self.config.parameters.get('depth_momentum', True):
            names.extend([
                'depth_momentum_bid', 'depth_momentum_ask',
                'depth_imbalance_change'
            ])
            
        return names
        
    @cached_feature(ttl=600)
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract order book depth features"""
        features = pd.DataFrame(index=data.index)
        
        try:
            if self.config.parameters.get('slope_analysis', True):
                self._compute_slope_features(features, data)
                
            if self.config.parameters.get('depth_ratios', True):
                self._compute_depth_ratios(features, data)
                
            if self.config.parameters.get('liquidity_analysis', True):
                self._compute_liquidity_features(features, data)
                
            if self.config.parameters.get('depth_momentum', True):
                self._compute_depth_momentum(features, data)
                
        except Exception as e:
            logger.error(f"Error computing depth features: {e}")
            
        return features
        
    def _compute_slope_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute order book slope and curvature"""
        try:
            levels = self.config.parameters.get('levels', 5)
            
            # Extract price and volume arrays
            bid_slopes = []
            ask_slopes = []
            bid_curves = []
            ask_curves = []
            
            for idx in range(len(data)):
                # Bid side analysis
                bid_prices = []
                bid_volumes = []
                ask_prices = []
                ask_volumes = []
                
                for i in range(1, levels + 1):
                    bid_price_col = f'bid_price_{i}'
                    bid_size_col = f'bid_size_{i}'
                    ask_price_col = f'ask_price_{i}'
                    ask_size_col = f'ask_size_{i}'
                    
                    if all(col in data.columns for col in [bid_price_col, bid_size_col, ask_price_col, ask_size_col]):
                        bid_price = data.iloc[idx][bid_price_col]
                        bid_volume = data.iloc[idx][bid_size_col]
                        ask_price = data.iloc[idx][ask_price_col]
                        ask_volume = data.iloc[idx][ask_size_col]
                        
                        if not pd.isna(bid_price) and not pd.isna(bid_volume) and bid_volume > 0:
                            bid_prices.append(bid_price)
                            bid_volumes.append(bid_volume)
                            
                        if not pd.isna(ask_price) and not pd.isna(ask_volume) and ask_volume > 0:
                            ask_prices.append(ask_price)
                            ask_volumes.append(ask_volume)
                            
                # Calculate slopes (volume vs price relationship)
                bid_slope = self._calculate_slope(bid_prices, bid_volumes)
                ask_slope = self._calculate_slope(ask_prices, ask_volumes)
                
                bid_slopes.append(bid_slope)
                ask_slopes.append(ask_slope)
                
                # Calculate curvature (second derivative)
                bid_curve = self._calculate_curvature(bid_prices, bid_volumes)
                ask_curve = self._calculate_curvature(ask_prices, ask_volumes)
                
                bid_curves.append(bid_curve)
                ask_curves.append(ask_curve)
                
            features['bid_slope'] = bid_slopes
            features['ask_slope'] = ask_slopes
            features['bid_curve'] = bid_curves
            features['ask_curve'] = ask_curves
            
            self.add_feature_description('bid_slope', "Bid side order book slope (volume vs price)")
            self.add_feature_description('ask_slope', "Ask side order book slope (volume vs price)")
            
        except Exception as e:
            logger.warning(f"Failed to compute slope features: {e}")
            
    def _calculate_slope(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate slope of volume vs price relationship"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
            
        try:
            # Simple linear regression slope
            prices_array = np.array(prices)
            volumes_array = np.array(volumes)
            
            if np.std(prices_array) == 0:
                return 0.0
                
            correlation = np.corrcoef(prices_array, volumes_array)[0, 1]
            slope = correlation * (np.std(volumes_array) / np.std(prices_array))
            
            return slope if not np.isnan(slope) else 0.0
            
        except:
            return 0.0
            
    def _calculate_curvature(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate curvature (second derivative) of order book"""
        if len(prices) < 3 or len(volumes) < 3:
            return 0.0
            
        try:
            # Numerical second derivative
            volumes_array = np.array(volumes)
            prices_array = np.array(prices)
            
            # Calculate first derivatives
            first_deriv = np.gradient(volumes_array, prices_array)
            
            if len(first_deriv) < 2:
                return 0.0
                
            # Calculate second derivative
            second_deriv = np.gradient(first_deriv, prices_array)
            
            # Return average curvature
            return np.mean(second_deriv) if len(second_deriv) > 0 else 0.0
            
        except:
            return 0.0
            
    def _compute_depth_ratios(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute ratios between different depth levels"""
        try:
            # Depth ratios (level 1 vs deeper levels)
            if all(col in data.columns for col in ['bid_size_1', 'bid_size_2']):
                features['depth_ratio_1_2'] = safe_divide(data['bid_size_1'], data['bid_size_2'])
                
            if all(col in data.columns for col in ['bid_size_1', 'bid_size_3']):
                features['depth_ratio_1_3'] = safe_divide(data['bid_size_1'], data['bid_size_3'])
                
            if all(col in data.columns for col in ['bid_size_1', 'bid_size_5']):
                features['depth_ratio_1_5'] = safe_divide(data['bid_size_1'], data['bid_size_5'])
                
            # Concentration measures (how much liquidity is at top vs spread out)
            bid_concentration = self._calculate_concentration(data, 'bid')
            ask_concentration = self._calculate_concentration(data, 'ask')
            
            features['bid_concentration'] = bid_concentration
            features['ask_concentration'] = ask_concentration
            
        except Exception as e:
            logger.warning(f"Failed to compute depth ratios: {e}")
            
    def _calculate_concentration(self, data: pd.DataFrame, side: str) -> pd.Series:
        """Calculate liquidity concentration (Gini coefficient style)"""
        levels = self.config.parameters.get('levels', 5)
        concentration = []
        
        for idx in range(len(data)):
            volumes = []
            
            for i in range(1, levels + 1):
                col = f'{side}_size_{i}'
                if col in data.columns:
                    volume = data.iloc[idx][col]
                    if not pd.isna(volume) and volume > 0:
                        volumes.append(volume)
                        
            if len(volumes) < 2:
                concentration.append(0.5)  # Neutral concentration
                continue
                
            # Calculate concentration using normalized standard deviation
            volumes_array = np.array(volumes)
            total_volume = np.sum(volumes_array)
            
            if total_volume == 0:
                concentration.append(0.5)
                continue
                
            # Normalized volumes
            norm_volumes = volumes_array / total_volume
            
            # Concentration measure (0 = evenly distributed, 1 = concentrated in first level)
            concentration_value = np.sum(np.cumsum(norm_volumes) ** 2) - 0.5
            concentration.append(max(0, min(1, concentration_value)))
            
        return pd.Series(concentration, index=data.index)
        
    def _compute_liquidity_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute effective spreads and market impact measures"""
        try:
            # Effective spreads for different volumes
            for volume_level in [1, 2, 5]:
                eff_spread = self._calculate_effective_spread(data, volume_level)
                features[f'effective_spread_{volume_level}'] = eff_spread
                
            # Market impact estimation
            market_impact_bid = self._calculate_market_impact(data, 'bid')
            market_impact_ask = self._calculate_market_impact(data, 'ask')
            
            features['market_impact_bid'] = market_impact_bid
            features['market_impact_ask'] = market_impact_ask
            
        except Exception as e:
            logger.warning(f"Failed to compute liquidity features: {e}")
            
    def _calculate_effective_spread(self, data: pd.DataFrame, volume_level: int) -> pd.Series:
        """Calculate effective spread for given volume level"""
        effective_spreads = []
        
        for idx in range(len(data)):
            bid_col = f'bid_price_{volume_level}'
            ask_col = f'ask_price_{volume_level}'
            
            if bid_col in data.columns and ask_col in data.columns:
                bid_price = data.iloc[idx][bid_col]
                ask_price = data.iloc[idx][ask_col]
                
                if not pd.isna(bid_price) and not pd.isna(ask_price):
                    mid_price = (bid_price + ask_price) / 2
                    eff_spread = (ask_price - bid_price) / mid_price if mid_price > 0 else 0
                    effective_spreads.append(eff_spread)
                else:
                    effective_spreads.append(np.nan)
            else:
                effective_spreads.append(np.nan)
                
        return pd.Series(effective_spreads, index=data.index)
        
    def _calculate_market_impact(self, data: pd.DataFrame, side: str) -> pd.Series:
        """Calculate estimated market impact for side"""
        impacts = []
        levels = self.config.parameters.get('levels', 5)
        
        for idx in range(len(data)):
            cumulative_volume = 0
            price_impact = 0
            
            reference_price = data.iloc[idx].get(f'{side}_price_1', np.nan)
            
            if pd.isna(reference_price):
                impacts.append(0)
                continue
                
            for i in range(1, levels + 1):
                price_col = f'{side}_price_{i}'
                size_col = f'{side}_size_{i}'
                
                if price_col in data.columns and size_col in data.columns:
                    price = data.iloc[idx][price_col]
                    size = data.iloc[idx][size_col]
                    
                    if not pd.isna(price) and not pd.isna(size) and size > 0:
                        cumulative_volume += size
                        
                        # Price impact (difference from reference price)
                        if side == 'bid':
                            impact = (reference_price - price) / reference_price
                        else:
                            impact = (price - reference_price) / reference_price
                            
                        price_impact = impact
                        
            impacts.append(price_impact)
            
        return pd.Series(impacts, index=data.index)
        
    def _compute_depth_momentum(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute momentum features for order book depth"""
        try:
            # Calculate depth momentum (change in total depth)
            if 'total_bid_volume' in data.columns and 'total_ask_volume' in data.columns:
                bid_momentum = data['total_bid_volume'].diff(5)  # 5-period change
                ask_momentum = data['total_ask_volume'].diff(5)
                
                features['depth_momentum_bid'] = bid_momentum
                features['depth_momentum_ask'] = ask_momentum
                
                # Depth imbalance change
                current_imbalance = safe_divide(
                    data['total_bid_volume'] - data['total_ask_volume'],
                    data['total_bid_volume'] + data['total_ask_volume']
                )
                
                imbalance_change = current_imbalance.diff(5)
                features['depth_imbalance_change'] = imbalance_change
                
        except Exception as e:
            logger.warning(f"Failed to compute depth momentum: {e}")


class MicropriceFatures(BaseFeatureExtractor):
    """
    Microprice and weighted price features
    
    Implements microprice calculation and related features that provide
    superior price discovery compared to simple mid-price. Research shows
    microprice outperforms mid-price for short-term predictions.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="MicropriceFatures",
            category="order_flow",
            priority=1,  # High priority - superior to mid-price
            parameters={
                'levels': [1, 3, 5],
                'microprice_variants': True,
                'weighted_prices': True,
                'price_momentum': True
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        max_level = max(self.config.parameters.get('levels', [1]))
        cols = []
        for i in range(1, max_level + 1):
            cols.extend([f'bid_price_{i}', f'ask_price_{i}', f'bid_size_{i}', f'ask_size_{i}'])
        return cols
        
    def get_min_periods(self) -> int:
        return 5
        
    def get_feature_names(self) -> List[str]:
        names = []
        levels = self.config.parameters.get('levels', [1, 3, 5])
        
        # Microprice features
        for level in levels:
            names.extend([
                f'microprice_{level}',
                f'microprice_vs_mid_{level}',
                f'microprice_trend_{level}'
            ])
            
        if self.config.parameters.get('microprice_variants', True):
            for level in levels:
                names.extend([
                    f'microprice_log_{level}',
                    f'microprice_linear_{level}',
                    f'microprice_size_weighted_{level}'
                ])
                
        if self.config.parameters.get('weighted_prices', True):
            for level in levels:
                names.extend([
                    f'vwap_bid_{level}',
                    f'vwap_ask_{level}',
                    f'size_weighted_mid_{level}'
                ])
                
        if self.config.parameters.get('price_momentum', True):
            for level in levels:
                names.extend([
                    f'microprice_momentum_{level}',
                    f'microprice_acceleration_{level}'
                ])
                
        return names
        
    @cached_feature(ttl=300)
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract microprice and weighted price features"""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Core microprice calculation
            self._compute_microprice_features(features, data)
            
            # Microprice variants
            if self.config.parameters.get('microprice_variants', True):
                self._compute_microprice_variants(features, data)
                
            # Volume weighted prices
            if self.config.parameters.get('weighted_prices', True):
                self._compute_weighted_prices(features, data)
                
            # Price momentum
            if self.config.parameters.get('price_momentum', True):
                self._compute_price_momentum(features, data)
                
        except Exception as e:
            logger.error(f"Error computing microprice features: {e}")
            
        return features
        
    def _compute_microprice_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute core microprice features"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5])
            
            for levels in levels_list:
                # Calculate microprice
                microprice = self._calculate_microprice(data, levels)
                features[f'microprice_{levels}'] = microprice
                
                # Microprice vs mid-price difference
                if 'bid_price_1' in data.columns and 'ask_price_1' in data.columns:
                    mid_price = (data['bid_price_1'] + data['ask_price_1']) / 2
                    microprice_vs_mid = safe_divide(microprice - mid_price, mid_price) * 10000  # In basis points
                    features[f'microprice_vs_mid_{levels}'] = microprice_vs_mid
                    
                # Microprice trend
                microprice_trend = microprice.diff(3)  # 3-period change
                features[f'microprice_trend_{levels}'] = microprice_trend
                
                self.add_feature_description(
                    f'microprice_{levels}',
                    f"Microprice ({levels} levels) - Superior to mid-price for predictions"
                )
                
        except Exception as e:
            logger.warning(f"Failed to compute microprice features: {e}")
            
    def _calculate_microprice(self, data: pd.DataFrame, levels: int) -> pd.Series:
        """Calculate microprice using volume-weighted bid/ask"""
        microprices = []
        
        for idx in range(len(data)):
            total_bid_volume = 0
            total_ask_volume = 0
            weighted_bid_price = 0
            weighted_ask_price = 0
            
            # Aggregate across levels
            for i in range(1, levels + 1):
                bid_price_col = f'bid_price_{i}'
                ask_price_col = f'ask_price_{i}'
                bid_size_col = f'bid_size_{i}'
                ask_size_col = f'ask_size_{i}'
                
                if all(col in data.columns for col in [bid_price_col, ask_price_col, bid_size_col, ask_size_col]):
                    bid_price = data.iloc[idx][bid_price_col]
                    ask_price = data.iloc[idx][ask_price_col]
                    bid_size = data.iloc[idx][bid_size_col]
                    ask_size = data.iloc[idx][ask_size_col]
                    
                    if not any(pd.isna([bid_price, ask_price, bid_size, ask_size])) and bid_size > 0 and ask_size > 0:
                        total_bid_volume += bid_size
                        total_ask_volume += ask_size
                        weighted_bid_price += bid_price * bid_size
                        weighted_ask_price += ask_price * ask_size
                        
            # Calculate microprice
            if total_bid_volume > 0 and total_ask_volume > 0:
                avg_bid_price = weighted_bid_price / total_bid_volume
                avg_ask_price = weighted_ask_price / total_ask_volume
                
                # Microprice formula: (bid_volume * ask_price + ask_volume * bid_price) / (bid_volume + ask_volume)
                total_volume = total_bid_volume + total_ask_volume
                microprice = (total_bid_volume * avg_ask_price + total_ask_volume * avg_bid_price) / total_volume
                microprices.append(microprice)
            else:
                microprices.append(np.nan)
                
        return pd.Series(microprices, index=data.index)
        
    def _compute_microprice_variants(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute different microprice calculation variants"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5])
            
            for levels in levels_list:
                # Log-weighted microprice (emphasizes price levels logarithmically)
                log_microprice = self._calculate_log_microprice(data, levels)
                features[f'microprice_log_{levels}'] = log_microprice
                
                # Linear-weighted microprice (weights by distance from mid)
                linear_microprice = self._calculate_linear_microprice(data, levels)
                features[f'microprice_linear_{levels}'] = linear_microprice
                
                # Size-weighted microprice (emphasizes larger orders)
                size_weighted_microprice = self._calculate_size_weighted_microprice(data, levels)
                features[f'microprice_size_weighted_{levels}'] = size_weighted_microprice
                
        except Exception as e:
            logger.warning(f"Failed to compute microprice variants: {e}")
            
    def _calculate_log_microprice(self, data: pd.DataFrame, levels: int) -> pd.Series:
        """Calculate log-weighted microprice"""
        log_microprices = []
        
        for idx in range(len(data)):
            weighted_price_sum = 0
            weight_sum = 0
            
            for i in range(1, levels + 1):
                # Log weighting: deeper levels get exponentially less weight
                log_weight = 1 / np.log(i + 1)
                
                bid_price_col = f'bid_price_{i}'
                ask_price_col = f'ask_price_{i}'
                bid_size_col = f'bid_size_{i}'
                ask_size_col = f'ask_size_{i}'
                
                if all(col in data.columns for col in [bid_price_col, ask_price_col, bid_size_col, ask_size_col]):
                    bid_price = data.iloc[idx][bid_price_col]
                    ask_price = data.iloc[idx][ask_price_col]
                    bid_size = data.iloc[idx][bid_size_col]
                    ask_size = data.iloc[idx][ask_size_col]
                    
                    if not any(pd.isna([bid_price, ask_price, bid_size, ask_size])):
                        mid_price = (bid_price + ask_price) / 2
                        volume_weight = bid_size + ask_size
                        combined_weight = log_weight * volume_weight
                        
                        weighted_price_sum += mid_price * combined_weight
                        weight_sum += combined_weight
                        
            if weight_sum > 0:
                log_microprices.append(weighted_price_sum / weight_sum)
            else:
                log_microprices.append(np.nan)
                
        return pd.Series(log_microprices, index=data.index)
        
    def _calculate_linear_microprice(self, data: pd.DataFrame, levels: int) -> pd.Series:
        """Calculate linear distance-weighted microprice"""
        linear_microprices = []
        
        for idx in range(len(data)):
            if 'bid_price_1' not in data.columns or 'ask_price_1' not in data.columns:
                linear_microprices.append(np.nan)
                continue
                
            reference_mid = (data.iloc[idx]['bid_price_1'] + data.iloc[idx]['ask_price_1']) / 2
            
            weighted_price_sum = 0
            weight_sum = 0
            
            for i in range(1, levels + 1):
                bid_price_col = f'bid_price_{i}'
                ask_price_col = f'ask_price_{i}'
                bid_size_col = f'bid_size_{i}'
                ask_size_col = f'ask_size_{i}'
                
                if all(col in data.columns for col in [bid_price_col, ask_price_col, bid_size_col, ask_size_col]):
                    bid_price = data.iloc[idx][bid_price_col]
                    ask_price = data.iloc[idx][ask_price_col]
                    bid_size = data.iloc[idx][bid_size_col]
                    ask_size = data.iloc[idx][ask_size_col]
                    
                    if not any(pd.isna([bid_price, ask_price, bid_size, ask_size])):
                        mid_price = (bid_price + ask_price) / 2
                        
                        # Linear weight: inverse of distance from reference mid
                        distance = abs(mid_price - reference_mid)
                        linear_weight = 1 / (1 + distance) if distance > 0 else 1
                        
                        volume_weight = bid_size + ask_size
                        combined_weight = linear_weight * volume_weight
                        
                        weighted_price_sum += mid_price * combined_weight
                        weight_sum += combined_weight
                        
            if weight_sum > 0:
                linear_microprices.append(weighted_price_sum / weight_sum)
            else:
                linear_microprices.append(np.nan)
                
        return pd.Series(linear_microprices, index=data.index)
        
    def _calculate_size_weighted_microprice(self, data: pd.DataFrame, levels: int) -> pd.Series:
        """Calculate size-weighted microprice (emphasizes larger orders)"""
        size_weighted_microprices = []
        
        for idx in range(len(data)):
            weighted_price_sum = 0
            weight_sum = 0
            
            for i in range(1, levels + 1):
                bid_price_col = f'bid_price_{i}'
                ask_price_col = f'ask_price_{i}'
                bid_size_col = f'bid_size_{i}'
                ask_size_col = f'ask_size_{i}'
                
                if all(col in data.columns for col in [bid_price_col, ask_price_col, bid_size_col, ask_size_col]):
                    bid_price = data.iloc[idx][bid_price_col]
                    ask_price = data.iloc[idx][ask_price_col]
                    bid_size = data.iloc[idx][bid_size_col]
                    ask_size = data.iloc[idx][ask_size_col]
                    
                    if not any(pd.isna([bid_price, ask_price, bid_size, ask_size])):
                        mid_price = (bid_price + ask_price) / 2
                        volume = bid_size + ask_size
                        
                        # Square root weighting for size (reduces impact of very large orders)
                        size_weight = np.sqrt(volume)
                        
                        weighted_price_sum += mid_price * size_weight
                        weight_sum += size_weight
                        
            if weight_sum > 0:
                size_weighted_microprices.append(weighted_price_sum / weight_sum)
            else:
                size_weighted_microprices.append(np.nan)
                
        return pd.Series(size_weighted_microprices, index=data.index)
        
    def _compute_weighted_prices(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute volume-weighted average prices for bid/ask sides"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5])
            
            for levels in levels_list:
                # VWAP for bid side
                vwap_bid = self._calculate_vwap_side(data, levels, 'bid')
                features[f'vwap_bid_{levels}'] = vwap_bid
                
                # VWAP for ask side
                vwap_ask = self._calculate_vwap_side(data, levels, 'ask')
                features[f'vwap_ask_{levels}'] = vwap_ask
                
                # Size-weighted mid price
                size_weighted_mid = (vwap_bid + vwap_ask) / 2
                features[f'size_weighted_mid_{levels}'] = size_weighted_mid
                
        except Exception as e:
            logger.warning(f"Failed to compute weighted prices: {e}")
            
    def _calculate_vwap_side(self, data: pd.DataFrame, levels: int, side: str) -> pd.Series:
        """Calculate VWAP for one side of the book"""
        vwaps = []
        
        for idx in range(len(data)):
            weighted_price_sum = 0
            volume_sum = 0
            
            for i in range(1, levels + 1):
                price_col = f'{side}_price_{i}'
                size_col = f'{side}_size_{i}'
                
                if price_col in data.columns and size_col in data.columns:
                    price = data.iloc[idx][price_col]
                    size = data.iloc[idx][size_col]
                    
                    if not pd.isna(price) and not pd.isna(size) and size > 0:
                        weighted_price_sum += price * size
                        volume_sum += size
                        
            if volume_sum > 0:
                vwaps.append(weighted_price_sum / volume_sum)
            else:
                vwaps.append(np.nan)
                
        return pd.Series(vwaps, index=data.index)
        
    def _compute_price_momentum(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute momentum features for microprice"""
        try:
            levels_list = self.config.parameters.get('levels', [1, 3, 5])
            
            for levels in levels_list:
                microprice_col = f'microprice_{levels}'
                
                if microprice_col in features.columns:
                    microprice_values = features[microprice_col]
                    
                    # Momentum (5-period change)
                    microprice_momentum = microprice_values.diff(5)
                    features[f'microprice_momentum_{levels}'] = microprice_momentum
                    
                    # Acceleration (momentum change)
                    microprice_acceleration = microprice_momentum.diff(3)
                    features[f'microprice_acceleration_{levels}'] = microprice_acceleration
                    
        except Exception as e:
            logger.warning(f"Failed to compute price momentum: {e}")