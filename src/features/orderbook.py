"""
Order Book Features - Core Implementation

5-level Order Book Imbalance (OBI) and microstructure features that form
the core predictive power of the system. Research shows order-flow consistently
outperforms traditional TA.

Optimized for sub-millisecond calculation with Numba JIT compilation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from numba import njit, prange
import logging

from .base import BaseFeatureExtractor, FeatureConfig, cached_feature
from src.core.utils import safe_divide

logger = logging.getLogger(__name__)


@njit(cache=True)
def calculate_obi_levels(bid_sizes: np.ndarray, ask_sizes: np.ndarray, levels: int) -> np.ndarray:
    """
    Calculate Order Book Imbalance for specified levels
    OBI = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume)
    """
    n = len(bid_sizes) // levels  # Number of snapshots
    obi = np.full(n, np.nan)
    
    for i in range(n):
        bid_sum = 0.0
        ask_sum = 0.0
        
        for level in range(levels):
            bid_idx = i * levels + level
            ask_idx = i * levels + level
            
            if bid_idx < len(bid_sizes) and ask_idx < len(ask_sizes):
                if not np.isnan(bid_sizes[bid_idx]) and not np.isnan(ask_sizes[ask_idx]):
                    bid_sum += bid_sizes[bid_idx]
                    ask_sum += ask_sizes[ask_idx]
        
        total_volume = bid_sum + ask_sum
        if total_volume > 0:
            obi[i] = (bid_sum - ask_sum) / total_volume
        else:
            obi[i] = 0.0
    
    return obi


@njit(cache=True)
def calculate_microprice(bid_prices: np.ndarray, ask_prices: np.ndarray,
                        bid_sizes: np.ndarray, ask_sizes: np.ndarray, levels: int) -> np.ndarray:
    """
    Calculate microprice using volume-weighted bid/ask
    Superior to mid-price for short-term predictions per research
    """
    n = len(bid_prices) // levels
    microprice = np.full(n, np.nan)
    
    for i in range(n):
        total_bid_volume = 0.0
        total_ask_volume = 0.0
        weighted_bid_price = 0.0
        weighted_ask_price = 0.0
        
        for level in range(levels):
            bid_idx = i * levels + level
            ask_idx = i * levels + level
            
            if (bid_idx < len(bid_prices) and ask_idx < len(ask_prices) and
                bid_idx < len(bid_sizes) and ask_idx < len(ask_sizes)):
                
                if (not np.isnan(bid_prices[bid_idx]) and not np.isnan(ask_prices[ask_idx]) and
                    not np.isnan(bid_sizes[bid_idx]) and not np.isnan(ask_sizes[ask_idx]) and
                    bid_sizes[bid_idx] > 0 and ask_sizes[ask_idx] > 0):
                    
                    total_bid_volume += bid_sizes[bid_idx]
                    total_ask_volume += ask_sizes[ask_idx]
                    weighted_bid_price += bid_prices[bid_idx] * bid_sizes[bid_idx]
                    weighted_ask_price += ask_prices[ask_idx] * ask_sizes[ask_idx]
        
        if total_bid_volume > 0 and total_ask_volume > 0:
            avg_bid_price = weighted_bid_price / total_bid_volume
            avg_ask_price = weighted_ask_price / total_ask_volume
            total_volume = total_bid_volume + total_ask_volume
            
            # Microprice formula
            microprice[i] = (total_bid_volume * avg_ask_price + total_ask_volume * avg_bid_price) / total_volume
    
    return microprice


@njit(cache=True)
def calculate_order_book_slope(prices: np.ndarray, sizes: np.ndarray, levels: int) -> np.ndarray:
    """Calculate order book slope (volume vs price relationship)"""
    n = len(prices) // levels
    slopes = np.full(n, np.nan)
    
    for i in range(n):
        x_sum = 0.0
        y_sum = 0.0
        xy_sum = 0.0
        x2_sum = 0.0
        count = 0
        
        for level in range(levels):
            idx = i * levels + level
            
            if (idx < len(prices) and idx < len(sizes) and
                not np.isnan(prices[idx]) and not np.isnan(sizes[idx]) and
                sizes[idx] > 0):
                
                x = prices[idx]
                y = sizes[idx]
                
                x_sum += x
                y_sum += y
                xy_sum += x * y
                x2_sum += x * x
                count += 1
        
        if count > 1:
            # Linear regression slope
            denominator = count * x2_sum - x_sum * x_sum
            if abs(denominator) > 1e-10:
                slopes[i] = (count * xy_sum - x_sum * y_sum) / denominator
            else:
                slopes[i] = 0.0
    
    return slopes


@njit(cache=True)
def calculate_weighted_mid_price(bid_prices: np.ndarray, ask_prices: np.ndarray,
                               bid_sizes: np.ndarray, ask_sizes: np.ndarray, levels: int) -> np.ndarray:
    """Calculate volume-weighted mid price"""
    n = len(bid_prices) // levels
    weighted_mid = np.full(n, np.nan)
    
    for i in range(n):
        total_volume = 0.0
        weighted_price_sum = 0.0
        
        for level in range(levels):
            bid_idx = i * levels + level
            ask_idx = i * levels + level
            
            if (bid_idx < len(bid_prices) and ask_idx < len(ask_prices) and
                bid_idx < len(bid_sizes) and ask_idx < len(ask_sizes)):
                
                if (not np.isnan(bid_prices[bid_idx]) and not np.isnan(ask_prices[ask_idx]) and
                    not np.isnan(bid_sizes[bid_idx]) and not np.isnan(ask_sizes[ask_idx]) and
                    bid_sizes[bid_idx] > 0 and ask_sizes[ask_idx] > 0):
                    
                    mid_price = (bid_prices[bid_idx] + ask_prices[ask_idx]) / 2
                    volume = bid_sizes[bid_idx] + ask_sizes[ask_idx]
                    
                    weighted_price_sum += mid_price * volume
                    total_volume += volume
        
        if total_volume > 0:
            weighted_mid[i] = weighted_price_sum / total_volume
    
    return weighted_mid


class OrderBookFeatures(BaseFeatureExtractor):
    """
    Core order book microstructure features
    
    Implements the 5-level OBI and related features that form the
    predictive core of the scalping strategy.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="OrderBookFeatures",
            category="order_flow",
            priority=1,  # Highest priority
            parameters={
                'levels': [1, 3, 5, 10],
                'imbalance_features': True,
                'microprice_features': True,
                'depth_features': True,
                'slope_features': True,
                'spread_features': True
            },
            max_latency_ms=15.0  # Critical path
        )
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        """Required order book columns"""
        required = []
        max_levels = max(self.config.parameters.get('levels', [5]))
        
        for i in range(1, max_levels + 1):
            required.extend([
                f'bid_price_{i}', f'ask_price_{i}',
                f'bid_size_{i}', f'ask_size_{i}'
            ])
        
        return required
        
    def get_min_periods(self) -> int:
        return 1
        
    def get_feature_names(self) -> List[str]:
        names = []
        levels = self.config.parameters.get('levels', [1, 3, 5, 10])
        
        # Basic price and spread features
        names.extend([
            'best_bid', 'best_ask', 'mid_price', 'spread_abs', 'spread_pct'
        ])
        
        # Order Book Imbalance (core feature)
        if self.config.parameters.get('imbalance_features', True):
            for level in levels:
                names.extend([
                    f'obi_{level}',
                    f'obi_{level}_momentum',
                    f'obi_{level}_volatility'
                ])
        
        # Microprice features (research superior to mid-price)
        if self.config.parameters.get('microprice_features', True):
            for level in levels:
                names.extend([
                    f'microprice_{level}',
                    f'microprice_vs_mid_{level}'
                ])
        
        # Depth features
        if self.config.parameters.get('depth_features', True):
            for level in levels:
                names.extend([
                    f'total_bid_volume_{level}',
                    f'total_ask_volume_{level}',
                    f'volume_ratio_{level}',
                    f'weighted_mid_price_{level}'
                ])
        
        # Slope features
        if self.config.parameters.get('slope_features', True):
            for level in levels:
                names.extend([
                    f'bid_slope_{level}',
                    f'ask_slope_{level}'
                ])
        
        # Additional spread features
        if self.config.parameters.get('spread_features', True):
            names.extend([
                'effective_spread_1',
                'effective_spread_3', 
                'effective_spread_5',
                'price_impact_bid',
                'price_impact_ask'
            ])
        
        return names
        
    @cached_feature(ttl=60)  # Short cache for order book data
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract order book features"""
        features = pd.DataFrame(index=data.index)
        
        # Basic price and spread features
        self._add_basic_features(features, data)
        
        # Order Book Imbalance features
        if self.config.parameters.get('imbalance_features', True):
            self._add_imbalance_features(features, data)
        
        # Microprice features
        if self.config.parameters.get('microprice_features', True):
            self._add_microprice_features(features, data)
        
        # Depth features
        if self.config.parameters.get('depth_features', True):
            self._add_depth_features(features, data)
        
        # Slope features
        if self.config.parameters.get('slope_features', True):
            self._add_slope_features(features, data)
        
        # Spread features
        if self.config.parameters.get('spread_features', True):
            self._add_spread_features(features, data)
        
        return features
    
    def _add_basic_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add basic price and spread features"""
        # Best bid/ask
        features['best_bid'] = data.get('bid_price_1', np.nan)
        features['best_ask'] = data.get('ask_price_1', np.nan)
        
        # Mid price
        features['mid_price'] = (features['best_bid'] + features['best_ask']) / 2
        
        # Spread
        features['spread_abs'] = features['best_ask'] - features['best_bid']
        features['spread_pct'] = (features['spread_abs'] / features['mid_price']) * 100
        
        # Add feature descriptions
        self.add_feature_description('spread_abs', 'Absolute bid-ask spread')
        self.add_feature_description('mid_price', 'Mid-price from best bid/ask')
    
    def _add_imbalance_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add Order Book Imbalance features (core research feature)"""
        levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
        
        for levels in levels_list:
            # Collect bid and ask sizes for this level
            bid_sizes = []
            ask_sizes = []
            
            for i in range(1, levels + 1):
                bid_col = f'bid_size_{i}'
                ask_col = f'ask_size_{i}'
                
                if bid_col in data.columns:
                    bid_sizes.append(data[bid_col].fillna(0).values)
                else:
                    bid_sizes.append(np.zeros(len(data)))
                    
                if ask_col in data.columns:
                    ask_sizes.append(data[ask_col].fillna(0).values)
                else:
                    ask_sizes.append(np.zeros(len(data)))
            
            # Calculate OBI
            total_bid = sum(bid_sizes)
            total_ask = sum(ask_sizes)
            total_volume = total_bid + total_ask
            
            obi = safe_divide(total_bid - total_ask, total_volume)
            features[f'obi_{levels}'] = obi
            
            # OBI momentum (5-period change)
            features[f'obi_{levels}_momentum'] = obi.diff(5)
            
            # OBI volatility (10-period rolling std)
            features[f'obi_{levels}_volatility'] = obi.rolling(10).std()
            
            # Add description for core feature
            if levels == 5:
                self.add_feature_description(
                    f'obi_{levels}',
                    'Order Book Imbalance (5-level) - Core research feature'
                )
    
    def _add_microprice_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add microprice features (superior to mid-price per research)"""
        levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
        
        for levels in levels_list:
            # Collect prices and sizes
            bid_prices = []
            ask_prices = []
            bid_sizes = []
            ask_sizes = []
            
            for i in range(1, levels + 1):
                # Prices
                bid_prices.append(data.get(f'bid_price_{i}', pd.Series(np.nan, index=data.index)).values)
                ask_prices.append(data.get(f'ask_price_{i}', pd.Series(np.nan, index=data.index)).values)
                
                # Sizes
                bid_sizes.append(data.get(f'bid_size_{i}', pd.Series(0, index=data.index)).fillna(0).values)
                ask_sizes.append(data.get(f'ask_size_{i}', pd.Series(0, index=data.index)).fillna(0).values)
            
            # Calculate microprice for each snapshot
            microprices = []
            
            for idx in range(len(data)):
                total_bid_volume = 0.0
                total_ask_volume = 0.0
                weighted_bid_price = 0.0
                weighted_ask_price = 0.0
                
                for level in range(levels):
                    bid_price = bid_prices[level][idx]
                    ask_price = ask_prices[level][idx]
                    bid_size = bid_sizes[level][idx]
                    ask_size = ask_sizes[level][idx]
                    
                    if (not pd.isna(bid_price) and not pd.isna(ask_price) and 
                        bid_size > 0 and ask_size > 0):
                        
                        total_bid_volume += bid_size
                        total_ask_volume += ask_size
                        weighted_bid_price += bid_price * bid_size
                        weighted_ask_price += ask_price * ask_size
                
                if total_bid_volume > 0 and total_ask_volume > 0:
                    avg_bid_price = weighted_bid_price / total_bid_volume
                    avg_ask_price = weighted_ask_price / total_ask_volume
                    total_volume = total_bid_volume + total_ask_volume
                    
                    # Microprice calculation
                    microprice = (total_bid_volume * avg_ask_price + total_ask_volume * avg_bid_price) / total_volume
                    microprices.append(microprice)
                else:
                    microprices.append(np.nan)
            
            features[f'microprice_{levels}'] = microprices
            
            # Microprice vs mid-price difference
            if 'mid_price' in features.columns:
                microprice_vs_mid = ((pd.Series(microprices) - features['mid_price']) / features['mid_price']) * 10000
                features[f'microprice_vs_mid_{levels}'] = microprice_vs_mid
            
            # Add description
            self.add_feature_description(
                f'microprice_{levels}',
                f'Microprice ({levels} levels) - Superior to mid-price for predictions'
            )
    
    def _add_depth_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add market depth features"""
        levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
        
        for levels in levels_list:
            # Total volumes
            total_bid_volume = 0
            total_ask_volume = 0
            
            for i in range(1, levels + 1):
                bid_col = f'bid_size_{i}'
                ask_col = f'ask_size_{i}'
                
                if bid_col in data.columns:
                    total_bid_volume += data[bid_col].fillna(0)
                if ask_col in data.columns:
                    total_ask_volume += data[ask_col].fillna(0)
            
            features[f'total_bid_volume_{levels}'] = total_bid_volume
            features[f'total_ask_volume_{levels}'] = total_ask_volume
            
            # Volume ratio
            total_volume = total_bid_volume + total_ask_volume
            features[f'volume_ratio_{levels}'] = safe_divide(total_bid_volume, total_volume)
            
            # Weighted mid price
            weighted_mid_prices = []
            
            for idx in range(len(data)):
                total_volume = 0.0
                weighted_price_sum = 0.0
                
                for level in range(1, levels + 1):
                    bid_price_col = f'bid_price_{level}'
                    ask_price_col = f'ask_price_{level}'
                    bid_size_col = f'bid_size_{level}'
                    ask_size_col = f'ask_size_{level}'
                    
                    if (bid_price_col in data.columns and ask_price_col in data.columns and
                        bid_size_col in data.columns and ask_size_col in data.columns):
                        
                        bid_price = data[bid_price_col].iloc[idx]
                        ask_price = data[ask_price_col].iloc[idx]
                        bid_size = data[bid_size_col].iloc[idx]
                        ask_size = data[ask_size_col].iloc[idx]
                        
                        if (not pd.isna(bid_price) and not pd.isna(ask_price) and
                            not pd.isna(bid_size) and not pd.isna(ask_size) and
                            bid_size > 0 and ask_size > 0):
                            
                            mid_price = (bid_price + ask_price) / 2
                            volume = bid_size + ask_size
                            
                            weighted_price_sum += mid_price * volume
                            total_volume += volume
                
                if total_volume > 0:
                    weighted_mid_prices.append(weighted_price_sum / total_volume)
                else:
                    weighted_mid_prices.append(np.nan)
            
            features[f'weighted_mid_price_{levels}'] = weighted_mid_prices
    
    def _add_slope_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add order book slope features"""
        levels_list = self.config.parameters.get('levels', [1, 3, 5, 10])
        
        for levels in levels_list:
            bid_slopes = []
            ask_slopes = []
            
            for idx in range(len(data)):
                # Bid slope calculation
                bid_prices = []
                bid_volumes = []
                
                for level in range(1, levels + 1):
                    price_col = f'bid_price_{level}'
                    size_col = f'bid_size_{level}'
                    
                    if price_col in data.columns and size_col in data.columns:
                        price = data[price_col].iloc[idx]
                        size = data[size_col].iloc[idx]
                        
                        if not pd.isna(price) and not pd.isna(size) and size > 0:
                            bid_prices.append(price)
                            bid_volumes.append(size)
                
                bid_slope = self._calculate_slope(bid_prices, bid_volumes)
                bid_slopes.append(bid_slope)
                
                # Ask slope calculation
                ask_prices = []
                ask_volumes = []
                
                for level in range(1, levels + 1):
                    price_col = f'ask_price_{level}'
                    size_col = f'ask_size_{level}'
                    
                    if price_col in data.columns and size_col in data.columns:
                        price = data[price_col].iloc[idx]
                        size = data[size_col].iloc[idx]
                        
                        if not pd.isna(price) and not pd.isna(size) and size > 0:
                            ask_prices.append(price)
                            ask_volumes.append(size)
                
                ask_slope = self._calculate_slope(ask_prices, ask_volumes)
                ask_slopes.append(ask_slope)
            
            features[f'bid_slope_{levels}'] = bid_slopes
            features[f'ask_slope_{levels}'] = ask_slopes
    
    def _add_spread_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add effective spread and market impact features"""
        # Effective spreads at different levels
        for level in [1, 3, 5]:
            bid_col = f'bid_price_{level}'
            ask_col = f'ask_price_{level}'
            
            if bid_col in data.columns and ask_col in data.columns:
                bid_price = data[bid_col]
                ask_price = data[ask_col]
                mid_price = (bid_price + ask_price) / 2
                
                effective_spread = safe_divide(ask_price - bid_price, mid_price)
                features[f'effective_spread_{level}'] = effective_spread
        
        # Price impact estimation
        if 'bid_price_5' in data.columns and 'ask_price_5' in data.columns:
            # Bid side impact (cost of selling)
            bid_price_1 = data.get('bid_price_1', np.nan)
            bid_price_5 = data.get('bid_price_5', np.nan)
            features['price_impact_bid'] = safe_divide(bid_price_1 - bid_price_5, bid_price_1)
            
            # Ask side impact (cost of buying)
            ask_price_1 = data.get('ask_price_1', np.nan)
            ask_price_5 = data.get('ask_price_5', np.nan)
            features['price_impact_ask'] = safe_divide(ask_price_5 - ask_price_1, ask_price_1)
    
    def _calculate_slope(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate slope of volume vs price relationship"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
        
        try:
            prices_array = np.array(prices)
            volumes_array = np.array(volumes)
            
            if np.std(prices_array) == 0:
                return 0.0
            
            # Simple linear regression
            correlation = np.corrcoef(prices_array, volumes_array)[0, 1]
            slope = correlation * (np.std(volumes_array) / np.std(prices_array))
            
            return slope if not np.isnan(slope) else 0.0
        except:
            return 0.0


# Export key functions
__all__ = ['OrderBookFeatures', 'calculate_obi_levels', 'calculate_microprice']