"""
Microstructure Features

Advanced market microstructure analysis including volume profile, smart money concepts,
and cross-asset correlations. These features capture institutional trading patterns,
liquidity dynamics, and regime changes critical for scalping success.

Key Components:
- Volume Profile Visible Range (VPVR) - Research core feature
- Point of Control (POC) analysis
- Smart Money Concepts (SMC) - liquidity sweeps, order blocks
- Cross-asset correlations (DXY, TIPS, SPY)
- Fair Value Gaps and imbalances
- Session-based regime detection

Performance Focus:
- Real-time computation optimized for < 25ms latency
- Memory efficient algorithms for high-frequency data
- Vectorized operations where possible

Time Complexity: O(n log n) for sorting operations, O(n) for most features
Space Complexity: O(n) with efficient windowing for memory management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import logging
from collections import defaultdict
from numba import njit

from .base import BaseFeatureExtractor, FeatureConfig, cached_feature
from src.core.utils import safe_divide, get_market_session, pips_to_price

logger = logging.getLogger(__name__)


class VolumeProfileFeatures(BaseFeatureExtractor):
    """
    Volume Profile Visible Range (VPVR) and Point of Control analysis
    
    Implements volume profile calculations that are central to the research
    findings. VPVR breakout detection is included in the minimal feature set
    due to its predictive power for scalping operations.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="VolumeProfileFeatures",
            category="market_structure",
            priority=1,  # High priority - in minimal set
            parameters={
                'lookback_bars': [50, 100, 200],
                'price_bins': 50,
                'volume_area': 0.7,  # 70% value area
                'breakout_threshold': 2.0,  # Standard deviations for breakout
                'poc_distance_levels': [0.5, 1.0, 2.0],  # Distance thresholds in pips
                'session_vpvr': True  # Calculate session-based VPVR
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
        
    def get_min_periods(self) -> int:
        return max(self.config.parameters.get('lookback_bars', [100]))
        
    def get_feature_names(self) -> List[str]:
        names = []
        lookbacks = self.config.parameters.get('lookback_bars', [50, 100, 200])
        poc_levels = self.config.parameters.get('poc_distance_levels', [0.5, 1.0, 2.0])
        
        # Core VPVR features
        for lookback in lookbacks:
            names.extend([
                f'vpvr_poc_{lookback}',
                f'vpvr_value_area_high_{lookback}',
                f'vpvr_value_area_low_{lookback}',
                f'vpvr_volume_at_poc_{lookback}',
                f'vpvr_breakout_{lookback}'  # Research core feature
            ])
            
        # POC distance features
        for level in poc_levels:
            names.extend([
                f'poc_distance_{level}',
                f'price_near_poc_{level}'
            ])
            
        # Volume distribution features
        for lookback in lookbacks:
            names.extend([
                f'volume_distribution_skew_{lookback}',
                f'volume_distribution_kurt_{lookback}',
                f'high_volume_node_count_{lookback}'
            ])
            
        # Session-based VPVR
        if self.config.parameters.get('session_vpvr', True):
            names.extend([
                'session_poc',
                'session_value_area_high',
                'session_value_area_low',
                'session_poc_distance'
            ])
            
        return names
        
    @cached_feature(ttl=900)  # 15 minute cache
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume profile features"""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Core VPVR calculations
            self._compute_vpvr_features(features, data)
            
            # POC distance analysis
            self._compute_poc_distance_features(features, data)
            
            # Volume distribution analysis
            self._compute_volume_distribution_features(features, data)
            
            # Session-based VPVR
            if self.config.parameters.get('session_vpvr', True):
                self._compute_session_vpvr_features(features, data)
                
        except Exception as e:
            logger.error(f"Error computing volume profile features: {e}")
            
        return features
        
    def _compute_vpvr_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute Volume Profile Visible Range features"""
        try:
            lookbacks = self.config.parameters.get('lookback_bars', [50, 100, 200])
            price_bins = self.config.parameters.get('price_bins', 50)
            volume_area = self.config.parameters.get('volume_area', 0.7)
            
            for lookback in lookbacks:
                poc_values = []
                vah_values = []
                val_values = []
                poc_volumes = []
                breakout_signals = []
                
                for i in range(len(data)):
                    start_idx = max(0, i - lookback + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx < lookback // 2:
                        # Not enough data
                        poc_values.append(np.nan)
                        vah_values.append(np.nan)
                        val_values.append(np.nan)
                        poc_volumes.append(np.nan)
                        breakout_signals.append(0)
                        continue
                        
                    # Extract window data
                    window_data = data.iloc[start_idx:end_idx]
                    
                    # Calculate volume profile
                    vp_result = self._calculate_volume_profile(
                        window_data, price_bins, volume_area
                    )
                    
                    poc_values.append(vp_result['poc'])
                    vah_values.append(vp_result['value_area_high'])
                    val_values.append(vp_result['value_area_low'])
                    poc_volumes.append(vp_result['poc_volume'])
                    
                    # Breakout detection
                    current_price = data.iloc[i]['close']
                    breakout = self._detect_vpvr_breakout(
                        current_price, vp_result, data.iloc[i]
                    )
                    breakout_signals.append(breakout)
                    
                # Store features
                features[f'vpvr_poc_{lookback}'] = poc_values
                features[f'vpvr_value_area_high_{lookback}'] = vah_values
                features[f'vpvr_value_area_low_{lookback}'] = val_values
                features[f'vpvr_volume_at_poc_{lookback}'] = poc_volumes
                features[f'vpvr_breakout_{lookback}'] = breakout_signals
                
                # Add descriptions
                self.add_feature_description(
                    f'vpvr_breakout_{lookback}',
                    f"VPVR breakout signal ({lookback} bars) - Research core feature"
                )
                
        except Exception as e:
            logger.warning(f"Failed to compute VPVR features: {e}")
            
    def _calculate_volume_profile(self, data: pd.DataFrame, bins: int, value_area: float) -> Dict[str, float]:
        """Calculate volume profile for given data window"""
        if data.empty:
            return {
                'poc': np.nan,
                'value_area_high': np.nan,
                'value_area_low': np.nan,
                'poc_volume': np.nan
            }
            
        try:
            # Price range
            min_price = data['low'].min()
            max_price = data['high'].max()
            
            if min_price >= max_price:
                return {
                    'poc': data['close'].iloc[-1],
                    'value_area_high': data['close'].iloc[-1],
                    'value_area_low': data['close'].iloc[-1],
                    'poc_volume': 0
                }
                
            # Create price bins
            price_step = (max_price - min_price) / bins
            price_levels = np.linspace(min_price, max_price, bins + 1)
            
            # Calculate volume at each price level
            volume_at_price = np.zeros(bins)
            
            for idx, row in data.iterrows():
                # Distribute volume across price range for each bar
                bar_volume = row['volume']
                bar_high = row['high']
                bar_low = row['low']
                
                # Find bins that intersect with this bar's range
                low_bin = max(0, int((bar_low - min_price) / price_step))
                high_bin = min(bins - 1, int((bar_high - min_price) / price_step))
                
                if low_bin <= high_bin:
                    # Distribute volume proportionally across intersecting bins
                    bin_count = high_bin - low_bin + 1
                    volume_per_bin = bar_volume / bin_count
                    
                    for bin_idx in range(low_bin, high_bin + 1):
                        volume_at_price[bin_idx] += volume_per_bin
                        
            # Find Point of Control (POC) - price level with highest volume
            poc_bin = np.argmax(volume_at_price)
            poc_price = price_levels[poc_bin] + price_step / 2
            poc_volume = volume_at_price[poc_bin]
            
            # Calculate Value Area (70% of volume around POC)
            total_volume = np.sum(volume_at_price)
            target_volume = total_volume * value_area
            
            # Expand around POC until we capture target volume
            value_area_volume = volume_at_price[poc_bin]
            low_idx = poc_bin
            high_idx = poc_bin
            
            while value_area_volume < target_volume and (low_idx > 0 or high_idx < bins - 1):
                # Expand to the side with more volume
                low_volume = volume_at_price[low_idx - 1] if low_idx > 0 else 0
                high_volume = volume_at_price[high_idx + 1] if high_idx < bins - 1 else 0
                
                if low_volume >= high_volume and low_idx > 0:
                    low_idx -= 1
                    value_area_volume += volume_at_price[low_idx]
                elif high_idx < bins - 1:
                    high_idx += 1
                    value_area_volume += volume_at_price[high_idx]
                else:
                    break
                    
            value_area_high = price_levels[high_idx + 1]
            value_area_low = price_levels[low_idx]
            
            return {
                'poc': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'poc_volume': poc_volume
            }
            
        except Exception as e:
            logger.warning(f"Volume profile calculation failed: {e}")
            return {
                'poc': data['close'].iloc[-1] if not data.empty else np.nan,
                'value_area_high': np.nan,
                'value_area_low': np.nan,
                'poc_volume': 0
            }
            
    def _detect_vpvr_breakout(self, current_price: float, vp_result: Dict[str, float], 
                            current_bar: pd.Series) -> int:
        """Detect VPVR breakout signals"""
        try:
            threshold = self.config.parameters.get('breakout_threshold', 2.0)
            
            # No breakout if invalid data
            if any(pd.isna([current_price, vp_result['value_area_high'], vp_result['value_area_low']])):
                return 0
                
            vah = vp_result['value_area_high']
            val = vp_result['value_area_low']
            
            # Calculate breakout distance in standard deviations
            value_area_range = vah - val
            
            if value_area_range <= 0:
                return 0
                
            # Upside breakout
            if current_price > vah:
                breakout_distance = (current_price - vah) / value_area_range
                if breakout_distance >= threshold:
                    return 1
                    
            # Downside breakout
            elif current_price < val:
                breakout_distance = (val - current_price) / value_area_range
                if breakout_distance >= threshold:
                    return -1
                    
            return 0
            
        except Exception:
            return 0
            
    def _compute_poc_distance_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute Point of Control distance features"""
        try:
            poc_levels = self.config.parameters.get('poc_distance_levels', [0.5, 1.0, 2.0])
            
            # Use the primary POC (100-bar lookback)
            poc_col = 'vpvr_poc_100'
            if poc_col not in features.columns:
                return
                
            current_prices = data['close']
            poc_prices = features[poc_col]
            
            for level in poc_levels:
                level_pips = pips_to_price(level, 'XAUUSD')
                
                # Distance from POC in pips
                poc_distance = (current_prices - poc_prices) / 0.01  # Convert to pips
                features[f'poc_distance_{level}'] = poc_distance
                
                # Binary feature: price near POC
                price_near_poc = (np.abs(current_prices - poc_prices) <= level_pips).astype(int)
                features[f'price_near_poc_{level}'] = price_near_poc
                
        except Exception as e:
            logger.warning(f"Failed to compute POC distance features: {e}")
            
    def _compute_volume_distribution_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute volume distribution characteristics"""
        try:
            lookbacks = self.config.parameters.get('lookback_bars', [50, 100, 200])
            
            for lookback in lookbacks:
                skew_values = []
                kurt_values = []
                node_counts = []
                
                for i in range(len(data)):
                    start_idx = max(0, i - lookback + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx < lookback // 2:
                        skew_values.append(np.nan)
                        kurt_values.append(np.nan)
                        node_counts.append(np.nan)
                        continue
                        
                    window_data = data.iloc[start_idx:end_idx]
                    
                    # Calculate volume distribution statistics
                    volumes = window_data['volume'].values
                    
                    if len(volumes) > 10:
                        # Skewness and kurtosis of volume distribution
                        volume_skew = stats.skew(volumes)
                        volume_kurt = stats.kurtosis(volumes)
                        
                        # High volume node count (above 75th percentile)
                        volume_threshold = np.percentile(volumes, 75)
                        high_volume_nodes = np.sum(volumes > volume_threshold)
                        
                        skew_values.append(volume_skew)
                        kurt_values.append(volume_kurt)
                        node_counts.append(high_volume_nodes)
                    else:
                        skew_values.append(np.nan)
                        kurt_values.append(np.nan)
                        node_counts.append(np.nan)
                        
                features[f'volume_distribution_skew_{lookback}'] = skew_values
                features[f'volume_distribution_kurt_{lookback}'] = kurt_values
                features[f'high_volume_node_count_{lookback}'] = node_counts
                
        except Exception as e:
            logger.warning(f"Failed to compute volume distribution features: {e}")
            
    def _compute_session_vpvr_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute session-based volume profile features"""
        try:
            if 'timestamp' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                logger.warning("Cannot compute session VPVR without timestamp data")
                return
                
            timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['timestamp'])
            
            session_poc = []
            session_vah = []
            session_val = []
            session_poc_dist = []
            
            current_session = None
            current_session_data = []
            
            for i, (timestamp, row) in enumerate(zip(timestamps, data.itertuples())):
                session = get_market_session(timestamp)
                
                # If session changed, calculate VPVR for previous session
                if session != current_session and current_session_data:
                    session_df = pd.DataFrame(current_session_data)
                    
                    if len(session_df) > 10:
                        vp_result = self._calculate_volume_profile(session_df, 30, 0.7)
                        last_poc = vp_result['poc']
                        last_vah = vp_result['value_area_high']
                        last_val = vp_result['value_area_low']
                    else:
                        last_poc = np.nan
                        last_vah = np.nan
                        last_val = np.nan
                        
                    # Backfill previous session
                    for j in range(len(current_session_data)):
                        session_poc.append(last_poc)
                        session_vah.append(last_vah)
                        session_val.append(last_val)
                        
                        # Distance from session POC
                        if not pd.isna(last_poc):
                            poc_distance = (current_session_data[j]['close'] - last_poc) / 0.01
                            session_poc_dist.append(poc_distance)
                        else:
                            session_poc_dist.append(np.nan)
                            
                    current_session_data = []
                    
                current_session = session
                current_session_data.append({
                    'high': row.high,
                    'low': row.low, 
                    'close': row.close,
                    'volume': row.volume
                })
                
                # For current bar, use last calculated values or NaN
                if session_poc:
                    session_poc.append(session_poc[-1])
                    session_vah.append(session_vah[-1])
                    session_val.append(session_val[-1])
                    session_poc_dist.append(session_poc_dist[-1])
                else:
                    session_poc.append(np.nan)
                    session_vah.append(np.nan)
                    session_val.append(np.nan)
                    session_poc_dist.append(np.nan)
                    
            # Ensure length matches
            while len(session_poc) < len(data):
                session_poc.append(np.nan)
                session_vah.append(np.nan)
                session_val.append(np.nan)
                session_poc_dist.append(np.nan)
                
            features['session_poc'] = session_poc[:len(data)]
            features['session_value_area_high'] = session_vah[:len(data)]
            features['session_value_area_low'] = session_val[:len(data)]
            features['session_poc_distance'] = session_poc_dist[:len(data)]
            
        except Exception as e:
            logger.warning(f"Failed to compute session VPVR features: {e}")


class SmartMoneyFeatures(BaseFeatureExtractor):
    """
    Smart Money Concepts (SMC) features
    
    Implements institutional trading pattern detection including liquidity sweeps,
    order blocks, fair value gaps, and break of structure analysis. These concepts
    help identify where institutional money is flowing.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="SmartMoneyFeatures",
            category="market_structure",
            priority=2,  # Medium-high priority
            parameters={
                'liquidity_sweep_lookback': 20,
                'liquidity_tolerance_pips': 3,
                'order_block_lookback': 20,
                'fair_value_gap_min_pips': 2,
                'break_of_structure_lookback': 50,
                'equal_highs_lows_tolerance': 3
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
        
    def get_min_periods(self) -> int:
        return max(
            self.config.parameters.get('liquidity_sweep_lookback', 20),
            self.config.parameters.get('break_of_structure_lookback', 50)
        )
        
    def get_feature_names(self) -> List[str]:
        names = [
            # Liquidity concepts
            'liquidity_sweep_high', 'liquidity_sweep_low',
            'equal_highs', 'equal_lows',
            'liquidity_grab_signal',
            
            # Order blocks
            'bullish_order_block', 'bearish_order_block',
            'order_block_distance', 'order_block_strength',
            
            # Fair value gaps
            'fair_value_gap_up', 'fair_value_gap_down',
            'fvg_size_pips', 'fvg_filled',
            
            # Structure breaks
            'break_of_structure_bull', 'break_of_structure_bear',
            'change_of_character', 'market_structure_shift',
            
            # Inducement
            'inducement_high', 'inducement_low',
            'premium_discount_zone'
        ]
        
        return names
        
    @cached_feature(ttl=600)
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract Smart Money Concepts features"""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Liquidity sweep detection
            self._compute_liquidity_features(features, data)
            
            # Order block identification
            self._compute_order_block_features(features, data)
            
            # Fair value gap detection
            self._compute_fair_value_gap_features(features, data)
            
            # Structure break analysis
            self._compute_structure_break_features(features, data)
            
            # Premium/discount zones
            self._compute_premium_discount_features(features, data)
            
        except Exception as e:
            logger.error(f"Error computing SMC features: {e}")
            
        return features
        
    def _compute_liquidity_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute liquidity sweep and equal highs/lows features"""
        try:
            lookback = self.config.parameters.get('liquidity_sweep_lookback', 20)
            tolerance_pips = self.config.parameters.get('equal_highs_lows_tolerance', 3)
            tolerance_price = pips_to_price(tolerance_pips, 'XAUUSD')
            
            sweep_high = []
            sweep_low = []
            equal_highs = []
            equal_lows = []
            liquidity_grab = []
            
            for i in range(len(data)):
                start_idx = max(0, i - lookback)
                end_idx = i + 1
                
                if end_idx - start_idx < lookback // 2:
                    sweep_high.append(0)
                    sweep_low.append(0)
                    equal_highs.append(0)
                    equal_lows.append(0)
                    liquidity_grab.append(0)
                    continue
                    
                window_data = data.iloc[start_idx:end_idx]
                current_bar = data.iloc[i]
                
                # Find recent highs and lows
                highs = window_data['high'].values
                lows = window_data['low'].values
                
                # Liquidity sweep detection
                sweep_high_signal = self._detect_liquidity_sweep(
                    current_bar['high'], current_bar['low'], highs, 'high', tolerance_price
                )
                sweep_low_signal = self._detect_liquidity_sweep(
                    current_bar['high'], current_bar['low'], lows, 'low', tolerance_price
                )
                
                sweep_high.append(sweep_high_signal)
                sweep_low.append(sweep_low_signal)
                
                # Equal highs/lows detection
                equal_high_signal = self._detect_equal_levels(highs, tolerance_price, 'high')
                equal_low_signal = self._detect_equal_levels(lows, tolerance_price, 'low')
                
                equal_highs.append(equal_high_signal)
                equal_lows.append(equal_low_signal)
                
                # Overall liquidity grab signal
                grab_signal = max(sweep_high_signal, sweep_low_signal, equal_high_signal, equal_low_signal)
                liquidity_grab.append(grab_signal)
                
            features['liquidity_sweep_high'] = sweep_high
            features['liquidity_sweep_low'] = sweep_low
            features['equal_highs'] = equal_highs
            features['equal_lows'] = equal_lows
            features['liquidity_grab_signal'] = liquidity_grab
            
            self.add_feature_description('liquidity_sweep_high', "High liquidity sweep detection")
            self.add_feature_description('liquidity_grab_signal', "Overall liquidity grab signal")
            
        except Exception as e:
            logger.warning(f"Failed to compute liquidity features: {e}")
            
    def _detect_liquidity_sweep(self, current_high: float, current_low: float, 
                              price_levels: np.ndarray, level_type: str, tolerance: float) -> int:
        """Detect liquidity sweep above/below previous levels"""
        try:
            if level_type == 'high':
                # Look for sweeps above previous highs
                recent_highs = price_levels[-5:]  # Last 5 periods
                max_high = np.max(recent_highs)
                
                # Current bar must sweep above and then close below
                if current_high > max_high + tolerance and current_low < max_high:
                    return 1
                    
            elif level_type == 'low':
                # Look for sweeps below previous lows
                recent_lows = price_levels[-5:]  # Last 5 periods
                min_low = np.min(recent_lows)
                
                # Current bar must sweep below and then close above
                if current_low < min_low - tolerance and current_high > min_low:
                    return 1
                    
            return 0
            
        except Exception:
            return 0
            
    def _detect_equal_levels(self, price_levels: np.ndarray, tolerance: float, level_type: str) -> int:
        """Detect equal highs or lows"""
        try:
            if len(price_levels) < 3:
                return 0
                
            # Find local extremes
            if level_type == 'high':
                # Look for multiple highs at similar levels
                recent_levels = price_levels[-10:]
                max_level = np.max(recent_levels)
                equal_count = np.sum(np.abs(recent_levels - max_level) <= tolerance)
            else:
                # Look for multiple lows at similar levels
                recent_levels = price_levels[-10:]
                min_level = np.min(recent_levels)
                equal_count = np.sum(np.abs(recent_levels - min_level) <= tolerance)
                
            # Signal if we have multiple touches at the same level
            return 1 if equal_count >= 2 else 0
            
        except Exception:
            return 0
            
    def _compute_order_block_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute order block identification features"""
        try:
            lookback = self.config.parameters.get('order_block_lookback', 20)
            
            bullish_blocks = []
            bearish_blocks = []
            block_distances = []
            block_strengths = []
            
            for i in range(len(data)):
                start_idx = max(0, i - lookback)
                end_idx = i + 1
                
                if end_idx - start_idx < 5:
                    bullish_blocks.append(0)
                    bearish_blocks.append(0)
                    block_distances.append(np.nan)
                    block_strengths.append(0)
                    continue
                    
                window_data = data.iloc[start_idx:end_idx]
                current_price = data.iloc[i]['close']
                
                # Find order blocks
                bullish_block = self._identify_order_block(window_data, 'bullish')
                bearish_block = self._identify_order_block(window_data, 'bearish')
                
                bullish_blocks.append(bullish_block['signal'])
                bearish_blocks.append(bearish_block['signal'])
                
                # Distance to nearest order block
                block_distance = min(
                    abs(current_price - bullish_block['price']) if bullish_block['price'] else float('inf'),
                    abs(current_price - bearish_block['price']) if bearish_block['price'] else float('inf')
                )
                
                block_distances.append(block_distance if block_distance != float('inf') else np.nan)
                
                # Block strength (volume-based)
                block_strength = max(bullish_block['strength'], bearish_block['strength'])
                block_strengths.append(block_strength)
                
            features['bullish_order_block'] = bullish_blocks
            features['bearish_order_block'] = bearish_blocks
            features['order_block_distance'] = block_distances
            features['order_block_strength'] = block_strengths
            
        except Exception as e:
            logger.warning(f"Failed to compute order block features: {e}")
            
    def _identify_order_block(self, data: pd.DataFrame, block_type: str) -> Dict[str, Any]:
        """Identify order blocks in price data"""
        try:
            if len(data) < 5:
                return {'signal': 0, 'price': None, 'strength': 0}
                
            # Look for strong moves followed by consolidation
            if block_type == 'bullish':
                # Strong up move followed by pullback
                returns = data['close'].pct_change()
                strong_moves = returns > 0.005  # 0.5% move
                
                if strong_moves.sum() > 0:
                    # Find the base of the move (order block zone)
                    strong_move_idx = strong_moves.idxmax()
                    base_price = data.loc[strong_move_idx, 'low']
                    volume_strength = data.loc[strong_move_idx, 'volume']
                    
                    return {
                        'signal': 1,
                        'price': base_price,
                        'strength': volume_strength / data['volume'].mean()
                    }
                    
            else:  # bearish
                # Strong down move followed by bounce
                returns = data['close'].pct_change()
                strong_moves = returns < -0.005  # -0.5% move
                
                if strong_moves.sum() > 0:
                    # Find the top of the move (order block zone)
                    strong_move_idx = strong_moves.idxmax()
                    top_price = data.loc[strong_move_idx, 'high']
                    volume_strength = data.loc[strong_move_idx, 'volume']
                    
                    return {
                        'signal': 1,
                        'price': top_price,
                        'strength': volume_strength / data['volume'].mean()
                    }
                    
            return {'signal': 0, 'price': None, 'strength': 0}
            
        except Exception:
            return {'signal': 0, 'price': None, 'strength': 0}
            
    def _compute_fair_value_gap_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute Fair Value Gap (FVG) features"""
        try:
            min_gap_pips = self.config.parameters.get('fair_value_gap_min_pips', 2)
            min_gap_price = pips_to_price(min_gap_pips, 'XAUUSD')
            
            fvg_up = []
            fvg_down = []
            fvg_sizes = []
            fvg_filled = []
            
            for i in range(2, len(data)):  # Need at least 3 bars
                # Three-bar pattern for FVG
                bar1 = data.iloc[i-2]
                bar2 = data.iloc[i-1]
                bar3 = data.iloc[i]
                
                # Bullish FVG: gap between bar1 high and bar3 low
                if bar1['high'] < bar3['low'] - min_gap_price:
                    fvg_up.append(1)
                    fvg_down.append(0)
                    gap_size = (bar3['low'] - bar1['high']) / 0.01  # In pips
                    fvg_sizes.append(gap_size)
                    
                    # Check if gap is filled
                    filled = bar3['low'] <= bar1['high']
                    fvg_filled.append(1 if filled else 0)
                    
                # Bearish FVG: gap between bar1 low and bar3 high
                elif bar1['low'] > bar3['high'] + min_gap_price:
                    fvg_up.append(0)
                    fvg_down.append(1)
                    gap_size = (bar1['low'] - bar3['high']) / 0.01  # In pips
                    fvg_sizes.append(gap_size)
                    
                    # Check if gap is filled
                    filled = bar3['high'] >= bar1['low']
                    fvg_filled.append(1 if filled else 0)
                    
                else:
                    fvg_up.append(0)
                    fvg_down.append(0)
                    fvg_sizes.append(0)
                    fvg_filled.append(0)
                    
            # Pad the beginning
            while len(fvg_up) < len(data):
                fvg_up.insert(0, 0)
                fvg_down.insert(0, 0)
                fvg_sizes.insert(0, 0)
                fvg_filled.insert(0, 0)
                
            features['fair_value_gap_up'] = fvg_up
            features['fair_value_gap_down'] = fvg_down
            features['fvg_size_pips'] = fvg_sizes
            features['fvg_filled'] = fvg_filled
            
        except Exception as e:
            logger.warning(f"Failed to compute FVG features: {e}")
            
    def _compute_structure_break_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute break of structure and change of character features"""
        try:
            lookback = self.config.parameters.get('break_of_structure_lookback', 50)
            
            bos_bull = []
            bos_bear = []
            choch = []
            mss = []
            
            for i in range(len(data)):
                start_idx = max(0, i - lookback)
                end_idx = i + 1
                
                if end_idx - start_idx < 10:
                    bos_bull.append(0)
                    bos_bear.append(0)
                    choch.append(0)
                    mss.append(0)
                    continue
                    
                window_data = data.iloc[start_idx:end_idx]
                
                # Identify market structure
                structure_analysis = self._analyze_market_structure(window_data)
                
                bos_bull.append(structure_analysis['bos_bullish'])
                bos_bear.append(structure_analysis['bos_bearish'])
                choch.append(structure_analysis['change_of_character'])
                mss.append(structure_analysis['market_structure_shift'])
                
            features['break_of_structure_bull'] = bos_bull
            features['break_of_structure_bear'] = bos_bear
            features['change_of_character'] = choch
            features['market_structure_shift'] = mss
            
        except Exception as e:
            logger.warning(f"Failed to compute structure break features: {e}")
            
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, int]:
        """Analyze market structure for breaks and character changes"""
        try:
            if len(data) < 10:
                return {
                    'bos_bullish': 0, 'bos_bearish': 0,
                    'change_of_character': 0, 'market_structure_shift': 0
                }
                
            # Find swing highs and lows
            highs = data['high'].values
            lows = data['low'].values
            
            # Simple swing detection (could be enhanced)
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(data) - 2):
                # Swing high: higher than 2 bars on each side
                if (highs[i] > highs[i-2] and highs[i] > highs[i-1] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    swing_highs.append((i, highs[i]))
                    
                # Swing low: lower than 2 bars on each side
                if (lows[i] < lows[i-2] and lows[i] < lows[i-1] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    swing_lows.append((i, lows[i]))
                    
            # Analyze structure breaks
            bos_bullish = 0
            bos_bearish = 0
            
            if len(swing_highs) >= 2:
                # Bullish BOS: break above previous swing high
                last_high = swing_highs[-1][1]
                current_high = data['high'].iloc[-1]
                if current_high > last_high:
                    bos_bullish = 1
                    
            if len(swing_lows) >= 2:
                # Bearish BOS: break below previous swing low
                last_low = swing_lows[-1][1]
                current_low = data['low'].iloc[-1]
                if current_low < last_low:
                    bos_bearish = 1
                    
            # Change of character (reversal of trend)
            change_of_character = 0
            if bos_bullish and len(swing_lows) >= 2:
                # Bullish CHoCH after bearish structure
                change_of_character = 1
            elif bos_bearish and len(swing_highs) >= 2:
                # Bearish CHoCH after bullish structure
                change_of_character = 1
                
            # Market structure shift (stronger signal)
            market_structure_shift = 0
            if change_of_character and (len(swing_highs) >= 3 or len(swing_lows) >= 3):
                market_structure_shift = 1
                
            return {
                'bos_bullish': bos_bullish,
                'bos_bearish': bos_bearish,
                'change_of_character': change_of_character,
                'market_structure_shift': market_structure_shift
            }
            
        except Exception:
            return {
                'bos_bullish': 0, 'bos_bearish': 0,
                'change_of_character': 0, 'market_structure_shift': 0
            }
            
    def _compute_premium_discount_features(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """Compute premium/discount zone features"""
        try:
            # Use 50-period range for premium/discount zones
            lookback = 50
            
            inducement_high = []
            inducement_low = []
            premium_discount = []
            
            for i in range(len(data)):
                start_idx = max(0, i - lookback)
                end_idx = i + 1
                
                if end_idx - start_idx < lookback // 2:
                    inducement_high.append(0)
                    inducement_low.append(0)
                    premium_discount.append(0)
                    continue
                    
                window_data = data.iloc[start_idx:end_idx]
                current_price = data.iloc[i]['close']
                
                # Calculate range
                range_high = window_data['high'].max()
                range_low = window_data['low'].min()
                range_mid = (range_high + range_low) / 2
                
                # Premium zone (upper 30%)
                premium_level = range_low + 0.7 * (range_high - range_low)
                # Discount zone (lower 30%)
                discount_level = range_low + 0.3 * (range_high - range_low)
                
                # Inducement signals
                inducement_high_signal = 1 if current_price > range_high * 1.001 else 0
                inducement_low_signal = 1 if current_price < range_low * 0.999 else 0
                
                # Premium/discount classification
                if current_price > premium_level:
                    pd_zone = 1  # Premium
                elif current_price < discount_level:
                    pd_zone = -1  # Discount
                else:
                    pd_zone = 0  # Equilibrium
                    
                inducement_high.append(inducement_high_signal)
                inducement_low.append(inducement_low_signal)
                premium_discount.append(pd_zone)
                
            features['inducement_high'] = inducement_high
            features['inducement_low'] = inducement_low
            features['premium_discount_zone'] = premium_discount
            
        except Exception as e:
            logger.warning(f"Failed to compute premium/discount features: {e}")


class CrossAssetFeatures(BaseFeatureExtractor):
    """
    Cross-asset correlation and spread features
    
    Analyzes relationships between XAUUSD and correlated assets (DXY, TIPS, SPY)
    for regime detection and signal confirmation. Cross-asset divergences often
    precede significant moves in gold.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        default_config = FeatureConfig(
            name="CrossAssetFeatures",
            category="cross_asset",
            priority=2,  # Medium priority
            parameters={
                'assets': ['DXY', 'TIPS', 'SPY'],
                'correlation_windows': [20, 60, 240],
                'spread_windows': [20, 60],
                'regime_detection': True,
                'divergence_detection': True
            }
        )
        
        super().__init__(config or default_config)
        
    def get_required_columns(self) -> List[str]:
        return ['close']
        
    def get_min_periods(self) -> int:
        return max(self.config.parameters.get('correlation_windows', [240]))
        
    def get_feature_names(self) -> List[str]:
        names = []
        assets = self.config.parameters.get('assets', ['DXY', 'TIPS', 'SPY'])
        corr_windows = self.config.parameters.get('correlation_windows', [20, 60, 240])
        spread_windows = self.config.parameters.get('spread_windows', [20, 60])
        
        for asset in assets:
            asset_lower = asset.lower()
            
            # Correlation features
            for window in corr_windows:
                names.extend([
                    f'{asset_lower}_correlation_{window}',
                    f'{asset_lower}_beta_{window}',
                    f'{asset_lower}_correlation_regime_{window}'
                ])
                
            # Spread features
            for window in spread_windows:
                names.extend([
                    f'{asset_lower}_spread_{window}',
                    f'{asset_lower}_spread_zscore_{window}',
                    f'{asset_lower}_spread_momentum_{window}'
                ])
                
            # Divergence features
            if self.config.parameters.get('divergence_detection', True):
                names.extend([
                    f'{asset_lower}_divergence_short',
                    f'{asset_lower}_divergence_medium',
                    f'{asset_lower}_relative_strength'
                ])
                
        # Regime features
        if self.config.parameters.get('regime_detection', True):
            names.extend([
                'risk_on_off_regime',
                'dollar_strength_regime',
                'inflation_expectations_regime'
            ])
            
        return names
        
    def extract_features(self, data: pd.DataFrame, cross_asset_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Extract cross-asset features (override to accept cross-asset data)"""
        features = pd.DataFrame(index=data.index)
        
        if not cross_asset_data:
            logger.warning("No cross-asset data provided for correlation analysis")
            # Return empty features with correct structure
            for feature_name in self.get_feature_names():
                features[feature_name] = np.nan
            return features
            
        try:
            # Correlation analysis
            self._compute_correlation_features(features, data, cross_asset_data)
            
            # Spread analysis
            self._compute_spread_features(features, data, cross_asset_data)
            
            # Divergence detection
            if self.config.parameters.get('divergence_detection', True):
                self._compute_divergence_features(features, data, cross_asset_data)
                
            # Regime detection
            if self.config.parameters.get('regime_detection', True):
                self._compute_regime_features(features, data, cross_asset_data)
                
        except Exception as e:
            logger.error(f"Error computing cross-asset features: {e}")
            
        return features
        
    def _compute_correlation_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                    cross_asset_data: Dict[str, pd.DataFrame]) -> None:
        """Compute rolling correlations with cross-assets"""
        try:
            assets = self.config.parameters.get('assets', ['DXY', 'TIPS', 'SPY'])
            windows = self.config.parameters.get('correlation_windows', [20, 60, 240])
            
            gold_returns = data['close'].pct_change()
            
            for asset in assets:
                if asset not in cross_asset_data:
                    continue
                    
                asset_data = cross_asset_data[asset]
                if 'close' not in asset_data.columns:
                    continue
                    
                # Align data
                aligned_data = self._align_timeseries(data, asset_data)
                if aligned_data is None:
                    continue
                    
                gold_aligned, asset_aligned = aligned_data
                asset_returns = asset_aligned['close'].pct_change()
                
                asset_lower = asset.lower()
                
                for window in windows:
                    # Rolling correlation
                    correlation = gold_aligned['close'].rolling(window).corr(asset_aligned['close'])
                    features[f'{asset_lower}_correlation_{window}'] = correlation
                    
                    # Rolling beta
                    cov = gold_returns.rolling(window).cov(asset_returns)
                    var = asset_returns.rolling(window).var()
                    beta = safe_divide(cov, var)
                    features[f'{asset_lower}_beta_{window}'] = beta
                    
                    # Correlation regime (high/low correlation periods)
                    corr_mean = correlation.rolling(window * 2).mean()
                    corr_regime = (correlation > corr_mean).astype(int)
                    features[f'{asset_lower}_correlation_regime_{window}'] = corr_regime
                    
        except Exception as e:
            logger.warning(f"Failed to compute correlation features: {e}")
            
    def _align_timeseries(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Align two time series DataFrames"""
        try:
            # Use index for alignment
            if isinstance(data1.index, pd.DatetimeIndex) and isinstance(data2.index, pd.DatetimeIndex):
                # Resample both to same frequency (use higher frequency)
                freq1 = pd.infer_freq(data1.index)
                freq2 = pd.infer_freq(data2.index)
                
                # Simple alignment - forward fill and align indices
                aligned_data1 = data1.reindex(data1.index.union(data2.index)).fillna(method='ffill')
                aligned_data2 = data2.reindex(data1.index.union(data2.index)).fillna(method='ffill')
                
                # Keep only overlapping periods
                common_index = aligned_data1.index.intersection(aligned_data2.index)
                
                if len(common_index) < 10:
                    return None
                    
                return (
                    aligned_data1.loc[common_index],
                    aligned_data2.loc[common_index]
                )
            else:
                # No datetime index, assume already aligned
                min_len = min(len(data1), len(data2))
                return (data1.iloc[:min_len], data2.iloc[:min_len])
                
        except Exception:
            return None
            
    def _compute_spread_features(self, features: pd.DataFrame, data: pd.DataFrame,
                               cross_asset_data: Dict[str, pd.DataFrame]) -> None:
        """Compute spread and relative performance features"""
        try:
            assets = self.config.parameters.get('assets', ['DXY', 'TIPS', 'SPY'])
            windows = self.config.parameters.get('spread_windows', [20, 60])
            
            for asset in assets:
                if asset not in cross_asset_data:
                    continue
                    
                asset_data = cross_asset_data[asset]
                if 'close' not in asset_data.columns:
                    continue
                    
                aligned_data = self._align_timeseries(data, asset_data)
                if aligned_data is None:
                    continue
                    
                gold_aligned, asset_aligned = aligned_data
                asset_lower = asset.lower()
                
                for window in windows:
                    # Price ratio spread
                    price_ratio = safe_divide(gold_aligned['close'], asset_aligned['close'])
                    features[f'{asset_lower}_spread_{window}'] = price_ratio
                    
                    # Z-score of spread
                    ratio_mean = price_ratio.rolling(window).mean()
                    ratio_std = price_ratio.rolling(window).std()
                    spread_zscore = safe_divide(price_ratio - ratio_mean, ratio_std)
                    features[f'{asset_lower}_spread_zscore_{window}'] = spread_zscore
                    
                    # Spread momentum
                    spread_momentum = price_ratio.diff(5)  # 5-period change
                    features[f'{asset_lower}_spread_momentum_{window}'] = spread_momentum
                    
        except Exception as e:
            logger.warning(f"Failed to compute spread features: {e}")
            
    def _compute_divergence_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                   cross_asset_data: Dict[str, pd.DataFrame]) -> None:
        """Compute divergence signals between assets"""
        try:
            assets = self.config.parameters.get('assets', ['DXY', 'TIPS', 'SPY'])
            
            gold_momentum_short = data['close'].pct_change(5)  # 5-period return
            gold_momentum_medium = data['close'].pct_change(20)  # 20-period return
            
            for asset in assets:
                if asset not in cross_asset_data:
                    continue
                    
                asset_data = cross_asset_data[asset]
                if 'close' not in asset_data.columns:
                    continue
                    
                aligned_data = self._align_timeseries(data, asset_data)
                if aligned_data is None:
                    continue
                    
                gold_aligned, asset_aligned = aligned_data
                asset_lower = asset.lower()
                
                # Asset momentum
                asset_momentum_short = asset_aligned['close'].pct_change(5)
                asset_momentum_medium = asset_aligned['close'].pct_change(20)
                
                # Divergence detection (opposite direction moves)
                divergence_short = np.where(
                    (gold_momentum_short > 0) & (asset_momentum_short < 0), 1,
                    np.where((gold_momentum_short < 0) & (asset_momentum_short > 0), -1, 0)
                )
                
                divergence_medium = np.where(
                    (gold_momentum_medium > 0) & (asset_momentum_medium < 0), 1,
                    np.where((gold_momentum_medium < 0) & (asset_momentum_medium > 0), -1, 0)
                )
                
                features[f'{asset_lower}_divergence_short'] = divergence_short
                features[f'{asset_lower}_divergence_medium'] = divergence_medium
                
                # Relative strength
                relative_strength = safe_divide(
                    gold_momentum_medium,
                    asset_momentum_medium
                )
                features[f'{asset_lower}_relative_strength'] = relative_strength
                
        except Exception as e:
            logger.warning(f"Failed to compute divergence features: {e}")
            
    def _compute_regime_features(self, features: pd.DataFrame, data: pd.DataFrame,
                               cross_asset_data: Dict[str, pd.DataFrame]) -> None:
        """Compute market regime features"""
        try:
            # Risk-on/risk-off regime (based on SPY performance)
            if 'SPY' in cross_asset_data and 'close' in cross_asset_data['SPY'].columns:
                spy_data = cross_asset_data['SPY']
                aligned_data = self._align_timeseries(data, spy_data)
                
                if aligned_data is not None:
                    gold_aligned, spy_aligned = aligned_data
                    
                    # Risk-on when SPY outperforming, risk-off when gold outperforming
                    spy_returns = spy_aligned['close'].pct_change(20)
                    gold_returns = gold_aligned['close'].pct_change(20)
                    
                    risk_regime = np.where(spy_returns > gold_returns, 1, -1)
                    features['risk_on_off_regime'] = risk_regime
                else:
                    features['risk_on_off_regime'] = 0
            else:
                features['risk_on_off_regime'] = 0
                
            # Dollar strength regime (based on DXY)
            if 'DXY' in cross_asset_data and 'close' in cross_asset_data['DXY'].columns:
                dxy_data = cross_asset_data['DXY']
                aligned_data = self._align_timeseries(data, dxy_data)
                
                if aligned_data is not None:
                    gold_aligned, dxy_aligned = aligned_data
                    
                    # Dollar strength when DXY rising
                    dxy_momentum = dxy_aligned['close'].pct_change(20)
                    dollar_regime = np.where(dxy_momentum > 0, 1, -1)
                    features['dollar_strength_regime'] = dollar_regime
                else:
                    features['dollar_strength_regime'] = 0
            else:
                features['dollar_strength_regime'] = 0
                
            # Inflation expectations regime (based on TIPS)
            if 'TIPS' in cross_asset_data and 'close' in cross_asset_data['TIPS'].columns:
                tips_data = cross_asset_data['TIPS']
                aligned_data = self._align_timeseries(data, tips_data)
                
                if aligned_data is not None:
                    gold_aligned, tips_aligned = aligned_data
                    
                    # Inflation expectations when TIPS outperforming
                    tips_momentum = tips_aligned['close'].pct_change(20)
                    inflation_regime = np.where(tips_momentum > 0, 1, -1)
                    features['inflation_expectations_regime'] = inflation_regime
                else:
                    features['inflation_expectations_regime'] = 0
            else:
                features['inflation_expectations_regime'] = 0
                
        except Exception as e:
            logger.warning(f"Failed to compute regime features: {e}")


class MicrostructureFeatures:
    """
    Consolidated microstructure features class
    
    Provides unified interface to all microstructure analysis features
    including volume profile, smart money concepts, and cross-asset analysis.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Initialize feature extractors
        self.volume_profile_features = VolumeProfileFeatures()
        self.smart_money_features = SmartMoneyFeatures()
        self.cross_asset_features = CrossAssetFeatures()
        
        # Register extractors
        self.extractors = {
            'volume_profile': self.volume_profile_features,
            'smart_money': self.smart_money_features,
            'cross_asset': self.cross_asset_features
        }
        
    def extract_all_features(self, data: pd.DataFrame, cross_asset_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Extract all microstructure features"""
        all_features = pd.DataFrame(index=data.index)
        
        for name, extractor in self.extractors.items():
            try:
                if name == 'cross_asset':
                    # Cross-asset extractor needs additional data
                    features = extractor.extract_features(data, cross_asset_data)
                else:
                    features = extractor.compute_with_performance_tracking(data)
                    
                if not features.empty:
                    all_features = pd.concat([all_features, features], axis=1)
                    
            except Exception as e:
                logger.error(f"Failed to extract {name} features: {e}")
                
        return all_features
        
    def get_minimal_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract minimal microstructure features for low-latency operation"""
        minimal_features = pd.DataFrame(index=data.index)
        
        try:
            # VPVR breakout (research core feature)
            vp_features = self.volume_profile_features.compute_with_performance_tracking(data)
            if 'vpvr_breakout_100' in vp_features.columns:
                minimal_features['vpvr_breakout'] = vp_features['vpvr_breakout_100']
                
            # Liquidity sweep signals
            smc_features = self.smart_money_features.compute_with_performance_tracking(data)
            if 'liquidity_grab_signal' in smc_features.columns:
                minimal_features['liquidity_sweep'] = smc_features['liquidity_grab_signal']
                
        except Exception as e:
            logger.error(f"Failed to extract minimal microstructure features: {e}")
            
        return minimal_features