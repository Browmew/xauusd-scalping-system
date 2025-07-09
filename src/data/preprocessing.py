"""
Data Preprocessing Module

High-performance data preprocessing pipeline for tick and orderbook data.
Handles cleaning, resampling, alignment, and preparation for feature engineering
with optimized algorithms for low-latency operations.

Key Features:
- Tick data cleaning and outlier removal
- Orderbook normalization and gap filling
- Cross-asset data synchronization
- Time-weighted resampling
- Quality scoring and validation
- Memory-efficient processing

Time Complexity: O(n log n) for sorting, O(n) for most operations
Space Complexity: O(n) where n is data size
"""

import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import logging
from numba import njit
from scipy import stats
import warnings

from src.core.config import get_config_manager
from src.core.utils import Timer, validate_data, safe_divide
from src.data.ingestion import TickData, OrderBookSnapshot, OrderBookLevel

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Statistics from data cleaning process"""
    total_records: int
    outliers_removed: int
    gaps_filled: int
    duplicates_removed: int
    invalid_records: int
    quality_score: float


@dataclass
class ResamplingResult:
    """Result from data resampling"""
    data: pd.DataFrame
    stats: CleaningStats
    metadata: Dict[str, Any]


class TickProcessor:
    """
    High-performance tick data processor
    
    Cleans, validates, and resamples tick data with outlier detection,
    gap filling, and quality scoring for downstream feature engineering.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Quality thresholds
        self.max_price_change_pct = self.config.get('max_price_change_pct', 2.0)
        self.min_volume_threshold = self.config.get('min_volume_threshold', 0.01)
        self.max_gap_seconds = self.config.get('max_gap_seconds', 10)
        self.outlier_zscore_threshold = self.config.get('outlier_zscore_threshold', 5.0)
        
        # Resampling parameters
        self.default_frequencies = ['1S', '5S', '10S', '30S', '1T', '5T', '15T']
        
    def process_ticks(
        self, 
        ticks: List[TickData],
        frequencies: Optional[List[str]] = None
    ) -> Dict[str, ResamplingResult]:
        """
        Process tick data and resample to multiple frequencies
        
        Args:
            ticks: List of tick data
            frequencies: List of resampling frequencies
            
        Returns:
            Dictionary mapping frequency to resampled data
        """
        if not ticks:
            return {}
            
        frequencies = frequencies or self.default_frequencies
        
        with Timer("tick_processing"):
            # Convert to DataFrame
            df = self._ticks_to_dataframe(ticks)
            
            # Clean data
            df_clean, cleaning_stats = self._clean_tick_data(df)
            
            # Resample to different frequencies
            results = {}
            for freq in frequencies:
                with Timer(f"resample_{freq}"):
                    resampled = self._resample_ticks(df_clean, freq)
                    results[freq] = ResamplingResult(
                        data=resampled,
                        stats=cleaning_stats,
                        metadata={'frequency': freq, 'original_ticks': len(ticks)}
                    )
                    
            logger.info(f"Processed {len(ticks)} ticks to {len(frequencies)} frequencies")
            return results
            
    def _ticks_to_dataframe(self, ticks: List[TickData]) -> pd.DataFrame:
        """Convert tick data to DataFrame"""
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'side': tick.side,
                'trade_id': tick.trade_id,
                'sequence': tick.sequence
            })
            
        df = pd.DataFrame(data)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
        return df
        
    def _clean_tick_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        """Clean tick data and return statistics"""
        if df.empty:
            return df, CleaningStats(0, 0, 0, 0, 0, 0.0)
            
        original_count = len(df)
        outliers_removed = 0
        gaps_filled = 0
        duplicates_removed = 0
        invalid_records = 0
        
        # Remove duplicates
        df_clean = df[~df.index.duplicated(keep='first')]
        duplicates_removed = original_count - len(df_clean)
        
        # Remove invalid records
        invalid_mask = (
            (df_clean['price'] <= 0) |
            (df_clean['volume'] < 0) |
            df_clean['price'].isna() |
            df_clean['volume'].isna()
        )
        invalid_records = invalid_mask.sum()
        df_clean = df_clean[~invalid_mask]
        
        # Remove price outliers using Z-score
        if len(df_clean) > 10:
            price_returns = df_clean['price'].pct_change()
            z_scores = np.abs(stats.zscore(price_returns.dropna()))
            outlier_mask = z_scores > self.outlier_zscore_threshold
            
            # Extend mask to original DataFrame size
            extended_mask = pd.Series(False, index=df_clean.index)
            extended_mask[price_returns.dropna().index] = outlier_mask
            
            outliers_removed = extended_mask.sum()
            df_clean = df_clean[~extended_mask]
            
        # Remove volume outliers
        if len(df_clean) > 10:
            volume_outliers = self._detect_volume_outliers(df_clean['volume'])
            df_clean = df_clean[~volume_outliers]
            outliers_removed += volume_outliers.sum()
            
        # Fill small gaps (optional)
        gaps_filled = self._fill_small_gaps(df_clean)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df_clean, original_count)
        
        stats = CleaningStats(
            total_records=original_count,
            outliers_removed=outliers_removed,
            gaps_filled=gaps_filled,
            duplicates_removed=duplicates_removed,
            invalid_records=invalid_records,
            quality_score=quality_score
        )
        
        return df_clean, stats
        
    def _detect_volume_outliers(self, volume_series: pd.Series) -> pd.Series:
        """Detect volume outliers using IQR method"""
        Q1 = volume_series.quantile(0.25)
        Q3 = volume_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR  # More lenient than usual 1.5
        upper_bound = Q3 + 3 * IQR
        
        return (volume_series < lower_bound) | (volume_series > upper_bound)
        
    def _fill_small_gaps(self, df: pd.DataFrame) -> int:
        """Fill small time gaps in data"""
        # For now, just return 0 (no gaps filled)
        # In production, implement forward fill for small gaps
        return 0
        
    def _calculate_quality_score(self, df_clean: pd.DataFrame, original_count: int) -> float:
        """Calculate data quality score (0-1)"""
        if original_count == 0:
            return 0.0
            
        # Base score from retention rate
        retention_rate = len(df_clean) / original_count
        
        # Adjust for data consistency
        if len(df_clean) > 1:
            time_consistency = 1.0 - (df_clean.index.to_series().diff().std().total_seconds() / 60)
            time_consistency = max(0.0, min(1.0, time_consistency))
        else:
            time_consistency = 0.5
            
        # Combine scores
        quality_score = 0.7 * retention_rate + 0.3 * time_consistency
        return min(1.0, quality_score)
        
    def _resample_ticks(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Resample tick data to specified frequency"""
        if df.empty:
            return df
            
        # Define aggregation functions
        agg_funcs = {
            'price': 'ohlc',  # Open, High, Low, Close
            'volume': 'sum',
            'side': lambda x: (x == 'buy').sum() - (x == 'sell').sum()  # Buy/sell imbalance
        }
        
        # Resample and aggregate
        resampled = df.groupby(pd.Grouper(freq=frequency)).agg(agg_funcs)
        
        # Flatten column names
        resampled.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                           for col in resampled.columns]
        
        # Rename price columns
        if 'price_open' in resampled.columns:
            resampled = resampled.rename(columns={
                'price_open': 'open',
                'price_high': 'high', 
                'price_low': 'low',
                'price_close': 'close'
            })
            
        # Remove empty periods
        resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Add additional features
        if not resampled.empty:
            resampled['typical_price'] = (resampled['high'] + resampled['low'] + resampled['close']) / 3
            resampled['price_range'] = resampled['high'] - resampled['low']
            resampled['volume_per_tick'] = safe_divide(resampled['volume'], 
                                                     resampled.groupby(level=0).size())
            
        return resampled


class OrderBookProcessor:
    """
    Order book data processor
    
    Processes Level 2 order book snapshots to extract microstructure features
    and calculate market depth metrics for order flow analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.levels = self.config.get('levels', 5)
        
    def process_orderbook_snapshots(
        self,
        snapshots: List[OrderBookSnapshot]
    ) -> pd.DataFrame:
        """
        Process order book snapshots into structured DataFrame
        
        Args:
            snapshots: List of order book snapshots
            
        Returns:
            DataFrame with order book features
        """
        if not snapshots:
            return pd.DataFrame()
            
        with Timer("orderbook_processing"):
            data = []
            
            for snapshot in snapshots:
                features = self._extract_orderbook_features(snapshot)
                features['timestamp'] = snapshot.timestamp
                data.append(features)
                
            df = pd.DataFrame(data)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                
                # Add derived features
                df = self._add_derived_features(df)
                
            logger.info(f"Processed {len(snapshots)} order book snapshots")
            return df
            
    def _extract_orderbook_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Extract features from single order book snapshot"""
        features = {}
        
        # Basic features
        features['bid_price'] = snapshot.bids[0].price if snapshot.bids else 0.0
        features['ask_price'] = snapshot.asks[0].price if snapshot.asks else 0.0
        features['spread'] = snapshot.get_spread()
        features['mid_price'] = snapshot.get_mid_price()
        
        # Imbalance features
        for levels in [1, 3, 5]:
            if levels <= len(snapshot.bids) and levels <= len(snapshot.asks):
                imbalance = snapshot.get_imbalance(levels)
                features[f'imbalance_{levels}'] = imbalance
                
        # Depth features
        features.update(self._calculate_depth_features(snapshot))
        
        # Slope features
        features.update(self._calculate_slope_features(snapshot))
        
        return features
        
    def _calculate_depth_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate market depth features"""
        features = {}
        
        # Total volume at each level
        for i in range(min(self.levels, len(snapshot.bids), len(snapshot.asks))):
            features[f'bid_volume_{i+1}'] = snapshot.bids[i].size
            features[f'ask_volume_{i+1}'] = snapshot.asks[i].size
            features[f'bid_price_{i+1}'] = snapshot.bids[i].price
            features[f'ask_price_{i+1}'] = snapshot.asks[i].price
            
        # Cumulative volumes
        bid_volumes = [level.size for level in snapshot.bids[:self.levels]]
        ask_volumes = [level.size for level in snapshot.asks[:self.levels]]
        
        features['total_bid_volume'] = sum(bid_volumes)
        features['total_ask_volume'] = sum(ask_volumes)
        features['total_volume'] = features['total_bid_volume'] + features['total_ask_volume']
        
        # Volume-weighted average prices
        if bid_volumes and sum(bid_volumes) > 0:
            bid_prices = [level.price for level in snapshot.bids[:self.levels]]
            features['vwap_bid'] = np.average(bid_prices, weights=bid_volumes)
            
        if ask_volumes and sum(ask_volumes) > 0:
            ask_prices = [level.price for level in snapshot.asks[:self.levels]]
            features['vwap_ask'] = np.average(ask_prices, weights=ask_volumes)
            
        return features
        
    def _calculate_slope_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate order book slope features"""
        features = {}
        
        if len(snapshot.bids) >= 3 and len(snapshot.asks) >= 3:
            # Bid slope (how quickly volume decreases)
            bid_prices = [level.price for level in snapshot.bids[:3]]
            bid_volumes = [level.size for level in snapshot.bids[:3]]
            
            if len(set(bid_prices)) > 1:  # Avoid division by zero
                bid_slope = np.polyfit(bid_prices, bid_volumes, 1)[0]
                features['bid_slope'] = bid_slope
                
            # Ask slope
            ask_prices = [level.price for level in snapshot.asks[:3]]
            ask_volumes = [level.size for level in snapshot.asks[:3]]
            
            if len(set(ask_prices)) > 1:
                ask_slope = np.polyfit(ask_prices, ask_volumes, 1)[0]
                features['ask_slope'] = ask_slope
                
        return features
        
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to order book DataFrame"""
        if df.empty:
            return df
            
        # Spread percentage
        df['spread_pct'] = safe_divide(df['spread'], df['mid_price']) * 100
        
        # Price momentum
        df['mid_price_change'] = df['mid_price'].diff()
        df['mid_price_change_pct'] = df['mid_price'].pct_change() * 100
        
        # Volume momentum
        if 'total_volume' in df.columns:
            df['volume_change'] = df['total_volume'].diff()
            df['volume_change_pct'] = df['total_volume'].pct_change() * 100
            
        # Imbalance momentum
        if 'imbalance_1' in df.columns:
            df['imbalance_change'] = df['imbalance_1'].diff()
            
        # Rolling statistics
        for window in [10, 30, 60]:
            df[f'mid_price_sma_{window}'] = df['mid_price'].rolling(window).mean()
            df[f'spread_sma_{window}'] = df['spread'].rolling(window).mean()
            
            if 'imbalance_1' in df.columns:
                df[f'imbalance_sma_{window}'] = df['imbalance_1'].rolling(window).mean()
                
        return df


class CrossAssetProcessor:
    """
    Cross-asset data processor
    
    Synchronizes and processes cross-asset data (DXY, TIPS, SPY) for 
    correlation analysis and regime detection features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.correlation_windows = self.config.get('correlation_windows', [20, 60, 240])
        
    def process_cross_asset_data(
        self,
        xauusd_data: pd.DataFrame,
        cross_asset_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Process cross-asset data and calculate correlations
        
        Args:
            xauusd_data: XAUUSD price data
            cross_asset_data: Dictionary of cross-asset DataFrames
            
        Returns:
            DataFrame with cross-asset features
        """
        if xauusd_data.empty or not cross_asset_data:
            return pd.DataFrame()
            
        with Timer("cross_asset_processing"):
            # Synchronize timestamps
            all_data = self._synchronize_data(xauusd_data, cross_asset_data)
            
            # Calculate features
            features_df = self._calculate_cross_asset_features(all_data)
            
            logger.info(f"Processed cross-asset data for {len(cross_asset_data)} assets")
            return features_df
            
    def _synchronize_data(
        self,
        xauusd_data: pd.DataFrame,
        cross_asset_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Synchronize data across all assets"""
        # Start with XAUUSD data
        if 'close' in xauusd_data.columns:
            synchronized = xauusd_data[['close']].copy()
            synchronized.columns = ['xauusd_close']
        else:
            synchronized = pd.DataFrame(index=xauusd_data.index)
            synchronized['xauusd_close'] = 0.0
            
        # Add cross-asset data
        for asset, data in cross_asset_data.items():
            if not data.empty and 'close' in data.columns:
                # Resample to match XAUUSD frequency
                resampled = data['close'].resample(synchronized.index.freq or '1T').last()
                synchronized[f'{asset.lower()}_close'] = resampled
                
        # Forward fill missing values
        synchronized = synchronized.fillna(method='ffill').fillna(method='bfill')
        
        return synchronized
        
    def _calculate_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-asset correlation and spread features"""
        features = pd.DataFrame(index=data.index)
        
        # Calculate returns for correlation analysis
        returns_data = data.pct_change()
        
        # Cross-correlations
        if 'xauusd_close' in data.columns:
            xauusd_returns = returns_data['xauusd_close']
            
            for col in returns_data.columns:
                if col != 'xauusd_close':
                    asset_name = col.replace('_close', '')
                    
                    # Rolling correlations
                    for window in self.correlation_windows:
                        corr = xauusd_returns.rolling(window).corr(returns_data[col])
                        features[f'{asset_name}_corr_{window}'] = corr
                        
                        # Beta calculation
                        cov = xauusd_returns.rolling(window).cov(returns_data[col])
                        var = returns_data[col].rolling(window).var()
                        beta = safe_divide(cov, var)
                        features[f'{asset_name}_beta_{window}'] = beta
                        
        # Relative strength
        for col in data.columns:
            if col != 'xauusd_close':
                asset_name = col.replace('_close', '')
                
                # Price ratio
                ratio = safe_divide(data['xauusd_close'], data[col])
                features[f'{asset_name}_ratio'] = ratio
                
                # Ratio momentum
                features[f'{asset_name}_ratio_change'] = ratio.pct_change()
                
                # Z-score of ratio
                for window in [20, 60]:
                    ratio_mean = ratio.rolling(window).mean()
                    ratio_std = ratio.rolling(window).std()
                    zscore = safe_divide(ratio - ratio_mean, ratio_std)
                    features[f'{asset_name}_ratio_zscore_{window}'] = zscore
                    
        return features


class DataPreprocessor:
    """
    Main data preprocessing coordinator
    
    Orchestrates tick processing, order book processing, and cross-asset
    data synchronization for the complete preprocessing pipeline.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Initialize processors
        tick_config = self.config.get('data', 'processing', {})
        self.tick_processor = TickProcessor(tick_config)
        self.orderbook_processor = OrderBookProcessor(tick_config)
        self.cross_asset_processor = CrossAssetProcessor(tick_config)
        
        # Performance monitoring
        self.processing_stats = {
            'ticks_processed': 0,
            'orderbooks_processed': 0,
            'cross_assets_processed': 0,
            'processing_time_ms': 0.0
        }
        
    def preprocess_data(
        self,
        ticks: List[TickData],
        orderbooks: List[OrderBookSnapshot],
        cross_asset_data: Dict[str, Any] = None,
        frequencies: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Complete data preprocessing pipeline
        
        Args:
            ticks: Raw tick data
            orderbooks: Raw order book snapshots
            cross_asset_data: Cross-asset price data
            frequencies: Resampling frequencies
            
        Returns:
            Dictionary of processed DataFrames by frequency
        """
        start_time = time.time()
        results = {}
        
        try:
            # Process tick data
            if ticks:
                tick_results = self.tick_processor.process_ticks(ticks, frequencies)
                for freq, result in tick_results.items():
                    results[f'ticks_{freq}'] = result.data
                    
                self.processing_stats['ticks_processed'] += len(ticks)
                
            # Process order book data
            if orderbooks:
                orderbook_df = self.orderbook_processor.process_orderbook_snapshots(orderbooks)
                results['orderbook'] = orderbook_df
                self.processing_stats['orderbooks_processed'] += len(orderbooks)
                
            # Process cross-asset data
            if cross_asset_data and 'ticks_1T' in results:
                cross_asset_df = self.cross_asset_processor.process_cross_asset_data(
                    results['ticks_1T'], cross_asset_data
                )
                results['cross_asset'] = cross_asset_df
                self.processing_stats['cross_assets_processed'] += len(cross_asset_data)
                
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_stats['processing_time_ms'] = processing_time
            
            logger.info(f"Data preprocessing completed in {processing_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
            
        return results
        
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.processing_stats.copy()
        
    def validate_processed_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Validate processed data quality"""
        validation_results = {}
        
        for data_type, df in data.items():
            if df.empty:
                validation_results[data_type] = False
                continue
                
            # Check for required columns based on data type
            if 'ticks' in data_type:
                required_cols = ['open', 'high', 'low', 'close', 'volume']
            elif 'orderbook' in data_type:
                required_cols = ['bid_price', 'ask_price', 'spread', 'mid_price']
            elif 'cross_asset' in data_type:
                required_cols = []  # Variable columns
            else:
                required_cols = []
                
            # Validate
            is_valid = validate_data(df, required_cols)
            validation_results[data_type] = is_valid
            
        return validation_results