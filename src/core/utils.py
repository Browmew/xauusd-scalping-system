"""
Core Utilities for XAUUSD Scalping System

High-performance utility functions optimized for low-latency trading operations.
Includes timing utilities, performance monitoring, memory management, and
mathematical helpers for financial calculations.

Time Complexity: O(1) for most operations unless noted
Space Complexity: O(1) per operation unless noted
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime, timezone, timedelta
from functools import wraps, lru_cache
from dataclasses import dataclass
import threading
import warnings
from contextlib import contextmanager
import gc
import sys
import os
from pathlib import Path
import hashlib
import json
from numba import njit
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result from timing operations"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any]


@dataclass
class MemorySnapshot:
    """System memory snapshot"""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    
    
@dataclass
class PerformanceSnapshot:
    """Complete system performance snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory: MemorySnapshot
    latency_ms: float
    throughput: float
    active_threads: int


class Timer:
    """
    High-precision timer for measuring operation latency
    
    Provides sub-millisecond timing precision for critical trading operations.
    Supports context manager and decorator patterns.
    """
    
    def __init__(self, operation: str = "operation"):
        self.operation = operation
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
        
    def start(self) -> None:
        """Start timing"""
        self.start_time = time.perf_counter()
        
    def stop(self) -> float:
        """Stop timing and return duration in milliseconds"""
        if self.start_time is None:
            raise ValueError("Timer not started")
            
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        return self.duration_ms
        
    def reset(self) -> None:
        """Reset timer"""
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        
    def elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        return (time.perf_counter() - self.start_time) * 1000
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        success = exc_type is None
        
        result = TimingResult(
            operation=self.operation,
            duration_ms=self.duration_ms,
            timestamp=datetime.now(timezone.utc),
            success=success,
            metadata={'exc_type': str(exc_type) if exc_type else None}
        )
        
        # Log if duration exceeds threshold
        if self.duration_ms > 50:  # 50ms threshold from research
            logger.warning(
                f"Operation '{self.operation}' exceeded latency budget: {self.duration_ms:.2f}ms"
            )
            
        return False  # Don't suppress exceptions


def timing(operation: str = None):
    """
    Decorator for timing function execution
    
    Args:
        operation: Name of the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(op_name) as timer:
                result = func(*args, **kwargs)
            return result
            
        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Real-time system performance monitoring
    
    Tracks CPU, memory, and application metrics with configurable sampling
    intervals. Provides alerts when performance degrades beyond thresholds.
    """
    
    def __init__(
        self, 
        sample_interval: float = 1.0,
        memory_threshold_mb: float = 500,
        cpu_threshold_pct: float = 80,
        latency_threshold_ms: float = 75
    ):
        self.sample_interval = sample_interval
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_pct = cpu_threshold_pct
        self.latency_threshold_ms = latency_threshold_ms
        
        self.process = psutil.Process()
        self.snapshots: List[PerformanceSnapshot] = []
        self.max_snapshots = 1000
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def start_monitoring(self) -> None:
        """Start background performance monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started performance monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped performance monitoring")
        
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring:
            try:
                snapshot = self.get_snapshot()
                
                with self._lock:
                    self.snapshots.append(snapshot)
                    if len(self.snapshots) > self.max_snapshots:
                        self.snapshots.pop(0)
                        
                # Check thresholds
                self._check_thresholds(snapshot)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                
    def _check_thresholds(self, snapshot: PerformanceSnapshot) -> None:
        """Check performance thresholds and log warnings"""
        if snapshot.memory.rss_mb > self.memory_threshold_mb:
            logger.warning(
                f"Memory usage high: {snapshot.memory.rss_mb:.1f}MB "
                f"(threshold: {self.memory_threshold_mb}MB)"
            )
            
        if snapshot.cpu_percent > self.cpu_threshold_pct:
            logger.warning(
                f"CPU usage high: {snapshot.cpu_percent:.1f}% "
                f"(threshold: {self.cpu_threshold_pct}%)"
            )
            
        if snapshot.latency_ms > self.latency_threshold_ms:
            logger.warning(
                f"Latency high: {snapshot.latency_ms:.1f}ms "
                f"(threshold: {self.latency_threshold_ms}ms)"
            )
            
    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        # Memory info
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        virtual_memory = psutil.virtual_memory()
        
        memory_snapshot = MemorySnapshot(
            timestamp=datetime.now(timezone.utc),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=virtual_memory.available / 1024 / 1024
        )
        
        # CPU info
        cpu_percent = self.process.cpu_percent()
        
        # Thread count
        active_threads = threading.active_count()
        
        # Estimate latency (time to get system info)
        start = time.perf_counter()
        psutil.cpu_percent()
        latency_ms = (time.perf_counter() - start) * 1000
        
        return PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory=memory_snapshot,
            latency_ms=latency_ms,
            throughput=0.0,  # To be set by application
            active_threads=active_threads
        )
        
    def get_recent_snapshots(self, minutes: int = 5) -> List[PerformanceSnapshot]:
        """Get snapshots from recent time window"""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        with self._lock:
            return [s for s in self.snapshots if s.timestamp >= cutoff]
            
    def get_average_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """Get average metrics over time window"""
        recent = self.get_recent_snapshots(minutes)
        
        if not recent:
            return {}
            
        return {
            'avg_cpu_percent': np.mean([s.cpu_percent for s in recent]),
            'avg_memory_mb': np.mean([s.memory.rss_mb for s in recent]),
            'avg_latency_ms': np.mean([s.latency_ms for s in recent]),
            'max_latency_ms': np.max([s.latency_ms for s in recent]),
            'sample_count': len(recent)
        }
        
    def reset_snapshots(self) -> None:
        """Clear stored snapshots"""
        with self._lock:
            self.snapshots.clear()


@contextmanager
def memory_limit(limit_mb: float):
    """
    Context manager to enforce memory limits
    
    Args:
        limit_mb: Memory limit in megabytes
    """
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - initial_memory
        
        if memory_used > limit_mb:
            logger.warning(f"Memory limit exceeded: {memory_used:.1f}MB > {limit_mb}MB")
            # Force garbage collection
            gc.collect()


@lru_cache(maxsize=1000)
def cached_datetime_parse(date_string: str) -> datetime:
    """Cached datetime parsing for performance"""
    return pd.to_datetime(date_string)


def batch_process(
    items: List[Any], 
    process_func: Callable, 
    batch_size: int = 1000,
    show_progress: bool = False
) -> List[Any]:
    """
    Process items in batches for memory efficiency
    
    Args:
        items: Items to process
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        show_progress: Whether to show progress
        
    Returns:
        List of processed results
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        with Timer(f"batch_process_{i//batch_size}"):
            batch_result = process_func(batch)
            results.extend(batch_result)
            
        if show_progress:
            batch_num = i // batch_size + 1
            print(f"Processed batch {batch_num}/{total_batches}", end='\r')
            
        # Force garbage collection every 10 batches
        if i % (batch_size * 10) == 0:
            gc.collect()
            
    return results


# Optimized numerical functions using Numba

@njit
def fast_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast exponential moving average calculation using Numba JIT
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    alpha = 2.0 / (period + 1.0)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
    return ema


@njit
def fast_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast simple moving average using Numba JIT
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    sma = np.full_like(prices, np.nan)
    
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
        
    return sma


@njit
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Fast RSI calculation using Numba JIT
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi = np.full(len(prices), np.nan)
    
    # Calculate RSI using Wilder's smoothing
    for i in range(period, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
            
    return rsi


@njit
def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Fast Average True Range calculation using Numba JIT
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    n = len(close)
    true_range = np.empty(n)
    
    # First true range is just high - low
    true_range[0] = high[0] - low[0]
    
    # Calculate true range for remaining periods
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        true_range[i] = max(tr1, max(tr2, tr3))
        
    # Calculate ATR using EMA smoothing
    atr = np.empty(n)
    atr[period-1] = np.mean(true_range[:period])
    
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
        
    # Fill initial values with NaN
    atr[:period-1] = np.nan
    
    return atr


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame structure and data quality
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if validation passes
    """
    if df.empty:
        logger.error("DataFrame is empty")
        return False
        
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
        
    # Check for sufficient data
    if len(df) < 100:
        logger.warning(f"Limited data available: {len(df)} rows")
        
    # Check for missing values
    missing_pct = df[required_columns].isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 5]
    
    if not high_missing.empty:
        logger.warning(f"High missing data percentage: {high_missing.to_dict()}")
        
    # Check for infinite values
    inf_cols = []
    for col in required_columns:
        if df[col].dtype in ['float64', 'float32']:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
                
    if inf_cols:
        logger.error(f"Infinite values found in columns: {inf_cols}")
        return False
        
    return True


def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray],
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safe division with default value for zero division
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)  
        default: Default value for zero division
        
    Returns:
        Division result with safe handling
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.divide(numerator, denominator)
        
    # Handle division by zero
    if isinstance(result, np.ndarray):
        result = np.where(np.isfinite(result), result, default)
    else:
        if not np.isfinite(result):
            result = default
            
    return result


def hash_data(data: Any) -> str:
    """
    Generate hash for data caching/versioning
    
    Args:
        data: Data to hash
        
    Returns:
        SHA256 hash string
    """
    if isinstance(data, pd.DataFrame):
        # Hash DataFrame index and values
        content = f"{data.index.tolist()}{data.values.tolist()}"
    elif isinstance(data, np.ndarray):
        content = data.tobytes()
    else:
        content = json.dumps(data, sort_keys=True, default=str)
        
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_market_session(timestamp: datetime) -> str:
    """
    Get market session for a given timestamp
    
    Args:
        timestamp: UTC timestamp
        
    Returns:
        Session name ('london', 'new_york', 'overlap', 'asian', 'closed')
    """
    # Convert to UTC if not already
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    elif timestamp.tzinfo != timezone.utc:
        timestamp = timestamp.astimezone(timezone.utc)
        
    hour = timestamp.hour
    
    # Define session hours in UTC
    if 8 <= hour < 17:  # London: 08:00-17:00 UTC
        if 13 <= hour < 17:  # Overlap with NY: 13:00-17:00 UTC
            return 'overlap'
        else:
            return 'london'
    elif 13 <= hour < 22:  # New York: 13:00-22:00 UTC
        return 'new_york'  
    elif 23 <= hour or hour < 8:  # Asian: 23:00-08:00 UTC
        return 'asian'
    else:
        return 'closed'


def pips_to_price(pips: float, symbol: str = "XAUUSD") -> float:
    """
    Convert pips to price for a given symbol
    
    Args:
        pips: Number of pips
        symbol: Trading symbol
        
    Returns:
        Price equivalent
    """
    # XAUUSD pip value (0.01)
    if symbol.upper() == "XAUUSD":
        return pips * 0.01
    else:
        # Default forex pip (0.0001)
        return pips * 0.0001


def price_to_pips(price_diff: float, symbol: str = "XAUUSD") -> float:
    """
    Convert price difference to pips
    
    Args:
        price_diff: Price difference
        symbol: Trading symbol
        
    Returns:
        Equivalent pips
    """
    # XAUUSD pip value (0.01)
    if symbol.upper() == "XAUUSD":
        return price_diff / 0.01
    else:
        # Default forex pip (0.0001)
        return price_diff / 0.0001


def calculate_position_size(
    account_balance: float,
    risk_pct: float,
    stop_loss_pips: float,
    symbol: str = "XAUUSD"
) -> float:
    """
    Calculate position size based on risk management rules
    
    Args:
        account_balance: Account balance in USD
        risk_pct: Risk percentage (e.g., 0.02 for 2%)
        stop_loss_pips: Stop loss in pips
        symbol: Trading symbol
        
    Returns:
        Position size in lots
    """
    # Risk amount in USD
    risk_amount = account_balance * risk_pct
    
    # Pip value for XAUUSD (approximately $1 per pip for 0.01 lot)
    if symbol.upper() == "XAUUSD":
        pip_value_per_lot = 100  # $100 per pip for 1 lot
    else:
        pip_value_per_lot = 10   # Standard forex
        
    # Calculate position size
    position_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
    
    # Round to reasonable precision
    return round(position_size, 2)


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        _performance_monitor.start_monitoring()
    return _performance_monitor