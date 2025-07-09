"""
Base Feature Engineering Framework

Core framework for feature extraction with performance optimization,
caching, and standardized interfaces. Provides the foundation for
all feature extractors in the XAUUSD scalping system.

Architecture:
- BaseFeatureExtractor: Abstract base class for all feature extractors
- FeatureEngine: Main coordinator for feature computation
- FeatureConfig: Configuration management for feature parameters
- FeatureImportance: SHAP-based feature importance analysis

Performance Optimizations:
- Vectorized operations using NumPy/Numba
- Lazy evaluation and caching
- Memory-efficient chunked processing
- Parallel computation for independent features

Time Complexity: O(n) for most features, O(n log n) for sorting-based
Space Complexity: O(n) with configurable memory limits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache, wraps
import hashlib
import pickle
import time
from pathlib import Path
import shap
import joblib

from src.core.config import get_config_manager
from src.core.utils import Timer, safe_divide, hash_data, get_performance_monitor
from src.core.logging import get_perf_logger

logger = logging.getLogger(__name__)
perf_logger = get_perf_logger()


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=highest priority (minimal set), 5=lowest
    category: str = "unknown"
    description: str = ""
    
    # Performance constraints
    max_latency_ms: float = 50.0
    max_memory_mb: float = 100.0
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Validation
    expected_output_shape: Optional[Tuple[int, ...]] = None
    value_range: Optional[Tuple[float, float]] = None


@dataclass
class FeatureImportance:
    """Feature importance metrics from SHAP analysis"""
    feature_name: str
    shap_importance: float
    permutation_importance: float
    correlation_with_target: float
    stability_score: float  # Consistency across time periods
    business_relevance: float  # Manual scoring 0-1
    overall_score: float
    
    def __post_init__(self):
        """Calculate overall score from components"""
        if self.overall_score == 0:
            self.overall_score = (
                0.4 * self.shap_importance +
                0.25 * self.permutation_importance +
                0.15 * abs(self.correlation_with_target) +
                0.1 * self.stability_score +
                0.1 * self.business_relevance
            )


class FeatureCache:
    """
    High-performance feature caching system
    
    Implements intelligent caching with TTL, memory limits, and hash-based
    invalidation to avoid recomputing expensive features.
    """
    
    def __init__(self, max_size_mb: float = 500, default_ttl: float = 3600):
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if datetime.now().timestamp() - entry['created'] < entry['ttl']:
                    self.access_times[key] = datetime.now()
                    return entry['value']
                else:
                    # Expired
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                        
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set cached value"""
        with self.lock:
            # Check memory usage and evict if necessary
            self._evict_if_needed()
            
            self.cache[key] = {
                'value': value,
                'created': datetime.now().timestamp(),
                'ttl': ttl or self.default_ttl,
                'size_bytes': self._estimate_size(value)
            }
            self.access_times[key] = datetime.now()
            
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value"""
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
            
    def _evict_if_needed(self) -> None:
        """Evict old entries if memory limit exceeded"""
        total_size = sum(entry['size_bytes'] for entry in self.cache.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Sort by access time (LRU)
            sorted_keys = sorted(
                self.access_times.keys(),
                key=lambda k: self.access_times[k]
            )
            
            # Remove oldest entries until under limit
            for key in sorted_keys:
                if total_size <= max_size_bytes:
                    break
                    
                if key in self.cache:
                    total_size -= self.cache[key]['size_bytes']
                    del self.cache[key]
                    del self.access_times[key]
                    
    def clear(self) -> None:
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_size = sum(entry['size_bytes'] for entry in self.cache.values())
            
            return {
                'entries': len(self.cache),
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_access_count', 1), 1)
            }


def cached_feature(ttl: float = 3600, cache_instance: Optional[FeatureCache] = None):
    """
    Decorator for caching feature computation results
    
    Args:
        ttl: Time to live in seconds
        cache_instance: Optional cache instance (uses global if None)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__name__}_{hash_data(args)}_{hash_data(kwargs)}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Get cache instance
            cache = cache_instance or getattr(self, '_cache', None)
            if not cache:
                # No caching, execute function directly
                return func(self, *args, **kwargs)
                
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Compute and cache result
            with Timer(f"feature_computation_{func.__name__}"):
                result = func(self, *args, **kwargs)
                cache.set(cache_key, result, ttl)
                
            return result
            
        return wrapper
    return decorator


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors
    
    Provides standard interface, caching, performance monitoring,
    and configuration management for feature computation.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig(name=self.__class__.__name__)
        self.cache = FeatureCache()
        self.performance_monitor = get_performance_monitor()
        
        # Performance tracking
        self.computation_times: List[float] = []
        self.memory_usage: List[float] = []
        
        # Feature metadata
        self.feature_names: List[str] = []
        self.feature_descriptions: Dict[str, str] = {}
        
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from input data
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with computed features
        """
        pass
        
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces"""
        pass
        
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format and requirements
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if validation passes
        """
        if data.empty:
            logger.warning(f"{self.__class__.__name__}: Empty input data")
            return False
            
        # Check for required columns (override in subclasses)
        required_columns = self.get_required_columns()
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            logger.error(f"{self.__class__.__name__}: Missing required columns: {missing_columns}")
            return False
            
        # Check for sufficient data
        min_periods = self.get_min_periods()
        if len(data) < min_periods:
            logger.warning(f"{self.__class__.__name__}: Insufficient data ({len(data)} < {min_periods})")
            return False
            
        return True
        
    def get_required_columns(self) -> List[str]:
        """Get list of required input columns"""
        return ['open', 'high', 'low', 'close', 'volume']
        
    def get_min_periods(self) -> int:
        """Get minimum number of periods required"""
        return 1
        
    def compute_with_performance_tracking(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features with performance tracking
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with computed features
        """
        if not self.validate_input(data):
            return pd.DataFrame()
            
        start_time = time.time()
        start_memory = self.performance_monitor.get_snapshot().memory.rss_mb
        
        try:
            with Timer(f"feature_extraction_{self.__class__.__name__}"):
                features = self.extract_features(data)
                
            # Track performance
            computation_time = time.time() - start_time
            end_memory = self.performance_monitor.get_snapshot().memory.rss_mb
            memory_delta = end_memory - start_memory
            
            self.computation_times.append(computation_time * 1000)  # Convert to ms
            self.memory_usage.append(memory_delta)
            
            # Check performance constraints
            if computation_time * 1000 > self.config.max_latency_ms:
                logger.warning(
                    f"{self.__class__.__name__} exceeded latency limit: "
                    f"{computation_time*1000:.2f}ms > {self.config.max_latency_ms}ms"
                )
                
            if memory_delta > self.config.max_memory_mb:
                logger.warning(
                    f"{self.__class__.__name__} exceeded memory limit: "
                    f"{memory_delta:.2f}MB > {self.config.max_memory_mb}MB"
                )
                
            # Log feature computation
            perf_logger.log_latency(
                f"feature_{self.__class__.__name__}",
                computation_time * 1000
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {self.__class__.__name__}: {e}")
            return pd.DataFrame()
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this extractor"""
        if not self.computation_times:
            return {}
            
        return {
            'avg_latency_ms': np.mean(self.computation_times),
            'max_latency_ms': np.max(self.computation_times),
            'min_latency_ms': np.min(self.computation_times),
            'std_latency_ms': np.std(self.computation_times),
            'avg_memory_mb': np.mean(self.memory_usage),
            'max_memory_mb': np.max(self.memory_usage),
            'total_computations': len(self.computation_times)
        }
        
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics"""
        self.computation_times.clear()
        self.memory_usage.clear()
        
    def add_feature_description(self, feature_name: str, description: str) -> None:
        """Add description for a feature"""
        self.feature_descriptions[feature_name] = description
        
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get all feature descriptions"""
        return self.feature_descriptions.copy()


class FeatureEngine:
    """
    Main feature engineering coordinator
    
    Orchestrates multiple feature extractors, manages dependencies,
    handles caching, and provides unified interface for feature computation.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Feature extractors registry
        self.extractors: Dict[str, BaseFeatureExtractor] = {}
        self.extractor_configs: Dict[str, FeatureConfig] = {}
        
        # Dependency graph
        self.dependency_graph: Dict[str, List[str]] = {}
        self.execution_order: List[str] = []
        
        # Global cache
        self.global_cache = FeatureCache(
            max_size_mb=self.config.get('features', 'parameters.memory_limit_gb', 16) * 1024,
            default_ttl=self.config.get('features', 'parameters.cache_ttl_hours', 24) * 3600
        )
        
        # Performance tracking
        self.feature_importance: Dict[str, FeatureImportance] = {}
        self.computation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Parallel execution
        self.use_parallel = self.config.get('features', 'parameters.compute_parallel', True)
        self.n_workers = self.config.get('features', 'parameters.n_workers', 8)
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
        
    def register_extractor(self, extractor: BaseFeatureExtractor, config: Optional[FeatureConfig] = None) -> None:
        """
        Register a feature extractor
        
        Args:
            extractor: Feature extractor instance
            config: Optional configuration override
        """
        name = extractor.__class__.__name__
        self.extractors[name] = extractor
        
        # Use provided config or extractor's config
        if config:
            self.extractor_configs[name] = config
            extractor.config = config
        else:
            self.extractor_configs[name] = extractor.config
            
        # Set up dependencies
        if extractor.config.depends_on:
            self.dependency_graph[name] = extractor.config.depends_on
        else:
            self.dependency_graph[name] = []
            
        # Update execution order
        self._update_execution_order()
        
        logger.info(f"Registered feature extractor: {name}")
        
    def _update_execution_order(self) -> None:
        """Update execution order based on dependency graph"""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
                
            temp_visited.add(node)
            for dependency in self.dependency_graph.get(node, []):
                if dependency in self.extractors:
                    dfs(dependency)
                    
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
            
        for extractor_name in self.extractors:
            if extractor_name not in visited:
                dfs(extractor_name)
                
        self.execution_order = order
        
    def extract_features(
        self,
        data: pd.DataFrame,
        feature_set: str = "extended",
        selected_extractors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract features using registered extractors
        
        Args:
            data: Input DataFrame with OHLCV data
            feature_set: "minimal" or "extended" feature set
            selected_extractors: Optional list of specific extractors to use
            
        Returns:
            DataFrame with all computed features
        """
        if data.empty:
            logger.warning("Cannot extract features from empty data")
            return pd.DataFrame()
            
        start_time = datetime.now()
        
        # Determine which extractors to use
        if selected_extractors:
            extractors_to_use = selected_extractors
        else:
            # Filter by feature set and priority
            extractors_to_use = []
            for name, config in self.extractor_configs.items():
                if not config.enabled:
                    continue
                    
                if feature_set == "minimal" and config.priority > 2:
                    continue  # Skip low-priority features for minimal set
                elif feature_set == "extended":
                    pass  # Include all enabled extractors
                    
                extractors_to_use.append(name)
                
        # Ensure dependencies are included
        extractors_to_use = self._resolve_dependencies(extractors_to_use)
        
        # Sort by execution order
        extractors_to_use = [name for name in self.execution_order if name in extractors_to_use]
        
        logger.info(f"Computing features using {len(extractors_to_use)} extractors: {extractors_to_use}")
        
        # Compute features
        all_features = pd.DataFrame(index=data.index)
        
        if self.use_parallel and len(extractors_to_use) > 1:
            # Parallel execution (for independent extractors)
            independent_extractors = [name for name in extractors_to_use 
                                    if not self.dependency_graph.get(name, [])]
            dependent_extractors = [name for name in extractors_to_use 
                                  if self.dependency_graph.get(name, [])]
            
            # Execute independent extractors in parallel
            if independent_extractors:
                features_dict = self._extract_features_parallel(data, independent_extractors)
                for name, features in features_dict.items():
                    if not features.empty:
                        all_features = pd.concat([all_features, features], axis=1)
                        
            # Execute dependent extractors sequentially
            for name in dependent_extractors:
                if name in self.extractors:
                    extractor = self.extractors[name]
                    features = extractor.compute_with_performance_tracking(data)
                    if not features.empty:
                        all_features = pd.concat([all_features, features], axis=1)
        else:
            # Sequential execution
            for name in extractors_to_use:
                if name in self.extractors:
                    extractor = self.extractors[name]
                    features = extractor.compute_with_performance_tracking(data)
                    if not features.empty:
                        all_features = pd.concat([all_features, features], axis=1)
                        
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Fill NaN values
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        # Log performance
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Feature extraction completed in {total_time:.2f}ms, {len(all_features.columns)} features")
        
        perf_logger.log_latency("feature_extraction_total", total_time)
        
        return all_features
        
    def _resolve_dependencies(self, extractor_names: List[str]) -> List[str]:
        """Resolve dependencies and return complete list of required extractors"""
        resolved = set(extractor_names)
        queue = list(extractor_names)
        
        while queue:
            current = queue.pop(0)
            dependencies = self.dependency_graph.get(current, [])
            
            for dep in dependencies:
                if dep not in resolved and dep in self.extractors:
                    resolved.add(dep)
                    queue.append(dep)
                    
        return list(resolved)
        
    def _extract_features_parallel(self, data: pd.DataFrame, extractor_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Extract features in parallel for independent extractors"""
        futures = {}
        
        # Submit tasks
        for name in extractor_names:
            if name in self.extractors:
                extractor = self.extractors[name]
                future = self.executor.submit(extractor.compute_with_performance_tracking, data)
                futures[name] = future
                
        # Collect results
        results = {}
        for name, future in futures.items():
            try:
                features = future.result(timeout=30)  # 30 second timeout
                results[name] = features
            except Exception as e:
                logger.error(f"Parallel feature extraction failed for {name}: {e}")
                results[name] = pd.DataFrame()
                
        return results
        
    def get_minimal_feature_set(self) -> List[str]:
        """Get the minimal feature set for low-latency operation"""
        minimal_features = self.config.get('features', 'minimal_features', [])
        
        # Also include features marked as priority 1
        for name, config in self.extractor_configs.items():
            if config.priority == 1 and config.enabled:
                extractor = self.extractors.get(name)
                if extractor:
                    minimal_features.extend(extractor.get_feature_names())
                    
        return list(set(minimal_features))  # Remove duplicates
        
    def compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any = None
    ) -> Dict[str, FeatureImportance]:
        """
        Compute comprehensive feature importance using SHAP and other methods
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model for SHAP analysis
            
        Returns:
            Dictionary mapping feature names to importance metrics
        """
        if X.empty or y.empty:
            return {}
            
        feature_importance = {}
        
        try:
            # SHAP importance (if model provided)
            shap_values = None
            if model is not None:
                try:
                    if hasattr(model, 'predict'):
                        explainer = shap.Explainer(model)
                        shap_values = explainer(X.values[:1000])  # Sample for performance
                        shap_importance = np.abs(shap_values.values).mean(axis=0)
                    else:
                        shap_importance = np.zeros(len(X.columns))
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {e}")
                    shap_importance = np.zeros(len(X.columns))
            else:
                shap_importance = np.zeros(len(X.columns))
                
            # Permutation importance (simplified)
            perm_importance = self._compute_permutation_importance(X, y)
            
            # Correlation with target
            correlations = X.corrwith(y).abs().fillna(0)
            
            # Stability score (variance of importance across time windows)
            stability_scores = self._compute_stability_scores(X, y)
            
            # Combine into FeatureImportance objects
            for i, feature_name in enumerate(X.columns):
                importance = FeatureImportance(
                    feature_name=feature_name,
                    shap_importance=shap_importance[i] if i < len(shap_importance) else 0.0,
                    permutation_importance=perm_importance.get(feature_name, 0.0),
                    correlation_with_target=correlations.get(feature_name, 0.0),
                    stability_score=stability_scores.get(feature_name, 0.5),
                    business_relevance=self._get_business_relevance(feature_name),
                    overall_score=0.0  # Will be calculated in __post_init__
                )
                feature_importance[feature_name] = importance
                
            self.feature_importance = feature_importance
            logger.info(f"Computed feature importance for {len(feature_importance)} features")
            
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            
        return feature_importance
        
    def _compute_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute simplified permutation importance"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        # Train simple model
        model = LinearRegression()
        model.fit(X.fillna(0), y)
        baseline_score = mean_squared_error(y, model.predict(X.fillna(0)))
        
        importance = {}
        for feature in X.columns:
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            permuted_score = mean_squared_error(y, model.predict(X_permuted.fillna(0)))
            importance[feature] = max(0, permuted_score - baseline_score) / baseline_score
            
        return importance
        
    def _compute_stability_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute feature stability across time windows"""
        # For now, return default scores
        # In production, implement rolling window analysis
        return {col: 0.5 for col in X.columns}
        
    def _get_business_relevance(self, feature_name: str) -> float:
        """Get business relevance score for feature"""
        # Order flow features get highest relevance (per research)
        if any(keyword in feature_name.lower() for keyword in ['obi', 'imbalance', 'microprice', 'depth']):
            return 1.0
        elif any(keyword in feature_name.lower() for keyword in ['volume', 'vpvr', 'cvd']):
            return 0.9
        elif any(keyword in feature_name.lower() for keyword in ['hma', 'gkyz', 'atr']):
            return 0.8
        elif any(keyword in feature_name.lower() for keyword in ['ema', 'sma', 'vwap']):
            return 0.6
        elif any(keyword in feature_name.lower() for keyword in ['rsi', 'macd', 'stoch']):
            return 0.4  # Traditional TA gets lower scores per research
        else:
            return 0.5
            
    def prune_features(self, importance_threshold: float = 0.01) -> List[str]:
        """
        Prune low-importance features based on threshold
        
        Args:
            importance_threshold: Minimum importance score to keep feature
            
        Returns:
            List of features to keep
        """
        if not self.feature_importance:
            logger.warning("No feature importance data available for pruning")
            return []
            
        # Sort by overall importance score
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        # Filter by threshold
        kept_features = [
            name for name, importance in sorted_features
            if importance.overall_score >= importance_threshold
        ]
        
        pruned_count = len(self.feature_importance) - len(kept_features)
        logger.info(f"Feature pruning: kept {len(kept_features)}, pruned {pruned_count}")
        
        return kept_features
        
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering statistics"""
        total_extractors = len(self.extractors)
        enabled_extractors = sum(1 for config in self.extractor_configs.values() if config.enabled)
        
        # Performance stats
        avg_latencies = {}
        for name, extractor in self.extractors.items():
            stats = extractor.get_performance_stats()
            if stats:
                avg_latencies[name] = stats.get('avg_latency_ms', 0)
                
        return {
            'total_extractors': total_extractors,
            'enabled_extractors': enabled_extractors,
            'avg_latencies_ms': avg_latencies,
            'cache_stats': self.global_cache.get_stats(),
            'feature_importance_computed': len(self.feature_importance),
            'execution_order': self.execution_order
        }
        
    def save_feature_importance(self, filepath: str) -> None:
        """Save feature importance to file"""
        try:
            # Convert to serializable format
            importance_data = {
                name: {
                    'shap_importance': imp.shap_importance,
                    'permutation_importance': imp.permutation_importance,
                    'correlation_with_target': imp.correlation_with_target,
                    'stability_score': imp.stability_score,
                    'business_relevance': imp.business_relevance,
                    'overall_score': imp.overall_score
                }
                for name, imp in self.feature_importance.items()
            }
            
            joblib.dump(importance_data, filepath)
            logger.info(f"Saved feature importance to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save feature importance: {e}")
            
    def load_feature_importance(self, filepath: str) -> bool:
        """Load feature importance from file"""
        try:
            importance_data = joblib.load(filepath)
            
            # Convert back to FeatureImportance objects
            self.feature_importance = {}
            for name, data in importance_data.items():
                self.feature_importance[name] = FeatureImportance(
                    feature_name=name,
                    shap_importance=data['shap_importance'],
                    permutation_importance=data['permutation_importance'],
                    correlation_with_target=data['correlation_with_target'],
                    stability_score=data['stability_score'],
                    business_relevance=data['business_relevance'],
                    overall_score=data['overall_score']
                )
                
            logger.info(f"Loaded feature importance from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load feature importance: {e}")
            return False