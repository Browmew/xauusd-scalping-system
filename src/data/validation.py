"""
Data Validation and Quality Control

Comprehensive data validation system for ensuring data quality and integrity
across all data sources. Implements real-time anomaly detection, quality
scoring, and automated data cleaning for the XAUUSD scalping system.

Key Features:
- Real-time data quality monitoring
- Statistical anomaly detection
- Schema validation
- Data completeness checking
- Quality score calculation
- Automated alerting system

Time Complexity: O(n) for most validation operations
Space Complexity: O(1) for streaming validation, O(n) for batch processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import warnings
from abc import ABC, abstractmethod

from src.core.config import get_config_manager
from src.core.utils import Timer, safe_divide
from src.data.ingestion import TickData, OrderBookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Data validation rule definition"""
    name: str
    description: str
    rule_type: str  # 'range', 'statistical', 'pattern', 'business'
    parameters: Dict[str, Any]
    severity: str = 'warning'  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result from data validation"""
    rule_name: str
    passed: bool
    severity: str
    message: str
    affected_records: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Comprehensive data quality metrics"""
    timestamp: datetime
    data_type: str
    total_records: int
    valid_records: int
    completeness_score: float  # 0-1
    accuracy_score: float      # 0-1
    consistency_score: float   # 0-1
    timeliness_score: float    # 0-1
    overall_score: float       # 0-1
    anomalies_detected: int
    validation_results: List[ValidationResult] = field(default_factory=list)


class ValidationRuleEngine:
    """
    Validation rule engine for data quality checks
    
    Applies configurable validation rules to incoming data and provides
    detailed reporting on data quality issues and anomalies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules: List[ValidationRule] = []
        self.load_default_rules()
        
    def load_default_rules(self) -> None:
        """Load default validation rules"""
        
        # Price validation rules
        self.add_rule(ValidationRule(
            name="price_range_check",
            description="Check if prices are within reasonable range",
            rule_type="range",
            parameters={
                'min_price': 1000.0,  # $1000 minimum for gold
                'max_price': 5000.0,  # $5000 maximum for gold
            },
            severity="error"
        ))
        
        self.add_rule(ValidationRule(
            name="price_jump_check",
            description="Detect unrealistic price jumps",
            rule_type="statistical",
            parameters={
                'max_change_pct': 2.0,  # 2% max change between ticks
                'lookback_periods': 10
            },
            severity="warning"
        ))
        
        # Volume validation rules
        self.add_rule(ValidationRule(
            name="volume_range_check",
            description="Check if volume is non-negative and reasonable",
            rule_type="range",
            parameters={
                'min_volume': 0.0,
                'max_volume': 1000.0  # 1000 lots maximum
            },
            severity="error"
        ))
        
        # Timestamp validation rules
        self.add_rule(ValidationRule(
            name="timestamp_order_check",
            description="Check timestamp ordering",
            rule_type="pattern",
            parameters={
                'allow_duplicates': False,
                'max_gap_seconds': 60
            },
            severity="warning"
        ))
        
        # Spread validation rules
        self.add_rule(ValidationRule(
            name="spread_range_check",
            description="Check if spreads are reasonable",
            rule_type="range",
            parameters={
                'min_spread': 0.1,   # 0.1 pip minimum
                'max_spread': 10.0   # 10 pips maximum
            },
            severity="warning"
        ))
        
        # Order book validation rules
        self.add_rule(ValidationRule(
            name="orderbook_integrity_check",
            description="Check order book bid/ask ordering",
            rule_type="business",
            parameters={
                'check_price_ordering': True,
                'check_volume_positive': True
            },
            severity="error"
        ))
        
    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule"""
        self.rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")
        
    def remove_rule(self, rule_name: str) -> bool:
        """Remove validation rule by name"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.info(f"Removed validation rule: {rule_name}")
                return True
        return False
        
    def validate_tick_data(self, ticks: List[TickData]) -> List[ValidationResult]:
        """Validate tick data against all applicable rules"""
        if not ticks:
            return []
            
        results = []
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([{
            'timestamp': tick.timestamp,
            'price': tick.price,
            'volume': tick.volume,
            'side': tick.side
        } for tick in ticks])
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            try:
                result = self._apply_rule_to_ticks(rule, df)
                results.append(result)
            except Exception as e:
                logger.error(f"Error applying rule {rule.name}: {e}")
                results.append(ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity="error",
                    message=f"Rule execution failed: {e}"
                ))
                
        return results
        
    def validate_orderbook_data(self, snapshots: List[OrderBookSnapshot]) -> List[ValidationResult]:
        """Validate order book data"""
        if not snapshots:
            return []
            
        results = []
        
        for rule in self.rules:
            if not rule.enabled or rule.rule_type != 'business':
                continue
                
            try:
                result = self._apply_rule_to_orderbook(rule, snapshots)
                results.append(result)
            except Exception as e:
                logger.error(f"Error applying rule {rule.name}: {e}")
                
        return results
        
    def _apply_rule_to_ticks(self, rule: ValidationRule, df: pd.DataFrame) -> ValidationResult:
        """Apply validation rule to tick data"""
        
        if rule.rule_type == "range":
            return self._apply_range_rule(rule, df)
        elif rule.rule_type == "statistical":
            return self._apply_statistical_rule(rule, df)
        elif rule.rule_type == "pattern":
            return self._apply_pattern_rule(rule, df)
        else:
            return ValidationResult(
                rule_name=rule.name,
                passed=True,
                severity=rule.severity,
                message="Rule type not applicable to tick data"
            )
            
    def _apply_range_rule(self, rule: ValidationRule, df: pd.DataFrame) -> ValidationResult:
        """Apply range validation rule"""
        params = rule.parameters
        violations = 0
        
        if rule.name == "price_range_check":
            violations = ((df['price'] < params['min_price']) | 
                         (df['price'] > params['max_price'])).sum()
        elif rule.name == "volume_range_check":
            violations = ((df['volume'] < params['min_volume']) | 
                         (df['volume'] > params['max_volume'])).sum()
        elif rule.name == "spread_range_check" and 'spread' in df.columns:
            violations = ((df['spread'] < params['min_spread']) | 
                         (df['spread'] > params['max_spread'])).sum()
            
        passed = violations == 0
        message = f"Found {violations} range violations" if violations > 0 else "All values within range"
        
        return ValidationResult(
            rule_name=rule.name,
            passed=passed,
            severity=rule.severity,
            message=message,
            affected_records=violations
        )
        
    def _apply_statistical_rule(self, rule: ValidationRule, df: pd.DataFrame) -> ValidationResult:
        """Apply statistical validation rule"""
        params = rule.parameters
        violations = 0
        
        if rule.name == "price_jump_check" and len(df) > 1:
            price_changes = df['price'].pct_change().abs() * 100
            max_change = params['max_change_pct']
            violations = (price_changes > max_change).sum()
            
        passed = violations == 0
        message = f"Found {violations} statistical anomalies" if violations > 0 else "No statistical anomalies"
        
        return ValidationResult(
            rule_name=rule.name,
            passed=passed,
            severity=rule.severity,
            message=message,
            affected_records=violations
        )
        
    def _apply_pattern_rule(self, rule: ValidationRule, df: pd.DataFrame) -> ValidationResult:
        """Apply pattern validation rule"""
        params = rule.parameters
        violations = 0
        
        if rule.name == "timestamp_order_check":
            # Check for timestamp ordering
            if not df['timestamp'].is_monotonic_increasing:
                violations += 1
                
            # Check for large gaps
            if len(df) > 1:
                time_diffs = df['timestamp'].diff().dt.total_seconds()
                large_gaps = (time_diffs > params['max_gap_seconds']).sum()
                violations += large_gaps
                
        passed = violations == 0
        message = f"Found {violations} pattern violations" if violations > 0 else "Pattern validation passed"
        
        return ValidationResult(
            rule_name=rule.name,
            passed=passed,
            severity=rule.severity,
            message=message,
            affected_records=violations
        )
        
    def _apply_rule_to_orderbook(self, rule: ValidationRule, snapshots: List[OrderBookSnapshot]) -> ValidationResult:
        """Apply validation rule to order book data"""
        violations = 0
        
        if rule.name == "orderbook_integrity_check":
            for snapshot in snapshots:
                # Check bid/ask price ordering
                if snapshot.bids and snapshot.asks:
                    if snapshot.bids[0].price >= snapshot.asks[0].price:
                        violations += 1
                        
                # Check bid price ordering (descending)
                for i in range(len(snapshot.bids) - 1):
                    if snapshot.bids[i].price <= snapshot.bids[i + 1].price:
                        violations += 1
                        
                # Check ask price ordering (ascending)
                for i in range(len(snapshot.asks) - 1):
                    if snapshot.asks[i].price >= snapshot.asks[i + 1].price:
                        violations += 1
                        
                # Check positive volumes
                for level in snapshot.bids + snapshot.asks:
                    if level.size <= 0:
                        violations += 1
                        
        passed = violations == 0
        message = f"Found {violations} order book integrity violations" if violations > 0 else "Order book integrity check passed"
        
        return ValidationResult(
            rule_name=rule.name,
            passed=passed,
            severity=rule.severity,
            message=message,
            affected_records=violations
        )


class AnomalyDetector:
    """
    Real-time anomaly detection for financial data
    
    Uses statistical methods and machine learning to detect anomalies
    in price, volume, and order book data in real-time.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Anomaly detection parameters
        self.zscore_threshold = self.config.get('zscore_threshold', 3.0)
        self.isolation_forest_contamination = self.config.get('contamination', 0.1)
        self.lookback_window = self.config.get('lookback_window', 100)
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=self.isolation_forest_contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Historical data for comparison
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.spread_history: List[float] = []
        
        # Fitted flag
        self.is_fitted = False
        
    def detect_price_anomalies(self, prices: Union[List[float], pd.Series]) -> List[bool]:
        """Detect price anomalies using Z-score method"""
        if len(prices) < 10:
            return [False] * len(prices)
            
        # Convert to numpy array
        prices_array = np.array(prices)
        
        # Calculate rolling Z-scores
        anomalies = []
        
        for i in range(len(prices_array)):
            if i < 10:  # Need minimum data for statistics
                anomalies.append(False)
                continue
                
            # Use last 20 points for statistical calculation
            window_start = max(0, i - 20)
            window_data = prices_array[window_start:i]
            
            if len(window_data) < 5:
                anomalies.append(False)
                continue
                
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std == 0:
                anomalies.append(False)
                continue
                
            zscore = abs(prices_array[i] - mean) / std
            anomalies.append(zscore > self.zscore_threshold)
            
        return anomalies
        
    def detect_volume_anomalies(self, volumes: Union[List[float], pd.Series]) -> List[bool]:
        """Detect volume anomalies using IQR method"""
        if len(volumes) < 10:
            return [False] * len(volumes)
            
        volumes_array = np.array(volumes)
        anomalies = []
        
        for i in range(len(volumes_array)):
            if i < 10:
                anomalies.append(False)
                continue
                
            # Use last 50 points for IQR calculation
            window_start = max(0, i - 50)
            window_data = volumes_array[window_start:i]
            
            if len(window_data) < 10:
                anomalies.append(False)
                continue
                
            Q1 = np.percentile(window_data, 25)
            Q3 = np.percentile(window_data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            is_anomaly = (volumes_array[i] < lower_bound) or (volumes_array[i] > upper_bound)
            anomalies.append(is_anomaly)
            
        return anomalies
        
    def detect_multivariate_anomalies(self, features: pd.DataFrame) -> List[bool]:
        """Detect multivariate anomalies using Isolation Forest"""
        if len(features) < self.lookback_window or features.empty:
            return [False] * len(features)
            
        try:
            # Prepare features
            features_clean = features.fillna(0).replace([np.inf, -np.inf], 0)
            
            if not self.is_fitted and len(features_clean) >= self.lookback_window:
                # Fit the model on recent data
                fit_data = features_clean.iloc[-self.lookback_window:].values
                fit_data_scaled = self.scaler.fit_transform(fit_data)
                self.isolation_forest.fit(fit_data_scaled)
                self.is_fitted = True
                
            if not self.is_fitted:
                return [False] * len(features)
                
            # Detect anomalies
            features_scaled = self.scaler.transform(features_clean.values)
            anomaly_scores = self.isolation_forest.decision_function(features_scaled)
            anomalies = self.isolation_forest.predict(features_scaled) == -1
            
            return anomalies.tolist()
            
        except Exception as e:
            logger.error(f"Error in multivariate anomaly detection: {e}")
            return [False] * len(features)
            
    def update_historical_data(self, price: float, volume: float, spread: float = None) -> None:
        """Update historical data for anomaly detection"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if spread is not None:
            self.spread_history.append(spread)
            
        # Keep only recent history
        max_history = self.lookback_window * 2
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.spread_history = self.spread_history[-max_history:]


class DataValidator:
    """
    Main data validation coordinator
    
    Orchestrates validation rules, anomaly detection, and quality scoring
    to provide comprehensive data quality assessment and monitoring.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Initialize components
        validation_config = self.config.get('data', 'validation', {})
        self.rule_engine = ValidationRuleEngine(validation_config)
        self.anomaly_detector = AnomalyDetector(validation_config)
        
        # Quality tracking
        self.quality_history: List[QualityMetrics] = []
        self.max_history = 1000
        
    def validate_tick_data(self, ticks: List[TickData]) -> QualityMetrics:
        """Comprehensive tick data validation"""
        if not ticks:
            return self._create_empty_metrics("tick_data")
            
        start_time = datetime.now(timezone.utc)
        
        with Timer("tick_validation"):
            # Rule-based validation
            validation_results = self.rule_engine.validate_tick_data(ticks)
            
            # Anomaly detection
            prices = [tick.price for tick in ticks]
            volumes = [tick.volume for tick in ticks]
            
            price_anomalies = self.anomaly_detector.detect_price_anomalies(prices)
            volume_anomalies = self.anomaly_detector.detect_volume_anomalies(volumes)
            
            total_anomalies = sum(price_anomalies) + sum(volume_anomalies)
            
            # Update historical data
            for tick in ticks[-10:]:  # Update with last 10 ticks
                self.anomaly_detector.update_historical_data(tick.price, tick.volume)
            
            # Calculate quality scores
            metrics = self._calculate_quality_metrics(
                data_type="tick_data",
                total_records=len(ticks),
                validation_results=validation_results,
                anomalies_detected=total_anomalies,
                timestamps=[tick.timestamp for tick in ticks]
            )
            
        self._store_quality_metrics(metrics)
        return metrics
        
    def validate_orderbook_data(self, snapshots: List[OrderBookSnapshot]) -> QualityMetrics:
        """Comprehensive order book data validation"""
        if not snapshots:
            return self._create_empty_metrics("orderbook_data")
            
        with Timer("orderbook_validation"):
            # Rule-based validation
            validation_results = self.rule_engine.validate_orderbook_data(snapshots)
            
            # Spread anomaly detection
            spreads = [snapshot.get_spread() for snapshot in snapshots if snapshot.bids and snapshot.asks]
            
            if spreads:
                # Update historical data
                for spread in spreads[-10:]:
                    if len(self.anomaly_detector.spread_history) > 0:
                        last_price = self.anomaly_detector.price_history[-1] if self.anomaly_detector.price_history else 2000
                        last_volume = self.anomaly_detector.volume_history[-1] if self.anomaly_detector.volume_history else 1
                        self.anomaly_detector.update_historical_data(last_price, last_volume, spread)
                        
            # Calculate quality scores
            metrics = self._calculate_quality_metrics(
                data_type="orderbook_data",
                total_records=len(snapshots),
                validation_results=validation_results,
                anomalies_detected=0,  # Implement specific orderbook anomaly detection if needed
                timestamps=[snapshot.timestamp for snapshot in snapshots]
            )
            
        self._store_quality_metrics(metrics)
        return metrics
        
    def validate_dataframe(self, df: pd.DataFrame, data_type: str) -> QualityMetrics:
        """Validate processed DataFrame"""
        if df.empty:
            return self._create_empty_metrics(data_type)
            
        with Timer(f"{data_type}_validation"):
            # Basic data quality checks
            total_records = len(df)
            missing_values = df.isnull().sum().sum()
            infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            
            # Calculate completeness
            total_cells = df.size
            valid_cells = total_cells - missing_values - infinite_values
            completeness_score = safe_divide(valid_cells, total_cells)
            
            # Detect multivariate anomalies if enough numeric data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            anomalies_detected = 0
            
            if len(numeric_cols) >= 3 and len(df) >= 10:
                try:
                    anomalies = self.anomaly_detector.detect_multivariate_anomalies(df[numeric_cols])
                    anomalies_detected = sum(anomalies)
                except Exception as e:
                    logger.warning(f"Failed to detect multivariate anomalies: {e}")
                    
            # Create validation results
            validation_results = []
            
            if missing_values > 0:
                validation_results.append(ValidationResult(
                    rule_name="missing_values_check",
                    passed=False,
                    severity="warning",
                    message=f"Found {missing_values} missing values",
                    affected_records=missing_values
                ))
                
            if infinite_values > 0:
                validation_results.append(ValidationResult(
                    rule_name="infinite_values_check",
                    passed=False,
                    severity="error",
                    message=f"Found {infinite_values} infinite values",
                    affected_records=infinite_values
                ))
                
            # Calculate quality metrics
            metrics = QualityMetrics(
                timestamp=datetime.now(timezone.utc),
                data_type=data_type,
                total_records=total_records,
                valid_records=total_records - missing_values - infinite_values,
                completeness_score=completeness_score,
                accuracy_score=1.0 - (anomalies_detected / max(total_records, 1)),
                consistency_score=1.0,  # Would need more complex logic
                timeliness_score=1.0,   # Assume current data is timely
                overall_score=0.0,      # Will be calculated
                anomalies_detected=anomalies_detected,
                validation_results=validation_results
            )
            
            # Calculate overall score
            metrics.overall_score = (
                0.3 * metrics.completeness_score +
                0.3 * metrics.accuracy_score +
                0.2 * metrics.consistency_score +
                0.2 * metrics.timeliness_score
            )
            
        self._store_quality_metrics(metrics)
        return metrics
        
    def _calculate_quality_metrics(
        self,
        data_type: str,
        total_records: int,
        validation_results: List[ValidationResult],
        anomalies_detected: int,
        timestamps: List[datetime]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        
        # Count validation failures by severity
        error_count = sum(1 for r in validation_results if r.severity == 'error' and not r.passed)
        warning_count = sum(1 for r in validation_results if r.severity == 'warning' and not r.passed)
        
        # Calculate completeness (assume complete if no missing data rule fails)
        missing_data_failures = sum(r.affected_records for r in validation_results 
                                   if 'missing' in r.rule_name.lower() and not r.passed)
        completeness_score = 1.0 - (missing_data_failures / max(total_records, 1))
        
        # Calculate accuracy (inverse of anomaly rate)
        accuracy_score = 1.0 - (anomalies_detected / max(total_records, 1))
        
        # Calculate consistency (inverse of validation failure rate)
        total_failures = sum(r.affected_records for r in validation_results if not r.passed)
        consistency_score = 1.0 - (total_failures / max(total_records, 1))
        
        # Calculate timeliness (based on timestamp gaps)
        timeliness_score = 1.0
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                         for i in range(1, len(timestamps))]
            avg_gap = np.mean(time_diffs)
            expected_gap = 1.0  # 1 second expected gap
            timeliness_score = max(0.0, 1.0 - abs(avg_gap - expected_gap) / expected_gap)
            
        # Calculate overall score
        overall_score = (
            0.3 * completeness_score +
            0.3 * accuracy_score +
            0.2 * consistency_score +
            0.2 * timeliness_score
        )
        
        return QualityMetrics(
            timestamp=datetime.now(timezone.utc),
            data_type=data_type,
            total_records=total_records,
            valid_records=total_records - total_failures,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            anomalies_detected=anomalies_detected,
            validation_results=validation_results
        )
        
    def _create_empty_metrics(self, data_type: str) -> QualityMetrics:
        """Create empty quality metrics for empty data"""
        return QualityMetrics(
            timestamp=datetime.now(timezone.utc),
            data_type=data_type,
            total_records=0,
            valid_records=0,
            completeness_score=0.0,
            accuracy_score=0.0,
            consistency_score=0.0,
            timeliness_score=0.0,
            overall_score=0.0,
            anomalies_detected=0,
            validation_results=[]
        )
        
    def _store_quality_metrics(self, metrics: QualityMetrics) -> None:
        """Store quality metrics in history"""
        self.quality_history.append(metrics)
        
        # Keep only recent history
        if len(self.quality_history) > self.max_history:
            self.quality_history = self.quality_history[-self.max_history:]
            
    def get_quality_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality summary for recent time period"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.quality_history if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {'error': 'No quality metrics available'}
            
        # Aggregate by data type
        summary = {}
        data_types = set(m.data_type for m in recent_metrics)
        
        for data_type in data_types:
            type_metrics = [m for m in recent_metrics if m.data_type == data_type]
            
            summary[data_type] = {
                'count': len(type_metrics),
                'avg_overall_score': np.mean([m.overall_score for m in type_metrics]),
                'avg_completeness': np.mean([m.completeness_score for m in type_metrics]),
                'avg_accuracy': np.mean([m.accuracy_score for m in type_metrics]),
                'total_anomalies': sum(m.anomalies_detected for m in type_metrics),
                'total_validation_failures': sum(len([r for r in m.validation_results if not r.passed]) 
                                                for m in type_metrics)
            }
            
        return summary
        
    def get_current_quality_status(self) -> Dict[str, Any]:
        """Get current data quality status"""
        if not self.quality_history:
            return {'status': 'no_data', 'message': 'No quality metrics available'}
            
        latest_metrics = {}
        for metrics in reversed(self.quality_history):
            if metrics.data_type not in latest_metrics:
                latest_metrics[metrics.data_type] = metrics
                
        overall_scores = [m.overall_score for m in latest_metrics.values()]
        avg_score = np.mean(overall_scores) if overall_scores else 0.0
        
        if avg_score >= 0.9:
            status = 'excellent'
        elif avg_score >= 0.8:
            status = 'good'
        elif avg_score >= 0.7:
            status = 'acceptable'
        elif avg_score >= 0.5:
            status = 'poor'
        else:
            status = 'critical'
            
        return {
            'status': status,
            'overall_score': avg_score,
            'data_types': {k: v.overall_score for k, v in latest_metrics.items()},
            'total_anomalies': sum(m.anomalies_detected for m in latest_metrics.values()),
            'message': f"Data quality is {status} (score: {avg_score:.2f})"
        }