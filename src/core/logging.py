"""
Structured Logging System

High-performance structured logging for the XAUUSD scalping system.
Provides millisecond precision timestamps, JSON formatting, and 
performance-optimized logging for low-latency trading operations.

Features:
- Structured JSON logging with contextual information
- Performance monitoring integration
- Trade and signal logging
- Configurable log levels and outputs
- Log rotation and retention
- Metrics extraction for monitoring

Time Complexity: O(1) for log operations
Space Complexity: O(1) per log entry
"""

import logging
import logging.handlers
import structlog
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
import psutil
import os
from contextlib import contextmanager
import time
import threading
from dataclasses import dataclass, asdict


@dataclass
class TradeLogEntry:
    """Structured trade log entry"""
    timestamp: datetime
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fill_price: Optional[float] = None
    pnl: Optional[float] = None
    commission: Optional[float] = None
    slippage_pips: Optional[float] = None
    latency_ms: Optional[float] = None
    strategy_signal: Optional[str] = None
    confidence: Optional[float] = None
    
    
@dataclass 
class SignalLogEntry:
    """Structured signal log entry"""
    timestamp: datetime
    signal_id: str
    symbol: str
    signal_type: str
    probability: float
    features: Dict[str, float]
    session: str
    volatility_regime: str
    spread_pips: float
    latency_ms: float
    action_taken: str


@dataclass
class PerformanceLogEntry:
    """System performance log entry"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    latency_ms: float
    throughput_msgs_sec: float
    active_positions: int
    pnl_today: float
    drawdown_pct: float


class ContextualFilter(logging.Filter):
    """Add contextual information to log records"""
    
    def filter(self, record):
        # Add system information
        record.pid = os.getpid()
        record.thread_id = threading.get_ident()
        
        # Add memory and CPU info for performance logs
        if hasattr(record, 'log_type') and record.log_type == 'performance':
            process = psutil.Process()
            record.cpu_percent = process.cpu_percent()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        # Base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
                
        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """High-performance logger for trading operations"""
    
    def __init__(self, name: str = "trading"):
        self.logger = structlog.get_logger(name)
        self._start_time = None
        
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.logger.info(
                "operation_completed",
                operation=operation,
                duration_ms=duration_ms
            )
    
    def log_trade(self, trade_entry: TradeLogEntry):
        """Log a trade with structured format"""
        self.logger.info(
            "trade_executed",
            log_type="trade",
            **asdict(trade_entry)
        )
        
    def log_signal(self, signal_entry: SignalLogEntry):
        """Log a trading signal"""
        self.logger.info(
            "signal_generated", 
            log_type="signal",
            **asdict(signal_entry)
        )
        
    def log_performance(self, perf_entry: PerformanceLogEntry):
        """Log system performance metrics"""
        self.logger.info(
            "performance_snapshot",
            log_type="performance", 
            **asdict(perf_entry)
        )
        
    def log_latency(self, operation: str, latency_ms: float):
        """Log latency measurement"""
        self.logger.info(
            "latency_measurement",
            log_type="latency",
            operation=operation,
            latency_ms=latency_ms
        )
        
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        context = context or {}
        self.logger.error(
            "error_occurred",
            log_type="error",
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
            exc_info=error
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    structured: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10
) -> PerformanceLogger:
    """
    Setup structured logging system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console_output: Enable console logging
        file_output: Enable file logging
        structured: Use structured JSON logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        PerformanceLogger instance
    """
    
    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if structured else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup formatters
    if structured:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(ContextualFilter())
        root_logger.addHandler(console_handler)
    
    # File handlers
    if file_output:
        # Main log file
        main_handler = logging.handlers.RotatingFileHandler(
            log_dir_path / "trading.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        main_handler.setFormatter(formatter)
        main_handler.addFilter(ContextualFilter())
        root_logger.addHandler(main_handler)
        
        # Trade-specific log file
        trade_handler = logging.handlers.RotatingFileHandler(
            log_dir_path / "trades.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        trade_handler.setFormatter(formatter)
        trade_handler.addFilter(ContextualFilter())
        
        # Add filter for trade logs only
        class TradeFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'log_type') and record.log_type in ['trade', 'signal']
        
        trade_handler.addFilter(TradeFilter())
        root_logger.addHandler(trade_handler)
        
        # Performance log file
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir_path / "performance.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        perf_handler.setFormatter(formatter)
        
        # Add filter for performance logs only
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'log_type') and record.log_type in ['performance', 'latency']
                
        perf_handler.addFilter(PerformanceFilter())
        root_logger.addHandler(perf_handler)
        
        # Error log file  
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir_path / "errors.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Financial data loggers
    logging.getLogger("data.ingestion").setLevel(logging.INFO)
    logging.getLogger("features").setLevel(logging.INFO)
    logging.getLogger("models").setLevel(logging.INFO)
    logging.getLogger("backtesting").setLevel(logging.INFO)
    logging.getLogger("live").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        log_level=log_level,
        log_dir=str(log_dir_path),
        console_output=console_output,
        file_output=file_output,
        structured=structured
    )
    
    return PerformanceLogger()


class LogMetricsCollector:
    """Collect metrics from log files for monitoring"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        
    def get_trade_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Extract trade metrics from logs"""
        trade_log = self.log_dir / "trades.log"
        if not trade_log.exists():
            return {}
            
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'avg_latency_ms': 0.0,
            'avg_slippage_pips': 0.0
        }
        
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        
        try:
            with open(trade_log, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        timestamp = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                        
                        if timestamp.timestamp() < cutoff_time:
                            continue
                            
                        if log_entry.get('log_type') == 'trade':
                            metrics['total_trades'] += 1
                            
                            pnl = log_entry.get('pnl', 0)
                            if pnl > 0:
                                metrics['winning_trades'] += 1
                            elif pnl < 0:
                                metrics['losing_trades'] += 1
                                
                            metrics['total_pnl'] += pnl
                            
                            if 'latency_ms' in log_entry:
                                metrics['avg_latency_ms'] += log_entry['latency_ms']
                                
                            if 'slippage_pips' in log_entry:
                                metrics['avg_slippage_pips'] += log_entry['slippage_pips']
                                
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
            # Calculate averages
            if metrics['total_trades'] > 0:
                metrics['avg_latency_ms'] /= metrics['total_trades']
                metrics['avg_slippage_pips'] /= metrics['total_trades']
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
            else:
                metrics['win_rate'] = 0.0
                
        except Exception as e:
            logging.error(f"Error collecting trade metrics: {e}")
            
        return metrics
        
    def get_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Extract performance metrics from logs"""
        perf_log = self.log_dir / "performance.log"
        if not perf_log.exists():
            return {}
            
        metrics = {
            'avg_cpu_percent': 0.0,
            'avg_memory_mb': 0.0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'samples': 0
        }
        
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        
        try:
            with open(perf_log, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        timestamp = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                        
                        if timestamp.timestamp() < cutoff_time:
                            continue
                            
                        if log_entry.get('log_type') in ['performance', 'latency']:
                            metrics['samples'] += 1
                            
                            if 'cpu_percent' in log_entry:
                                metrics['avg_cpu_percent'] += log_entry['cpu_percent']
                                
                            if 'memory_mb' in log_entry:
                                metrics['avg_memory_mb'] += log_entry['memory_mb']
                                
                            if 'latency_ms' in log_entry:
                                latency = log_entry['latency_ms']
                                metrics['avg_latency_ms'] += latency
                                metrics['max_latency_ms'] = max(metrics['max_latency_ms'], latency)
                                
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
            # Calculate averages
            if metrics['samples'] > 0:
                metrics['avg_cpu_percent'] /= metrics['samples']
                metrics['avg_memory_mb'] /= metrics['samples'] 
                metrics['avg_latency_ms'] /= metrics['samples']
                
        except Exception as e:
            logging.error(f"Error collecting performance metrics: {e}")
            
        return metrics


# Global performance logger instance
_perf_logger: Optional[PerformanceLogger] = None


def get_perf_logger() -> PerformanceLogger:
    """Get or create global performance logger"""
    global _perf_logger
    if _perf_logger is None:
        _perf_logger = setup_logging()
    return _perf_logger