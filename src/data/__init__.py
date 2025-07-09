"""
Data Layer for XAUUSD Scalping System

High-performance data ingestion, processing, and storage system optimized
for tick-level and Level 2 orderbook data. Supports multiple data sources,
real-time streaming, and millisecond-precision alignment.

Key Components:
- Tick data ingestion from CME GC futures
- Level 2 orderbook processing (5 levels)
- Cross-asset data (DXY, TIPS, SPY)
- News calendar integration
- Dynamic spread modeling
- Data validation and quality control
- Storage optimization with compression

Performance Targets:
- < 1ms data ingestion latency
- 100MB/day storage efficiency
- 99.9% data quality score
"""

from .ingestion import (
    DataIngestionEngine,
    TickDataProvider,
    OrderBookProvider,
    NewsProvider
)

from .preprocessing import (
    DataPreprocessor,
    TickProcessor,
    OrderBookProcessor,
    CrossAssetProcessor
)

from .validation import (
    DataValidator,
    QualityMetrics,
    AnomalyDetector
)

from .storage import (
    DataStorage,
    TickStorage,
    OrderBookStorage,
    FeatureStorage
)

__all__ = [
    "DataIngestionEngine",
    "TickDataProvider", 
    "OrderBookProvider",
    "NewsProvider",
    "DataPreprocessor",
    "TickProcessor",
    "OrderBookProcessor", 
    "CrossAssetProcessor",
    "DataValidator",
    "QualityMetrics",
    "AnomalyDetector",
    "DataStorage",
    "TickStorage",
    "OrderBookStorage", 
    "FeatureStorage"
]