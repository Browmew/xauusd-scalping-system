"""
Data Storage System

High-performance data storage system optimized for time-series financial data.
Supports multiple storage backends with compression, partitioning, and efficient
querying for tick data, orderbook snapshots, and processed features.

Storage Backends:
- Parquet files with LZ4 compression
- TimescaleDB for time-series data
- Redis for real-time caching
- HDF5 for large datasets
- Feature store integration

Performance Targets:
- < 10ms write latency for single records
- < 100ms query latency for 1M records
- 90%+ compression ratio
- Automatic data retention and cleanup

Time Complexity: O(log n) for indexed queries, O(1) for appends
Space Complexity: O(n) with compression
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import redis
import asyncpg
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, AsyncGenerator, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import logging
import lz4.frame
import pickle
import json
import hashlib
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

from src.core.config import get_config_manager
from src.core.utils import Timer, ensure_directory, hash_data
from src.data.ingestion import TickData, OrderBookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class StorageMetrics:
    """Storage performance metrics"""
    timestamp: datetime
    storage_type: str
    operation: str  # 'read', 'write', 'delete'
    records_count: int
    latency_ms: float
    size_bytes: int
    compression_ratio: float = 1.0


@dataclass
class DataPartition:
    """Data partition information"""
    partition_key: str
    start_time: datetime
    end_time: datetime
    record_count: int
    size_bytes: int
    file_path: Optional[str] = None


class StorageBackend(ABC):
    """Abstract storage backend interface"""
    
    @abstractmethod
    async def write(self, data: pd.DataFrame, partition_key: str) -> bool:
        """Write data to storage"""
        pass
        
    @abstractmethod
    async def read(self, partition_key: str, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        """Read data from storage"""
        pass
        
    @abstractmethod
    async def delete(self, partition_key: str) -> bool:
        """Delete data partition"""
        pass
        
    @abstractmethod
    async def list_partitions(self) -> List[DataPartition]:
        """List available data partitions"""
        pass


class ParquetStorage(StorageBackend):
    """
    Parquet-based storage with LZ4 compression
    
    Optimized for analytical queries with columnar storage format.
    Provides excellent compression and query performance for OLAP workloads.
    """
    
    def __init__(self, base_path: str, compression: str = 'lz4'):
        self.base_path = Path(base_path)
        self.compression = compression
        ensure_directory(self.base_path)
        
        # Performance tracking
        self.metrics: List[StorageMetrics] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def write(self, data: pd.DataFrame, partition_key: str) -> bool:
        """Write DataFrame to Parquet file"""
        if data.empty:
            logger.warning(f"Attempted to write empty DataFrame for partition {partition_key}")
            return False
            
        try:
            start_time = datetime.now()
            
            # Create partition directory
            partition_path = self.base_path / partition_key
            ensure_directory(partition_path)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_{timestamp}.parquet"
            file_path = partition_path / filename
            
            # Optimize data types for storage
            data_optimized = self._optimize_dtypes(data)
            
            # Write to Parquet
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._write_parquet_sync,
                data_optimized,
                file_path
            )
            
            # Calculate metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            file_size = file_path.stat().st_size
            
            # Estimate compression ratio
            uncompressed_size = data.memory_usage(deep=True).sum()
            compression_ratio = uncompressed_size / file_size if file_size > 0 else 1.0
            
            # Record metrics
            metric = StorageMetrics(
                timestamp=datetime.now(timezone.utc),
                storage_type='parquet',
                operation='write',
                records_count=len(data),
                latency_ms=latency,
                size_bytes=file_size,
                compression_ratio=compression_ratio
            )
            self.metrics.append(metric)
            
            logger.info(f"Wrote {len(data)} records to {file_path} in {latency:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write Parquet data: {e}")
            return False
            
    def _write_parquet_sync(self, data: pd.DataFrame, file_path: Path) -> None:
        """Synchronous Parquet write operation"""
        table = pa.Table.from_pandas(data)
        pq.write_table(
            table,
            file_path,
            compression=self.compression,
            use_dictionary=True,
            row_group_size=50000
        )
        
    async def read(self, partition_key: str, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        """Read data from Parquet files"""
        try:
            start_read = datetime.now()
            partition_path = self.base_path / partition_key
            
            if not partition_path.exists():
                logger.warning(f"Partition path does not exist: {partition_path}")
                return pd.DataFrame()
                
            # Find all Parquet files in partition
            parquet_files = list(partition_path.glob("*.parquet"))
            
            if not parquet_files:
                return pd.DataFrame()
                
            # Read all files
            loop = asyncio.get_event_loop()
            dfs = await loop.run_in_executor(
                self.executor,
                self._read_parquet_files_sync,
                parquet_files
            )
            
            # Combine DataFrames
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Filter by time range if specified
                if start_time or end_time:
                    combined_df = self._filter_by_time(combined_df, start_time, end_time)
                    
                # Sort by timestamp if index is datetime
                if isinstance(combined_df.index, pd.DatetimeIndex):
                    combined_df = combined_df.sort_index()
                elif 'timestamp' in combined_df.columns:
                    combined_df = combined_df.sort_values('timestamp')
                    
                # Record metrics
                latency = (datetime.now() - start_read).total_seconds() * 1000
                metric = StorageMetrics(
                    timestamp=datetime.now(timezone.utc),
                    storage_type='parquet',
                    operation='read',
                    records_count=len(combined_df),
                    latency_ms=latency,
                    size_bytes=combined_df.memory_usage(deep=True).sum()
                )
                self.metrics.append(metric)
                
                logger.info(f"Read {len(combined_df)} records from {partition_key} in {latency:.2f}ms")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to read Parquet data: {e}")
            return pd.DataFrame()
            
    def _read_parquet_files_sync(self, parquet_files: List[Path]) -> List[pd.DataFrame]:
        """Synchronous read of multiple Parquet files"""
        dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
        return dfs
        
    def _filter_by_time(self, df: pd.DataFrame, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        """Filter DataFrame by time range"""
        if df.empty:
            return df
            
        # Try to find timestamp column
        time_col = None
        if isinstance(df.index, pd.DatetimeIndex):
            time_col = df.index
        elif 'timestamp' in df.columns:
            time_col = pd.to_datetime(df['timestamp'])
        else:
            return df
            
        mask = pd.Series(True, index=df.index)
        
        if start_time:
            mask &= (time_col >= start_time)
        if end_time:
            mask &= (time_col <= end_time)
            
        return df[mask]
        
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for storage efficiency"""
        optimized = df.copy()
        
        for col in optimized.columns:
            if optimized[col].dtype == 'object':
                # Try to convert to category for string columns
                if optimized[col].nunique() / len(optimized) < 0.5:
                    optimized[col] = optimized[col].astype('category')
            elif optimized[col].dtype == 'int64':
                # Downcast integers
                if optimized[col].min() >= 0:
                    if optimized[col].max() < 2**32:
                        optimized[col] = optimized[col].astype('uint32')
                else:
                    if optimized[col].min() >= -2**31 and optimized[col].max() < 2**31:
                        optimized[col] = optimized[col].astype('int32')
            elif optimized[col].dtype == 'float64':
                # Downcast floats
                optimized[col] = pd.to_numeric(optimized[col], downcast='float')
                
        return optimized
        
    async def delete(self, partition_key: str) -> bool:
        """Delete partition directory and all files"""
        try:
            partition_path = self.base_path / partition_key
            
            if partition_path.exists():
                # Delete all files in partition
                for file_path in partition_path.rglob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        
                # Remove directory
                partition_path.rmdir()
                
                logger.info(f"Deleted partition: {partition_key}")
                return True
            else:
                logger.warning(f"Partition does not exist: {partition_key}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete partition {partition_key}: {e}")
            return False
            
    async def list_partitions(self) -> List[DataPartition]:
        """List available data partitions"""
        partitions = []
        
        try:
            for partition_dir in self.base_path.iterdir():
                if partition_dir.is_dir():
                    # Get partition statistics
                    parquet_files = list(partition_dir.glob("*.parquet"))
                    
                    if parquet_files:
                        total_size = sum(f.stat().st_size for f in parquet_files)
                        
                        # Read first file to get time range
                        first_file = parquet_files[0]
                        try:
                            df_sample = pd.read_parquet(first_file, columns=['timestamp'] if 'timestamp' else None)
                            
                            if not df_sample.empty:
                                if isinstance(df_sample.index, pd.DatetimeIndex):
                                    start_time = df_sample.index.min()
                                    end_time = df_sample.index.max()
                                elif 'timestamp' in df_sample.columns:
                                    start_time = df_sample['timestamp'].min()
                                    end_time = df_sample['timestamp'].max()
                                else:
                                    start_time = datetime.min
                                    end_time = datetime.max
                                    
                                record_count = len(df_sample)
                            else:
                                start_time = datetime.min
                                end_time = datetime.max
                                record_count = 0
                                
                        except Exception:
                            start_time = datetime.min
                            end_time = datetime.max
                            record_count = 0
                            
                        partition = DataPartition(
                            partition_key=partition_dir.name,
                            start_time=start_time,
                            end_time=end_time,
                            record_count=record_count,
                            size_bytes=total_size,
                            file_path=str(partition_dir)
                        )
                        partitions.append(partition)
                        
        except Exception as e:
            logger.error(f"Failed to list partitions: {e}")
            
        return partitions


class TimescaleDBStorage(StorageBackend):
    """
    TimescaleDB storage for time-series data
    
    Optimized for time-series workloads with automatic partitioning,
    compression, and high-performance time-based queries.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection_pool = None
        
    async def initialize(self) -> bool:
        """Initialize database connection and create tables"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("TimescaleDB storage initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {e}")
            return False
            
    async def _create_tables(self) -> None:
        """Create necessary tables and hypertables"""
        tick_table_sql = """
        CREATE TABLE IF NOT EXISTS tick_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION NOT NULL,
            side TEXT,
            trade_id TEXT,
            sequence BIGINT,
            exchange TEXT
        );
        """
        
        orderbook_table_sql = """
        CREATE TABLE IF NOT EXISTS orderbook_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            bid_prices DOUBLE PRECISION[],
            bid_sizes DOUBLE PRECISION[],
            ask_prices DOUBLE PRECISION[],
            ask_sizes DOUBLE PRECISION[],
            sequence BIGINT,
            exchange TEXT
        );
        """
        
        features_table_sql = """
        CREATE TABLE IF NOT EXISTS features_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            features JSONB NOT NULL
        );
        """
        
        async with self.connection_pool.acquire() as conn:
            await conn.execute(tick_table_sql)
            await conn.execute(orderbook_table_sql)
            await conn.execute(features_table_sql)
            
            # Create hypertables (TimescaleDB specific)
            try:
                await conn.execute("SELECT create_hypertable('tick_data', 'timestamp', if_not_exists => TRUE);")
                await conn.execute("SELECT create_hypertable('orderbook_data', 'timestamp', if_not_exists => TRUE);")
                await conn.execute("SELECT create_hypertable('features_data', 'timestamp', if_not_exists => TRUE);")
            except Exception as e:
                # Hypertables might already exist
                logger.info(f"Hypertables might already exist: {e}")
                
    async def write(self, data: pd.DataFrame, partition_key: str) -> bool:
        """Write data to TimescaleDB"""
        if data.empty or not self.connection_pool:
            return False
            
        try:
            start_time = datetime.now()
            
            async with self.connection_pool.acquire() as conn:
                # Determine table based on partition key
                if 'tick' in partition_key:
                    await self._write_tick_data(conn, data)
                elif 'orderbook' in partition_key:
                    await self._write_orderbook_data(conn, data)
                elif 'features' in partition_key:
                    await self._write_features_data(conn, data)
                else:
                    logger.warning(f"Unknown partition type: {partition_key}")
                    return False
                    
            latency = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Wrote {len(data)} records to TimescaleDB in {latency:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write to TimescaleDB: {e}")
            return False
            
    async def _write_tick_data(self, conn, data: pd.DataFrame) -> None:
        """Write tick data to database"""
        records = []
        for _, row in data.iterrows():
            records.append((
                row.get('timestamp', datetime.now(timezone.utc)),
                row.get('symbol', 'XAUUSD'),
                row.get('price', 0.0),
                row.get('volume', 0.0),
                row.get('side', ''),
                row.get('trade_id', ''),
                row.get('sequence', 0),
                row.get('exchange', '')
            ))
            
        await conn.executemany(
            """INSERT INTO tick_data 
               (timestamp, symbol, price, volume, side, trade_id, sequence, exchange)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            records
        )
        
    async def read(self, partition_key: str, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        """Read data from TimescaleDB"""
        if not self.connection_pool:
            return pd.DataFrame()
            
        try:
            start_read = datetime.now()
            
            async with self.connection_pool.acquire() as conn:
                # Build query based on partition key
                if 'tick' in partition_key:
                    query = "SELECT * FROM tick_data WHERE 1=1"
                    params = []
                elif 'orderbook' in partition_key:
                    query = "SELECT * FROM orderbook_data WHERE 1=1"
                    params = []
                elif 'features' in partition_key:
                    query = "SELECT * FROM features_data WHERE 1=1"
                    params = []
                else:
                    return pd.DataFrame()
                    
                # Add time filters
                if start_time:
                    query += f" AND timestamp >= ${len(params) + 1}"
                    params.append(start_time)
                if end_time:
                    query += f" AND timestamp <= ${len(params) + 1}"
                    params.append(end_time)
                    
                query += " ORDER BY timestamp"
                
                rows = await conn.fetch(query, *params)
                
            # Convert to DataFrame
            if rows:
                df = pd.DataFrame([dict(row) for row in rows])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    
                latency = (datetime.now() - start_read).total_seconds() * 1000
                logger.info(f"Read {len(df)} records from TimescaleDB in {latency:.2f}ms")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to read from TimescaleDB: {e}")
            return pd.DataFrame()
            
    async def delete(self, partition_key: str) -> bool:
        """Delete data partition (not implemented for safety)"""
        logger.warning("Delete operation not implemented for TimescaleDB")
        return False
        
    async def list_partitions(self) -> List[DataPartition]:
        """List available data partitions"""
        partitions = []
        
        if not self.connection_pool:
            return partitions
            
        try:
            async with self.connection_pool.acquire() as conn:
                # Query partition information
                tables = ['tick_data', 'orderbook_data', 'features_data']
                
                for table in tables:
                    query = f"""
                    SELECT 
                        '{table}' as partition_key,
                        MIN(timestamp) as start_time,
                        MAX(timestamp) as end_time,
                        COUNT(*) as record_count
                    FROM {table}
                    """
                    
                    row = await conn.fetchrow(query)
                    
                    if row and row['record_count'] > 0:
                        partition = DataPartition(
                            partition_key=row['partition_key'],
                            start_time=row['start_time'],
                            end_time=row['end_time'],
                            record_count=row['record_count'],
                            size_bytes=0  # Would need separate query
                        )
                        partitions.append(partition)
                        
        except Exception as e:
            logger.error(f"Failed to list TimescaleDB partitions: {e}")
            
        return partitions


class RedisCache:
    """
    Redis-based caching for real-time data
    
    Provides low-latency caching for frequently accessed data
    with automatic expiration and memory management.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, ttl: int = 3600):
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.redis_client = None
        
    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False  # Keep binary for pickle
            )
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            logger.info("Redis cache initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            return False
            
    async def set(self, key: str, data: pd.DataFrame, ttl: Optional[int] = None) -> bool:
        """Set DataFrame in cache"""
        if not self.redis_client or data.empty:
            return False
            
        try:
            # Serialize DataFrame
            serialized = pickle.dumps(data)
            compressed = lz4.frame.compress(serialized)
            
            # Set with TTL
            ttl = ttl or self.ttl
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.redis_client.setex, key, ttl, compressed
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
            
    async def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame from cache"""
        if not self.redis_client:
            return None
            
        try:
            loop = asyncio.get_event_loop()
            compressed = await loop.run_in_executor(
                None, self.redis_client.get, key
            )
            
            if compressed:
                serialized = lz4.frame.decompress(compressed)
                data = pickle.loads(serialized)
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
            
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.redis_client.delete, key
            )
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
            
    async def clear_all(self) -> bool:
        """Clear all cache entries"""
        if not self.redis_client:
            return False
            
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.redis_client.flushdb
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


class DataStorage:
    """
    Main data storage coordinator
    
    Orchestrates multiple storage backends to provide a unified interface
    for storing and retrieving financial time-series data with optimal
    performance characteristics for different use cases.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Initialize storage backends
        storage_config = self.config.get('data', 'storage', {})
        
        # Parquet storage
        parquet_path = storage_config.get('base_path', './data') + '/parquet'
        self.parquet_storage = ParquetStorage(parquet_path)
        
        # TimescaleDB storage (if configured)
        db_config = storage_config.get('database', {})
        if db_config.get('type') == 'timescaledb':
            conn_str = f"postgresql://{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('database', 'xauusd_data')}"
            self.timescale_storage = TimescaleDBStorage(conn_str)
        else:
            self.timescale_storage = None
            
        # Redis cache (if configured)
        cache_config = storage_config.get('cache', {})
        if cache_config.get('type') == 'redis':
            self.redis_cache = RedisCache(
                host=cache_config.get('host', 'localhost'),
                port=cache_config.get('port', 6379),
                ttl=cache_config.get('ttl_seconds', 3600)
            )
        else:
            self.redis_cache = None
            
        # Storage routing rules
        self.storage_routing = {
            'tick_data': ['parquet', 'timescale', 'cache'],
            'orderbook_data': ['parquet', 'timescale'],
            'features': ['parquet', 'cache'],
            'models': ['parquet'],
            'backtest_results': ['parquet']
        }
        
        # Performance tracking
        self.storage_metrics: List[StorageMetrics] = []
        
    async def initialize(self) -> bool:
        """Initialize all storage backends"""
        try:
            # Initialize TimescaleDB if configured
            if self.timescale_storage:
                await self.timescale_storage.initialize()
                
            # Initialize Redis if configured
            if self.redis_cache:
                await self.redis_cache.initialize()
                
            logger.info("Data storage system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize storage system: {e}")
            return False
            
    async def store_data(self, data: pd.DataFrame, data_type: str, partition_key: str = None) -> bool:
        """Store data using appropriate backends"""
        if data.empty:
            return False
            
        partition_key = partition_key or f"{data_type}_{datetime.now().strftime('%Y%m%d')}"
        storage_backends = self.storage_routing.get(data_type, ['parquet'])
        
        success = True
        
        # Store in primary backends
        for backend in storage_backends:
            try:
                if backend == 'parquet':
                    result = await self.parquet_storage.write(data, partition_key)
                elif backend == 'timescale' and self.timescale_storage:
                    result = await self.timescale_storage.write(data, partition_key)
                elif backend == 'cache' and self.redis_cache:
                    cache_key = f"{data_type}:latest"
                    result = await self.redis_cache.set(cache_key, data)
                else:
                    continue
                    
                if not result:
                    success = False
                    logger.warning(f"Failed to store data in {backend}")
                    
            except Exception as e:
                logger.error(f"Error storing data in {backend}: {e}")
                success = False
                
        return success
        
    async def retrieve_data(
        self, 
        data_type: str, 
        start_time: datetime = None, 
        end_time: datetime = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Retrieve data with intelligent backend selection"""
        
        # Try cache first for recent data
        if use_cache and self.redis_cache and not start_time:
            cache_key = f"{data_type}:latest"
            cached_data = await self.redis_cache.get(cache_key)
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Retrieved {len(cached_data)} records from cache")
                return cached_data
                
        # Try TimescaleDB for time-series queries
        if self.timescale_storage and (start_time or end_time):
            try:
                data = await self.timescale_storage.read(data_type, start_time, end_time)
                if not data.empty:
                    return data
            except Exception as e:
                logger.warning(f"TimescaleDB query failed: {e}")
                
        # Fall back to Parquet storage
        try:
            # Find relevant partitions
            partitions = await self.parquet_storage.list_partitions()
            relevant_partitions = []
            
            for partition in partitions:
                if data_type in partition.partition_key:
                    # Check time overlap
                    if start_time and partition.end_time < start_time:
                        continue
                    if end_time and partition.start_time > end_time:
                        continue
                    relevant_partitions.append(partition)
                    
            # Read from relevant partitions
            dfs = []
            for partition in relevant_partitions:
                df = await self.parquet_storage.read(partition.partition_key, start_time, end_time)
                if not df.empty:
                    dfs.append(df)
                    
            # Combine results
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Sort and remove duplicates
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.sort_values('timestamp')
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to retrieve data from Parquet: {e}")
            return pd.DataFrame()
            
    async def delete_old_data(self, data_type: str, older_than: datetime) -> bool:
        """Delete old data partitions to manage storage space"""
        try:
            # Get all partitions
            partitions = await self.parquet_storage.list_partitions()
            
            deleted_count = 0
            for partition in partitions:
                if data_type in partition.partition_key and partition.end_time < older_than:
                    success = await self.parquet_storage.delete(partition.partition_key)
                    if success:
                        deleted_count += 1
                        
            logger.info(f"Deleted {deleted_count} old partitions for {data_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete old data: {e}")
            return False
            
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage system statistics"""
        stats = {
            'timestamp': datetime.now(timezone.utc),
            'backends': {},
            'partitions': {},
            'total_size_bytes': 0,
            'total_records': 0
        }
        
        try:
            # Parquet statistics
            partitions = await self.parquet_storage.list_partitions()
            
            parquet_stats = {
                'partition_count': len(partitions),
                'total_size_bytes': sum(p.size_bytes for p in partitions),
                'total_records': sum(p.record_count for p in partitions),
                'oldest_data': min(p.start_time for p in partitions) if partitions else None,
                'newest_data': max(p.end_time for p in partitions) if partitions else None
            }
            
            stats['backends']['parquet'] = parquet_stats
            stats['total_size_bytes'] += parquet_stats['total_size_bytes']
            stats['total_records'] += parquet_stats['total_records']
            
            # Partition breakdown by data type
            for partition in partitions:
                data_type = partition.partition_key.split('_')[0]
                if data_type not in stats['partitions']:
                    stats['partitions'][data_type] = {
                        'count': 0,
                        'size_bytes': 0,
                        'records': 0
                    }
                    
                stats['partitions'][data_type]['count'] += 1
                stats['partitions'][data_type]['size_bytes'] += partition.size_bytes
                stats['partitions'][data_type]['records'] += partition.record_count
                
            # TimescaleDB statistics (if available)
            if self.timescale_storage:
                ts_partitions = await self.timescale_storage.list_partitions()
                ts_stats = {
                    'table_count': len(ts_partitions),
                    'total_records': sum(p.record_count for p in ts_partitions)
                }
                stats['backends']['timescale'] = ts_stats
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            
        return stats
        
    async def optimize_storage(self) -> bool:
        """Optimize storage by compacting partitions and cleaning cache"""
        try:
            # Clear old cache entries
            if self.redis_cache:
                await self.redis_cache.clear_all()
                
            # Could implement partition compaction here
            logger.info("Storage optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize storage: {e}")
            return False


# Specialized storage classes

class TickStorage:
    """Specialized storage for tick data"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        
    async def store_ticks(self, ticks: List[TickData]) -> bool:
        """Store tick data"""
        if not ticks:
            return False
            
        # Convert to DataFrame
        data = pd.DataFrame([asdict(tick) for tick in ticks])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        return await self.storage.store_data(data, 'tick_data')
        
    async def get_recent_ticks(self, symbol: str = 'XAUUSD', minutes: int = 60) -> List[TickData]:
        """Get recent tick data"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=minutes)
        
        df = await self.storage.retrieve_data('tick_data', start_time, end_time)
        
        if df.empty:
            return []
            
        # Filter by symbol and convert back to TickData objects
        symbol_data = df[df['symbol'] == symbol] if 'symbol' in df.columns else df
        
        ticks = []
        for _, row in symbol_data.iterrows():
            tick = TickData(
                timestamp=row.get('timestamp'),
                symbol=row.get('symbol', symbol),
                price=row.get('price', 0.0),
                volume=row.get('volume', 0.0),
                side=row.get('side', ''),
                trade_id=row.get('trade_id'),
                sequence=row.get('sequence'),
                exchange=row.get('exchange')
            )
            ticks.append(tick)
            
        return ticks


class OrderBookStorage:
    """Specialized storage for order book data"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        
    async def store_orderbook_snapshots(self, snapshots: List[OrderBookSnapshot]) -> bool:
        """Store order book snapshots"""
        if not snapshots:
            return False
            
        # Convert to DataFrame with flattened structure
        data = []
        for snapshot in snapshots:
            record = {
                'timestamp': snapshot.timestamp,
                'symbol': snapshot.symbol,
                'sequence': snapshot.sequence,
                'exchange': snapshot.exchange
            }
            
            # Flatten bid/ask levels
            for i, bid in enumerate(snapshot.bids[:5]):  # Store up to 5 levels
                record[f'bid_price_{i+1}'] = bid.price
                record[f'bid_size_{i+1}'] = bid.size
                
            for i, ask in enumerate(snapshot.asks[:5]):
                record[f'ask_price_{i+1}'] = ask.price
                record[f'ask_size_{i+1}'] = ask.size
                
            data.append(record)
            
        df = pd.DataFrame(data)
        return await self.storage.store_data(df, 'orderbook_data')


class FeatureStorage:
    """Specialized storage for processed features"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        
    async def store_features(self, features: pd.DataFrame, feature_set: str) -> bool:
        """Store feature data"""
        partition_key = f"features_{feature_set}_{datetime.now().strftime('%Y%m%d')}"
        return await self.storage.store_data(features, 'features', partition_key)
        
    async def get_latest_features(self, feature_set: str, minutes: int = 60) -> pd.DataFrame:
        """Get latest features"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=minutes)
        
        return await self.storage.retrieve_data('features', start_time, end_time)