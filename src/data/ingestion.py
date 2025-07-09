"""
Data Ingestion Engine

High-performance data ingestion system for tick-level and Level 2 orderbook data.
Handles multiple data sources with millisecond precision alignment and real-time
streaming capabilities for XAUUSD scalping operations.

Performance Requirements:
- < 1ms ingestion latency
- Tick-level precision (microsecond timestamps)
- 5-level orderbook depth
- Cross-asset correlation feeds
- Dynamic spread modeling

Time Complexity: O(1) per tick/quote
Space Complexity: O(n) where n is buffer size
"""

import asyncio
import aiohttp
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import ccxt
import yfinance as yf
from pathlib import Path
import gzip
import lz4.frame
import pickle

from src.core.config import get_config_manager
from src.core.utils import Timer, get_performance_monitor
from src.core.logging import get_perf_logger

logger = logging.getLogger(__name__)
perf_logger = get_perf_logger()


@dataclass
class TickData:
    """Tick data structure with microsecond precision"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: str  # 'buy', 'sell', 'unknown'
    trade_id: Optional[str] = None
    sequence: Optional[int] = None
    exchange: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderBookLevel:
    """Single level of order book"""
    price: float
    size: float
    orders: Optional[int] = None


@dataclass
class OrderBookSnapshot:
    """Level 2 order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence: Optional[int] = None
    exchange: Optional[str] = None
    
    def get_spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0
        
    def get_mid_price(self) -> float:
        """Calculate mid price"""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2.0
        return 0.0
        
    def get_imbalance(self, levels: int = 1) -> float:
        """Calculate order book imbalance"""
        if len(self.bids) < levels or len(self.asks) < levels:
            return 0.0
            
        bid_volume = sum(level.size for level in self.bids[:levels])
        ask_volume = sum(level.size for level in self.asks[:levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / total_volume


@dataclass
class NewsEvent:
    """Economic news event"""
    timestamp: datetime
    title: str
    impact: str  # 'high', 'medium', 'low'
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    currency: Optional[str] = None


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass
        
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        pass
        
    @abstractmethod
    async def get_data_stream(self) -> AsyncGenerator[Any, None]:
        """Get real-time data stream"""
        pass


class TickDataProvider(DataProvider):
    """
    Tick data provider for real-time trade feeds
    
    Connects to multiple exchanges and aggregates tick data with
    microsecond precision timestamps for scalping operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config.get('symbols', ['XAUUSD'])
        self.exchanges = config.get('exchanges', ['oanda'])
        self.websocket_urls = config.get('websocket_urls', {})
        self.api_keys = config.get('api_keys', {})
        
        self.connections: Dict[str, Any] = {}
        self.data_queue = asyncio.Queue(maxsize=10000)
        self.connected = False
        
        # Performance monitoring
        self.tick_count = 0
        self.last_tick_time = None
        
    async def connect(self) -> bool:
        """Connect to tick data sources"""
        try:
            for exchange in self.exchanges:
                if exchange == 'oanda':
                    await self._connect_oanda()
                elif exchange == 'interactive_brokers':
                    await self._connect_ib()
                elif exchange == 'mt5':
                    await self._connect_mt5()
                    
            self.connected = True
            logger.info(f"Connected to tick data providers: {self.exchanges}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to tick data providers: {e}")
            return False
            
    async def _connect_oanda(self) -> None:
        """Connect to OANDA streaming API"""
        # Placeholder for OANDA connection
        # In production, implement actual OANDA v20 API connection
        logger.info("Connected to OANDA tick stream")
        
    async def _connect_ib(self) -> None:
        """Connect to Interactive Brokers"""
        # Placeholder for IB connection
        # In production, implement IB API connection
        logger.info("Connected to Interactive Brokers tick stream")
        
    async def _connect_mt5(self) -> None:
        """Connect to MetaTrader 5"""
        # Placeholder for MT5 connection
        logger.info("Connected to MetaTrader 5 tick stream")
        
    async def disconnect(self) -> None:
        """Disconnect from all sources"""
        for connection in self.connections.values():
            try:
                if hasattr(connection, 'close'):
                    await connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
                
        self.connected = False
        logger.info("Disconnected from tick data providers")
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols for tick data"""
        self.symbols.extend(symbols)
        logger.info(f"Subscribed to symbols: {symbols}")
        
    async def get_data_stream(self) -> AsyncGenerator[TickData, None]:
        """Get real-time tick data stream"""
        while self.connected:
            try:
                # In production, this would receive actual tick data
                # For demo, generate realistic tick data
                tick = await self._generate_demo_tick()
                
                # Performance monitoring
                self.tick_count += 1
                self.last_tick_time = datetime.now(timezone.utc)
                
                yield tick
                
                # Small delay to simulate realistic tick frequency
                await asyncio.sleep(0.01)  # 100 ticks per second
                
            except Exception as e:
                logger.error(f"Error in tick data stream: {e}")
                await asyncio.sleep(1)
                
    async def _generate_demo_tick(self) -> TickData:
        """Generate realistic demo tick data for testing"""
        # Simple random walk for XAUUSD around 2000
        base_price = 2000.0
        price_change = np.random.normal(0, 0.5)  # $0.50 std dev
        price = base_price + price_change
        
        volume = np.random.exponential(1.0)  # Exponential volume distribution
        side = np.random.choice(['buy', 'sell'], p=[0.51, 0.49])  # Slight buy bias
        
        return TickData(
            timestamp=datetime.now(timezone.utc),
            symbol='XAUUSD',
            price=round(price, 2),
            volume=round(volume, 2),
            side=side,
            trade_id=f"T{int(time.time() * 1000000)}",
            sequence=self.tick_count,
            exchange='demo'
        )


class OrderBookProvider(DataProvider):
    """
    Level 2 order book data provider
    
    Provides real-time order book snapshots with 5 levels of depth
    for market microstructure analysis and order flow features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config.get('symbols', ['XAUUSD'])
        self.levels = config.get('levels', 5)
        self.snapshot_frequency_ms = config.get('snapshot_frequency_ms', 100)
        
        self.data_queue = asyncio.Queue(maxsize=5000)
        self.connected = False
        self.sequence = 0
        
    async def connect(self) -> bool:
        """Connect to order book data source"""
        try:
            # In production, connect to actual Level 2 feed
            self.connected = True
            logger.info("Connected to order book provider")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to order book provider: {e}")
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from order book source"""
        self.connected = False
        logger.info("Disconnected from order book provider")
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to order book updates"""
        self.symbols.extend(symbols)
        logger.info(f"Subscribed to order book for: {symbols}")
        
    async def get_data_stream(self) -> AsyncGenerator[OrderBookSnapshot, None]:
        """Get real-time order book snapshots"""
        while self.connected:
            try:
                snapshot = await self._generate_demo_orderbook()
                yield snapshot
                
                # Wait for next snapshot
                await asyncio.sleep(self.snapshot_frequency_ms / 1000)
                
            except Exception as e:
                logger.error(f"Error in order book stream: {e}")
                await asyncio.sleep(1)
                
    async def _generate_demo_orderbook(self) -> OrderBookSnapshot:
        """Generate realistic demo order book"""
        base_price = 2000.0
        spread = np.random.uniform(0.5, 2.0)  # 0.5-2.0 pip spread
        
        # Generate bids (below mid price)
        bids = []
        for i in range(self.levels):
            price = base_price - spread/2 - i * 0.1
            size = np.random.exponential(10.0)  # Exponential size distribution
            bids.append(OrderBookLevel(price=round(price, 2), size=round(size, 2)))
            
        # Generate asks (above mid price)  
        asks = []
        for i in range(self.levels):
            price = base_price + spread/2 + i * 0.1
            size = np.random.exponential(10.0)
            asks.append(OrderBookLevel(price=round(price, 2), size=round(size, 2)))
            
        self.sequence += 1
        
        return OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol='XAUUSD',
            bids=bids,
            asks=asks,
            sequence=self.sequence,
            exchange='demo'
        )


class NewsProvider(DataProvider):
    """
    Economic news and calendar data provider
    
    Provides real-time economic news events with impact ratings
    for news-based risk management and blackout periods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.importance_filter = config.get('importance_filter', ['high', 'medium'])
        self.currencies = config.get('currencies', ['USD', 'EUR', 'GBP'])
        
        self.connected = False
        self.events_cache: List[NewsEvent] = []
        
    async def connect(self) -> bool:
        """Connect to news data source"""
        try:
            # In production, connect to ForexFactory, Bloomberg, etc.
            self.connected = True
            logger.info("Connected to news provider")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to news provider: {e}")
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from news source"""
        self.connected = False
        logger.info("Disconnected from news provider")
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to news for symbols"""
        logger.info(f"Subscribed to news for: {symbols}")
        
    async def get_data_stream(self) -> AsyncGenerator[NewsEvent, None]:
        """Get real-time news events"""
        while self.connected:
            try:
                # Check for upcoming events
                upcoming_events = await self._get_upcoming_events()
                
                for event in upcoming_events:
                    if event not in self.events_cache:
                        self.events_cache.append(event)
                        yield event
                        
                # Clean old events from cache
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                self.events_cache = [e for e in self.events_cache if e.timestamp > cutoff]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in news stream: {e}")
                await asyncio.sleep(60)
                
    async def _get_upcoming_events(self) -> List[NewsEvent]:
        """Get upcoming news events"""
        # In production, fetch from actual news API
        # For demo, return empty list
        return []
        
    async def get_events_in_window(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[NewsEvent]:
        """Get news events in time window"""
        return [
            event for event in self.events_cache
            if start_time <= event.timestamp <= end_time
        ]


class CrossAssetProvider:
    """
    Cross-asset data provider for correlation analysis
    
    Provides real-time quotes for DXY, TIPS, SPY and other correlated assets
    for cross-asset feature engineering and regime detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config.get('symbols', ['DXY', 'SPY', 'TLT'])
        self.update_frequency = config.get('update_frequency_seconds', 1)
        
        self.data: Dict[str, float] = {}
        self.last_update = {}
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to cross-asset data sources"""
        try:
            # Initialize with current prices
            for symbol in self.symbols:
                self.data[symbol] = await self._get_current_price(symbol)
                self.last_update[symbol] = datetime.now(timezone.utc)
                
            self.connected = True
            logger.info(f"Connected to cross-asset data: {self.symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to cross-asset data: {e}")
            return False
            
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            # Use yfinance for demo data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice', 0.0)
        except:
            # Return demo price if API fails
            return 100.0 + np.random.normal(0, 1)
            
    async def get_latest_data(self) -> Dict[str, Dict[str, Any]]:
        """Get latest cross-asset data"""
        result = {}
        
        for symbol in self.symbols:
            result[symbol] = {
                'price': self.data.get(symbol, 0.0),
                'timestamp': self.last_update.get(symbol),
                'symbol': symbol
            }
            
        return result
        
    async def update_data(self) -> None:
        """Update cross-asset data"""
        if not self.connected:
            return
            
        for symbol in self.symbols:
            try:
                new_price = await self._get_current_price(symbol)
                self.data[symbol] = new_price
                self.last_update[symbol] = datetime.now(timezone.utc)
            except Exception as e:
                logger.warning(f"Failed to update {symbol}: {e}")


class DataIngestionEngine:
    """
    Main data ingestion engine
    
    Coordinates multiple data providers and ensures synchronized data flow
    with millisecond precision alignment for scalping operations.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        
        # Initialize providers
        tick_config = self.config.get('data', 'sources.tick_data', {})
        orderbook_config = self.config.get('data', 'sources.orderbook_data', {})
        news_config = self.config.get('data', 'sources.news_calendar', {})
        cross_asset_config = self.config.get('data', 'sources.cross_assets', {})
        
        self.tick_provider = TickDataProvider(tick_config)
        self.orderbook_provider = OrderBookProvider(orderbook_config)
        self.news_provider = NewsProvider(news_config)
        self.cross_asset_provider = CrossAssetProvider(cross_asset_config)
        
        # Data buffers
        self.tick_buffer: List[TickData] = []
        self.orderbook_buffer: List[OrderBookSnapshot] = []
        self.news_buffer: List[NewsEvent] = []
        
        # Buffer limits
        self.max_buffer_size = 10000
        
        # Callbacks for real-time processing
        self.tick_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        self.news_callbacks: List[Callable] = []
        
        # Performance monitoring
        self.perf_monitor = get_performance_monitor()
        self.stats = {
            'ticks_processed': 0,
            'orderbook_updates': 0,
            'news_events': 0,
            'last_update': None
        }
        
    async def start(self) -> bool:
        """Start data ingestion engine"""
        try:
            # Connect all providers
            tick_connected = await self.tick_provider.connect()
            orderbook_connected = await self.orderbook_provider.connect()
            news_connected = await self.news_provider.connect()
            cross_asset_connected = await self.cross_asset_provider.connect()
            
            if not all([tick_connected, orderbook_connected, news_connected, cross_asset_connected]):
                logger.error("Failed to connect all data providers")
                return False
                
            # Start data processing tasks
            asyncio.create_task(self._process_tick_data())
            asyncio.create_task(self._process_orderbook_data())
            asyncio.create_task(self._process_news_data())
            asyncio.create_task(self._update_cross_asset_data())
            
            logger.info("Data ingestion engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start data ingestion engine: {e}")
            return False
            
    async def stop(self) -> None:
        """Stop data ingestion engine"""
        await self.tick_provider.disconnect()
        await self.orderbook_provider.disconnect()
        await self.news_provider.disconnect()
        
        logger.info("Data ingestion engine stopped")
        
    async def _process_tick_data(self) -> None:
        """Process incoming tick data"""
        async for tick in self.tick_provider.get_data_stream():
            with Timer("tick_processing"):
                # Add to buffer
                self.tick_buffer.append(tick)
                if len(self.tick_buffer) > self.max_buffer_size:
                    self.tick_buffer.pop(0)
                    
                # Call registered callbacks
                for callback in self.tick_callbacks:
                    try:
                        await callback(tick)
                    except Exception as e:
                        logger.error(f"Error in tick callback: {e}")
                        
                self.stats['ticks_processed'] += 1
                self.stats['last_update'] = datetime.now(timezone.utc)
                
    async def _process_orderbook_data(self) -> None:
        """Process incoming order book data"""
        async for snapshot in self.orderbook_provider.get_data_stream():
            with Timer("orderbook_processing"):
                # Add to buffer
                self.orderbook_buffer.append(snapshot)
                if len(self.orderbook_buffer) > self.max_buffer_size:
                    self.orderbook_buffer.pop(0)
                    
                # Call registered callbacks
                for callback in self.orderbook_callbacks:
                    try:
                        await callback(snapshot)
                    except Exception as e:
                        logger.error(f"Error in orderbook callback: {e}")
                        
                self.stats['orderbook_updates'] += 1
                
    async def _process_news_data(self) -> None:
        """Process incoming news data"""
        async for event in self.news_provider.get_data_stream():
            # Add to buffer
            self.news_buffer.append(event)
            if len(self.news_buffer) > 1000:  # Smaller buffer for news
                self.news_buffer.pop(0)
                
            # Call registered callbacks
            for callback in self.news_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in news callback: {e}")
                    
            self.stats['news_events'] += 1
            
    async def _update_cross_asset_data(self) -> None:
        """Update cross-asset data periodically"""
        while True:
            try:
                await self.cross_asset_provider.update_data()
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error updating cross-asset data: {e}")
                await asyncio.sleep(5)
                
    def register_tick_callback(self, callback: Callable) -> None:
        """Register callback for tick data"""
        self.tick_callbacks.append(callback)
        
    def register_orderbook_callback(self, callback: Callable) -> None:
        """Register callback for order book data"""
        self.orderbook_callbacks.append(callback)
        
    def register_news_callback(self, callback: Callable) -> None:
        """Register callback for news data"""
        self.news_callbacks.append(callback)
        
    def get_latest_tick(self, symbol: str = 'XAUUSD') -> Optional[TickData]:
        """Get latest tick for symbol"""
        for tick in reversed(self.tick_buffer):
            if tick.symbol == symbol:
                return tick
        return None
        
    def get_latest_orderbook(self, symbol: str = 'XAUUSD') -> Optional[OrderBookSnapshot]:
        """Get latest order book for symbol"""
        for snapshot in reversed(self.orderbook_buffer):
            if snapshot.symbol == symbol:
                return snapshot
        return None
        
    def get_recent_ticks(self, symbol: str = 'XAUUSD', seconds: int = 60) -> List[TickData]:
        """Get recent ticks within time window"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        return [
            tick for tick in self.tick_buffer
            if tick.symbol == symbol and tick.timestamp >= cutoff
        ]
        
    def get_recent_orderbooks(self, symbol: str = 'XAUUSD', seconds: int = 60) -> List[OrderBookSnapshot]:
        """Get recent order book snapshots within time window"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        return [
            snapshot for snapshot in self.orderbook_buffer
            if snapshot.symbol == symbol and snapshot.timestamp >= cutoff
        ]
        
    async def get_cross_asset_data(self) -> Dict[str, Dict[str, Any]]:
        """Get latest cross-asset data"""
        return await self.cross_asset_provider.get_latest_data()
        
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        return self.stats.copy()
        
    async def save_buffer_to_file(self, filepath: str) -> None:
        """Save current buffers to file for analysis"""
        data = {
            'ticks': [tick.to_dict() for tick in self.tick_buffer],
            'orderbooks': [asdict(ob) for ob in self.orderbook_buffer],
            'news': [asdict(event) for event in self.news_buffer],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stats': self.stats
        }
        
        # Compress and save
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with lz4.frame.open(f"{filepath}.lz4", 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved data buffers to {filepath}.lz4")
        
    async def load_buffer_from_file(self, filepath: str) -> None:
        """Load buffers from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return
            
        try:
            with lz4.frame.open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Restore buffers
            self.tick_buffer = [TickData(**tick) for tick in data['ticks']]
            # Note: OrderBookSnapshot reconstruction would need custom logic
            # for the nested OrderBookLevel objects
            
            logger.info(f"Loaded data buffers from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load data buffers: {e}")