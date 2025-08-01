#!/usr/bin/env python3

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue, Empty
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
import warnings

from src.data.ingestion.realtime_data_sources import MarketDataPoint, DataFrequency
from src.data.ingestion.enhanced_data_pipeline import EnhancedDataPipeline
from src.models.hmm.hmm_engine import AdvancedBaumWelchHMM
from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class StreamingEventType(Enum):
    """Types of streaming events"""
    PRICE_UPDATE = "price_update"
    REGIME_CHANGE = "regime_change"
    RISK_ALERT = "risk_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    MARKET_INDICATOR = "market_indicator"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"


@dataclass
class StreamingEvent:
    """Real-time streaming event"""
    event_id: str
    event_type: StreamingEventType
    timestamp: datetime
    symbol: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientSubscription:
    """Client subscription configuration"""
    client_id: str
    symbols: Set[str]
    event_types: Set[StreamingEventType]
    websocket: Optional[Any] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active: bool = True


class RealTimeStreamingService:
    """Real-time streaming service for portfolio data"""
    
    def __init__(self, enhanced_pipeline: EnhancedDataPipeline):
        self.pipeline = enhanced_pipeline
        self.clients: Dict[str, ClientSubscription] = {}
        self.event_queue = Queue()
        self.running = False
        self.server = None
        self.port = 8765
        
        # Real-time analytics components
        self.hmm_model = None
        self.risk_monitor = None
        self.portfolio_weights = {}
        self.current_prices = {}
        self.regime_history = []
        
        # Performance tracking
        self.events_processed = 0
        self.start_time = None
        self.last_regime_update = None
        
        # Background tasks
        self.heartbeat_task = None
        self.analytics_task = None
        
    async def initialize(self, hmm_components: int = 3):
        """Initialize streaming service components"""
        logger.info("Initializing real-time streaming service")
        
        try:
            # Initialize HMM model for regime detection
            self.hmm_model = AdvancedBaumWelchHMM(
                n_components=hmm_components,
                random_state=42
            )
            
            # Initialize risk monitor
            from src.risk.monitoring.risk_monitor import RiskLimits
            risk_limits = RiskLimits(
                max_portfolio_volatility=0.25,
                max_individual_weight=0.20,
                max_drawdown=0.15,
                max_var_95=0.05
            )
            self.risk_monitor = RealTimeRiskMonitor(risk_limits)
            
            # Setup data pipeline callbacks
            self.pipeline.add_streaming_callback(
                symbols=[],  # Will be updated based on subscriptions
                callback_function=self._handle_market_data_update
            )
            
            logger.info("Streaming service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing streaming service: {e}")
            raise
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server"""
        self.port = port
        self.running = True
        self.start_time = datetime.now()
        
        try:
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.analytics_task = asyncio.create_task(self._analytics_loop())
            
            # Start data pipeline streaming
            await self.pipeline.start_streaming()
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_client_connection,
                host,
                port
            )
            
            logger.info(f"Streaming server started on {host}:{port}")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error starting streaming server: {e}")
            raise
    
    async def stop_server(self):
        """Stop streaming server"""
        logger.info("Stopping streaming server")
        
        self.running = False
        
        # Stop background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.analytics_task:
            self.analytics_task.cancel()
        
        # Disconnect all clients
        for client_id in list(self.clients.keys()):
            await self._disconnect_client(client_id)
        
        # Stop data pipeline
        await self.pipeline.stop_streaming()
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Streaming server stopped")
    
    async def _handle_client_connection(self, websocket, path):
        """Handle new WebSocket client connection"""
        client_id = str(uuid.uuid4())
        
        try:
            logger.info(f"New client connected: {client_id}")
            
            # Register client
            self.clients[client_id] = ClientSubscription(
                client_id=client_id,
                symbols=set(),
                event_types=set(),
                websocket=websocket
            )
            
            # Send welcome message
            await self._send_to_client(client_id, {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "server_time": datetime.now().isoformat()
            })
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, json.loads(message))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            await self._disconnect_client(client_id)
    
    async def _handle_client_message(self, client_id: str, message: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        
        try:
            msg_type = message.get("type")
            
            if msg_type == "subscribe":
                await self._handle_subscription(client_id, message)
            
            elif msg_type == "unsubscribe":
                await self._handle_unsubscription(client_id, message)
            
            elif msg_type == "heartbeat":
                await self._handle_heartbeat(client_id)
            
            elif msg_type == "get_status":
                await self._send_status_update(client_id)
            
            elif msg_type == "get_portfolio":
                await self._send_portfolio_update(client_id)
            
            elif msg_type == "update_portfolio":
                await self._handle_portfolio_update(client_id, message)
            
            else:
                await self._send_to_client(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
        
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self._send_to_client(client_id, {
                "type": "error",
                "message": str(e)
            })
    
    async def _handle_subscription(self, client_id: str, message: Dict[str, Any]):
        """Handle client subscription requests"""
        
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        # Update symbols
        if "symbols" in message:
            new_symbols = set(message["symbols"])
            client.symbols.update(new_symbols)
            logger.info(f"Client {client_id} subscribed to symbols: {new_symbols}")
        
        # Update event types
        if "events" in message:
            new_events = set(StreamingEventType(event) for event in message["events"])
            client.event_types.update(new_events)
            logger.info(f"Client {client_id} subscribed to events: {new_events}")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            "type": "subscription_confirmed",
            "symbols": list(client.symbols),
            "events": [event.value for event in client.event_types]
        })
        
        # Update pipeline subscriptions
        await self._update_pipeline_subscriptions()
    
    async def _handle_unsubscription(self, client_id: str, message: Dict[str, Any]):
        """Handle client unsubscription requests"""
        
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        # Remove symbols
        if "symbols" in message:
            remove_symbols = set(message["symbols"])
            client.symbols -= remove_symbols
            logger.info(f"Client {client_id} unsubscribed from symbols: {remove_symbols}")
        
        # Remove event types
        if "events" in message:
            remove_events = set(StreamingEventType(event) for event in message["events"])
            client.event_types -= remove_events
            logger.info(f"Client {client_id} unsubscribed from events: {remove_events}")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            "type": "unsubscription_confirmed",
            "symbols": list(client.symbols),
            "events": [event.value for event in client.event_types]
        })
        
        # Update pipeline subscriptions
        await self._update_pipeline_subscriptions()
    
    async def _handle_heartbeat(self, client_id: str):
        """Handle client heartbeat"""
        if client_id in self.clients:
            self.clients[client_id].last_heartbeat = datetime.now()
            await self._send_to_client(client_id, {
                "type": "heartbeat_ack",
                "server_time": datetime.now().isoformat()
            })
    
    async def _handle_portfolio_update(self, client_id: str, message: Dict[str, Any]):
        """Handle portfolio weight updates from client"""
        
        try:
            if "weights" in message:
                self.portfolio_weights.update(message["weights"])
                logger.info(f"Portfolio weights updated by client {client_id}")
                
                # Broadcast portfolio update to all relevant clients
                await self._broadcast_event(StreamingEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=StreamingEventType.PORTFOLIO_UPDATE,
                    timestamp=datetime.now(),
                    symbol=None,
                    data={
                        "weights": self.portfolio_weights,
                        "updated_by": client_id
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error handling portfolio update: {e}")
    
    async def _update_pipeline_subscriptions(self):
        """Update data pipeline subscriptions based on client needs"""
        
        # Collect all symbols from all clients
        all_symbols = set()
        for client in self.clients.values():
            all_symbols.update(client.symbols)
        
        if all_symbols:
            logger.info(f"Updating pipeline subscriptions for {len(all_symbols)} symbols")
            # Note: This would require modifying the pipeline to update subscriptions
            # For now, we'll log the intent
    
    async def _disconnect_client(self, client_id: str):
        """Disconnect and cleanup client"""
        
        if client_id in self.clients:
            client = self.clients[client_id]
            
            if client.websocket:
                try:
                    await client.websocket.close()
                except:
                    pass
            
            del self.clients[client_id]
            logger.info(f"Client {client_id} disconnected and cleaned up")
    
    async def _send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to specific client"""
        
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        try:
            if client.websocket and client.active:
                await client.websocket.send(json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Error sending to client {client_id}: {e}")
            await self._disconnect_client(client_id)
    
    async def _broadcast_event(self, event: StreamingEvent):
        """Broadcast event to relevant clients"""
        
        message_data = {
            "type": "event",
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "symbol": event.symbol,
            "data": event.data,
            "metadata": event.metadata
        }
        
        # Send to relevant clients
        for client in self.clients.values():
            # Check if client is interested in this event type
            if event.event_type not in client.event_types:
                continue
            
            # Check if client is interested in this symbol (if symbol-specific)
            if event.symbol and event.symbol not in client.symbols:
                continue
            
            await self._send_to_client(client.client_id, message_data)
        
        self.events_processed += 1
    
    def _handle_market_data_update(self, data_point: MarketDataPoint):
        """Handle market data updates from pipeline"""
        
        try:
            # Update current prices
            self.current_prices[data_point.symbol] = {
                "price": data_point.close,
                "timestamp": data_point.timestamp,
                "volume": data_point.volume
            }
            
            # Create streaming event
            event = StreamingEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamingEventType.PRICE_UPDATE,
                timestamp=data_point.timestamp,
                symbol=data_point.symbol,
                data={
                    "open": data_point.open,
                    "high": data_point.high,
                    "low": data_point.low,
                    "close": data_point.close,
                    "volume": data_point.volume,
                    "source": data_point.source
                }
            )
            
            # Queue event for broadcasting
            self.event_queue.put(event)
            
        except Exception as e:
            logger.error(f"Error handling market data update: {e}")
    
    async def _heartbeat_loop(self):
        """Background task for client heartbeat monitoring"""
        
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(minutes=5)
                
                # Check for timed out clients
                timeout_clients = []
                for client_id, client in self.clients.items():
                    if client.last_heartbeat < timeout_threshold:
                        timeout_clients.append(client_id)
                
                # Disconnect timed out clients
                for client_id in timeout_clients:
                    logger.warning(f"Client {client_id} timed out")
                    await self._disconnect_client(client_id)
                
                # Process queued events
                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get_nowait()
                        await self._broadcast_event(event)
                    except Empty:
                        break
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
    
    async def _analytics_loop(self):
        """Background task for real-time analytics"""
        
        while self.running:
            try:
                # Regime detection
                if len(self.current_prices) >= 5:  # Need minimum data
                    await self._update_regime_detection()
                
                # Risk monitoring
                if self.portfolio_weights and self.current_prices:
                    await self._update_risk_monitoring()
                
                # System status updates
                await self._broadcast_system_status()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_regime_detection(self):
        """Update regime detection based on current market data"""
        
        try:
            if not self.hmm_model or len(self.current_prices) < 10:
                return
            
            # Prepare feature data from current prices
            symbols = list(self.current_prices.keys())
            price_data = []
            
            for symbol in symbols:
                price_info = self.current_prices[symbol]
                price_data.append({
                    'symbol': symbol,
                    'price': price_info['price'],
                    'timestamp': price_info['timestamp']
                })
            
            if len(price_data) >= 5:
                # Create simple features for regime detection
                prices = pd.Series([p['price'] for p in price_data])
                returns = prices.pct_change().dropna()
                
                if len(returns) >= 3:
                    features = pd.DataFrame({
                        'returns': returns,
                        'volatility': returns.rolling(3).std().fillna(0),
                        'momentum': returns.rolling(3).sum().fillna(0)
                    }).dropna()
                    
                    if len(features) >= 3:
                        # Get regime prediction
                        current_regime = self.hmm_model.predict_regimes(features.tail(1))
                        
                        if len(current_regime) > 0:
                            regime_id = current_regime[0]
                            
                            # Check if regime changed
                            if (not self.regime_history or 
                                self.regime_history[-1]['regime'] != regime_id):
                                
                                # Broadcast regime change
                                event = StreamingEvent(
                                    event_id=str(uuid.uuid4()),
                                    event_type=StreamingEventType.REGIME_CHANGE,
                                    timestamp=datetime.now(),
                                    symbol=None,
                                    data={
                                        "new_regime": int(regime_id),
                                        "previous_regime": self.regime_history[-1]['regime'] if self.regime_history else None,
                                        "confidence": 0.8,  # Placeholder
                                        "features": features.tail(1).to_dict('records')[0]
                                    }
                                )
                                
                                await self._broadcast_event(event)
                                
                                # Update history
                                self.regime_history.append({
                                    'timestamp': datetime.now(),
                                    'regime': int(regime_id)
                                })
                                
                                # Keep only recent history
                                if len(self.regime_history) > 100:
                                    self.regime_history = self.regime_history[-100:]
                                
                                self.last_regime_update = datetime.now()
        
        except Exception as e:
            logger.error(f"Error updating regime detection: {e}")
    
    async def _update_risk_monitoring(self):
        """Update risk monitoring and generate alerts"""
        
        try:
            if not self.risk_monitor or not self.portfolio_weights:
                return
            
            # Convert current prices to returns data
            symbols = list(self.portfolio_weights.keys())
            current_returns = {}
            
            for symbol in symbols:
                if symbol in self.current_prices:
                    # Simple return calculation (would need historical data for proper calculation)
                    current_returns[symbol] = 0.0  # Placeholder
            
            if current_returns:
                # Create portfolio weights series
                weights = pd.Series(self.portfolio_weights)
                returns_data = pd.DataFrame([current_returns])
                
                # Monitor risk (simplified)
                if len(returns_data) > 0:
                    # Generate risk alert if needed
                    portfolio_value = sum(weights * 100000)  # Assume $100k portfolio
                    
                    if portfolio_value != 0:  # Avoid division by zero
                        event = StreamingEvent(
                            event_id=str(uuid.uuid4()),
                            event_type=StreamingEventType.RISK_ALERT,
                            timestamp=datetime.now(),
                            symbol=None,
                            data={
                                "alert_type": "portfolio_update",
                                "portfolio_value": portfolio_value,
                                "message": "Portfolio risk metrics updated"
                            }
                        )
                        
                        await self._broadcast_event(event)
        
        except Exception as e:
            logger.error(f"Error updating risk monitoring: {e}")
    
    async def _broadcast_system_status(self):
        """Broadcast system status update"""
        
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            status_event = StreamingEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamingEventType.SYSTEM_STATUS,
                timestamp=datetime.now(),
                symbol=None,
                data={
                    "status": "running",
                    "uptime_seconds": uptime,
                    "connected_clients": len(self.clients),
                    "events_processed": self.events_processed,
                    "symbols_tracked": len(self.current_prices),
                    "last_regime_update": self.last_regime_update.isoformat() if self.last_regime_update else None
                }
            )
            
            await self._broadcast_event(status_event)
        
        except Exception as e:
            logger.error(f"Error broadcasting system status: {e}")
    
    async def _send_status_update(self, client_id: str):
        """Send status update to specific client"""
        
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            status_data = {
                "type": "status",
                "status": "running",
                "uptime_seconds": uptime,
                "connected_clients": len(self.clients),
                "events_processed": self.events_processed,
                "symbols_tracked": len(self.current_prices),
                "pipeline_status": self.pipeline.get_pipeline_status(),
                "current_regime": self.regime_history[-1] if self.regime_history else None
            }
            
            await self._send_to_client(client_id, status_data)
        
        except Exception as e:
            logger.error(f"Error sending status update: {e}")
    
    async def _send_portfolio_update(self, client_id: str):
        """Send portfolio update to specific client"""
        
        try:
            portfolio_data = {
                "type": "portfolio",
                "weights": self.portfolio_weights,
                "current_prices": self.current_prices,
                "last_updated": datetime.now().isoformat()
            }
            
            await self._send_to_client(client_id, portfolio_data)
        
        except Exception as e:
            logger.error(f"Error sending portfolio update: {e}")


# Factory function to create streaming service
def create_streaming_service(enhanced_pipeline: EnhancedDataPipeline) -> RealTimeStreamingService:
    """Factory function to create real-time streaming service"""
    
    service = RealTimeStreamingService(enhanced_pipeline)
    logger.info("Created real-time streaming service")
    return service