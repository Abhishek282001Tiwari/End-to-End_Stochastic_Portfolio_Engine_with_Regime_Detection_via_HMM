#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from abc import ABC, abstractmethod

from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Order fill execution"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    market_impact: float
    venue: str = "SIM"


@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime


@dataclass
class TradingCosts:
    """Trading cost configuration"""
    commission_rate: float = 0.001  # 0.1% commission
    bid_ask_spread: float = 0.0005  # 0.05% bid-ask spread
    market_impact_linear: float = 0.0001  # Linear market impact coefficient
    market_impact_sqrt: float = 0.001  # Square root market impact coefficient
    slippage_std: float = 0.0002  # Standard deviation of random slippage
    min_commission: float = 1.0  # Minimum commission per trade
    max_impact: float = 0.01  # Maximum market impact (1%)


class MarketImpactModel(ABC):
    """Abstract base class for market impact models"""
    
    @abstractmethod
    def calculate_impact(self, order: Order, market_data: Dict[str, Any]) -> float:
        """Calculate market impact for an order"""
        pass


class LinearMarketImpactModel(MarketImpactModel):
    """Linear market impact model"""
    
    def __init__(self, impact_coefficient: float = 0.0001):
        self.impact_coefficient = impact_coefficient
    
    def calculate_impact(self, order: Order, market_data: Dict[str, Any]) -> float:
        """Linear impact proportional to order size"""
        volume = market_data.get('volume', 1000000)  # Default volume
        price = market_data.get('price', 100)  # Default price
        
        order_value = order.quantity * price
        volume_value = volume * price
        
        # Impact as percentage of order value relative to daily volume
        if volume_value > 0:
            participation_rate = order_value / volume_value
            impact = self.impact_coefficient * participation_rate
            
            # Apply direction (market impact increases price for buys, decreases for sells)
            if order.side == OrderSide.BUY:
                return min(impact, 0.01)  # Cap at 1%
            else:
                return max(-impact, -0.01)  # Cap at -1%
        
        return 0.0


class SquareRootMarketImpactModel(MarketImpactModel):
    """Square root market impact model (Kyle 1985)"""
    
    def __init__(self, impact_coefficient: float = 0.001):
        self.impact_coefficient = impact_coefficient
    
    def calculate_impact(self, order: Order, market_data: Dict[str, Any]) -> float:
        """Square root impact model"""
        volume = market_data.get('volume', 1000000)
        price = market_data.get('price', 100)
        
        order_value = order.quantity * price
        volume_value = volume * price
        
        if volume_value > 0:
            participation_rate = order_value / volume_value
            impact = self.impact_coefficient * np.sqrt(participation_rate)
            
            if order.side == OrderSide.BUY:
                return min(impact, 0.01)
            else:
                return max(-impact, -0.01)
        
        return 0.0


class AdvancedMarketImpactModel(MarketImpactModel):
    """Advanced market impact model with volatility and liquidity adjustments"""
    
    def __init__(self, base_impact: float = 0.001, volatility_factor: float = 0.5, liquidity_factor: float = 0.3):
        self.base_impact = base_impact
        self.volatility_factor = volatility_factor
        self.liquidity_factor = liquidity_factor
    
    def calculate_impact(self, order: Order, market_data: Dict[str, Any]) -> float:
        """Advanced impact model considering volatility and liquidity"""
        volume = market_data.get('volume', 1000000)
        price = market_data.get('price', 100)
        volatility = market_data.get('volatility', 0.02)  # Daily volatility
        avg_volume = market_data.get('avg_volume', volume)  # 20-day average volume
        
        order_value = order.quantity * price
        volume_value = volume * price
        
        if volume_value > 0:
            # Base participation rate impact
            participation_rate = order_value / volume_value
            base_impact = self.base_impact * np.sqrt(participation_rate)
            
            # Volatility adjustment (higher volatility = higher impact)
            volatility_adj = 1 + self.volatility_factor * (volatility / 0.02 - 1)
            
            # Liquidity adjustment (lower relative volume = higher impact)
            liquidity_ratio = volume / avg_volume if avg_volume > 0 else 1
            liquidity_adj = 1 + self.liquidity_factor * (1 / liquidity_ratio - 1)
            
            # Combined impact
            total_impact = base_impact * volatility_adj * liquidity_adj
            
            if order.side == OrderSide.BUY:
                return min(total_impact, 0.02)  # Cap at 2%
            else:
                return max(-total_impact, -0.02)
        
        return 0.0


class RealisticTradingSimulator:
    """Realistic trading simulator with market microstructure effects"""
    
    def __init__(self, trading_costs: TradingCosts, market_impact_model: MarketImpactModel):
        self.trading_costs = trading_costs
        self.market_impact_model = market_impact_model
        self.positions: Dict[str, Position] = {}
        self.cash = 0.0
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_counter = 0
        self.fill_counter = 0
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_market_impact = 0.0
        self.trade_count = 0
    
    def reset(self, initial_cash: float):
        """Reset simulator state"""
        self.positions.clear()
        self.cash = initial_cash
        self.orders.clear()
        self.fills.clear()
        self.order_counter = 0
        self.fill_counter = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_market_impact = 0.0
        self.trade_count = 0
    
    def submit_order(self, symbol: str, side: OrderSide, quantity: float, 
                    order_type: OrderType = OrderType.MARKET, price: Optional[float] = None) -> str:
        """Submit trading order"""
        
        order_id = f"ORDER_{self.order_counter:06d}"
        self.order_counter += 1
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=abs(quantity),
            price=price,
            timestamp=datetime.now()
        )
        
        self.orders[order_id] = order
        logger.debug(f"Submitted {order.side.value} order for {order.quantity} shares of {symbol}")
        
        return order_id
    
    def process_orders(self, market_data: Dict[str, Dict[str, Any]], timestamp: datetime) -> List[Fill]:
        """Process pending orders against market data"""
        
        new_fills = []
        
        for order_id, order in list(self.orders.items()):
            if order.status == OrderStatus.PENDING and order.symbol in market_data:
                fills = self._execute_order(order, market_data[order.symbol], timestamp)
                new_fills.extend(fills)
                
                # Remove filled or cancelled orders
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    del self.orders[order_id]
        
        return new_fills
    
    def _execute_order(self, order: Order, symbol_data: Dict[str, Any], timestamp: datetime) -> List[Fill]:
        """Execute individual order"""
        
        fills = []
        
        try:
            # Get market price
            if order.order_type == OrderType.MARKET:
                # For market orders, use mid price with bid-ask spread
                mid_price = symbol_data.get('close', symbol_data.get('price', 100))
                
                # Apply bid-ask spread
                if order.side == OrderSide.BUY:
                    execution_price = mid_price * (1 + self.trading_costs.bid_ask_spread / 2)
                else:
                    execution_price = mid_price * (1 - self.trading_costs.bid_ask_spread / 2)
                
                # Calculate market impact
                market_impact = self.market_impact_model.calculate_impact(order, symbol_data)
                execution_price *= (1 + market_impact)
                
                # Add random slippage
                slippage = np.random.normal(0, self.trading_costs.slippage_std)
                execution_price *= (1 + slippage)
                
                # Calculate commission
                trade_value = order.quantity * execution_price
                commission = max(
                    trade_value * self.trading_costs.commission_rate,
                    self.trading_costs.min_commission
                )
                
                # Check if we have enough cash (for buys) or shares (for sells)
                if order.side == OrderSide.BUY:
                    total_cost = trade_value + commission
                    if self.cash >= total_cost:
                        # Execute the trade
                        fill = self._create_fill(order, execution_price, order.quantity, 
                                               commission, slippage, market_impact, timestamp)
                        fills.append(fill)
                        self._update_position(fill)
                        order.status = OrderStatus.FILLED
                    else:
                        logger.warning(f"Insufficient cash for order {order.order_id}")
                        order.status = OrderStatus.REJECTED
                
                else:  # SELL
                    position = self.positions.get(order.symbol)
                    if position and position.quantity >= order.quantity:
                        # Execute the trade
                        fill = self._create_fill(order, execution_price, order.quantity,
                                               commission, slippage, market_impact, timestamp)
                        fills.append(fill)
                        self._update_position(fill)
                        order.status = OrderStatus.FILLED
                    else:
                        logger.warning(f"Insufficient shares for order {order.order_id}")
                        order.status = OrderStatus.REJECTED
            
            else:
                # Handle limit orders (simplified - assume immediate fill if price is favorable)
                current_price = symbol_data.get('close', 100)
                
                if ((order.side == OrderSide.BUY and current_price <= order.price) or
                    (order.side == OrderSide.SELL and current_price >= order.price)):
                    
                    # Convert to market-like execution at limit price
                    execution_price = order.price
                    
                    # Still apply some market impact and slippage (reduced)
                    market_impact = self.market_impact_model.calculate_impact(order, symbol_data) * 0.5
                    execution_price *= (1 + market_impact)
                    
                    slippage = np.random.normal(0, self.trading_costs.slippage_std * 0.5)
                    execution_price *= (1 + slippage)
                    
                    trade_value = order.quantity * execution_price
                    commission = max(
                        trade_value * self.trading_costs.commission_rate,
                        self.trading_costs.min_commission
                    )
                    
                    # Execute similar to market order logic
                    if order.side == OrderSide.BUY and self.cash >= (trade_value + commission):
                        fill = self._create_fill(order, execution_price, order.quantity,
                                               commission, slippage, market_impact, timestamp)
                        fills.append(fill)
                        self._update_position(fill)
                        order.status = OrderStatus.FILLED
                    elif order.side == OrderSide.SELL:
                        position = self.positions.get(order.symbol)
                        if position and position.quantity >= order.quantity:
                            fill = self._create_fill(order, execution_price, order.quantity,
                                                   commission, slippage, market_impact, timestamp)
                            fills.append(fill)
                            self._update_position(fill)
                            order.status = OrderStatus.FILLED
        
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
        
        return fills
    
    def _create_fill(self, order: Order, price: float, quantity: float, 
                    commission: float, slippage: float, market_impact: float, 
                    timestamp: datetime) -> Fill:
        """Create fill record"""
        
        fill_id = f"FILL_{self.fill_counter:06d}"
        self.fill_counter += 1
        
        fill = Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage,
            market_impact=market_impact
        )
        
        self.fills.append(fill)
        
        # Update tracking
        self.total_commission += commission
        self.total_slippage += abs(slippage * price * quantity)
        self.total_market_impact += abs(market_impact * price * quantity)
        self.trade_count += 1
        
        return fill
    
    def _update_position(self, fill: Fill):
        """Update position based on fill"""
        
        if fill.symbol not in self.positions:
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=0.0,
                avg_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                last_updated=fill.timestamp
            )
        
        position = self.positions[fill.symbol]
        
        if fill.side == OrderSide.BUY:
            # Update average price for buys
            total_cost = position.quantity * position.avg_price + fill.quantity * fill.price
            total_quantity = position.quantity + fill.quantity
            
            if total_quantity > 0:
                position.avg_price = total_cost / total_quantity
            
            position.quantity += fill.quantity
            
            # Update cash
            self.cash -= (fill.quantity * fill.price + fill.commission)
        
        else:  # SELL
            # Calculate realized P&L
            if position.quantity > 0:
                realized_pnl = (fill.price - position.avg_price) * fill.quantity
                position.realized_pnl += realized_pnl
            
            position.quantity -= fill.quantity
            
            # Update cash
            self.cash += (fill.quantity * fill.price - fill.commission)
            
            # If position is closed, reset average price
            if position.quantity == 0:
                position.avg_price = 0.0
        
        position.last_updated = fill.timestamp
    
    def update_positions_market_value(self, market_data: Dict[str, Dict[str, Any]]):
        """Update position market values and unrealized P&L"""
        
        for symbol, position in self.positions.items():
            if symbol in market_data and position.quantity != 0:
                current_price = market_data[symbol].get('close', market_data[symbol].get('price', 0))
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def get_portfolio_value(self, market_data: Dict[str, Dict[str, Any]]) -> float:
        """Calculate total portfolio value"""
        
        self.update_positions_market_value(market_data)
        
        total_value = self.cash
        for position in self.positions.values():
            total_value += position.market_value
        
        return total_value
    
    def get_portfolio_weights(self, market_data: Dict[str, Dict[str, Any]]) -> pd.Series:
        """Get current portfolio weights"""
        
        portfolio_value = self.get_portfolio_value(market_data)
        
        if portfolio_value <= 0:
            return pd.Series(dtype=float)
        
        weights = {}
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                weights[symbol] = position.market_value / portfolio_value
        
        return pd.Series(weights)
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics"""
        
        if self.trade_count == 0:
            return {}
        
        total_volume = sum(fill.quantity * fill.price for fill in self.fills)
        
        # Calculate cost breakdown
        commission_rate = self.total_commission / total_volume if total_volume > 0 else 0
        slippage_rate = self.total_slippage / total_volume if total_volume > 0 else 0
        impact_rate = self.total_market_impact / total_volume if total_volume > 0 else 0
        
        total_trading_costs = self.total_commission + self.total_slippage + self.total_market_impact
        
        # Analyze fill quality
        buy_fills = [f for f in self.fills if f.side == OrderSide.BUY]
        sell_fills = [f for f in self.fills if f.side == OrderSide.SELL]
        
        return {
            'total_trades': self.trade_count,
            'total_volume': total_volume,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_market_impact': self.total_market_impact,
            'total_trading_costs': total_trading_costs,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'market_impact_rate': impact_rate,
            'total_cost_rate': total_trading_costs / total_volume if total_volume > 0 else 0,
            'buy_trades': len(buy_fills),
            'sell_trades': len(sell_fills),
            'avg_trade_size': total_volume / self.trade_count if self.trade_count > 0 else 0,
            'avg_commission_per_trade': self.total_commission / self.trade_count if self.trade_count > 0 else 0,
            'positions_held': len([p for p in self.positions.values() if p.quantity != 0]),
            'realized_pnl': sum(p.realized_pnl for p in self.positions.values()),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values())
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get positions summary as DataFrame"""
        
        if not self.positions:
            return pd.DataFrame()
        
        position_data = []
        for position in self.positions.values():
            if position.quantity != 0:
                position_data.append({
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'last_updated': position.last_updated
                })
        
        return pd.DataFrame(position_data)
    
    def get_fill_history(self) -> pd.DataFrame:
        """Get fill history as DataFrame"""
        
        if not self.fills:
            return pd.DataFrame()
        
        fill_data = []
        for fill in self.fills:
            fill_data.append({
                'fill_id': fill.fill_id,
                'order_id': fill.order_id,
                'symbol': fill.symbol,
                'side': fill.side.value,
                'quantity': fill.quantity,
                'price': fill.price,
                'timestamp': fill.timestamp,
                'commission': fill.commission,
                'slippage': fill.slippage,
                'market_impact': fill.market_impact,
                'trade_value': fill.quantity * fill.price
            })
        
        return pd.DataFrame(fill_data)


def create_trading_simulator(
    commission_rate: float = 0.001,
    bid_ask_spread: float = 0.0005,
    market_impact_model: str = "advanced",
    **kwargs
) -> RealisticTradingSimulator:
    """Factory function to create configured trading simulator"""
    
    # Create trading costs configuration
    trading_costs = TradingCosts(
        commission_rate=commission_rate,
        bid_ask_spread=bid_ask_spread,
        **{k: v for k, v in kwargs.items() if k in TradingCosts.__dataclass_fields__}
    )
    
    # Create market impact model
    if market_impact_model == "linear":
        impact_model = LinearMarketImpactModel()
    elif market_impact_model == "sqrt":
        impact_model = SquareRootMarketImpactModel()
    else:  # advanced
        impact_model = AdvancedMarketImpactModel()
    
    simulator = RealisticTradingSimulator(trading_costs, impact_model)
    logger.info(f"Created trading simulator with {market_impact_model} market impact model")
    
    return simulator