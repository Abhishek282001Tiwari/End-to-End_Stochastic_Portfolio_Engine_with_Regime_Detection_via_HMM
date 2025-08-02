#!/usr/bin/env python3
"""
Portfolio Data Generator

Creates comprehensive portfolio datasets for backtesting and simulation:
- Portfolio holdings and weights over time
- Transaction history with realistic costs
- Benchmark data and performance attribution
- Risk attribution by factors and sectors
- Corporate actions and dividend data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

class TransactionType(Enum):
    """Types of portfolio transactions"""
    BUY = "Buy"
    SELL = "Sell"
    DIVIDEND = "Dividend"
    SPLIT = "Split"
    SPINOFF = "Spinoff"
    REBALANCE = "Rebalance"

class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequencies"""
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    ANNUALLY = "Annually"

@dataclass
class Transaction:
    """Individual portfolio transaction"""
    date: datetime
    symbol: str
    transaction_type: TransactionType
    quantity: float
    price: float
    commission: float
    market_impact: float
    notes: str = ""

@dataclass
class PortfolioSnapshot:
    """Portfolio holdings at a point in time"""
    date: datetime
    holdings: Dict[str, float]  # symbol -> quantity
    weights: Dict[str, float]   # symbol -> weight
    cash_balance: float
    total_value: float

class PortfolioDataGenerator:
    """Generate realistic portfolio data for backtesting"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize portfolio data generator
        
        Args:
            random_seed: Random seed for reproducible generation
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Default transaction costs
        self.commission_rates = {
            'equity': 0.001,      # 0.1% commission
            'bond': 0.0005,       # 0.05% commission  
            'etf': 0.0005,        # 0.05% commission
            'international': 0.002 # 0.2% commission
        }
        
        # Market impact models
        self.market_impact_params = {
            'linear_coefficient': 0.1,   # Price impact per % of volume
            'sqrt_coefficient': 0.05,    # Square root component
            'fixed_cost': 0.0001         # Fixed spread cost
        }
        
        # Dividend yields by sector
        self.dividend_yields = {
            'Technology': 0.015,    # 1.5% yield
            'Financials': 0.035,    # 3.5% yield
            'Healthcare': 0.025,    # 2.5% yield
            'Consumer': 0.028,      # 2.8% yield
            'Industrial': 0.022,    # 2.2% yield
            'Energy': 0.045,        # 4.5% yield
            'Utilities': 0.040,     # 4.0% yield
            'Telecom': 0.045,       # 4.5% yield
            'Bonds': 0.030,         # 3.0% yield
            'REITs': 0.050          # 5.0% yield
        }
    
    def generate_portfolio_strategy(self, 
                                  strategy_type: str = "mean_variance",
                                  risk_target: float = 0.15,
                                  return_target: float = 0.08) -> Dict[str, Any]:
        """
        Generate portfolio strategy parameters
        
        Args:
            strategy_type: Type of strategy (mean_variance, risk_parity, etc.)
            risk_target: Target portfolio volatility
            return_target: Target portfolio return
            
        Returns:
            Strategy configuration dictionary
        """
        
        strategies = {
            "mean_variance": {
                "name": "Mean Variance Optimization",
                "risk_target": risk_target,
                "return_target": return_target,
                "max_weight": 0.25,
                "min_weight": 0.01,
                "turnover_target": 0.5,
                "rebalance_frequency": RebalanceFrequency.MONTHLY,
                "constraints": {
                    "sector_max": 0.40,
                    "single_stock_max": 0.10,
                    "cash_min": 0.02
                }
            },
            
            "risk_parity": {
                "name": "Risk Parity",
                "risk_target": risk_target,
                "equal_risk_contribution": True,
                "max_weight": 0.20,
                "min_weight": 0.02,
                "turnover_target": 0.3,
                "rebalance_frequency": RebalanceFrequency.QUARTERLY,
                "constraints": {
                    "sector_max": 0.35,
                    "leverage_max": 1.0
                }
            },
            
            "momentum": {
                "name": "Momentum Strategy",
                "lookback_period": 126,  # 6 months
                "risk_target": risk_target,
                "max_weight": 0.15,
                "min_weight": 0.0,
                "turnover_target": 1.2,
                "rebalance_frequency": RebalanceFrequency.MONTHLY,
                "constraints": {
                    "top_n_assets": 20,
                    "momentum_threshold": 0.05
                }
            },
            
            "equal_weight": {
                "name": "Equal Weight",
                "risk_target": None,
                "return_target": None,
                "equal_weights": True,
                "max_weight": None,
                "min_weight": None,
                "turnover_target": 0.1,
                "rebalance_frequency": RebalanceFrequency.QUARTERLY,
                "constraints": {}
            }
        }
        
        return strategies.get(strategy_type, strategies["mean_variance"])
    
    def generate_portfolio_weights(self,
                                 asset_returns: pd.DataFrame,
                                 strategy_config: Dict[str, Any],
                                 start_date: str,
                                 end_date: str) -> pd.DataFrame:
        """
        Generate portfolio weights over time based on strategy
        
        Args:
            asset_returns: DataFrame of asset returns
            strategy_config: Strategy configuration
            start_date: Start date for portfolio
            end_date: End date for portfolio
            
        Returns:
            DataFrame with portfolio weights over time
        """
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        rebalance_freq = strategy_config.get('rebalance_frequency', RebalanceFrequency.MONTHLY)
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(dates, rebalance_freq)
        
        # Initialize weights DataFrame
        weights_df = pd.DataFrame(index=dates, columns=asset_returns.columns)
        
        # Generate initial weights
        current_weights = self._generate_initial_weights(asset_returns.columns, strategy_config)
        
        # Fill weights for each period
        for i, date in enumerate(dates):
            # Check if rebalancing date
            if date in rebalance_dates and i > 0:
                # Calculate new weights based on strategy
                lookback_returns = asset_returns.loc[:date].tail(
                    strategy_config.get('lookback_period', 252)
                )
                current_weights = self._calculate_strategy_weights(
                    lookback_returns, strategy_config, current_weights
                )
            
            weights_df.loc[date] = current_weights
        
        # Forward fill any missing values
        weights_df = weights_df.fillna(method='ffill')
        
        return weights_df
    
    def generate_transaction_history(self,
                                   weights_df: pd.DataFrame,
                                   price_data: Dict[str, pd.DataFrame],
                                   initial_capital: float = 1000000) -> List[Transaction]:
        """
        Generate realistic transaction history from weight changes
        
        Args:
            weights_df: Portfolio weights over time
            price_data: Price data for assets
            initial_capital: Starting portfolio value
            
        Returns:
            List of transactions
        """
        
        transactions = []
        current_holdings = {}  # symbol -> quantity
        current_cash = initial_capital
        
        for i, (date, weights) in enumerate(weights_df.iterrows()):
            if i == 0:
                # Initial portfolio construction
                for symbol, weight in weights.items():
                    if weight > 0 and symbol in price_data:
                        price = price_data[symbol].loc[date, 'Close'] if date in price_data[symbol].index else None
                        if price and price > 0:
                            target_value = initial_capital * weight
                            quantity = target_value / price
                            
                            # Calculate transaction costs
                            commission = self._calculate_commission(symbol, target_value)
                            market_impact = self._calculate_market_impact(symbol, quantity, price)
                            
                            transactions.append(Transaction(
                                date=date,
                                symbol=symbol,
                                transaction_type=TransactionType.BUY,
                                quantity=quantity,
                                price=price,
                                commission=commission,
                                market_impact=market_impact,
                                notes="Initial purchase"
                            ))
                            
                            current_holdings[symbol] = quantity
                            current_cash -= target_value + commission + market_impact
            
            else:
                # Rebalancing transactions
                prev_weights = weights_df.iloc[i-1]
                
                # Check if significant weight change
                weight_changes = (weights - prev_weights).abs()
                significant_changes = weight_changes[weight_changes > 0.01]  # > 1% change
                
                if len(significant_changes) > 0:
                    # Calculate current portfolio value
                    portfolio_value = current_cash
                    for symbol, quantity in current_holdings.items():
                        if symbol in price_data and date in price_data[symbol].index:
                            price = price_data[symbol].loc[date, 'Close']
                            portfolio_value += quantity * price
                    
                    # Execute rebalancing trades
                    for symbol in significant_changes.index:
                        if symbol in price_data and date in price_data[symbol].index:
                            price = price_data[symbol].loc[date, 'Close']
                            
                            target_value = portfolio_value * weights[symbol]
                            current_value = current_holdings.get(symbol, 0) * price
                            
                            trade_value = target_value - current_value
                            
                            if abs(trade_value) > 1000:  # Minimum trade size
                                trade_quantity = trade_value / price
                                transaction_type = TransactionType.BUY if trade_quantity > 0 else TransactionType.SELL
                                
                                commission = self._calculate_commission(symbol, abs(trade_value))
                                market_impact = self._calculate_market_impact(symbol, abs(trade_quantity), price)
                                
                                transactions.append(Transaction(
                                    date=date,
                                    symbol=symbol,
                                    transaction_type=transaction_type,
                                    quantity=abs(trade_quantity),
                                    price=price,
                                    commission=commission,
                                    market_impact=market_impact,
                                    notes="Rebalancing"
                                ))
                                
                                # Update holdings
                                current_holdings[symbol] = current_holdings.get(symbol, 0) + trade_quantity
                                current_cash -= trade_value + commission + market_impact
        
        return transactions
    
    def generate_dividend_history(self,
                                holdings_df: pd.DataFrame,
                                price_data: Dict[str, pd.DataFrame],
                                asset_sectors: Dict[str, str]) -> List[Transaction]:
        """
        Generate dividend payment history
        
        Args:
            holdings_df: Holdings quantities over time
            price_data: Price data for assets
            asset_sectors: Asset to sector mapping
            
        Returns:
            List of dividend transactions
        """
        
        dividend_transactions = []
        
        # Generate quarterly dividend dates
        start_date = holdings_df.index.min()
        end_date = holdings_df.index.max()
        
        # Quarterly dividend dates (end of March, June, September, December)
        dividend_dates = []
        current_date = start_date.replace(month=3, day=31)
        while current_date <= end_date:
            if current_date >= start_date:
                dividend_dates.append(current_date)
            
            # Next quarter
            if current_date.month == 3:
                current_date = current_date.replace(month=6, day=30)
            elif current_date.month == 6:
                current_date = current_date.replace(month=9, day=30)
            elif current_date.month == 9:
                current_date = current_date.replace(month=12, day=31)
            else:
                current_date = current_date.replace(year=current_date.year + 1, month=3, day=31)
        
        # Generate dividend payments
        for dividend_date in dividend_dates:
            # Find closest business day
            business_date = dividend_date
            while business_date.weekday() >= 5:  # Weekend
                business_date -= timedelta(days=1)
            
            if business_date in holdings_df.index:
                holdings = holdings_df.loc[business_date]
                
                for symbol, quantity in holdings.items():
                    if quantity > 0 and symbol in asset_sectors:
                        sector = asset_sectors[symbol]
                        annual_yield = self.dividend_yields.get(sector, 0.02)
                        quarterly_yield = annual_yield / 4
                        
                        if symbol in price_data and business_date in price_data[symbol].index:
                            price = price_data[symbol].loc[business_date, 'Close']
                            dividend_per_share = price * quarterly_yield
                            total_dividend = quantity * dividend_per_share
                            
                            if total_dividend > 1:  # Minimum dividend threshold
                                dividend_transactions.append(Transaction(
                                    date=business_date,
                                    symbol=symbol,
                                    transaction_type=TransactionType.DIVIDEND,
                                    quantity=quantity,
                                    price=dividend_per_share,
                                    commission=0,
                                    market_impact=0,
                                    notes=f"Quarterly dividend payment"
                                ))
        
        return dividend_transactions
    
    def generate_benchmark_data(self,
                              portfolio_returns: pd.Series,
                              benchmark_type: str = "market") -> pd.DataFrame:
        """
        Generate benchmark comparison data
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_type: Type of benchmark
            
        Returns:
            DataFrame with benchmark data
        """
        
        if benchmark_type == "market":
            # Market benchmark (slightly lower return, lower volatility)
            benchmark_returns = (
                portfolio_returns * 0.8 +  # 80% correlation
                np.random.normal(0.0003, 0.010, len(portfolio_returns))  # Independent component
            )
        
        elif benchmark_type == "sector":
            # Sector-specific benchmark
            benchmark_returns = (
                portfolio_returns * 0.9 +  # 90% correlation
                np.random.normal(0.0001, 0.008, len(portfolio_returns))
            )
        
        elif benchmark_type == "risk_free":
            # Risk-free rate (constant)
            risk_free_rate = 0.03 / 252  # 3% annual
            benchmark_returns = pd.Series(risk_free_rate, index=portfolio_returns.index)
        
        else:
            # Custom benchmark
            benchmark_returns = np.random.normal(0.0004, 0.012, len(portfolio_returns))
            benchmark_returns = pd.Series(benchmark_returns, index=portfolio_returns.index)
        
        # Create benchmark DataFrame
        benchmark_prices = 100 * (1 + benchmark_returns).cumprod()
        
        benchmark_df = pd.DataFrame({
            'Close': benchmark_prices,
            'Returns': benchmark_returns,
            'Cumulative_Returns': (1 + benchmark_returns).cumprod() - 1
        }, index=portfolio_returns.index)
        
        return benchmark_df
    
    def generate_factor_exposures(self,
                                asset_list: List[str],
                                dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate factor exposure data for performance attribution
        
        Args:
            asset_list: List of asset symbols
            dates: Date index
            
        Returns:
            DataFrame with factor exposures
        """
        
        # Define common factors
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Low_Vol', 'Sector']
        
        # Generate factor loadings for each asset
        factor_data = {}
        
        for asset in asset_list:
            factor_loadings = {}
            
            # Market factor (beta)
            factor_loadings['Market'] = np.random.normal(1.0, 0.3)  # Beta around 1.0
            
            # Size factor (negative = large cap, positive = small cap)
            factor_loadings['Size'] = np.random.normal(0.0, 0.5)
            
            # Value factor (negative = growth, positive = value)  
            factor_loadings['Value'] = np.random.normal(0.0, 0.4)
            
            # Momentum factor
            factor_loadings['Momentum'] = np.random.normal(0.0, 0.3)
            
            # Quality factor
            factor_loadings['Quality'] = np.random.normal(0.0, 0.3)
            
            # Low volatility factor
            factor_loadings['Low_Vol'] = np.random.normal(0.0, 0.2)
            
            # Sector factor (simplified as single value)
            factor_loadings['Sector'] = np.random.normal(0.0, 0.6)
            
            factor_data[asset] = factor_loadings
        
        # Create time-varying factor exposures (with some drift)
        factor_df = pd.DataFrame(index=dates)
        
        for factor in factors:
            for asset in asset_list:
                base_loading = factor_data[asset][factor]
                
                # Add time variation (random walk)
                time_series = [base_loading]
                for i in range(1, len(dates)):
                    change = np.random.normal(0, 0.01)  # Small random changes
                    time_series.append(time_series[-1] + change)
                
                factor_df[f"{asset}_{factor}"] = time_series
        
        return factor_df
    
    def generate_performance_attribution(self,
                                       portfolio_returns: pd.Series,
                                       portfolio_weights: pd.DataFrame,
                                       factor_exposures: pd.DataFrame,
                                       benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Generate performance attribution data
        
        Args:
            portfolio_returns: Portfolio return series
            portfolio_weights: Portfolio weights over time
            factor_exposures: Factor exposure data
            benchmark_returns: Benchmark return series
            
        Returns:
            DataFrame with attribution results
        """
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        
        # Simple Brinson attribution
        attribution_data = []
        
        for date in portfolio_returns.index[1:]:  # Skip first date
            if date in portfolio_weights.index:
                weights = portfolio_weights.loc[date]
                
                # Asset contribution
                asset_contributions = {}
                for asset in weights.index:
                    if weights[asset] > 0:
                        # Simplified contribution calculation
                        asset_return = np.random.normal(0.0005, 0.015)  # Mock asset return
                        contribution = weights[asset] * asset_return
                        asset_contributions[asset] = contribution
                
                # Factor attribution (simplified)
                factor_contributions = {
                    'Market': np.random.normal(0.0003, 0.008),
                    'Size': np.random.normal(0.0, 0.003),
                    'Value': np.random.normal(0.0, 0.004),
                    'Momentum': np.random.normal(0.0, 0.005),
                    'Selection': np.random.normal(0.0, 0.006),
                    'Interaction': np.random.normal(0.0, 0.002)
                }
                
                attribution_row = {
                    'Date': date,
                    'Portfolio_Return': portfolio_returns[date],
                    'Benchmark_Return': benchmark_returns[date],
                    'Excess_Return': excess_returns[date],
                    **factor_contributions,
                    **{f'Asset_{k}': v for k, v in asset_contributions.items()}
                }
                
                attribution_data.append(attribution_row)
        
        return pd.DataFrame(attribution_data)
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: RebalanceFrequency) -> List[datetime]:
        """Get rebalancing dates based on frequency"""
        
        rebalance_dates = []
        
        if frequency == RebalanceFrequency.DAILY:
            rebalance_dates = dates.tolist()
        
        elif frequency == RebalanceFrequency.WEEKLY:
            # Rebalance every Friday
            rebalance_dates = [d for d in dates if d.weekday() == 4]
        
        elif frequency == RebalanceFrequency.MONTHLY:
            # Last business day of each month
            monthly_ends = dates.to_series().groupby([dates.year, dates.month]).last()
            rebalance_dates = monthly_ends.tolist()
        
        elif frequency == RebalanceFrequency.QUARTERLY:
            # End of quarters
            quarterly_ends = dates.to_series().groupby([dates.year, dates.quarter]).last()
            rebalance_dates = quarterly_ends.tolist()
        
        elif frequency == RebalanceFrequency.ANNUALLY:
            # End of years
            yearly_ends = dates.to_series().groupby(dates.year).last()
            rebalance_dates = yearly_ends.tolist()
        
        return rebalance_dates
    
    def _generate_initial_weights(self, asset_list: List[str], strategy_config: Dict[str, Any]) -> Dict[str, float]:
        """Generate initial portfolio weights"""
        
        if strategy_config.get('equal_weights', False):
            # Equal weight strategy
            weight = 1.0 / len(asset_list)
            return {asset: weight for asset in asset_list}
        
        else:
            # Random initial weights with constraints
            max_weight = strategy_config.get('max_weight', 0.25)
            min_weight = strategy_config.get('min_weight', 0.01)
            
            # Generate random weights
            weights = np.random.dirichlet(np.ones(len(asset_list)))
            
            # Apply constraints
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()  # Renormalize
            
            return {asset: weight for asset, weight in zip(asset_list, weights)}
    
    def _calculate_strategy_weights(self,
                                  returns_data: pd.DataFrame,
                                  strategy_config: Dict[str, Any],
                                  current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate new portfolio weights based on strategy"""
        
        strategy_name = strategy_config.get('name', 'Mean Variance Optimization')
        
        if 'Equal Weight' in strategy_name:
            # Equal weight
            weight = 1.0 / len(returns_data.columns)
            return {asset: weight for asset in returns_data.columns}
        
        elif 'Momentum' in strategy_name:
            # Momentum strategy
            lookback = strategy_config.get('lookback_period', 126)
            momentum_scores = returns_data.tail(lookback).mean()
            
            # Select top assets
            top_n = strategy_config.get('top_n_assets', len(returns_data.columns))
            top_assets = momentum_scores.nlargest(top_n).index
            
            # Equal weight among top assets
            new_weights = {}
            for asset in returns_data.columns:
                if asset in top_assets:
                    new_weights[asset] = 1.0 / len(top_assets)
                else:
                    new_weights[asset] = 0.0
            
            return new_weights
        
        elif 'Risk Parity' in strategy_name:
            # Risk parity (simplified)
            volatilities = returns_data.std()
            inv_vol_weights = 1 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            
            return inv_vol_weights.to_dict()
        
        else:
            # Mean variance (simplified - equal weight with drift)
            # Add small random changes to current weights
            new_weights = {}
            for asset in current_weights:
                change = np.random.normal(0, 0.02)  # 2% std drift
                new_weights[asset] = max(0.01, current_weights[asset] + change)
            
            # Renormalize
            total_weight = sum(new_weights.values())
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
            
            return new_weights
    
    def _calculate_commission(self, symbol: str, trade_value: float) -> float:
        """Calculate commission cost for a trade"""
        
        # Determine asset type for commission rate
        if any(bond in symbol for bond in ['AGG', 'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'TIP', 'VTEB']):
            asset_type = 'bond'
        elif any(intl in symbol for intl in ['VEA', 'VWO', 'EFA', 'EEM']):
            asset_type = 'international'
        elif any(etf in symbol for etf in ['GLD', 'SLV', 'USO', 'XLF', 'XLK', 'XLE']):
            asset_type = 'etf'
        else:
            asset_type = 'equity'
        
        commission_rate = self.commission_rates.get(asset_type, 0.001)
        return trade_value * commission_rate
    
    def _calculate_market_impact(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate market impact cost"""
        
        # Simplified market impact model
        trade_value = abs(quantity * price)
        
        # Assume average daily volume (simplified)
        avg_daily_volume = 1000000  # $1M average daily volume
        
        # Participation rate
        participation_rate = trade_value / avg_daily_volume
        
        # Market impact components
        linear_impact = self.market_impact_params['linear_coefficient'] * participation_rate
        sqrt_impact = self.market_impact_params['sqrt_coefficient'] * np.sqrt(participation_rate)
        fixed_impact = self.market_impact_params['fixed_cost']
        
        total_impact_rate = linear_impact + sqrt_impact + fixed_impact
        
        return trade_value * total_impact_rate
    
    def save_portfolio_data(self, datasets: Dict[str, Any], output_dir: str = "data/generated/portfolios"):
        """Save portfolio datasets to files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save different types of data
        for name, data in datasets.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(os.path.join(output_dir, f"{name}.csv"))
            elif isinstance(data, list) and len(data) > 0:
                # Convert transaction list to DataFrame
                if isinstance(data[0], Transaction):
                    transactions_df = pd.DataFrame([
                        {
                            'Date': t.date,
                            'Symbol': t.symbol,
                            'Type': t.transaction_type.value,
                            'Quantity': t.quantity,
                            'Price': t.price,
                            'Commission': t.commission,
                            'Market_Impact': t.market_impact,
                            'Notes': t.notes
                        } for t in data
                    ])
                    transactions_df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        
        print(f"Portfolio data saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Create generator
    generator = PortfolioDataGenerator(random_seed=42)
    
    # Generate sample asset returns
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    returns_data = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (len(dates), len(assets))),
        index=dates,
        columns=assets
    )
    
    # Generate strategy configuration
    strategy_config = generator.generate_portfolio_strategy("mean_variance")
    
    # Generate portfolio weights
    weights_df = generator.generate_portfolio_weights(
        returns_data, strategy_config, '2023-01-01', '2023-12-31'
    )
    
    # Mock price data
    price_data = {}
    for asset in assets:
        price_data[asset] = pd.DataFrame({
            'Close': 100 * (1 + returns_data[asset]).cumprod()
        }, index=dates)
    
    # Generate transaction history
    transactions = generator.generate_transaction_history(weights_df, price_data)
    
    # Create datasets
    portfolio_datasets = {
        'weights': weights_df,
        'transactions': transactions,
        'strategy_config': pd.DataFrame([strategy_config])
    }
    
    # Save data
    generator.save_portfolio_data(portfolio_datasets)
    
    print(f"Generated {len(transactions)} transactions")
    print(f"Portfolio weights shape: {weights_df.shape}")