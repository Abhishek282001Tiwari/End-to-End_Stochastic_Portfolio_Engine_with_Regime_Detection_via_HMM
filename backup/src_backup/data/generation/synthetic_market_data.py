#!/usr/bin/env python3
"""
Synthetic Market Data Generator

Creates realistic financial time series data with proper statistical properties:
- Volatility clustering and fat tails
- Realistic correlation structures
- Multiple market regimes
- OHLCV data with proper relationships
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

@dataclass
class AssetConfig:
    """Configuration for individual asset data generation"""
    symbol: str
    sector: str
    initial_price: float
    annual_drift: float  # Expected annual return
    annual_volatility: float  # Annual volatility
    correlation_group: str  # For generating correlated movements
    market_beta: float = 1.0  # Beta to market factor
    mean_reversion_speed: float = 0.1  # Speed of mean reversion
    jump_intensity: float = 0.02  # Probability of price jumps
    jump_size_std: float = 0.05  # Standard deviation of jump sizes

@dataclass
class MarketRegime:
    """Market regime configuration"""
    name: str
    duration_mean: int  # Average regime duration in days
    volatility_multiplier: float  # Volatility adjustment factor
    drift_adjustment: float  # Return adjustment factor
    correlation_multiplier: float  # Correlation adjustment factor
    
class SyntheticMarketDataGenerator:
    """Generate realistic synthetic market data for portfolio testing"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the market data generator
        
        Args:
            random_seed: Random seed for reproducible data generation
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Define comprehensive asset universe
        self.asset_universe = self._create_asset_universe()
        
        # Define market regimes
        self.market_regimes = self._create_market_regimes()
        
        # Market factor for systematic risk
        self.market_factor = None
        
    def _create_asset_universe(self) -> Dict[str, AssetConfig]:
        """Create comprehensive asset universe across sectors"""
        
        assets = {}
        
        # Technology Sector (12 stocks)
        tech_stocks = {
            'AAPL': AssetConfig('AAPL', 'Technology', 150.0, 0.12, 0.25, 'tech', 1.2, 0.15, 0.03, 0.06),
            'MSFT': AssetConfig('MSFT', 'Technology', 300.0, 0.11, 0.23, 'tech', 1.1, 0.12, 0.02, 0.05),
            'GOOGL': AssetConfig('GOOGL', 'Technology', 2500.0, 0.13, 0.28, 'tech', 1.3, 0.18, 0.04, 0.07),
            'AMZN': AssetConfig('AMZN', 'Technology', 3200.0, 0.15, 0.35, 'tech', 1.4, 0.20, 0.05, 0.08),
            'TSLA': AssetConfig('TSLA', 'Technology', 800.0, 0.20, 0.55, 'tech', 1.8, 0.25, 0.08, 0.12),
            'META': AssetConfig('META', 'Technology', 350.0, 0.10, 0.32, 'tech', 1.3, 0.16, 0.04, 0.07),
            'NVDA': AssetConfig('NVDA', 'Technology', 450.0, 0.25, 0.45, 'tech', 1.6, 0.22, 0.06, 0.10),
            'CRM': AssetConfig('CRM', 'Technology', 220.0, 0.08, 0.30, 'tech', 1.2, 0.14, 0.03, 0.06),
            'ORCL': AssetConfig('ORCL', 'Technology', 85.0, 0.06, 0.22, 'tech', 0.9, 0.10, 0.02, 0.04),
            'ADBE': AssetConfig('ADBE', 'Technology', 550.0, 0.09, 0.27, 'tech', 1.1, 0.13, 0.03, 0.05),
            'INTC': AssetConfig('INTC', 'Technology', 55.0, 0.04, 0.28, 'tech', 1.0, 0.12, 0.02, 0.05),
            'AMD': AssetConfig('AMD', 'Technology', 120.0, 0.18, 0.40, 'tech', 1.5, 0.20, 0.05, 0.08)
        }
        
        # Financial Sector (8 stocks)
        financial_stocks = {
            'JPM': AssetConfig('JPM', 'Financials', 140.0, 0.08, 0.24, 'financial', 1.1, 0.12, 0.03, 0.05),
            'BAC': AssetConfig('BAC', 'Financials', 38.0, 0.07, 0.26, 'financial', 1.2, 0.14, 0.03, 0.06),
            'WFC': AssetConfig('WFC', 'Financials', 45.0, 0.06, 0.25, 'financial', 1.1, 0.13, 0.02, 0.05),
            'GS': AssetConfig('GS', 'Financials', 380.0, 0.09, 0.28, 'financial', 1.3, 0.15, 0.04, 0.06),
            'MS': AssetConfig('MS', 'Financials', 95.0, 0.08, 0.27, 'financial', 1.2, 0.14, 0.03, 0.06),
            'C': AssetConfig('C', 'Financials', 52.0, 0.05, 0.29, 'financial', 1.3, 0.16, 0.03, 0.07),
            'BRK-B': AssetConfig('BRK-B', 'Financials', 290.0, 0.10, 0.18, 'financial', 0.8, 0.08, 0.01, 0.03),
            'AXP': AssetConfig('AXP', 'Financials', 165.0, 0.07, 0.23, 'financial', 1.0, 0.11, 0.02, 0.05)
        }
        
        # Healthcare & Pharmaceuticals (8 stocks)
        healthcare_stocks = {
            'JNJ': AssetConfig('JNJ', 'Healthcare', 170.0, 0.06, 0.16, 'healthcare', 0.7, 0.08, 0.01, 0.03),
            'PFE': AssetConfig('PFE', 'Healthcare', 45.0, 0.05, 0.20, 'healthcare', 0.8, 0.10, 0.02, 0.04),
            'UNH': AssetConfig('UNH', 'Healthcare', 480.0, 0.09, 0.19, 'healthcare', 0.9, 0.09, 0.02, 0.04),
            'ABBV': AssetConfig('ABBV', 'Healthcare', 135.0, 0.07, 0.21, 'healthcare', 0.8, 0.10, 0.02, 0.04),
            'MRK': AssetConfig('MRK', 'Healthcare', 88.0, 0.06, 0.18, 'healthcare', 0.7, 0.09, 0.01, 0.03),
            'TMO': AssetConfig('TMO', 'Healthcare', 520.0, 0.08, 0.20, 'healthcare', 0.9, 0.10, 0.02, 0.04),
            'ABT': AssetConfig('ABT', 'Healthcare', 110.0, 0.07, 0.17, 'healthcare', 0.8, 0.08, 0.01, 0.03),
            'LLY': AssetConfig('LLY', 'Healthcare', 280.0, 0.10, 0.22, 'healthcare', 0.9, 0.11, 0.02, 0.04)
        }
        
        # Consumer & Retail (7 stocks)
        consumer_stocks = {
            'PG': AssetConfig('PG', 'Consumer', 145.0, 0.05, 0.15, 'consumer', 0.6, 0.07, 0.01, 0.02),
            'KO': AssetConfig('KO', 'Consumer', 58.0, 0.04, 0.16, 'consumer', 0.6, 0.08, 0.01, 0.03),
            'PEP': AssetConfig('PEP', 'Consumer', 165.0, 0.05, 0.16, 'consumer', 0.7, 0.08, 0.01, 0.03),
            'WMT': AssetConfig('WMT', 'Consumer', 145.0, 0.06, 0.18, 'consumer', 0.7, 0.09, 0.01, 0.03),
            'HD': AssetConfig('HD', 'Consumer', 320.0, 0.08, 0.21, 'consumer', 1.0, 0.10, 0.02, 0.04),
            'MCD': AssetConfig('MCD', 'Consumer', 250.0, 0.07, 0.17, 'consumer', 0.8, 0.08, 0.01, 0.03),
            'NKE': AssetConfig('NKE', 'Consumer', 120.0, 0.09, 0.24, 'consumer', 1.1, 0.12, 0.02, 0.05)
        }
        
        # Industrial & Energy (6 stocks)
        industrial_stocks = {
            'XOM': AssetConfig('XOM', 'Energy', 85.0, 0.03, 0.32, 'energy', 1.2, 0.16, 0.04, 0.08),
            'CVX': AssetConfig('CVX', 'Energy', 155.0, 0.04, 0.28, 'energy', 1.1, 0.14, 0.03, 0.07),
            'CAT': AssetConfig('CAT', 'Industrial', 210.0, 0.06, 0.26, 'industrial', 1.3, 0.13, 0.03, 0.06),
            'BA': AssetConfig('BA', 'Industrial', 200.0, 0.08, 0.35, 'industrial', 1.4, 0.18, 0.05, 0.09),
            'GE': AssetConfig('GE', 'Industrial', 95.0, 0.05, 0.30, 'industrial', 1.2, 0.15, 0.03, 0.07),
            'LMT': AssetConfig('LMT', 'Industrial', 450.0, 0.07, 0.20, 'industrial', 0.9, 0.10, 0.02, 0.04)
        }
        
        # Telecommunications & Utilities (4 stocks)
        telecom_stocks = {
            'VZ': AssetConfig('VZ', 'Telecom', 55.0, 0.03, 0.18, 'utility', 0.5, 0.09, 0.01, 0.03),
            'T': AssetConfig('T', 'Telecom', 28.0, 0.02, 0.20, 'utility', 0.6, 0.10, 0.01, 0.04),
            'NEE': AssetConfig('NEE', 'Utilities', 85.0, 0.05, 0.16, 'utility', 0.5, 0.08, 0.01, 0.02),
            'SO': AssetConfig('SO', 'Utilities', 68.0, 0.04, 0.15, 'utility', 0.4, 0.07, 0.01, 0.02)
        }
        
        # Bond ETFs and Indices
        bond_etfs = {
            'AGG': AssetConfig('AGG', 'Bonds', 110.0, 0.03, 0.08, 'bonds', -0.2, 0.04, 0.001, 0.01),
            'TLT': AssetConfig('TLT', 'Bonds', 140.0, 0.04, 0.15, 'bonds', -0.4, 0.06, 0.002, 0.02),
            'IEF': AssetConfig('IEF', 'Bonds', 115.0, 0.03, 0.10, 'bonds', -0.3, 0.05, 0.001, 0.015),
            'SHY': AssetConfig('SHY', 'Bonds', 86.0, 0.02, 0.05, 'bonds', -0.1, 0.02, 0.0005, 0.005),
            'HYG': AssetConfig('HYG', 'Bonds', 88.0, 0.05, 0.12, 'bonds', 0.3, 0.08, 0.003, 0.03),
            'LQD': AssetConfig('LQD', 'Bonds', 125.0, 0.04, 0.09, 'bonds', -0.1, 0.05, 0.002, 0.02),
            'TIP': AssetConfig('TIP', 'Bonds', 120.0, 0.03, 0.08, 'bonds', -0.2, 0.04, 0.001, 0.01),
            'VTEB': AssetConfig('VTEB', 'Bonds', 55.0, 0.03, 0.06, 'bonds', -0.1, 0.03, 0.001, 0.01)
        }
        
        # Commodity ETFs
        commodity_etfs = {
            'GLD': AssetConfig('GLD', 'Commodities', 180.0, 0.05, 0.20, 'commodities', -0.1, 0.10, 0.02, 0.05),
            'SLV': AssetConfig('SLV', 'Commodities', 25.0, 0.08, 0.35, 'commodities', 0.2, 0.18, 0.04, 0.08),
            'USO': AssetConfig('USO', 'Commodities', 45.0, 0.02, 0.45, 'commodities', 0.8, 0.25, 0.06, 0.12),
            'DBA': AssetConfig('DBA', 'Commodities', 18.0, 0.04, 0.25, 'commodities', 0.3, 0.12, 0.03, 0.06),
            'PDBC': AssetConfig('PDBC', 'Commodities', 22.0, 0.03, 0.22, 'commodities', 0.4, 0.11, 0.02, 0.05),
            'IAU': AssetConfig('IAU', 'Commodities', 35.0, 0.05, 0.20, 'commodities', -0.1, 0.10, 0.02, 0.05)
        }
        
        # International ETFs
        international_etfs = {
            'VEA': AssetConfig('VEA', 'International', 50.0, 0.07, 0.18, 'international', 0.8, 0.09, 0.02, 0.04),
            'VWO': AssetConfig('VWO', 'International', 52.0, 0.09, 0.25, 'international', 1.1, 0.13, 0.03, 0.06),
            'EFA': AssetConfig('EFA', 'International', 75.0, 0.07, 0.19, 'international', 0.9, 0.10, 0.02, 0.04),
            'EEM': AssetConfig('EEM', 'International', 48.0, 0.08, 0.24, 'international', 1.2, 0.12, 0.03, 0.06)
        }
        
        # Sector ETFs
        sector_etfs = {
            'XLF': AssetConfig('XLF', 'Sector', 35.0, 0.08, 0.24, 'financial', 1.1, 0.12, 0.03, 0.05),
            'XLK': AssetConfig('XLK', 'Sector', 155.0, 0.12, 0.25, 'tech', 1.2, 0.15, 0.03, 0.06),
            'XLE': AssetConfig('XLE', 'Sector', 85.0, 0.03, 0.30, 'energy', 1.2, 0.15, 0.04, 0.08),
            'XLV': AssetConfig('XLV', 'Sector', 125.0, 0.07, 0.18, 'healthcare', 0.8, 0.09, 0.02, 0.04),
            'XLI': AssetConfig('XLI', 'Sector', 95.0, 0.06, 0.24, 'industrial', 1.2, 0.12, 0.03, 0.06),
            'XLP': AssetConfig('XLP', 'Sector', 75.0, 0.05, 0.16, 'consumer', 0.7, 0.08, 0.01, 0.03)
        }
        
        # Combine all assets
        assets.update(tech_stocks)
        assets.update(financial_stocks)
        assets.update(healthcare_stocks)
        assets.update(consumer_stocks)
        assets.update(industrial_stocks)
        assets.update(telecom_stocks)
        assets.update(bond_etfs)
        assets.update(commodity_etfs)
        assets.update(international_etfs)
        assets.update(sector_etfs)
        
        return assets
    
    def _create_market_regimes(self) -> List[MarketRegime]:
        """Define market regimes with different characteristics"""
        return [
            MarketRegime("Bull Market", 180, 0.8, 0.3, 0.9),      # Low vol, positive drift, lower correlation
            MarketRegime("Bear Market", 120, 1.5, -0.4, 1.3),    # High vol, negative drift, higher correlation  
            MarketRegime("High Volatility", 60, 2.0, 0.0, 1.4),  # Very high vol, neutral drift, high correlation
            MarketRegime("Low Volatility", 90, 0.6, 0.1, 0.8),   # Very low vol, slight positive drift, low correlation
            MarketRegime("Sideways Market", 150, 1.0, 0.0, 1.0)  # Normal vol, neutral drift, normal correlation
        ]
    
    def generate_market_data(self, 
                           start_date: str = "2020-01-01",
                           end_date: str = "2024-01-01", 
                           frequency: str = "D") -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive market data for all assets
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation  
            frequency: Data frequency ('D' for daily, 'H' for hourly)
            
        Returns:
            Dictionary containing DataFrames for different asset classes
        """
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_periods = len(dates)
        
        # Generate regime sequence
        regime_sequence = self._generate_regime_sequence(n_periods)
        
        # Generate market factor (systematic risk)
        market_factor = self._generate_market_factor(dates, regime_sequence)
        self.market_factor = market_factor
        
        # Generate data for each asset class
        all_data = {}
        
        # Group assets by correlation group for correlated generation
        correlation_groups = {}
        for symbol, config in self.asset_universe.items():
            group = config.correlation_group
            if group not in correlation_groups:
                correlation_groups[group] = []
            correlation_groups[group].append((symbol, config))
        
        # Generate correlated asset data by group
        for group_name, assets in correlation_groups.items():
            group_data = self._generate_correlated_asset_group(
                assets, dates, regime_sequence, market_factor
            )
            all_data.update(group_data)
        
        # Organize data by asset class
        organized_data = self._organize_data_by_class(all_data, dates)
        
        return organized_data
    
    def _generate_regime_sequence(self, n_periods: int) -> np.ndarray:
        """Generate sequence of market regimes over time"""
        
        regimes = []
        current_period = 0
        
        while current_period < n_periods:
            # Select random regime
            regime_idx = np.random.choice(len(self.market_regimes))
            regime = self.market_regimes[regime_idx]
            
            # Generate regime duration
            duration = max(10, int(np.random.exponential(regime.duration_mean)))
            duration = min(duration, n_periods - current_period)
            
            # Add regime periods
            regimes.extend([regime_idx] * duration)
            current_period += duration
        
        return np.array(regimes[:n_periods])
    
    def _generate_market_factor(self, dates: pd.DatetimeIndex, 
                              regime_sequence: np.ndarray) -> pd.Series:
        """Generate market-wide systematic factor"""
        
        n_periods = len(dates)
        returns = np.zeros(n_periods)
        
        for i in range(1, n_periods):
            regime = self.market_regimes[regime_sequence[i]]
            
            # Base market parameters
            daily_drift = 0.08 / 252  # 8% annual drift
            daily_vol = 0.16 / np.sqrt(252)  # 16% annual volatility
            
            # Adjust for regime
            adjusted_drift = daily_drift + regime.drift_adjustment / 252
            adjusted_vol = daily_vol * regime.volatility_multiplier
            
            # Generate return with possible jumps
            base_return = np.random.normal(adjusted_drift, adjusted_vol)
            
            # Add jumps during high volatility regimes
            if regime.name in ["Bear Market", "High Volatility"] and np.random.random() < 0.02:
                jump_size = np.random.normal(0, 0.05) * (-1 if regime.name == "Bear Market" else 1)
                base_return += jump_size
            
            returns[i] = base_return
        
        # Convert to price series
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.Series(prices, index=dates, name='market_factor')
    
    def _generate_correlated_asset_group(self, 
                                       assets: List[Tuple[str, AssetConfig]], 
                                       dates: pd.DatetimeIndex,
                                       regime_sequence: np.ndarray,
                                       market_factor: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate correlated asset data for a group"""
        
        n_periods = len(dates)
        n_assets = len(assets)
        
        if n_assets == 1:
            # Single asset - no correlation structure needed
            symbol, config = assets[0]
            asset_data = self._generate_single_asset_data(
                config, dates, regime_sequence, market_factor
            )
            return {symbol: asset_data}
        
        # Generate correlation matrix for the group
        base_correlation = 0.3 + 0.4 * np.random.random()  # Base correlation 0.3-0.7
        correlation_matrix = self._generate_correlation_matrix(n_assets, base_correlation)
        
        # Generate multivariate returns
        all_returns = np.zeros((n_periods, n_assets))
        
        for i in range(1, n_periods):
            regime = self.market_regimes[regime_sequence[i]]
            
            # Generate independent innovations
            innovations = np.random.multivariate_normal(
                np.zeros(n_assets), 
                correlation_matrix * regime.correlation_multiplier
            )
            
            # Generate asset-specific returns
            for j, (symbol, config) in enumerate(assets):
                # Base parameters
                daily_drift = config.annual_drift / 252
                daily_vol = config.annual_volatility / np.sqrt(252)
                
                # Regime adjustments
                adjusted_drift = daily_drift + regime.drift_adjustment / 252
                adjusted_vol = daily_vol * regime.volatility_multiplier
                
                # Market factor exposure
                market_return = market_factor.pct_change().iloc[i] if i > 0 else 0
                beta_component = config.market_beta * market_return
                
                # Idiosyncratic component
                idiosyncratic = innovations[j] * adjusted_vol
                
                # Mean reversion component
                if i > 10:  # Need some history
                    recent_returns = all_returns[i-10:i, j]
                    mean_return = np.mean(recent_returns)
                    mean_reversion = -config.mean_reversion_speed * mean_return
                else:
                    mean_reversion = 0
                
                # Combine components
                total_return = adjusted_drift + beta_component + idiosyncratic + mean_reversion
                
                # Add jumps
                if np.random.random() < config.jump_intensity:
                    jump = np.random.normal(0, config.jump_size_std)
                    total_return += jump
                
                all_returns[i, j] = total_return
        
        # Convert returns to OHLCV data for each asset
        group_data = {}
        for j, (symbol, config) in enumerate(assets):
            returns = all_returns[:, j]
            asset_data = self._returns_to_ohlcv(returns, config.initial_price, dates, config)
            group_data[symbol] = asset_data
        
        return group_data
    
    def _generate_single_asset_data(self, 
                                  config: AssetConfig,
                                  dates: pd.DatetimeIndex,
                                  regime_sequence: np.ndarray, 
                                  market_factor: pd.Series) -> pd.DataFrame:
        """Generate data for a single asset"""
        
        n_periods = len(dates)
        returns = np.zeros(n_periods)
        
        for i in range(1, n_periods):
            regime = self.market_regimes[regime_sequence[i]]
            
            # Base parameters
            daily_drift = config.annual_drift / 252
            daily_vol = config.annual_volatility / np.sqrt(252)
            
            # Regime adjustments
            adjusted_drift = daily_drift + regime.drift_adjustment / 252
            adjusted_vol = daily_vol * regime.volatility_multiplier
            
            # Market factor exposure
            market_return = market_factor.pct_change().iloc[i] if i > 0 else 0
            beta_component = config.market_beta * market_return
            
            # Idiosyncratic component
            idiosyncratic = np.random.normal(0, adjusted_vol)
            
            # Mean reversion
            if i > 10:
                recent_returns = returns[i-10:i]
                mean_return = np.mean(recent_returns)
                mean_reversion = -config.mean_reversion_speed * mean_return
            else:
                mean_reversion = 0
            
            # Combine components
            total_return = adjusted_drift + beta_component + idiosyncratic + mean_reversion
            
            # Add jumps
            if np.random.random() < config.jump_intensity:
                jump = np.random.normal(0, config.jump_size_std)
                total_return += jump
            
            returns[i] = total_return
        
        # Convert to OHLCV
        return self._returns_to_ohlcv(returns, config.initial_price, dates, config)
    
    def _generate_correlation_matrix(self, n_assets: int, base_correlation: float) -> np.ndarray:
        """Generate realistic correlation matrix"""
        
        if n_assets == 1:
            return np.array([[1.0]])
        
        # Start with identity matrix
        corr_matrix = np.eye(n_assets)
        
        # Fill off-diagonal elements
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # Add some randomness to base correlation
                correlation = base_correlation + np.random.normal(0, 0.1)
                correlation = np.clip(correlation, 0.1, 0.9)  # Keep reasonable bounds
                
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize diagonal to 1
        diag_sqrt = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        return corr_matrix
    
    def _returns_to_ohlcv(self, 
                         returns: np.ndarray, 
                         initial_price: float,
                         dates: pd.DatetimeIndex,
                         config: AssetConfig) -> pd.DataFrame:
        """Convert return series to OHLCV format"""
        
        n_periods = len(returns)
        
        # Generate price series
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        
        for i in range(n_periods):
            if i == 0:
                # First day
                open_price = initial_price
                close_price = prices[i] if prices[i] > 0 else initial_price
            else:
                open_price = prices[i-1]
                close_price = prices[i]
            
            # Generate high/low based on intraday volatility
            daily_vol = config.annual_volatility / np.sqrt(252)
            intraday_range = abs(returns[i]) + np.random.exponential(daily_vol * 0.5)
            
            high_price = max(open_price, close_price) * (1 + intraday_range * 0.5)
            low_price = min(open_price, close_price) * (1 - intraday_range * 0.5)
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume (higher volume during high volatility)
            base_volume = 1000000  # Base daily volume
            volatility_multiplier = 1 + abs(returns[i]) * 10  # Higher vol = higher volume
            volume = int(base_volume * volatility_multiplier * np.random.lognormal(0, 0.3))
            
            data.append({
                'Date': dates[i],
                'Open': round(open_price, 2),
                'High': round(high_price, 2), 
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume,
                'Adj_Close': round(close_price, 2)  # Assuming no dividends/splits for now
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        return df
    
    def _organize_data_by_class(self, 
                              all_data: Dict[str, pd.DataFrame],
                              dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """Organize generated data by asset class"""
        
        organized = {
            'equities': {},
            'bonds': {},
            'commodities': {},
            'international': {},
            'sectors': {},
            'market_data': {}
        }
        
        for symbol, data in all_data.items():
            config = self.asset_universe[symbol]
            
            if config.sector in ['Technology', 'Financials', 'Healthcare', 'Consumer', 'Industrial', 'Energy', 'Telecom', 'Utilities']:
                organized['equities'][symbol] = data
            elif config.sector == 'Bonds':
                organized['bonds'][symbol] = data
            elif config.sector == 'Commodities':
                organized['commodities'][symbol] = data
            elif config.sector == 'International':
                organized['international'][symbol] = data
            elif config.sector == 'Sector':
                organized['sectors'][symbol] = data
        
        # Add market factor and regime data
        organized['market_data']['market_factor'] = pd.DataFrame({
            'Close': self.market_factor,
            'Volume': 0  # Market factor doesn't have volume
        })
        
        return organized
    
    def generate_economic_indicators(self, 
                                   start_date: str = "2020-01-01",
                                   end_date: str = "2024-01-01") -> pd.DataFrame:
        """Generate synthetic economic indicators"""
        
        dates = pd.date_range(start=start_date, end=end_date, freq='M')  # Monthly data
        n_periods = len(dates)
        
        # Initialize indicators
        indicators = {}
        
        # GDP Growth (quarterly, interpolated to monthly)
        gdp_growth = []
        current_gdp = 2.5  # Starting at 2.5% annual growth
        for i in range(n_periods):
            # Random walk with mean reversion
            change = np.random.normal(0, 0.2) - 0.1 * (current_gdp - 2.5)
            current_gdp += change
            current_gdp = np.clip(current_gdp, -3.0, 8.0)  # Reasonable bounds
            gdp_growth.append(current_gdp)
        
        indicators['GDP_Growth'] = gdp_growth
        
        # Inflation Rate  
        inflation = []
        current_inflation = 2.0  # Start at 2% target
        for i in range(n_periods):
            change = np.random.normal(0, 0.15) - 0.05 * (current_inflation - 2.0)
            current_inflation += change
            current_inflation = np.clip(current_inflation, -1.0, 6.0)
            inflation.append(current_inflation)
        
        indicators['Inflation_Rate'] = inflation
        
        # Unemployment Rate
        unemployment = []
        current_unemployment = 5.0  # Start at 5%
        for i in range(n_periods):
            change = np.random.normal(0, 0.1) - 0.02 * (current_unemployment - 5.0)
            current_unemployment += change
            current_unemployment = np.clip(current_unemployment, 2.0, 12.0)
            unemployment.append(current_unemployment)
        
        indicators['Unemployment_Rate'] = unemployment
        
        # Federal Funds Rate
        fed_rate = []
        current_rate = 2.0  # Start at 2%
        for i in range(n_periods):
            # Fed rate responds to inflation and unemployment
            target_rate = max(0, indicators['Inflation_Rate'][i] + 0.5 - 0.2 * (indicators['Unemployment_Rate'][i] - 5.0))
            change = 0.1 * (target_rate - current_rate) + np.random.normal(0, 0.05)
            current_rate += change
            current_rate = np.clip(current_rate, 0.0, 8.0)
            fed_rate.append(current_rate)
        
        indicators['Fed_Funds_Rate'] = fed_rate
        
        # 10-Year Treasury Yield
        treasury_yield = []
        for i in range(n_periods):
            # Treasury yield related to fed rate + term premium
            term_premium = 1.0 + 0.5 * np.sin(i * 2 * np.pi / 12)  # Seasonal component
            yield_rate = indicators['Fed_Funds_Rate'][i] + term_premium + np.random.normal(0, 0.1)
            yield_rate = np.clip(yield_rate, 0.5, 8.0)
            treasury_yield.append(yield_rate)
        
        indicators['Treasury_10Y'] = treasury_yield
        
        # VIX (Fear Index)
        vix = []
        for i in range(n_periods):
            # VIX increases with economic uncertainty
            base_vix = 20 + 5 * abs(indicators['GDP_Growth'][i] - 2.5) + 3 * abs(indicators['Inflation_Rate'][i] - 2.0)
            vix_level = base_vix + np.random.normal(0, 3)
            vix_level = np.clip(vix_level, 10, 80)
            vix.append(vix_level)
        
        indicators['VIX'] = vix
        
        # Create DataFrame
        econ_data = pd.DataFrame(indicators, index=dates)
        
        return econ_data
    
    def save_data(self, data: Dict[str, any], output_dir: str = "data/generated"):
        """Save generated data to files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save equity data
        if 'equities' in data:
            equity_dir = os.path.join(output_dir, 'equities')
            os.makedirs(equity_dir, exist_ok=True)
            
            for symbol, df in data['equities'].items():
                df.to_csv(os.path.join(equity_dir, f"{symbol}.csv"))
        
        # Save other asset classes
        for asset_class in ['bonds', 'commodities', 'international', 'sectors']:
            if asset_class in data:
                class_dir = os.path.join(output_dir, asset_class)
                os.makedirs(class_dir, exist_ok=True)
                
                for symbol, df in data[asset_class].items():
                    df.to_csv(os.path.join(class_dir, f"{symbol}.csv"))
        
        # Save market data
        if 'market_data' in data:
            market_dir = os.path.join(output_dir, 'market_data')
            os.makedirs(market_dir, exist_ok=True)
            
            for name, df in data['market_data'].items():
                df.to_csv(os.path.join(market_dir, f"{name}.csv"))
        
        print(f"Data saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Create generator
    generator = SyntheticMarketDataGenerator(random_seed=42)
    
    # Generate comprehensive market data
    market_data = generator.generate_market_data(
        start_date="2020-01-01", 
        end_date="2024-01-01"
    )
    
    # Generate economic indicators
    econ_data = generator.generate_economic_indicators(
        start_date="2020-01-01",
        end_date="2024-01-01"
    )
    
    # Save data
    market_data['economic_indicators'] = {'indicators': econ_data}
    generator.save_data(market_data)
    
    # Print summary
    print("Generated data summary:")
    for asset_class, assets in market_data.items():
        if isinstance(assets, dict):
            print(f"{asset_class}: {len(assets)} assets")
        else:
            print(f"{asset_class}: data generated")