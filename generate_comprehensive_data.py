#!/usr/bin/env python3
"""
Comprehensive Data Generation Script

Generates complete dataset for the Stochastic Portfolio Engine including:
- Synthetic market data for 70+ assets across all classes
- Regime detection training data with ground truth labels
- Portfolio simulation data with realistic transaction costs
- Economic indicators and macroeconomic time series
- Data validation and quality reports
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.data.generation.synthetic_market_data import SyntheticMarketDataGenerator
from src.data.generation.regime_data_generator import RegimeDataGenerator
from src.data.generation.portfolio_data_generator import PortfolioDataGenerator
from src.data.generation.data_validator import DataValidator

def create_directory_structure():
    """Create organized directory structure for generated data"""
    
    directories = [
        'data/generated',
        'data/generated/market_data',
        'data/generated/market_data/equities', 
        'data/generated/market_data/bonds',
        'data/generated/market_data/commodities',
        'data/generated/market_data/international',
        'data/generated/market_data/sectors',
        'data/generated/market_data/economic_indicators',
        'data/generated/regimes',
        'data/generated/portfolios',
        'data/generated/validation',
        'data/generated/metadata'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure created")

def generate_market_data(start_date: str = "2015-01-01", end_date: str = "2024-01-01"):
    """Generate comprehensive market data"""
    
    print("\n🔄 Generating synthetic market data...")
    
    # Initialize generator
    market_generator = SyntheticMarketDataGenerator(random_seed=42)
    
    # Generate market data
    market_data = market_generator.generate_market_data(
        start_date=start_date,
        end_date=end_date,
        frequency="D"
    )
    
    # Generate economic indicators
    econ_indicators = market_generator.generate_economic_indicators(
        start_date=start_date,
        end_date=end_date
    )
    
    # Add economic indicators to market data
    market_data['economic_indicators'] = {'indicators': econ_indicators}
    
    # Save market data
    market_generator.save_data(market_data, "data/generated/market_data")
    
    print(f"✅ Generated market data for {sum(len(assets) for assets in market_data.values() if isinstance(assets, dict))} assets")
    
    return market_data

def generate_regime_data(start_date: str = "2015-01-01", end_date: str = "2024-01-01"):
    """Generate labeled regime detection data"""
    
    print("\n🔄 Generating regime detection data...")
    
    # Initialize generator
    regime_generator = RegimeDataGenerator(random_seed=42)
    
    # Generate labeled dataset
    regime_data = regime_generator.create_labeled_dataset(
        start_date=start_date,
        end_date=end_date
    )
    
    # Generate validation datasets
    validation_sets = regime_generator.create_validation_datasets(n_datasets=10)
    
    # Save regime data
    regime_generator.save_regime_data(regime_data, "data/generated/regimes")
    
    # Save validation sets
    for i, val_set in enumerate(validation_sets):
        val_dir = f"data/generated/regimes/validation_set_{i}"
        os.makedirs(val_dir, exist_ok=True)
        
        for name, data in val_set.items():
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                data.to_csv(os.path.join(val_dir, f"{name}.csv"))
    
    print(f"✅ Generated regime data with {len(regime_data)} datasets and {len(validation_sets)} validation sets")
    
    return regime_data

def generate_portfolio_data(market_data: dict, start_date: str = "2020-01-01", end_date: str = "2024-01-01"):
    """Generate portfolio simulation data"""
    
    print("\n🔄 Generating portfolio simulation data...")
    
    # Initialize generator
    portfolio_generator = PortfolioDataGenerator(random_seed=42)
    
    # Extract equity data for portfolio construction
    equity_symbols = list(market_data.get('equities', {}).keys())
    if len(equity_symbols) < 10:
        print("⚠️ Not enough equity data for portfolio generation")
        return {}
    
    # Select subset of assets for portfolio
    selected_assets = equity_symbols[:20]  # Top 20 assets
    
    # Create returns data
    returns_data = {}
    price_data = {}
    
    for symbol in selected_assets:
        if symbol in market_data['equities']:
            asset_data = market_data['equities'][symbol]
            if 'Close' in asset_data.columns:
                returns = asset_data['Close'].pct_change().dropna()
                returns_data[symbol] = returns
                price_data[symbol] = asset_data
    
    if len(returns_data) < 5:
        print("⚠️ Insufficient return data for portfolio generation")
        return {}
    
    returns_df = pd.DataFrame(returns_data).dropna()
    
    # Generate multiple portfolio strategies
    strategies = ['mean_variance', 'risk_parity', 'momentum', 'equal_weight']
    portfolio_datasets = {}
    
    for strategy in strategies:
        print(f"  Generating {strategy} portfolio...")
        
        # Generate strategy configuration
        strategy_config = portfolio_generator.generate_portfolio_strategy(strategy)
        
        # Generate portfolio weights
        weights_df = portfolio_generator.generate_portfolio_weights(
            returns_df, strategy_config, start_date, end_date
        )
        
        # Generate transaction history
        transactions = portfolio_generator.generate_transaction_history(
            weights_df, price_data
        )
        
        # Generate benchmark data
        portfolio_returns = (returns_df * weights_df.shift(1)).sum(axis=1).dropna()
        benchmark_data = portfolio_generator.generate_benchmark_data(portfolio_returns)
        
        # Store strategy data
        portfolio_datasets[strategy] = {
            'weights': weights_df,
            'transactions': transactions,
            'benchmark': benchmark_data,
            'config': strategy_config
        }
    
    # Save portfolio data
    for strategy, data in portfolio_datasets.items():
        strategy_dir = f"data/generated/portfolios/{strategy}"
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save weights
        data['weights'].to_csv(os.path.join(strategy_dir, "weights.csv"))
        
        # Save transactions
        if data['transactions']:
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
                } for t in data['transactions']
            ])
            transactions_df.to_csv(os.path.join(strategy_dir, "transactions.csv"), index=False)
        
        # Save benchmark
        data['benchmark'].to_csv(os.path.join(strategy_dir, "benchmark.csv"))
        
        # Save config
        config_df = pd.DataFrame([data['config']])
        config_df.to_csv(os.path.join(strategy_dir, "config.csv"), index=False)
    
    print(f"✅ Generated portfolio data for {len(strategies)} strategies")
    
    return portfolio_datasets

def validate_generated_data(market_data: dict):
    """Validate the quality of generated data"""
    
    print("\n🔄 Validating generated data quality...")
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate equity data
    equity_data = market_data.get('equities', {})
    if equity_data:
        # Sample a few assets for validation
        sample_assets = dict(list(equity_data.items())[:5])
        
        # Run validation
        validation_report = validator.validate_portfolio_data(sample_assets)
        
        # Export validation report
        validator.export_report(validation_report, "data/generated/validation/data_quality_report.txt")
        
        # Create summary
        validation_summary = {
            'overall_score': validation_report.overall_score,
            'total_tests': len(validation_report.validation_results),
            'passed_tests': sum(1 for r in validation_report.validation_results if r.passed),
            'timestamp': validation_report.timestamp.isoformat(),
            'recommendations': validation_report.recommendations
        }
        
        # Save summary as JSON
        import json
        with open("data/generated/validation/validation_summary.json", 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        
        print(f"✅ Data validation complete - Overall Score: {validation_report.overall_score:.2f}")
        print(f"   Passed: {validation_summary['passed_tests']}/{validation_summary['total_tests']} tests")
    
    return validation_report if 'validation_report' in locals() else None

def create_data_documentation():
    """Create comprehensive documentation for generated data"""
    
    print("\n🔄 Creating data documentation...")
    
    # Create README for generated data
    readme_content = """# Generated Portfolio Engine Dataset

This dataset contains comprehensive synthetic financial data for testing and demonstrating the Stochastic Portfolio Engine.

## Dataset Overview

### Market Data
- **Equities**: 45+ individual stocks across 6 major sectors
- **Bonds**: 8 bond ETFs covering different duration and credit qualities  
- **Commodities**: 6 commodity ETFs including precious metals, energy, and agriculture
- **International**: 4 international/emerging market ETFs
- **Sectors**: 6 sector-specific ETFs
- **Total Assets**: 70+ instruments with complete OHLCV data

### Time Coverage
- **Start Date**: 2015-01-01
- **End Date**: 2024-01-01  
- **Frequency**: Daily
- **Total Periods**: ~2,300 trading days

### Data Features
- Realistic volatility clustering and fat-tailed return distributions
- Proper correlation structures within and across asset classes
- Multiple market regimes (bull, bear, high volatility, low volatility, crisis, recovery)
- Corporate actions (dividends, splits) 
- Transaction costs and market impact modeling

## Directory Structure

```
data/generated/
├── market_data/           # Raw market data by asset class
│   ├── equities/         # Individual stock data
│   ├── bonds/            # Bond ETF data
│   ├── commodities/      # Commodity ETF data
│   ├── international/    # International ETF data
│   ├── sectors/          # Sector ETF data
│   └── economic_indicators/ # Macro indicators
├── regimes/              # Regime detection training data
│   ├── returns.csv       # Labeled return series
│   ├── features.csv      # Regime detection features
│   ├── regime_probabilities.csv # Ground truth probabilities
│   └── validation_set_*/ # Multiple validation datasets
├── portfolios/           # Portfolio simulation data
│   ├── mean_variance/    # Mean variance strategy
│   ├── risk_parity/      # Risk parity strategy  
│   ├── momentum/         # Momentum strategy
│   └── equal_weight/     # Equal weight strategy
├── validation/           # Data quality reports
└── metadata/            # Data dictionaries and schemas
```

## Usage Examples

### Loading Market Data
```python
import pandas as pd

# Load equity data
aapl_data = pd.read_csv('data/generated/market_data/equities/AAPL.csv', 
                        index_col=0, parse_dates=True)

# Load economic indicators  
econ_data = pd.read_csv('data/generated/market_data/economic_indicators/indicators.csv',
                        index_col=0, parse_dates=True)
```

### Loading Regime Data
```python
# Load labeled regime data
regime_labels = pd.read_csv('data/generated/regimes/returns.csv', 
                           index_col=0, parse_dates=True)

# Load regime features
regime_features = pd.read_csv('data/generated/regimes/features.csv',
                             index_col=0, parse_dates=True)
```

### Loading Portfolio Data
```python
# Load portfolio weights
weights = pd.read_csv('data/generated/portfolios/mean_variance/weights.csv',
                     index_col=0, parse_dates=True)

# Load transaction history
transactions = pd.read_csv('data/generated/portfolios/mean_variance/transactions.csv',
                          parse_dates=['Date'])
```

## Data Quality

The generated data has been validated for:
- Statistical realism (proper return distributions, volatility clustering)
- Cross-asset relationships (realistic correlations)
- Time series properties (stationarity, autocorrelation)
- Data completeness and consistency

See `validation/data_quality_report.txt` for detailed quality metrics.

## Asset Universe

### Technology Sector (12 stocks)
AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, CRM, ORCL, ADBE, INTC, AMD

### Financial Sector (8 stocks)  
JPM, BAC, WFC, GS, MS, C, BRK-B, AXP

### Healthcare & Pharmaceuticals (8 stocks)
JNJ, PFE, UNH, ABBV, MRK, TMO, ABT, LLY

### Consumer & Retail (7 stocks)
PG, KO, PEP, WMT, HD, MCD, NKE

### Industrial & Energy (6 stocks)
XOM, CVX, CAT, BA, GE, LMT

### Telecommunications & Utilities (4 stocks)
VZ, T, NEE, SO

### Bond ETFs (8 instruments)
AGG, TLT, IEF, SHY, HYG, LQD, TIP, VTEB

### Commodity ETFs (6 instruments) 
GLD, SLV, USO, DBA, PDBC, IAU

### International ETFs (4 instruments)
VEA, VWO, EFA, EEM

### Sector ETFs (6 instruments)
XLF, XLK, XLE, XLV, XLI, XLP

## Economic Indicators

Monthly macroeconomic time series including:
- GDP Growth Rate (%)
- Inflation Rate (%)  
- Unemployment Rate (%)
- Federal Funds Rate (%)
- 10-Year Treasury Yield (%)
- VIX Fear Index

## Regime Definitions

Seven distinct market regimes with different characteristics:
1. **Bull Market**: High returns, low volatility, positive momentum
2. **Bear Market**: Negative returns, high volatility, negative skew
3. **High Volatility**: Extreme volatility, fat tails, no clear direction
4. **Low Volatility**: Low volatility, steady positive returns
5. **Sideways Market**: Range-bound, neutral returns
6. **Crisis**: Extreme negative returns, very high volatility
7. **Recovery**: Strong positive returns after crisis periods

## Notes

- All data is synthetic and generated for demonstration purposes
- Statistical properties match real market characteristics
- Use for backtesting, model development, and system testing
- Not suitable for live trading decisions

Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
"""
    
    with open("data/generated/README.md", 'w') as f:
        f.write(readme_content)
    
    # Create data dictionary
    data_dictionary = {
        'OHLCV_columns': {
            'Date': 'Trading date (YYYY-MM-DD)',
            'Open': 'Opening price',
            'High': 'Highest price during trading session',
            'Low': 'Lowest price during trading session', 
            'Close': 'Closing price',
            'Volume': 'Trading volume (shares/units)',
            'Adj_Close': 'Dividend/split adjusted closing price'
        },
        'regime_columns': {
            'returns': 'Daily return',
            'regime': 'Regime label (text)',
            'regime_code': 'Regime code (0-6)',
            'confidence': 'Regime identification confidence (0-1)'
        },
        'economic_indicators': {
            'GDP_Growth': 'GDP growth rate (%, annualized)',
            'Inflation_Rate': 'Inflation rate (%, annualized)', 
            'Unemployment_Rate': 'Unemployment rate (%)',
            'Fed_Funds_Rate': 'Federal funds rate (%)',
            'Treasury_10Y': '10-year treasury yield (%)',
            'VIX': 'Volatility index (fear gauge)'
        }
    }
    
    import json
    with open("data/generated/metadata/data_dictionary.json", 'w') as f:
        json.dump(data_dictionary, f, indent=2)
    
    print("✅ Documentation created")

def main():
    """Main data generation pipeline"""
    
    print("🚀 Starting Comprehensive Data Generation for Portfolio Engine")
    print("=" * 70)
    
    # Configuration
    START_DATE = "2015-01-01"
    END_DATE = "2024-01-01"
    PORTFOLIO_START = "2020-01-01"
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Generate market data  
    market_data = generate_market_data(START_DATE, END_DATE)
    
    # Step 3: Generate regime data
    regime_data = generate_regime_data(START_DATE, END_DATE)
    
    # Step 4: Generate portfolio data
    portfolio_data = generate_portfolio_data(market_data, PORTFOLIO_START, END_DATE)
    
    # Step 5: Validate data quality
    validation_report = validate_generated_data(market_data)
    
    # Step 6: Create documentation
    create_data_documentation()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DATA GENERATION COMPLETE!")
    print("=" * 70)
    
    # Print summary statistics
    total_assets = sum(len(assets) for assets in market_data.values() if isinstance(assets, dict))
    print(f"📊 Generated data for {total_assets} financial instruments")
    print(f"📅 Date range: {START_DATE} to {END_DATE}")
    print(f"🎯 Regime detection: {len(regime_data)} labeled datasets")
    print(f"💼 Portfolio strategies: {len(portfolio_data)} different approaches")
    
    if validation_report:
        print(f"✅ Data quality score: {validation_report.overall_score:.2f}/1.00")
    
    print(f"\n📁 All data saved to: data/generated/")
    print(f"📖 Documentation: data/generated/README.md")
    print(f"🔍 Quality report: data/generated/validation/data_quality_report.txt")
    
    print("\n🎉 Ready for portfolio engine testing and analysis!")

if __name__ == "__main__":
    main()