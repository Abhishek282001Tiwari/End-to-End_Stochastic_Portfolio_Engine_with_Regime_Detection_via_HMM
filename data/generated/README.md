# Generated Portfolio Engine Dataset

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

Generated on: 2025-08-02 01:33:34
