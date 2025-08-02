# End-to-End Stochastic Portfolio Engine with Regime Detection via HMM

A comprehensive portfolio optimization system that uses Hidden Markov Models (HMM) to detect market regimes and dynamically adjust portfolio allocation strategies based on detected market conditions.

## 🎯 Key Features

### Core Architecture
- **Data Infrastructure Layer**: Robust data pipeline with real-time streaming capabilities
- **HMM Regime Detection Engine**: Sophisticated market regime identification using multiple observable variables
- **Stochastic Portfolio Optimization**: Multiple optimization frameworks with regime-conditional strategies
- **Comprehensive Backtesting**: Walk-forward analysis with realistic market microstructure simulation
- **Real-time Risk Management**: Dynamic hedging strategies and position monitoring
- **Performance Analytics**: Advanced attribution analysis and visualization

### Advanced Capabilities
- **Multi-Asset Universe**: Supports equities, bonds, commodities, currencies, and derivatives
- **Factor Models**: Fama-French, statistical, and macroeconomic factor frameworks
- **Risk Models**: VaR, CVaR, expected shortfall, and drawdown constraints
- **Transaction Cost Modeling**: Realistic bid-ask spreads and market impact functions
- **Regime-Aware Optimization**: Portfolio weights adjust based on detected market regimes

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/End-to-End_Stochastic_Portfolio_Engine_with_Regime_Detection_via_HMM.git
cd End-to-End_Stochastic_Portfolio_Engine_with_Regime_Detection_via_HMM
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the portfolio engine**:
```bash
python main.py
```

### Basic Usage

```python
import asyncio
from src.utils.config import get_config
from src.data.ingestion.data_sources import create_data_pipeline
from src.models.hmm.hmm_engine import RegimeDetectionHMM
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine

async def run_portfolio_engine():
    # Load configuration
    config = get_config()
    
    # Create data pipeline
    data_pipeline = create_data_pipeline()
    
    # Fetch market data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    price_data = await data_pipeline.fetch_asset_universe(symbols, start_date, end_date)
    features_data = await data_pipeline.create_features_dataset(symbols, start_date, end_date)
    
    # Train HMM regime detection model
    hmm_model = RegimeDetectionHMM(n_components=3)
    hmm_model.fit(features_data)
    
    # Portfolio optimization
    optimizer = PortfolioOptimizationEngine()
    expected_returns = price_data.pct_change().mean() * 252
    covariance_matrix = price_data.pct_change().cov() * 252
    
    result = optimizer.optimize_portfolio(
        'mean_variance',
        expected_returns.values,
        covariance_matrix.values
    )
    
    print(f"Optimal weights: {result['weights']}")
    print(f"Expected return: {result['expected_return']:.2%}")
    print(f"Expected volatility: {result['expected_volatility']:.2%}")

# Run the engine
asyncio.run(run_portfolio_engine())
```

## 📊 System Architecture

### 1. Data Infrastructure Layer (`src/data/`)
- **Ingestion** (`ingestion/`): Multi-source data fetching with Yahoo Finance, Alpha Vantage, and custom APIs
- **Processing** (`processing/`): Data cleaning, feature engineering, and technical indicators
- **Validation** (`validation/`): Comprehensive data quality checks and anomaly detection
- **Storage** (`storage/`): Time-series optimized database with proper indexing

### 2. HMM Regime Detection (`src/models/hmm/`)
- **Engine** (`hmm_engine.py`): Core HMM implementation with Baum-Welch algorithm
- **Analyzer** (`regime_analyzer.py`): Regime characterization and transition analysis
- **Features**: Bull/Bear/Sideways market detection using returns, volatility, VIX, yield curves

### 3. Portfolio Optimization (`src/optimization/`)
- **Optimizers** (`portfolio/stochastic_optimizer.py`):
  - Mean-Variance Optimization
  - Black-Litterman Model
  - Risk Parity
  - Monte Carlo Optimization
  - Regime-Aware Optimization
- **Risk Measures** (`objectives/risk_measures.py`): VaR, CVaR, Sortino ratio, Calmar ratio
- **Constraints** (`constraints/`): Position limits, sector constraints, leverage controls

### 4. Factor Models (`src/models/factors/`)
- **Fama-French Models**: 3-factor, 5-factor, and momentum models
- **Statistical Models**: PCA and Factor Analysis
- **Macroeconomic Models**: GDP, inflation, interest rates, credit spreads

### 5. Backtesting Engine (`src/backtesting/`)
- **Engine** (`engine/backtesting_engine.py`): Walk-forward analysis with realistic execution
- **Attribution** (`attribution/`): Brinson, factor-based, and regime-based attribution
- **Scenarios** (`scenarios/`): Stress testing and scenario analysis

### 6. Risk Management (`src/risk/`)
- **Monitoring** (`monitoring/risk_monitor.py`): Real-time risk alerts and dashboards
- **Hedging** (`hedging/dynamic_hedging.py`): Delta-neutral, VaR-based, and tail risk hedging
- **Exposure** (`exposure/`): Sector, geography, and factor exposure management

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)

```yaml
data:
  sources:
    yahoo: true
    alpha_vantage: false
  refresh_frequency: "1H"
  lookback_days: 252

hmm:
  n_components: 3  # Bull, Bear, Sideways
  covariance_type: "full"
  features:
    - "returns"
    - "volatility" 
    - "volume"
    - "vix"
    - "yield_curve_slope"

portfolio:
  optimization:
    method: "mean_variance"
    rebalance_frequency: "1W"
    transaction_costs: 0.001
  constraints:
    max_weight: 0.2
    min_weight: 0.0
    leverage: 1.0
  risk:
    target_volatility: 0.15
    max_drawdown: 0.2
    var_confidence: 0.05

backtesting:
  start_date: "2010-01-01"
  end_date: "2023-12-31"
  initial_capital: 1000000
  benchmark: "SPY"
```

## 📈 Usage Examples

### 1. Regime Detection and Analysis

```python
from src.models.hmm.hmm_engine import RegimeDetectionHMM
from src.models.hmm.regime_analyzer import RegimeAnalyzer

# Train HMM model
hmm_model = RegimeDetectionHMM(n_components=3)
hmm_model.fit(features_data)

# Analyze regimes
analyzer = RegimeAnalyzer(hmm_model)
regime_analysis = analyzer.analyze_regime_characteristics(features_data, price_data)

# Generate regime report
report = analyzer.generate_regime_report(features_data, price_data)
print(report)
```

### 2. Portfolio Optimization with Multiple Methods

```python
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine

engine = PortfolioOptimizationEngine()

# Compare different optimization methods
comparison = engine.compare_optimizers(
    expected_returns,
    covariance_matrix,
    methods=['mean_variance', 'black_litterman', 'risk_parity']
)

print(comparison)
```

### 3. Backtesting with Regime Detection

```python
from src.backtesting.engine.backtesting_engine import BacktestingEngine, BacktestConfig

config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=1000000,
    rebalance_frequency='M',
    optimization_method='mean_variance'
)

engine = BacktestingEngine(config, portfolio_optimizer, hmm_model)
results = engine.run_backtest(asset_returns, benchmark_returns, features_data)

print(f"Total Return: {results.performance_metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {results.performance_metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results.performance_metrics['max_drawdown']:.2%}")
```

### 4. Real-time Risk Monitoring

```python
from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor, RiskLimits

# Set risk limits
risk_limits = RiskLimits(
    max_portfolio_volatility=0.20,
    max_individual_weight=0.15,
    max_drawdown=0.15
)

# Create risk monitor
risk_monitor = RealTimeRiskMonitor(risk_limits)

# Monitor portfolio
alerts = risk_monitor.monitor_portfolio_risk(
    portfolio_weights=current_weights,
    returns_data=recent_returns,
    sector_mapping=sector_map
)

for alert in alerts:
    print(f"{alert.level.value}: {alert.message}")
```

### 5. Dynamic Hedging

```python
from src.risk.hedging.dynamic_hedging import DynamicHedgingEngine

hedging_engine = DynamicHedgingEngine()

# Define hedge instruments
hedge_instruments = {
    'VIX_futures': vix_returns,
    'Treasury_futures': treasury_returns,
    'Currency_futures': currency_returns
}

# Calculate optimal hedges
hedge_summary = hedging_engine.calculate_optimal_hedges(
    portfolio_returns=portfolio_returns,
    portfolio_value=1000000,
    hedge_instruments=hedge_instruments,
    current_regime=current_regime,
    hedging_budget=0.05
)

print(f"Number of recommended hedges: {hedge_summary['number_of_hedges']}")
print(f"Total hedge cost: ${hedge_summary['total_hedge_cost']:,.2f}")
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_hmm_engine.py
pytest tests/test_portfolio_optimizer.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Performance Metrics

The system calculates comprehensive performance metrics including:

### Return Metrics
- Total Return
- Annualized Return
- Rolling Returns
- Win Rate and Win/Loss Ratio

### Risk Metrics
- Volatility (annualized)
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Skewness and Kurtosis

### Risk-Adjusted Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio
- Omega Ratio

### Benchmark Comparison
- Alpha and Beta
- Tracking Error
- Up/Down Capture Ratios
- Correlation

## 🔬 Advanced Features

### 1. Regime-Conditional Optimization
The system adjusts portfolio allocation based on detected market regimes:
- **Bull Market**: Higher risk tolerance, growth-oriented allocation
- **Bear Market**: Defensive positioning, tail risk hedging
- **Sideways Market**: Range-bound strategies, mean reversion

### 2. Transaction Cost Modeling
Realistic implementation costs including:
- Bid-ask spreads
- Market impact functions
- Commission structures
- Timing costs

### 3. Factor Risk Management
- Factor exposure monitoring
- Risk factor decomposition
- Factor-based hedging strategies
- Attribution analysis

### 4. Stress Testing
- Historical scenario analysis
- Monte Carlo simulations
- Tail risk assessment
- Regime transition stress tests

## 📝 Output Files

The system generates several output files:

- `portfolio_engine_results.csv`: Summary results and configuration
- `portfolio_returns_timeseries.csv`: Daily portfolio and benchmark returns
- `portfolio_weights_history.csv`: Historical portfolio weights
- `portfolio_transactions.csv`: All trading transactions
- `logs/portfolio_engine.log`: Detailed system logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hidden Markov Models**: Based on research in regime detection for financial markets
- **Portfolio Optimization**: Implementation of modern portfolio theory and advanced optimization techniques
- **Risk Management**: Industry best practices for institutional portfolio management
- **Data Sources**: Yahoo Finance for market data (replace with your preferred data provider)

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation in the `docs/` directory

---

**Disclaimer**: This is a research and educational tool. Past performance does not guarantee future results. Always consult with qualified financial professionals before making investment decisions.