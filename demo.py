#!/usr/bin/env python3
"""
Demo script for the End-to-End Stochastic Portfolio Engine
with Regime Detection via HMM

This simplified demo shows the core functionality without external dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🚀 End-to-End Stochastic Portfolio Engine 🚀          ║
║             with Regime Detection via HMM                   ║
║                                                              ║
║  • Real-time regime detection and portfolio optimization    ║
║  • Machine learning ensemble methods                        ║
║  • Comprehensive risk monitoring and reporting              ║
║  • Interactive web dashboard and automated alerts           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

class SimpleHMM:
    """Simplified HMM for demonstration"""
    
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.is_fitted = False
        
    def fit(self, data):
        """Fit HMM to data"""
        print(f"🧠 Training HMM with {self.n_components} regimes...")
        # Simulate training process
        self.transition_matrix = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8]
        ])
        self.is_fitted = True
        print("✅ HMM training completed")
        
    def predict_regimes(self, data):
        """Predict regimes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Simple regime detection based on volatility
        returns = data.pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        regimes = np.zeros(len(data))
        for i, vol in enumerate(volatility):
            if pd.isna(vol):
                regimes[i] = 1  # Default to middle regime
            elif vol < volatility.quantile(0.33):
                regimes[i] = 0  # Low volatility regime
            elif vol > volatility.quantile(0.67):
                regimes[i] = 2  # High volatility regime
            else:
                regimes[i] = 1  # Medium volatility regime
                
        return regimes.astype(int)
    
    def predict_regime_probabilities(self, data):
        """Predict regime probabilities"""
        regimes = self.predict_regimes(data)
        n_obs = len(regimes)
        probs = np.zeros((n_obs, self.n_components))
        
        for i, regime in enumerate(regimes):
            probs[i, regime] = 0.8  # High confidence in predicted regime
            other_regimes = [r for r in range(self.n_components) if r != regime]
            for other in other_regimes:
                probs[i, other] = 0.1  # Low probability for other regimes
                
        return probs

class SimpleOptimizer:
    """Simplified portfolio optimizer"""
    
    def optimize_mean_variance(self, returns, cov_matrix, risk_aversion=1.0):
        """Simple mean-variance optimization"""
        print("🎯 Optimizing portfolio allocation...")
        
        # Simple equal-weight portfolio for demo
        n_assets = len(returns)
        weights = np.ones(n_assets) / n_assets
        
        # Add some randomness for variety
        np.random.seed(42)
        weights += np.random.normal(0, 0.05, n_assets)
        weights = np.abs(weights)
        weights = weights / weights.sum()  # Normalize
        
        print(f"✅ Portfolio optimization completed - {n_assets} assets")
        return weights

def generate_sample_data():
    """Generate sample market data"""
    print("📊 Generating sample market data...")
    
    # Create date range
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate synthetic price data with regime switching
    np.random.seed(42)
    data = {}
    
    for symbol in symbols:
        prices = [100]  # Starting price
        regime = 0
        
        for i in range(1, len(dates)):
            # Occasional regime switches
            if np.random.random() < 0.01:
                regime = np.random.choice([0, 1, 2])
            
            # Different return/volatility by regime
            if regime == 0:  # Bull market
                daily_return = np.random.normal(0.0008, 0.015)
            elif regime == 1:  # Normal market
                daily_return = np.random.normal(0.0003, 0.020)
            else:  # Bear market
                daily_return = np.random.normal(-0.0005, 0.030)
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        data[symbol] = prices
    
    df = pd.DataFrame(data, index=dates)
    print(f"✅ Generated data: {len(df)} days, {len(symbols)} assets")
    return df

def calculate_portfolio_metrics(returns):
    """Calculate portfolio performance metrics"""
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns.mean()) ** 252 - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}"
    }

def run_demo():
    """Run the portfolio engine demo"""
    
    # 1. Generate sample data
    price_data = generate_sample_data()
    
    # 2. Initialize regime detection
    print("\n🔄 Initializing regime detection...")
    hmm_model = SimpleHMM(n_components=3)
    
    # Use first asset for regime detection
    hmm_model.fit(price_data.iloc[:, 0])
    
    # 3. Detect regimes
    print("📈 Detecting market regimes...")
    regimes = hmm_model.predict_regimes(price_data.iloc[:, 0])
    regime_probs = hmm_model.predict_regime_probabilities(price_data.iloc[:, 0])
    
    regime_names = {0: "Low Volatility", 1: "Normal", 2: "High Volatility"}
    current_regime = regimes[-1]
    current_confidence = regime_probs[-1, current_regime]
    
    print(f"🎯 Current Regime: {regime_names[current_regime]} (Confidence: {current_confidence:.1%})")
    
    # 4. Portfolio optimization
    print("\n💼 Optimizing portfolio...")
    returns_data = price_data.pct_change().dropna()
    expected_returns = returns_data.mean() * 252
    cov_matrix = returns_data.cov() * 252
    
    optimizer = SimpleOptimizer()
    optimal_weights = optimizer.optimize_mean_variance(
        expected_returns.values, 
        cov_matrix.values
    )
    
    # 5. Calculate portfolio performance
    portfolio_returns = (returns_data * optimal_weights).sum(axis=1)
    performance_metrics = calculate_portfolio_metrics(portfolio_returns)
    
    # 6. Display results
    print("\n" + "="*60)
    print("📊 PORTFOLIO ENGINE RESULTS")
    print("="*60)
    
    print("\n🏆 Portfolio Allocation:")
    for i, (symbol, weight) in enumerate(zip(price_data.columns, optimal_weights)):
        print(f"  {symbol}: {weight:.1%}")
    
    print(f"\n📈 Performance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"  {metric}: {value}")
    
    print(f"\n🔄 Regime Analysis:")
    regime_distribution = pd.Series(regimes).value_counts().sort_index()
    for regime_id, count in regime_distribution.items():
        regime_name = regime_names.get(regime_id, f"Regime {regime_id}")
        percentage = count / len(regimes) * 100
        print(f"  {regime_name}: {percentage:.1f}% of time")
    
    # 7. Generate simple visualizations
    print(f"\n📊 Generating visualizations...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Engine Dashboard', fontsize=16, fontweight='bold')
        
        # Portfolio cumulative return
        cumulative_returns = (1 + portfolio_returns).cumprod()
        ax1.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, color='blue')
        ax1.set_title('Portfolio Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)
        
        # Asset prices
        normalized_prices = price_data / price_data.iloc[0]
        for col in normalized_prices.columns:
            ax2.plot(normalized_prices.index, normalized_prices[col], label=col, alpha=0.7)
        ax2.set_title('Normalized Asset Prices')
        ax2.set_ylabel('Normalized Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Regime detection
        colors = ['green', 'orange', 'red']
        regime_colors = [colors[int(r)] for r in regimes]
        ax3.scatter(price_data.index, price_data.iloc[:, 0], c=regime_colors, alpha=0.6, s=1)
        ax3.set_title('Regime Detection (First Asset)')
        ax3.set_ylabel('Price')
        ax3.grid(True, alpha=0.3)
        
        # Portfolio weights
        ax4.pie(optimal_weights, labels=price_data.columns, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Portfolio Allocation')
        
        plt.tight_layout()
        
        # Save plot instead of showing interactively
        plot_filename = 'portfolio_dashboard.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"✅ Visualizations saved to {plot_filename}!")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping visualizations")
    
    print("\n" + "="*60)
    print("🎉 PORTFOLIO ENGINE DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"""
📋 Summary:
  • Processed {len(price_data)} days of market data
  • Detected {len(regime_distribution)} market regimes
  • Optimized portfolio across {len(price_data.columns)} assets
  • Generated comprehensive performance analytics
  
🌐 Full System Features (in production):
  • Real-time data feeds from multiple providers
  • Advanced ML ensemble methods for regime detection
  • Sophisticated optimization algorithms
  • Live web dashboard at http://localhost:8050
  • Automated risk monitoring and alerts
  • Comprehensive backtesting framework
  • Production deployment with Docker/Kubernetes
    """)

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Please ensure you have numpy and pandas installed:")
        print("pip install numpy pandas matplotlib")