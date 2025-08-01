#!/usr/bin/env python3
"""
Customizable Portfolio Engine Demo
Modify the settings below to analyze your own portfolio
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the demo functions
from demo import *

def run_custom_portfolio():
    """Run portfolio analysis with your custom settings"""
    
    print("🎯 Running Custom Portfolio Analysis...")
    
    # 🔧 CUSTOMIZE THESE SETTINGS:
    
    # Your portfolio symbols (change these to your preferred stocks)
    custom_symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']  # Tech focus
    # custom_symbols = ['SPY', 'QQQ', 'VTI', 'VXUS', 'BND']     # ETF focus
    # custom_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS']        # Banking focus
    
    # Time period (adjust for your analysis period)
    start_date_str = '2020-01-01'
    end_date_str = '2023-12-01'
    
    # Risk settings
    risk_aversion = 1.0  # Higher = more conservative, Lower = more aggressive
    
    print(f"📊 Analyzing portfolio: {custom_symbols}")
    print(f"📅 Period: {start_date_str} to {end_date_str}")
    print(f"⚖️ Risk Aversion: {risk_aversion}")
    
    # Generate data with your symbols
    price_data = generate_custom_data(custom_symbols, start_date_str, end_date_str)
    
    # Run the analysis
    results = analyze_custom_portfolio(price_data, risk_aversion)
    
    # Display results
    display_custom_results(results, custom_symbols)
    
    return results

def generate_custom_data(symbols, start_date, end_date):
    """Generate sample data for custom symbols"""
    print(f"📊 Generating data for {len(symbols)} symbols...")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    np.random.seed(42)  # For reproducible results
    
    data = {}
    
    for i, symbol in enumerate(symbols):
        prices = [100]  # Starting price
        regime = 0
        
        # Each symbol gets slightly different characteristics
        base_drift = 0.0003 + (i * 0.0001)  # Slight variation between assets
        base_vol = 0.018 + (i * 0.002)      # Different volatilities
        
        for day in range(1, len(dates)):
            # Regime switching
            if np.random.random() < 0.008:  # Regime change probability
                regime = np.random.choice([0, 1, 2])
            
            # Different parameters by regime
            if regime == 0:    # Bull market
                daily_return = np.random.normal(base_drift * 2, base_vol * 0.8)
            elif regime == 1:  # Normal market
                daily_return = np.random.normal(base_drift, base_vol)
            else:             # Bear market
                daily_return = np.random.normal(-base_drift, base_vol * 1.5)
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1))  # Prevent negative prices
        
        data[symbol] = prices
    
    df = pd.DataFrame(data, index=dates)
    print(f"✅ Generated {len(df)} days of data")
    return df

def analyze_custom_portfolio(price_data, risk_aversion):
    """Analyze the custom portfolio"""
    
    # Initialize models
    hmm_model = SimpleHMM(n_components=3)
    optimizer = SimpleOptimizer()
    
    # Train regime detection on market average
    market_index = price_data.mean(axis=1)  # Simple market proxy
    hmm_model.fit(market_index)
    
    # Detect regimes
    regimes = hmm_model.predict_regimes(market_index)
    regime_probs = hmm_model.predict_regime_probabilities(market_index)
    
    # Calculate returns
    returns_data = price_data.pct_change().dropna()
    expected_returns = returns_data.mean() * 252
    cov_matrix = returns_data.cov() * 252
    
    # Optimize portfolio
    optimal_weights = optimizer.optimize_mean_variance(
        expected_returns.values, 
        cov_matrix.values
    )
    
    # Calculate portfolio performance
    portfolio_returns = (returns_data * optimal_weights).sum(axis=1)
    performance_metrics = calculate_portfolio_metrics(portfolio_returns)
    
    return {
        'price_data': price_data,
        'regimes': regimes,
        'regime_probs': regime_probs,
        'optimal_weights': optimal_weights,
        'portfolio_returns': portfolio_returns,
        'performance_metrics': performance_metrics,
        'returns_data': returns_data
    }

def display_custom_results(results, symbols):
    """Display the analysis results"""
    
    print("\n" + "="*60)
    print("🎯 YOUR CUSTOM PORTFOLIO ANALYSIS")
    print("="*60)
    
    print(f"\n🏆 Optimized Portfolio Allocation:")
    for symbol, weight in zip(symbols, results['optimal_weights']):
        print(f"  {symbol}: {weight:.1%}")
    
    print(f"\n📈 Performance Metrics:")
    for metric, value in results['performance_metrics'].items():
        print(f"  {metric}: {value}")
    
    # Regime analysis
    regimes = results['regimes']
    regime_names = {0: "Low Volatility", 1: "Normal", 2: "High Volatility"}
    regime_distribution = pd.Series(regimes).value_counts().sort_index()
    
    print(f"\n🔄 Market Regime Analysis:")
    for regime_id, count in regime_distribution.items():
        regime_name = regime_names.get(regime_id, f"Regime {regime_id}")
        percentage = count / len(regimes) * 100
        print(f"  {regime_name}: {percentage:.1f}% of time")
    
    current_regime = regimes[-1]
    current_confidence = results['regime_probs'][-1, current_regime]
    print(f"\n🎯 Current Market Regime: {regime_names[current_regime]} (Confidence: {current_confidence:.1%})")
    
    # Generate visualization
    try:
        create_custom_visualization(results, symbols)
        print(f"\n📊 Custom dashboard saved to 'my_portfolio_dashboard.png'")
    except Exception as e:
        print(f"\n⚠️ Visualization failed: {e}")

def create_custom_visualization(results, symbols):
    """Create visualization for custom portfolio"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'My Custom Portfolio Dashboard - {", ".join(symbols)}', fontsize=16, fontweight='bold')
    
    price_data = results['price_data']
    portfolio_returns = results['portfolio_returns']
    regimes = results['regimes']
    optimal_weights = results['optimal_weights']
    
    # Portfolio cumulative return
    cumulative_returns = (1 + portfolio_returns).cumprod()
    ax1.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, color='blue')
    ax1.set_title('My Portfolio Cumulative Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True, alpha=0.3)
    
    # Asset prices (normalized)
    normalized_prices = price_data / price_data.iloc[0]
    for col in normalized_prices.columns:
        ax2.plot(normalized_prices.index, normalized_prices[col], label=col, alpha=0.7)
    ax2.set_title('My Assets - Normalized Prices')
    ax2.set_ylabel('Normalized Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Regime detection
    colors = ['green', 'orange', 'red']
    regime_colors = [colors[int(r)] for r in regimes]
    market_proxy = price_data.mean(axis=1)
    ax3.scatter(price_data.index, market_proxy, c=regime_colors, alpha=0.6, s=1)
    ax3.set_title('Market Regime Detection')
    ax3.set_ylabel('Market Index')
    ax3.grid(True, alpha=0.3)
    
    # Portfolio weights
    ax4.pie(optimal_weights, labels=symbols, autopct='%1.1f%%', startangle=90)
    ax4.set_title('My Portfolio Allocation')
    
    plt.tight_layout()
    plt.savefig('my_portfolio_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = run_custom_portfolio()
    
    print(f"""
    
🎉 Analysis Complete! Here's what you can do next:

📊 **View Your Results:**
   • Open 'my_portfolio_dashboard.png' to see your custom analysis
   
🔧 **Customize Further:**
   • Edit the symbols list in this file
   • Adjust the time period or risk aversion
   • Re-run: python3 run_my_portfolio.py
   
📈 **Try Different Strategies:**
   • Tech stocks: ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']
   • Blue chips: ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO']
   • ETFs: ['SPY', 'QQQ', 'VTI', 'VXUS', 'BND']
   
🚀 **Go Production:**
   • Install full system: pip install -r requirements.txt
   • Run: python3 start.py
   • Access dashboard: http://localhost:8050
    """)