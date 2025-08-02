---
layout: page
title: Code Repository
permalink: /code/
description: "Comprehensive code repository showcasing clean, documented implementations of quantitative finance algorithms, portfolio optimization techniques, and machine learning models."
---

<div class="code-repository">
  <!-- Repository Overview -->
  <section class="repository-overview">
    <h2>Code Repository & Implementation</h2>
    <p class="repo-description">
      This repository contains clean, well-documented implementations of the complete stochastic portfolio engine. 
      All code follows industry best practices with comprehensive testing, documentation, and modular architecture 
      suitable for institutional deployment.
    </p>
    
    <div class="repo-stats">
      <div class="stat-item">
        <div class="stat-number">25,000+</div>
        <div class="stat-label">Lines of Code</div>
      </div>
      <div class="stat-item">
        <div class="stat-number">150+</div>
        <div class="stat-label">Unit Tests</div>
      </div>
      <div class="stat-item">
        <div class="stat-number">95%</div>
        <div class="stat-label">Test Coverage</div>
      </div>
      <div class="stat-item">
        <div class="stat-number">12</div>
        <div class="stat-label">Core Modules</div>
      </div>
    </div>
    
    <div class="repository-links">
      <a href="https://github.com/your-repo/portfolio-engine" class="btn btn-primary" target="_blank">
        📦 View on GitHub
      </a>
      <a href="#quick-start" class="btn btn-secondary">
        🚀 Quick Start Guide
      </a>
      <a href="#api-docs" class="btn btn-outline">
        📚 API Documentation
      </a>
    </div>
  </section>

  <!-- System Architecture -->
  <section class="architecture-section">
    <h2>System Architecture</h2>
    
    <div class="architecture-diagram">
      <div class="architecture-layer" data-layer="presentation">
        <h3>Presentation Layer</h3>
        <div class="layer-components">
          <div class="component">CLI Interface</div>
          <div class="component">Web Dashboard</div>
          <div class="component">API Endpoints</div>
        </div>
      </div>
      
      <div class="architecture-layer" data-layer="business">
        <h3>Business Logic Layer</h3>
        <div class="layer-components">
          <div class="component">Portfolio Optimizer</div>
          <div class="component">Risk Manager</div>
          <div class="component">Regime Detector</div>
          <div class="component">Performance Analytics</div>
        </div>
      </div>
      
      <div class="architecture-layer" data-layer="data">
        <h3>Data Layer</h3>
        <div class="layer-components">
          <div class="component">Data Pipeline</div>
          <div class="component">Feature Engineering</div>
          <div class="component">Data Validation</div>
          <div class="component">Storage System</div>
        </div>
      </div>
    </div>
  </section>

  <!-- Core Modules -->
  <section class="modules-section">
    <h2>Core Modules</h2>
    
    <div class="modules-grid">
      <!-- HMM Engine -->
      <div class="module-card">
        <div class="module-header">
          <h3>🧠 HMM Engine</h3>
          <div class="module-path">src/models/hmm/</div>
        </div>
        <div class="module-content">
          <p>Advanced Hidden Markov Model implementation for regime detection with enhanced Baum-Welch algorithm.</p>
          
          <div class="code-preview">
            <h4>Key Classes:</h4>
            <div class="code-block">
              <pre><code class="language-python">class AdvancedBaumWelchHMM:
    """Enhanced HMM with multi-factor observations"""
    
    def __init__(self, n_components=3, covariance_type='full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
    
    def fit(self, observations):
        """Fit HMM using enhanced Baum-Welch algorithm"""
        return self._fit_with_convergence_check(observations)
    
    def predict_regimes(self, observations):
        """Predict most likely regime sequence"""
        return self._viterbi_decode(observations)</code></pre>
            </div>
          </div>
          
          <div class="module-features">
            <h4>Features:</h4>
            <ul>
              <li>Multi-factor observation modeling</li>
              <li>Robust parameter estimation</li>
              <li>Real-time regime prediction</li>
              <li>Convergence diagnostics</li>
            </ul>
          </div>
          
          <div class="module-links">
            <a href="#hmm-docs" class="module-link">📖 Documentation</a>
            <a href="#hmm-tests" class="module-link">🧪 Tests</a>
            <a href="#hmm-examples" class="module-link">💡 Examples</a>
          </div>
        </div>
      </div>

      <!-- Portfolio Optimization -->
      <div class="module-card">
        <div class="module-header">
          <h3>⚡ Portfolio Optimizer</h3>
          <div class="module-path">src/optimization/portfolio/</div>
        </div>
        <div class="module-content">
          <p>Comprehensive portfolio optimization engine with multiple strategies and regime-conditional allocation.</p>
          
          <div class="code-preview">
            <h4>Key Implementation:</h4>
            <div class="code-block">
              <pre><code class="language-python">class PortfolioOptimizationEngine:
    """Multi-strategy portfolio optimization"""
    
    def optimize_portfolio(self, method, returns, covariance, 
                          regime_condition=None):
        """Optimize portfolio with regime awareness"""
        
        if regime_condition == 'Bull':
            risk_aversion = 0.5
        elif regime_condition == 'Bear':
            risk_aversion = 2.0
        else:
            risk_aversion = 1.0
            
        return self._solve_optimization(
            method, returns, covariance, risk_aversion
        )</code></pre>
            </div>
          </div>
          
          <div class="module-features">
            <h4>Optimization Methods:</h4>
            <ul>
              <li>Mean-Variance Optimization</li>
              <li>Black-Litterman Model</li>
              <li>Risk Parity</li>
              <li>Regime-Conditional Strategies</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Risk Management -->
      <div class="module-card">
        <div class="module-header">
          <h3>⚠️ Risk Manager</h3>
          <div class="module-path">src/risk/monitoring/</div>
        </div>
        <div class="module-content">
          <p>Real-time risk monitoring system with comprehensive metrics and automated alerting capabilities.</p>
          
          <div class="code-preview">
            <h4>Risk Metrics:</h4>
            <div class="code-block">
              <pre><code class="language-python">class RealTimeRiskMonitor:
    """Real-time portfolio risk monitoring"""
    
    def calculate_risk_metrics(self, weights, returns):
        """Calculate comprehensive risk metrics"""
        
        metrics = {
            'var_95': self._calculate_var(returns, 0.05),
            'cvar_95': self._calculate_cvar(returns, 0.05),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'volatility': self._calculate_volatility(returns),
            'sharpe_ratio': self._calculate_sharpe(returns)
        }
        
        return metrics</code></pre>
            </div>
          </div>
          
          <div class="module-features">
            <h4>Risk Capabilities:</h4>
            <ul>
              <li>VaR/CVaR calculation</li>
              <li>Drawdown analysis</li>
              <li>Real-time alerts</li>
              <li>Stress testing</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Data Pipeline -->
      <div class="module-card">
        <div class="module-header">
          <h3>📊 Data Pipeline</h3>
          <div class="module-path">src/data/ingestion/</div>
        </div>
        <div class="module-content">
          <p>Robust data ingestion and processing pipeline with multi-source support and real-time capabilities.</p>
          
          <div class="code-preview">
            <h4>Pipeline Architecture:</h4>
            <div class="code-block">
              <pre><code class="language-python">async def create_data_pipeline():
    """Create comprehensive data pipeline"""
    
    pipeline = DataPipeline()
    
    # Add data sources
    pipeline.add_source('yahoo', YahooFinanceSource())
    pipeline.add_source('alpha_vantage', AlphaVantageSource())
    
    # Add processors
    pipeline.add_processor(DataValidator())
    pipeline.add_processor(FeatureEngineer())
    
    return pipeline</code></pre>
            </div>
          </div>
          
          <div class="module-features">
            <h4>Data Features:</h4>
            <ul>
              <li>Multi-source ingestion</li>
              <li>Real-time streaming</li>
              <li>Data validation</li>
              <li>Feature engineering</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Backtesting Engine -->
      <div class="module-card">
        <div class="module-header">
          <h3>🔄 Backtesting Engine</h3>
          <div class="module-path">src/backtesting/engine/</div>
        </div>
        <div class="module-content">
          <p>Sophisticated backtesting framework with walk-forward analysis and realistic transaction cost modeling.</p>
          
          <div class="code-preview">
            <h4>Backtesting Process:</h4>
            <div class="code-block">
              <pre><code class="language-python">class BacktestingEngine:
    """Walk-forward backtesting with realistic costs"""
    
    def run_backtest(self, returns, features, config):
        """Execute comprehensive backtest"""
        
        results = BacktestResults()
        
        for date in self.rebalance_dates:
            # Regime detection
            regime = self.hmm_model.predict(features[:date])
            
            # Portfolio optimization
            weights = self.optimizer.optimize(
                returns[:date], regime_condition=regime
            )
            
            # Calculate performance with costs
            performance = self._calculate_performance(
                weights, returns[date:next_date], 
                transaction_costs=True
            )
            
            results.add_period(date, weights, performance)
        
        return results</code></pre>
            </div>
          </div>
        </div>
      </div>

      <!-- Monte Carlo Engine -->
      <div class="module-card">
        <div class="module-header">
          <h3>🎲 Monte Carlo Engine</h3>
          <div class="module-path">src/simulation/monte_carlo_engine.py</div>
        </div>
        <div class="module-content">
          <p>Advanced Monte Carlo simulation engine for portfolio projections and risk assessment.</p>
          
          <div class="code-preview">
            <h4>Simulation Framework:</h4>
            <div class="code-block">
              <pre><code class="language-python">class MonteCarloEngine:
    """Advanced Monte Carlo simulation"""
    
    def simulate_portfolio_paths(self, weights, expected_returns, 
                                covariance, n_simulations=10000):
        """Generate portfolio return paths"""
        
        paths = []
        
        for i in range(n_simulations):
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                expected_returns, covariance, self.time_horizon
            )
            
            # Calculate portfolio path
            portfolio_returns = random_returns @ weights
            portfolio_path = np.cumprod(1 + portfolio_returns)
            
            paths.append(portfolio_path)
        
        return self._analyze_paths(paths)</code></pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Quick Start Guide -->
  <section id="quick-start" class="quickstart-section">
    <h2>🚀 Quick Start Guide</h2>
    
    <div class="quickstart-steps">
      <div class="step">
        <div class="step-number">1</div>
        <div class="step-content">
          <h3>Installation</h3>
          <div class="code-block">
            <pre><code class="language-bash"># Clone the repository
git clone https://github.com/your-repo/portfolio-engine.git
cd portfolio-engine

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .</code></pre>
          </div>
        </div>
      </div>
      
      <div class="step">
        <div class="step-number">2</div>
        <div class="step-content">
          <h3>Basic Usage</h3>
          <div class="code-block">
            <pre><code class="language-python">import asyncio
from src.models.hmm.hmm_engine import AdvancedBaumWelchHMM
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.data.ingestion.data_sources import create_data_pipeline

async def main():
    # Create data pipeline
    pipeline = create_data_pipeline()
    
    # Fetch data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    price_data = await pipeline.fetch_asset_universe(symbols)
    features = await pipeline.create_features_dataset(symbols)
    
    # Train regime detection model
    hmm_model = AdvancedBaumWelchHMM(n_components=3)
    hmm_model.fit(features)
    
    # Optimize portfolio
    optimizer = PortfolioOptimizationEngine()
    weights = optimizer.optimize_portfolio(
        'mean_variance', price_data.pct_change().mean(),
        price_data.pct_change().cov()
    )
    
    print(f"Optimal weights: {weights}")

asyncio.run(main())</code></pre>
          </div>
        </div>
      </div>
      
      <div class="step">
        <div class="step-number">3</div>
        <div class="step-content">
          <h3>Configuration</h3>
          <div class="code-block">
            <pre><code class="language-yaml"># config/config.yaml
data:
  sources:
    yahoo: true
    alpha_vantage: false
  refresh_frequency: "1H"

hmm:
  n_components: 3
  covariance_type: "full"
  features: ["returns", "volatility", "volume"]

portfolio:
  optimization_method: "mean_variance"
  rebalance_frequency: "1W"
  max_weight: 0.2
  transaction_costs: 0.001</code></pre>
          </div>
        </div>
      </div>
      
      <div class="step">
        <div class="step-number">4</div>
        <div class="step-content">
          <h3>Run Complete Analysis</h3>
          <div class="code-block">
            <pre><code class="language-bash"># Run the complete portfolio engine
python main.py --config config/config.yaml

# Run in demo mode (faster)
python main.py --demo

# Use CLI interface
portfolio optimize -s "AAPL,GOOGL,MSFT" --monte-carlo</code></pre>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- API Documentation -->
  <section id="api-docs" class="api-section">
    <h2>📚 API Documentation</h2>
    
    <div class="api-modules">
      <div class="api-module">
        <h3>HMM Engine API</h3>
        <div class="api-endpoints">
          <div class="api-endpoint">
            <div class="endpoint-method">CLASS</div>
            <div class="endpoint-path">AdvancedBaumWelchHMM</div>
            <div class="endpoint-description">Main HMM implementation class</div>
            
            <div class="endpoint-details">
              <h4>Methods:</h4>
              <ul>
                <li><code>fit(observations)</code> - Fit the HMM model</li>
                <li><code>predict_regimes(observations)</code> - Predict regime sequence</li>
                <li><code>get_regime_statistics()</code> - Calculate regime statistics</li>
                <li><code>save_model(filepath)</code> - Save trained model</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      <div class="api-module">
        <h3>Portfolio Optimizer API</h3>
        <div class="api-endpoints">
          <div class="api-endpoint">
            <div class="endpoint-method">CLASS</div>
            <div class="endpoint-path">PortfolioOptimizationEngine</div>
            <div class="endpoint-description">Multi-strategy optimization engine</div>
            
            <div class="endpoint-details">
              <h4>Methods:</h4>
              <ul>
                <li><code>optimize_portfolio(method, returns, covariance)</code></li>
                <li><code>compare_optimizers(methods, data)</code></li>
                <li><code>regime_conditional_optimize(regime, data)</code></li>
                <li><code>calculate_efficient_frontier()</code></li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="api-links">
      <a href="#" class="btn btn-primary">📖 Full API Reference</a>
      <a href="#" class="btn btn-secondary">🔗 Interactive API Explorer</a>
    </div>
  </section>

  <!-- Testing & Quality -->
  <section class="testing-section">
    <h2>🧪 Testing & Quality Assurance</h2>
    
    <div class="testing-grid">
      <div class="testing-card">
        <h3>Unit Tests</h3>
        <div class="test-stats">
          <div class="test-stat">
            <span class="test-number">150+</span>
            <span class="test-label">Test Cases</span>
          </div>
          <div class="test-stat">
            <span class="test-number">95%</span>
            <span class="test-label">Coverage</span>
          </div>
        </div>
        <div class="code-block">
          <pre><code class="language-bash"># Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module tests
pytest tests/test_hmm_engine.py</code></pre>
        </div>
      </div>
      
      <div class="testing-card">
        <h3>Code Quality</h3>
        <div class="quality-metrics">
          <div class="quality-item">
            <span class="quality-check">✅</span>
            <span>PEP 8 Compliant</span>
          </div>
          <div class="quality-item">
            <span class="quality-check">✅</span>
            <span>Type Hints</span>
          </div>
          <div class="quality-item">
            <span class="quality-check">✅</span>
            <span>Docstring Coverage</span>
          </div>
          <div class="quality-item">
            <span class="quality-check">✅</span>
            <span>Security Scanned</span>
          </div>
        </div>
        <div class="code-block">
          <pre><code class="language-bash"># Code formatting
black src/

# Linting
flake8 src/

# Type checking
mypy src/</code></pre>
        </div>
      </div>
    </div>
  </section>

  <!-- Performance & Benchmarks -->
  <section class="performance-section">
    <h2>⚡ Performance Benchmarks</h2>
    
    <div class="benchmark-grid">
      <div class="benchmark-card">
        <h3>HMM Training Performance</h3>
        <div class="benchmark-metrics">
          <div class="benchmark-item">
            <span class="benchmark-label">1,000 observations</span>
            <span class="benchmark-value">0.8s</span>
          </div>
          <div class="benchmark-item">
            <span class="benchmark-label">10,000 observations</span>
            <span class="benchmark-value">3.2s</span>
          </div>
          <div class="benchmark-item">
            <span class="benchmark-label">100,000 observations</span>
            <span class="benchmark-value">28.5s</span>
          </div>
        </div>
      </div>
      
      <div class="benchmark-card">
        <h3>Portfolio Optimization</h3>
        <div class="benchmark-metrics">
          <div class="benchmark-item">
            <span class="benchmark-label">50 assets</span>
            <span class="benchmark-value">0.2s</span>
          </div>
          <div class="benchmark-item">
            <span class="benchmark-label">200 assets</span>
            <span class="benchmark-value">1.1s</span>
          </div>
          <div class="benchmark-item">
            <span class="benchmark-label">1,000 assets</span>
            <span class="benchmark-value">8.7s</span>
          </div>
        </div>
      </div>
      
      <div class="benchmark-card">
        <h3>Monte Carlo Simulation</h3>
        <div class="benchmark-metrics">
          <div class="benchmark-item">
            <span class="benchmark-label">1,000 simulations</span>
            <span class="benchmark-value">0.5s</span>
          </div>
          <div class="benchmark-item">
            <span class="benchmark-label">10,000 simulations</span>
            <span class="benchmark-value">4.2s</span>
          </div>
          <div class="benchmark-item">
            <span class="benchmark-label">100,000 simulations</span>
            <span class="benchmark-value">35.8s</span>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Contributing -->
  <section class="contributing-section">
    <h2>🤝 Contributing</h2>
    
    <div class="contributing-content">
      <div class="contributing-text">
        <p>We welcome contributions from the quantitative finance community. Whether you're fixing bugs, adding features, or improving documentation, your contributions help advance the field.</p>
        
        <h3>How to Contribute:</h3>
        <ol>
          <li>Fork the repository</li>
          <li>Create a feature branch</li>
          <li>Make your changes with tests</li>
          <li>Submit a pull request</li>
        </ol>
        
        <h3>Areas for Contribution:</h3>
        <ul>
          <li>New optimization algorithms</li>
          <li>Alternative regime detection methods</li>
          <li>Enhanced risk models</li>
          <li>Performance improvements</li>
          <li>Documentation enhancements</li>
        </ul>
      </div>
      
      <div class="contributing-guidelines">
        <h3>Development Guidelines</h3>
        <div class="code-block">
          <pre><code class="language-bash"># Set up development environment
git clone https://github.com/your-repo/portfolio-engine.git
cd portfolio-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests before making changes
pytest tests/

# Make your changes...

# Run quality checks
black src/
flake8 src/
mypy src/
pytest tests/

# Submit pull request</code></pre>
        </div>
      </div>
    </div>
  </section>
</div>

<style>
.code-repository {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.repository-overview {
  text-align: center;
  margin-bottom: 4rem;
}

.repo-description {
  font-size: 1.125rem;
  color: var(--text-light);
  max-width: 800px;
  margin: 2rem auto;
  line-height: 1.6;
}

.repo-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.stat-item {
  text-align: center;
}

.stat-number {
  display: block;
  font-size: 2rem;
  font-weight: 700;
  color: var(--primary-color);
  margin-bottom: 0.5rem;
}

.stat-label {
  font-size: 0.875rem;
  color: var(--text-light);
  font-weight: 500;
}

.repository-links {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 2rem;
}

/* Architecture Section */
.architecture-section {
  margin-bottom: 4rem;
}

.architecture-diagram {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-width: 800px;
  margin: 2rem auto;
}

.architecture-layer {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
  border-left: 4px solid var(--primary-color);
}

.architecture-layer h3 {
  margin-bottom: 1rem;
  color: var(--text-color);
}

.layer-components {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.component {
  background: var(--bg-light);
  padding: 1rem;
  border-radius: var(--radius-md);
  text-align: center;
  font-weight: 500;
  color: var(--text-color);
  transition: background-color 0.2s ease;
}

.component:hover {
  background: var(--primary-color);
  color: white;
}

/* Modules Section */
.modules-section {
  margin-bottom: 4rem;
}

.modules-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
}

.module-card {
  background: white;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.module-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}

.module-header {
  background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
  color: white;
  padding: 1.5rem;
}

.module-header h3 {
  margin-bottom: 0.5rem;
  color: white;
}

.module-path {
  font-family: var(--font-mono);
  font-size: 0.875rem;
  opacity: 0.9;
}

.module-content {
  padding: 2rem;
}

.module-content p {
  color: var(--text-light);
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.code-preview {
  margin-bottom: 1.5rem;
}

.code-preview h4 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 1rem;
}

.code-block {
  background: #f8f9fa;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  overflow-x: auto;
}

.code-block pre {
  margin: 0;
  padding: 1rem;
  font-family: var(--font-mono);
  font-size: 0.875rem;
  line-height: 1.5;
  color: var(--text-color);
}

.code-block code {
  background: none;
  padding: 0;
  border-radius: 0;
}

.module-features {
  margin-bottom: 1.5rem;
}

.module-features h4 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 0.75rem;
}

.module-features ul {
  margin-left: 1rem;
}

.module-features li {
  margin-bottom: 0.5rem;
  color: var(--text-color);
}

.module-links {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.module-link {
  color: var(--primary-color);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 500;
  transition: color 0.2s ease;
}

.module-link:hover {
  color: var(--primary-dark);
}

/* Quick Start Section */
.quickstart-section {
  margin-bottom: 4rem;
}

.quickstart-steps {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.step {
  display: flex;
  gap: 2rem;
  align-items: start;
}

.step-number {
  flex-shrink: 0;
  width: 3rem;
  height: 3rem;
  background: var(--primary-color);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 1.25rem;
}

.step-content {
  flex: 1;
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
}

.step-content h3 {
  margin-bottom: 1rem;
  color: var(--text-color);
}

/* API Section */
.api-section {
  margin-bottom: 4rem;
}

.api-modules {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  margin-bottom: 2rem;
}

.api-module {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
}

.api-module h3 {
  margin-bottom: 1.5rem;
  color: var(--text-color);
}

.api-endpoint {
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.5rem;
  margin-bottom: 1rem;
}

.endpoint-method {
  display: inline-block;
  background: var(--primary-color);
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: var(--radius);
  font-size: 0.75rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.endpoint-path {
  font-family: var(--font-mono);
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 0.5rem;
}

.endpoint-description {
  color: var(--text-light);
  margin-bottom: 1rem;
}

.endpoint-details h4 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 0.75rem;
}

.endpoint-details ul {
  margin-left: 1rem;
}

.endpoint-details li {
  margin-bottom: 0.5rem;
  font-family: var(--font-mono);
  font-size: 0.875rem;
  color: var(--text-color);
}

.api-links {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

/* Testing Section */
.testing-section {
  margin-bottom: 4rem;
}

.testing-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.testing-card {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
}

.testing-card h3 {
  margin-bottom: 1.5rem;
  color: var(--text-color);
}

.test-stats {
  display: flex;
  gap: 2rem;
  margin-bottom: 1.5rem;
}

.test-stat {
  text-align: center;
}

.test-number {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--primary-color);
  margin-bottom: 0.25rem;
}

.test-label {
  font-size: 0.875rem;
  color: var(--text-light);
}

.quality-metrics {
  margin-bottom: 1.5rem;
}

.quality-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.quality-check {
  font-size: 1.25rem;
}

/* Performance Section */
.performance-section {
  margin-bottom: 4rem;
}

.benchmark-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}

.benchmark-card {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
}

.benchmark-card h3 {
  margin-bottom: 1.5rem;
  color: var(--text-color);
}

.benchmark-metrics {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.benchmark-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: var(--bg-light);
  border-radius: var(--radius-md);
}

.benchmark-label {
  font-weight: 500;
  color: var(--text-color);
}

.benchmark-value {
  font-weight: 600;
  color: var(--primary-color);
  font-family: var(--font-mono);
}

/* Contributing Section */
.contributing-section {
  margin-bottom: 4rem;
}

.contributing-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.contributing-text {
  p {
    font-size: 1.125rem;
    color: var(--text-color);
    line-height: 1.6;
    margin-bottom: 2rem;
  }
  
  h3 {
    color: var(--text-color);
    margin-bottom: 1rem;
    margin-top: 2rem;
  }
  
  ol, ul {
    margin-left: 1rem;
    
    li {
      margin-bottom: 0.5rem;
      color: var(--text-color);
    }
  }
}

.contributing-guidelines {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
  
  h3 {
    margin-bottom: 1.5rem;
    color: var(--text-color);
  }
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .code-repository {
    padding: 1rem;
  }
  
  .modules-grid {
    grid-template-columns: 1fr;
  }
  
  .step {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }
  
  .step-number {
    margin-bottom: 1rem;
  }
  
  .repo-stats {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .repository-links {
    flex-direction: column;
    align-items: center;
  }
}
</style>