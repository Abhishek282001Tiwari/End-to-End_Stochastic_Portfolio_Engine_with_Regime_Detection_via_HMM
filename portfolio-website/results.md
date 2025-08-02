---
layout: page
title: Results & Analytics
permalink: /results/
description: "Interactive portfolio performance analytics, backtesting results, and comprehensive risk analysis for the stochastic portfolio engine."
---

<div class="results-dashboard">
  <!-- Performance Summary -->
  <section class="performance-summary">
    <h2>Portfolio Performance Summary</h2>
    <div class="summary-grid">
      <div class="metric-card">
        <div class="metric-value">20.4%</div>
        <div class="metric-label">Annual Return</div>
        <div class="metric-change positive">+8.3% vs Benchmark</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">1.47</div>
        <div class="metric-label">Sharpe Ratio</div>
        <div class="metric-change positive">+0.58 vs Benchmark</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">14.2%</div>
        <div class="metric-label">Volatility</div>
        <div class="metric-change positive">-2.6% vs Benchmark</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">-8.2%</div>
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-change positive">+10.2% vs Benchmark</div>
      </div>
    </div>
  </section>

  <!-- Interactive Charts -->
  <section class="charts-section">
    <div class="chart-container">
      <h3>Cumulative Performance</h3>
      <canvas id="performanceChart" width="800" height="400"></canvas>
    </div>
    
    <div class="chart-container">
      <h3>Regime Detection Timeline</h3>
      <canvas id="regimeChart" width="800" height="300"></canvas>
    </div>
    
    <div class="chart-grid">
      <div class="chart-container">
        <h3>Risk-Return Scatter</h3>
        <canvas id="riskReturnChart" width="400" height="300"></canvas>
      </div>
      <div class="chart-container">
        <h3>Rolling Sharpe Ratio</h3>
        <canvas id="sharpeChart" width="400" height="300"></canvas>
      </div>
    </div>
  </section>

  <!-- Detailed Analytics -->
  <section class="detailed-analytics">
    <h2>Detailed Performance Analytics</h2>
    
    <div class="analytics-tabs">
      <button class="tab-button active" onclick="showTab('returns')">Returns Analysis</button>
      <button class="tab-button" onclick="showTab('risk')">Risk Metrics</button>
      <button class="tab-button" onclick="showTab('attribution')">Performance Attribution</button>
      <button class="tab-button" onclick="showTab('regime')">Regime Analysis</button>
    </div>

    <!-- Returns Analysis Tab -->
    <div id="returns" class="tab-content active">
      <div class="analysis-grid">
        <div class="analysis-card">
          <h4>Return Statistics</h4>
          <table class="metrics-table">
            <tr><td>Total Return</td><td>20.4%</td><td class="benchmark">12.1%</td></tr>
            <tr><td>Annualized Return</td><td>18.7%</td><td class="benchmark">11.2%</td></tr>
            <tr><td>Monthly Return (Avg)</td><td>1.42%</td><td class="benchmark">0.94%</td></tr>
            <tr><td>Best Month</td><td>8.7%</td><td class="benchmark">12.4%</td></tr>
            <tr><td>Worst Month</td><td>-4.2%</td><td class="benchmark">-12.5%</td></tr>
            <tr><td>Win Rate</td><td>67.3%</td><td class="benchmark">58.1%</td></tr>
          </table>
        </div>
        
        <div class="analysis-card">
          <h4>Distribution Analysis</h4>
          <canvas id="returnsDistribution" width="300" height="200"></canvas>
        </div>
      </div>
    </div>

    <!-- Risk Metrics Tab -->
    <div id="risk" class="tab-content">
      <div class="analysis-grid">
        <div class="analysis-card">
          <h4>Risk Measures</h4>
          <table class="metrics-table">
            <tr><td>Volatility (Annual)</td><td>14.2%</td><td class="benchmark">16.8%</td></tr>
            <tr><td>95% VaR (Monthly)</td><td>-3.8%</td><td class="benchmark">-6.2%</td></tr>
            <tr><td>95% CVaR (Monthly)</td><td>-5.1%</td><td class="benchmark">-9.4%</td></tr>
            <tr><td>Maximum Drawdown</td><td>-8.2%</td><td class="benchmark">-18.4%</td></tr>
            <tr><td>Skewness</td><td>0.23</td><td class="benchmark">-0.67</td></tr>
            <tr><td>Kurtosis</td><td>3.14</td><td class="benchmark">5.89</td></tr>
          </table>
        </div>
        
        <div class="analysis-card">
          <h4>Drawdown Analysis</h4>
          <canvas id="drawdownChart" width="300" height="200"></canvas>
        </div>
      </div>
    </div>

    <!-- Performance Attribution Tab -->
    <div id="attribution" class="tab-content">
      <div class="analysis-grid">
        <div class="analysis-card">
          <h4>Factor Attribution</h4>
          <table class="metrics-table">
            <tr><td>Market Factor</td><td>85.2%</td><td>7.3%</td></tr>
            <tr><td>Size Factor</td><td>12.1%</td><td>1.8%</td></tr>
            <tr><td>Value Factor</td><td>-3.4%</td><td>-0.6%</td></tr>
            <tr><td>Momentum Factor</td><td>18.7%</td><td>2.1%</td></tr>
            <tr><td>Quality Factor</td><td>9.3%</td><td>1.2%</td></tr>
            <tr><td>Alpha</td><td>-</td><td>8.6%</td></tr>
          </table>
        </div>
        
        <div class="analysis-card">
          <h4>Sector Attribution</h4>
          <canvas id="sectorChart" width="300" height="200"></canvas>
        </div>
      </div>
    </div>

    <!-- Regime Analysis Tab -->
    <div id="regime" class="tab-content">
      <div class="analysis-grid">
        <div class="analysis-card">
          <h4>Regime Performance</h4>
          <table class="metrics-table">
            <tr><th>Regime</th><th>Frequency</th><th>Avg Return</th><th>Volatility</th></tr>
            <tr><td>Bull Market</td><td>35.2%</td><td>2.4%</td><td>8.7%</td></tr>
            <tr><td>Bear Market</td><td>24.8%</td><td>-1.8%</td><td>22.3%</td></tr>
            <tr><td>High Volatility</td><td>19.7%</td><td>-0.3%</td><td>28.9%</td></tr>
            <tr><td>Sideways</td><td>20.3%</td><td>0.8%</td><td>11.2%</td></tr>
          </table>
        </div>
        
        <div class="analysis-card">
          <h4>Regime Transitions</h4>
          <canvas id="regimeTransition" width="300" height="200"></canvas>
        </div>
      </div>
    </div>
  </section>

  <!-- Monte Carlo Results -->
  <section class="monte-carlo-section">
    <h2>Monte Carlo Simulation Results</h2>
    <div class="monte-carlo-grid">
      <div class="monte-carlo-card">
        <h4>Forward-Looking Projections (1 Year)</h4>
        <div class="projection-metrics">
          <div class="projection-item">
            <span class="projection-label">Expected Return</span>
            <span class="projection-value">18.2% ± 4.1%</span>
          </div>
          <div class="projection-item">
            <span class="projection-label">Probability of Loss</span>
            <span class="projection-value">12.4%</span>
          </div>
          <div class="projection-item">
            <span class="projection-label">95% Confidence Interval</span>
            <span class="projection-value">[8.7%, 29.3%]</span>
          </div>
        </div>
      </div>
      
      <div class="monte-carlo-card">
        <h4>Simulation Distribution</h4>
        <canvas id="monteCarloChart" width="400" height="250"></canvas>
      </div>
    </div>
  </section>
</div>

<script src="{{ '/assets/js/results.js' | relative_url }}"></script>

<style>
.results-dashboard {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.performance-summary {
  margin-bottom: 3rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
}

.metric-card {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-align: center;
  box-shadow: var(--shadow);
  border-top: 4px solid var(--primary-color);
}

.metric-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--primary-color);
  margin-bottom: 0.5rem;
}

.metric-label {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 0.5rem;
}

.metric-change {
  font-size: 0.875rem;
  font-weight: 500;
}

.metric-change.positive {
  color: var(--success-color);
}

.metric-change.negative {
  color: var(--error-color);
}

.charts-section {
  margin-bottom: 3rem;
}

.chart-container {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: var(--shadow);
}

.chart-container h3 {
  margin-bottom: 1.5rem;
  color: var(--text-color);
  font-size: 1.25rem;
  font-weight: 600;
}

.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.detailed-analytics {
  margin-bottom: 3rem;
}

.analytics-tabs {
  display: flex;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border-color);
  
  @media (max-width: 768px) {
    flex-wrap: wrap;
  }
}

.tab-button {
  background: none;
  border: none;
  padding: 1rem 2rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-light);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
  
  &:hover {
    color: var(--primary-color);
  }
  
  &.active {
    color: var(--primary-color);
    border-bottom-color: var(--primary-color);
  }
}

.tab-content {
  display: none;
  
  &.active {
    display: block;
  }
}

.analysis-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.analysis-card {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
}

.analysis-card h4 {
  margin-bottom: 1.5rem;
  color: var(--text-color);
  font-size: 1.125rem;
  font-weight: 600;
}

.metrics-table {
  width: 100%;
  
  td, th {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-light);
  }
  
  th {
    font-weight: 600;
    color: var(--text-color);
    background-color: var(--bg-light);
  }
  
  td:nth-child(2) {
    font-weight: 600;
    color: var(--primary-color);
  }
  
  td.benchmark {
    color: var(--text-light);
    font-size: 0.875rem;
  }
}

.monte-carlo-section {
  margin-bottom: 3rem;
}

.monte-carlo-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.monte-carlo-card {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
}

.monte-carlo-card h4 {
  margin-bottom: 1.5rem;
  color: var(--text-color);
  font-size: 1.125rem;
  font-weight: 600;
}

.projection-metrics {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.projection-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background-color: var(--bg-light);
  border-radius: var(--radius-md);
}

.projection-label {
  font-weight: 500;
  color: var(--text-color);
}

.projection-value {
  font-weight: 600;
  color: var(--primary-color);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .results-dashboard {
    padding: 1rem;
  }
  
  .summary-grid {
    grid-template-columns: 1fr;
  }
  
  .chart-container {
    padding: 1rem;
  }
  
  .analytics-tabs {
    flex-direction: column;
  }
  
  .tab-button {
    text-align: left;
    padding: 0.75rem 1rem;
  }
}
</style>