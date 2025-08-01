import pytest
import numpy as np
import pandas as pd
from src.optimization.portfolio.stochastic_optimizer import (
    MeanVarianceOptimizer, 
    BlackLittermanOptimizer,
    RiskParityOptimizer,
    MonteCarloOptimizer,
    RegimeAwareOptimizer,
    PortfolioOptimizationEngine
)


@pytest.fixture
def sample_market_data():
    np.random.seed(42)
    n_assets = 5
    n_periods = 252
    
    returns = np.random.normal(0.08/252, 0.2/np.sqrt(252), (n_periods, n_assets))
    
    returns_df = pd.DataFrame(
        returns,
        columns=[f'Asset_{i}' for i in range(n_assets)],
        index=pd.date_range('2023-01-01', periods=n_periods, freq='D')
    )
    
    expected_returns = returns_df.mean() * 252
    covariance_matrix = returns_df.cov() * 252
    
    return {
        'returns': returns_df,
        'expected_returns': expected_returns.values,
        'covariance_matrix': covariance_matrix.values,
        'asset_names': returns_df.columns.tolist()
    }


class TestMeanVarianceOptimizer:
    def test_init(self):
        optimizer = MeanVarianceOptimizer(
            risk_aversion=2.0,
            transaction_costs=0.001,
            max_weight=0.3,
            min_weight=0.05
        )
        
        assert optimizer.risk_aversion == 2.0
        assert optimizer.transaction_costs == 0.001
        assert optimizer.max_weight == 0.3
        assert optimizer.min_weight == 0.05
    
    def test_optimize_basic(self, sample_market_data):
        optimizer = MeanVarianceOptimizer()
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'expected_volatility' in result
        assert 'sharpe_ratio' in result
        assert 'status' in result
        
        weights = result['weights']
        assert len(weights) == len(sample_market_data['expected_returns'])
        assert np.allclose(weights.sum(), 1.0, rtol=1e-3)
        assert np.all(weights >= -1e-6)
        assert np.all(weights <= 0.2 + 1e-6)
    
    def test_optimize_with_current_weights(self, sample_market_data):
        optimizer = MeanVarianceOptimizer(transaction_costs=0.01)
        
        current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix'],
            current_weights=current_weights
        )
        
        assert 'weights' in result
        assert result['status'] in ['optimal', 'optimal_inaccurate']
    
    def test_optimize_constraints(self, sample_market_data):
        optimizer = MeanVarianceOptimizer(
            max_weight=0.1,
            min_weight=0.1,
            leverage=0.8
        )
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        weights = result['weights']
        assert np.allclose(weights.sum(), 0.8, rtol=1e-3)
        assert np.all(weights >= 0.1 - 1e-6)
        assert np.all(weights <= 0.1 + 1e-6)


class TestBlackLittermanOptimizer:
    def test_init(self):
        optimizer = BlackLittermanOptimizer(
            tau=0.05,
            risk_aversion=2.5
        )
        
        assert optimizer.tau == 0.05
        assert optimizer.risk_aversion == 2.5
    
    def test_optimize_without_views(self, sample_market_data):
        optimizer = BlackLittermanOptimizer()
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'expected_volatility' in result
        assert len(result['weights']) == len(sample_market_data['expected_returns'])
    
    def test_optimize_with_views(self, sample_market_data):
        optimizer = BlackLittermanOptimizer()
        
        views_matrix = np.array([[1, 0, -1, 0, 0]])
        views_returns = np.array([0.05])
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix'],
            views_matrix=views_matrix,
            views_returns=views_returns
        )
        
        assert 'weights' in result
        assert result['status'] in ['optimal', 'optimal_inaccurate']


class TestRiskParityOptimizer:
    def test_init(self):
        target_rc = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        optimizer = RiskParityOptimizer(target_risk_contributions=target_rc)
        
        assert np.allclose(optimizer.target_risk_contributions, target_rc)
    
    def test_optimize_equal_risk(self, sample_market_data):
        optimizer = RiskParityOptimizer()
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'expected_volatility' in result
        
        weights = result['weights']
        assert len(weights) == len(sample_market_data['expected_returns'])
        assert np.allclose(weights.sum(), 1.0, rtol=1e-3)
        assert np.all(weights >= 0.01 - 1e-6)
    
    def test_optimize_custom_risk_contributions(self, sample_market_data):
        target_rc = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        optimizer = RiskParityOptimizer(target_risk_contributions=target_rc)
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        assert 'weights' in result
        assert result['status'] in ['optimal', 'failed']


class TestMonteCarloOptimizer:
    def test_init(self):
        base_optimizer = MeanVarianceOptimizer()
        optimizer = MonteCarloOptimizer(
            n_simulations=1000,
            confidence_level=0.1,
            base_optimizer=base_optimizer
        )
        
        assert optimizer.n_simulations == 1000
        assert optimizer.confidence_level == 0.1
        assert optimizer.base_optimizer == base_optimizer
    
    def test_optimize(self, sample_market_data):
        optimizer = MonteCarloOptimizer(n_simulations=500)
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        assert 'weights' in result
        assert 'var' in result
        assert 'cvar' in result
        assert 'weight_std' in result
        assert 'n_successful_optimizations' in result
        
        weights = result['weights']
        assert len(weights) == len(sample_market_data['expected_returns'])
        
        if result['status'] == 'optimal':
            assert np.allclose(weights.sum(), 1.0, rtol=1e-2)


class TestRegimeAwareOptimizer:
    def test_init(self):
        regime_optimizers = {
            0: MeanVarianceOptimizer(risk_aversion=3.0),
            1: RiskParityOptimizer(),
            2: MeanVarianceOptimizer(risk_aversion=1.0)
        }
        
        optimizer = RegimeAwareOptimizer(regime_optimizers)
        
        assert len(optimizer.regime_optimizers) == 3
        assert 0 in optimizer.regime_optimizers
        assert 1 in optimizer.regime_optimizers
        assert 2 in optimizer.regime_optimizers
    
    def test_optimize(self, sample_market_data):
        regime_optimizers = {
            0: MeanVarianceOptimizer(risk_aversion=3.0),
            1: RiskParityOptimizer(),
            2: MeanVarianceOptimizer(risk_aversion=1.0)
        }
        
        covariance_matrices = {
            0: sample_market_data['covariance_matrix'],
            1: sample_market_data['covariance_matrix'] * 0.8,
            2: sample_market_data['covariance_matrix'] * 1.2
        }
        
        regime_probabilities = np.array([0.3, 0.4, 0.3])
        current_regime = 1
        
        optimizer = RegimeAwareOptimizer(regime_optimizers)
        
        result = optimizer.optimize(
            sample_market_data['expected_returns'],
            covariance_matrices,
            regime_probabilities,
            current_regime
        )
        
        assert 'weights' in result
        assert 'regime_weights' in result
        assert 'regime_probabilities' in result
        assert 'regime_metrics' in result
        
        weights = result['weights']
        assert len(weights) == len(sample_market_data['expected_returns'])


class TestPortfolioOptimizationEngine:
    def test_init(self):
        engine = PortfolioOptimizationEngine()
        
        assert 'mean_variance' in engine.optimizers
        assert 'black_litterman' in engine.optimizers
        assert 'risk_parity' in engine.optimizers
        assert 'monte_carlo' in engine.optimizers
    
    def test_optimize_portfolio(self, sample_market_data):
        engine = PortfolioOptimizationEngine()
        
        result = engine.optimize_portfolio(
            'mean_variance',
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'expected_volatility' in result
    
    def test_add_optimizer(self, sample_market_data):
        engine = PortfolioOptimizationEngine()
        
        custom_optimizer = MeanVarianceOptimizer(risk_aversion=5.0)
        engine.add_optimizer('custom_mv', custom_optimizer)
        
        assert 'custom_mv' in engine.optimizers
        
        result = engine.optimize_portfolio(
            'custom_mv',
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix']
        )
        
        assert 'weights' in result
    
    def test_compare_optimizers(self, sample_market_data):
        engine = PortfolioOptimizationEngine()
        
        comparison = engine.compare_optimizers(
            sample_market_data['expected_returns'],
            sample_market_data['covariance_matrix'],
            methods=['mean_variance', 'risk_parity']
        )
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'method' in comparison.columns
        assert 'expected_return' in comparison.columns
        assert 'expected_volatility' in comparison.columns
        assert 'sharpe_ratio' in comparison.columns
    
    def test_unknown_method(self, sample_market_data):
        engine = PortfolioOptimizationEngine()
        
        with pytest.raises(ValueError):
            engine.optimize_portfolio(
                'unknown_method',
                sample_market_data['expected_returns'],
                sample_market_data['covariance_matrix']
            )


@pytest.mark.parametrize("optimizer_class", [
    MeanVarianceOptimizer,
    RiskParityOptimizer,
])
def test_optimizer_edge_cases(optimizer_class, sample_market_data):
    optimizer = optimizer_class()
    
    result = optimizer.optimize(
        sample_market_data['expected_returns'],
        sample_market_data['covariance_matrix']
    )
    
    assert 'weights' in result
    assert len(result['weights']) == len(sample_market_data['expected_returns'])


def test_singular_covariance_matrix():
    optimizer = MeanVarianceOptimizer()
    
    expected_returns = np.array([0.1, 0.12, 0.08])
    
    singular_cov = np.array([
        [0.04, 0.02, 0.02],
        [0.02, 0.01, 0.01],
        [0.02, 0.01, 0.01]
    ])
    
    result = optimizer.optimize(expected_returns, singular_cov)
    
    assert 'weights' in result


def test_negative_expected_returns(sample_market_data):
    optimizer = MeanVarianceOptimizer()
    
    negative_returns = -np.abs(sample_market_data['expected_returns'])
    
    result = optimizer.optimize(
        negative_returns,
        sample_market_data['covariance_matrix']
    )
    
    assert 'weights' in result