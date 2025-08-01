import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.hmm.hmm_engine import RegimeDetectionHMM, OnlineHMMUpdater, EnsembleRegimeDetector


@pytest.fixture
def sample_market_data():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    n_samples = len(dates)
    
    bull_regime = np.random.normal(0.001, 0.015, n_samples // 3)
    bear_regime = np.random.normal(-0.002, 0.025, n_samples // 3)
    sideways_regime = np.random.normal(0.0, 0.010, n_samples - 2 * (n_samples // 3))
    
    returns = np.concatenate([bull_regime, bear_regime, sideways_regime])
    volatility = np.abs(returns) + np.random.normal(0, 0.005, n_samples)
    volume = np.random.lognormal(10, 0.5, n_samples)
    vix = 20 + np.random.normal(0, 5, n_samples)
    yield_curve = np.random.normal(0.02, 0.01, n_samples)
    
    return pd.DataFrame({
        'market_return': returns,
        'volatility': volatility,
        'volume': volume,
        'vix': vix,
        'yield_curve_slope': yield_curve
    }, index=dates[:n_samples])


class TestRegimeDetectionHMM:
    def test_init(self):
        hmm = RegimeDetectionHMM(n_components=3, covariance_type="full")
        assert hmm.n_components == 3
        assert hmm.covariance_type == "full"
        assert not hmm.is_fitted
    
    def test_fit(self, sample_market_data):
        hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        hmm.fit(sample_market_data)
        
        assert hmm.is_fitted
        assert hmm.model is not None
        assert hasattr(hmm.model, 'means_')
        assert hasattr(hmm.model, 'covars_')
        assert hasattr(hmm.model, 'transmat_')
    
    def test_predict_regimes(self, sample_market_data):
        hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        hmm.fit(sample_market_data)
        
        test_data = sample_market_data.tail(100)
        states = hmm.predict_regimes(test_data)
        
        assert len(states) == len(test_data)
        assert all(0 <= state < 3 for state in states)
    
    def test_predict_regime_probabilities(self, sample_market_data):
        hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        hmm.fit(sample_market_data)
        
        test_data = sample_market_data.tail(50)
        probabilities = hmm.predict_regime_probabilities(test_data)
        
        assert probabilities.shape == (len(test_data), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10)
        assert np.all(probabilities >= 0)
    
    def test_score(self, sample_market_data):
        hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        hmm.fit(sample_market_data)
        
        test_data = sample_market_data.tail(100)
        score = hmm.score(test_data)
        
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_get_regime_statistics(self, sample_market_data):
        hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        hmm.fit(sample_market_data)
        
        stats = hmm.get_regime_statistics(sample_market_data)
        
        assert 'regime_distribution' in stats
        assert 'average_regime_duration' in stats
        assert 'transition_probabilities' in stats
        assert 'regime_means' in stats
        assert 'regime_covariances' in stats
        assert 'regime_names' in stats
        
        assert len(stats['regime_distribution']) == 3
        assert np.allclose(stats['regime_distribution'].sum(), 1.0)
    
    def test_get_regime_summary(self, sample_market_data):
        hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        hmm.fit(sample_market_data)
        
        summary = hmm.get_regime_summary(sample_market_data)
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(sample_market_data)
        assert 'regime' in summary.columns
        assert 'regime_name' in summary.columns
        assert 'confidence' in summary.columns
        
        assert all(0 <= regime < 3 for regime in summary['regime'])
        assert all(0 <= conf <= 1 for conf in summary['confidence'])


class TestOnlineHMMUpdater:
    def test_init(self, sample_market_data):
        base_hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        base_hmm.fit(sample_market_data.head(500))
        
        updater = OnlineHMMUpdater(base_hmm, update_window=50)
        
        assert updater.base_model == base_hmm
        assert updater.update_window == 50
        assert len(updater.data_buffer) == 0
    
    def test_update_with_new_data(self, sample_market_data):
        base_hmm = RegimeDetectionHMM(n_components=3, random_state=42)
        base_hmm.fit(sample_market_data.head(500))
        
        updater = OnlineHMMUpdater(base_hmm, update_window=5)
        
        for i in range(6):
            new_data = sample_market_data.iloc[500 + i*10:500 + (i+1)*10]
            result = updater.update_with_new_data(new_data)
            
            if i < 4:
                assert not result
            else:
                assert result is not None


class TestEnsembleRegimeDetector:
    def test_init_and_fit(self, sample_market_data):
        models = [
            RegimeDetectionHMM(n_components=3, random_state=42),
            RegimeDetectionHMM(n_components=3, random_state=43),
            RegimeDetectionHMM(n_components=3, random_state=44)
        ]
        
        ensemble = EnsembleRegimeDetector(models)
        ensemble.fit(sample_market_data)
        
        assert len(ensemble.models) == 3
        assert all(model.is_fitted for model in ensemble.models)
    
    def test_predict_regime_probabilities(self, sample_market_data):
        models = [
            RegimeDetectionHMM(n_components=3, random_state=42),
            RegimeDetectionHMM(n_components=3, random_state=43)
        ]
        
        ensemble = EnsembleRegimeDetector(models)
        ensemble.fit(sample_market_data)
        
        test_data = sample_market_data.tail(50)
        probabilities = ensemble.predict_regime_probabilities(test_data)
        
        assert probabilities.shape == (len(test_data), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10)
    
    def test_predict_regimes(self, sample_market_data):
        models = [
            RegimeDetectionHMM(n_components=3, random_state=42),
            RegimeDetectionHMM(n_components=3, random_state=43)
        ]
        
        ensemble = EnsembleRegimeDetector(models)
        ensemble.fit(sample_market_data)
        
        test_data = sample_market_data.tail(50)
        states = ensemble.predict_regimes(test_data)
        
        assert len(states) == len(test_data)
        assert all(0 <= state < 3 for state in states)


@pytest.mark.parametrize("n_components", [2, 3, 4])
def test_different_regime_numbers(sample_market_data, n_components):
    hmm = RegimeDetectionHMM(n_components=n_components, random_state=42)
    hmm.fit(sample_market_data)
    
    assert hmm.n_components == n_components
    assert hmm.is_fitted
    
    states = hmm.predict_regimes(sample_market_data.tail(100))
    assert all(0 <= state < n_components for state in states)


def test_empty_data():
    hmm = RegimeDetectionHMM(n_components=3)
    
    empty_data = pd.DataFrame()
    
    with pytest.raises(Exception):
        hmm.fit(empty_data)


def test_insufficient_data():
    hmm = RegimeDetectionHMM(n_components=3)
    
    small_data = pd.DataFrame({
        'feature1': [0.1, 0.2],
        'feature2': [0.3, 0.4]
    })
    
    with pytest.raises(Exception):
        hmm.fit(small_data)