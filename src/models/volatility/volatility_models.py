import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import warnings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class GARCHModel:
    def __init__(
        self,
        vol_model: str = "GARCH",
        p: int = 1,
        q: int = 1,
        distribution: str = "normal"
    ):
        self.vol_model = vol_model
        self.p = p
        self.q = q
        self.distribution = distribution
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns: pd.Series) -> 'GARCHModel':
        logger.info(f"Fitting {self.vol_model}({self.p},{self.q}) model")
        
        returns_clean = returns.dropna() * 100
        
        self.model = arch_model(
            returns_clean,
            vol=self.vol_model,
            p=self.p,
            q=self.q,
            dist=self.distribution,
            rescale=False
        )
        
        try:
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            logger.info("GARCH model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            raise
        
        return self
    
    def forecast(self, horizon: int = 1) -> Dict[str, np.ndarray]:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.fitted_model.forecast(horizon=horizon)
        
        return {
            "mean": forecast.mean.values[-1] / 100,
            "variance": forecast.variance.values[-1] / 10000,
            "volatility": np.sqrt(forecast.variance.values[-1] / 10000)
        }
    
    def get_conditional_volatility(self) -> pd.Series:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return np.sqrt(self.fitted_model.conditional_volatility / 100)
    
    def get_model_summary(self) -> str:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return str(self.fitted_model.summary())


class RealizedVolatilityModel:
    def __init__(self, frequency: str = "5min"):
        self.frequency = frequency
        self.realized_vol = None
        
    def calculate_realized_volatility(
        self,
        high_freq_data: pd.DataFrame,
        method: str = "standard"
    ) -> pd.Series:
        logger.info(f"Calculating realized volatility using {method} method")
        
        if method == "standard":
            returns = high_freq_data.pct_change().dropna()
            daily_rv = returns.groupby(returns.index.date).apply(
                lambda x: np.sqrt(np.sum(x ** 2) * 252)
            )
            
        elif method == "bipower":
            returns = high_freq_data.pct_change().dropna()
            daily_rv = returns.groupby(returns.index.date).apply(
                lambda x: self._bipower_variation(x) * 252
            )
            
        elif method == "two_scale":
            returns = high_freq_data.pct_change().dropna()
            daily_rv = returns.groupby(returns.index.date).apply(
                lambda x: self._two_scale_estimator(x) * 252
            )
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.realized_vol = pd.Series(daily_rv.values, index=pd.to_datetime(daily_rv.index))
        return self.realized_vol
    
    def _bipower_variation(self, returns: pd.Series) -> float:
        abs_returns = np.abs(returns)
        bipower = np.sum(abs_returns[:-1] * abs_returns[1:])
        return (np.pi / 2) * bipower
    
    def _two_scale_estimator(self, returns: pd.Series) -> float:
        n = len(returns)
        if n < 4:
            return np.sum(returns ** 2)
        
        rv_all = np.sum(returns ** 2)
        
        sub_returns = returns[::2]
        rv_sub = 2 * np.sum(sub_returns ** 2)
        
        return rv_all - (rv_sub - rv_all) / (n / 2 - 1)


class HestonModel:
    def __init__(self):
        self.params = None
        self.fitted = False
        
    def fit(
        self,
        prices: pd.Series,
        initial_params: Optional[Dict[str, float]] = None
    ) -> 'HestonModel':
        logger.info("Fitting Heston stochastic volatility model")
        
        if initial_params is None:
            initial_params = {
                'kappa': 2.0,  # Mean reversion speed
                'theta': 0.04,  # Long-term variance
                'sigma': 0.3,   # Volatility of volatility  
                'rho': -0.7,    # Correlation
                'v0': 0.04      # Initial variance
            }
        
        returns = prices.pct_change().dropna()
        
        def heston_log_likelihood(params):
            kappa, theta, sigma, rho, v0 = params
            
            if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0 or abs(rho) >= 1:
                return np.inf
            
            dt = 1/252
            n = len(returns)
            
            log_likelihood = 0
            v_t = v0
            
            for i in range(n):
                ret = returns.iloc[i]
                
                mu_v = kappa * (theta - v_t) * dt
                sigma_v = sigma * np.sqrt(v_t * dt)
                
                v_next = v_t + mu_v + sigma_v * np.random.normal()
                v_next = max(v_next, 1e-6)
                
                mu_s = -0.5 * v_t * dt
                sigma_s = np.sqrt(v_t * dt)
                
                log_likelihood += -0.5 * np.log(2 * np.pi * v_t * dt) - (ret - mu_s) ** 2 / (2 * v_t * dt)
                
                v_t = v_next
            
            return -log_likelihood
        
        bounds = [
            (0.1, 10.0),   # kappa
            (0.001, 1.0),  # theta
            (0.01, 2.0),   # sigma
            (-0.99, 0.99), # rho
            (0.001, 1.0)   # v0
        ]
        
        x0 = list(initial_params.values())
        
        try:
            result = minimize(
                heston_log_likelihood,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                self.params = dict(zip(initial_params.keys(), result.x))
                self.fitted = True
                logger.info("Heston model fitted successfully")
            else:
                logger.error("Heston model fitting failed")
                raise ValueError("Optimization failed")
                
        except Exception as e:
            logger.error(f"Error fitting Heston model: {e}")
            raise
        
        return self
    
    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
        
        dt = T / n_steps
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = self.params['v0']
        
        for i in range(n_steps):
            Z1 = np.random.normal(0, 1, n_paths)
            Z2 = np.random.normal(0, 1, n_paths)
            Z2 = self.params['rho'] * Z1 + np.sqrt(1 - self.params['rho']**2) * Z2
            
            v[:, i+1] = np.maximum(
                v[:, i] + self.params['kappa'] * (self.params['theta'] - v[:, i]) * dt +
                self.params['sigma'] * np.sqrt(v[:, i] * dt) * Z2,
                1e-6
            )
            
            S[:, i+1] = S[:, i] * np.exp(
                -0.5 * v[:, i] * dt + np.sqrt(v[:, i] * dt) * Z1
            )
        
        return S, v


class VolatilityRegimeModel:
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.regime_parameters = {}
        self.transition_matrix = None
        self.fitted = False
        
    def fit(self, returns: pd.Series) -> 'VolatilityRegimeModel':
        logger.info(f"Fitting volatility regime model with {self.n_regimes} regimes")
        
        from hmmlearn import hmm
        
        returns_clean = returns.dropna().values.reshape(-1, 1)
        
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100
        )
        
        try:
            model.fit(returns_clean)
            
            states = model.predict(returns_clean)
            
            for regime in range(self.n_regimes):
                regime_returns = returns_clean[states == regime]
                
                if len(regime_returns) > 0:
                    self.regime_parameters[regime] = {
                        'mean': np.mean(regime_returns),
                        'volatility': np.std(regime_returns),
                        'frequency': np.sum(states == regime) / len(states)
                    }
            
            self.transition_matrix = model.transmat_
            self.fitted = True
            
            logger.info("Volatility regime model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting volatility regime model: {e}")
            raise
        
        return self
    
    def predict_regime_volatility(
        self,
        current_regime: int,
        horizon: int = 1
    ) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        regime_probs = np.zeros(self.n_regimes)
        regime_probs[current_regime] = 1.0
        
        volatilities = []
        
        for h in range(horizon):
            expected_vol = 0
            for regime, prob in enumerate(regime_probs):
                if regime in self.regime_parameters:
                    expected_vol += prob * self.regime_parameters[regime]['volatility']
            
            volatilities.append(expected_vol)
            regime_probs = regime_probs @ self.transition_matrix
        
        return np.array(volatilities)


class VolatilityForecaster:
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        
    def add_model(self, name: str, model: Any):
        self.models[name] = model
        
    def fit_all_models(self, returns: pd.Series):
        logger.info("Fitting all volatility models")
        
        for name, model in self.models.items():
            try:
                logger.info(f"Fitting {name} model")
                model.fit(returns)
            except Exception as e:
                logger.error(f"Error fitting {name} model: {e}")
    
    def generate_forecasts(self, horizon: int = 30) -> pd.DataFrame:
        logger.info(f"Generating volatility forecasts for {horizon} periods")
        
        forecast_results = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'forecast'):
                    forecast = model.forecast(horizon=horizon)
                    if isinstance(forecast, dict) and 'volatility' in forecast:
                        forecast_results[name] = forecast['volatility']
                    else:
                        forecast_results[name] = forecast
                        
                elif hasattr(model, 'predict_regime_volatility'):
                    forecast_results[name] = model.predict_regime_volatility(0, horizon)
                    
            except Exception as e:
                logger.error(f"Error generating forecast for {name}: {e}")
                forecast_results[name] = np.full(horizon, np.nan)
        
        return pd.DataFrame(forecast_results)
    
    def calculate_ensemble_forecast(
        self,
        forecasts: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        if weights is None:
            weights = {col: 1.0 / len(forecasts.columns) for col in forecasts.columns}
        
        ensemble = pd.Series(0, index=forecasts.index)
        
        for model_name, weight in weights.items():
            if model_name in forecasts.columns:
                ensemble += weight * forecasts[model_name].fillna(0)
        
        return ensemble
    
    def evaluate_forecast_accuracy(
        self,
        forecasts: pd.DataFrame,
        realized_volatility: pd.Series
    ) -> pd.DataFrame:
        results = []
        
        common_index = forecasts.index.intersection(realized_volatility.index)
        
        for model_name in forecasts.columns:
            forecast_values = forecasts.loc[common_index, model_name].dropna()
            realized_values = realized_volatility.loc[forecast_values.index]
            
            if len(forecast_values) > 0:
                mse = np.mean((forecast_values - realized_values) ** 2)
                mae = np.mean(np.abs(forecast_values - realized_values))
                mape = np.mean(np.abs((forecast_values - realized_values) / realized_values)) * 100
                
                results.append({
                    'model': model_name,
                    'mse': mse,
                    'mae': mae,
                    'mape': mape,
                    'n_observations': len(forecast_values)
                })
        
        return pd.DataFrame(results)