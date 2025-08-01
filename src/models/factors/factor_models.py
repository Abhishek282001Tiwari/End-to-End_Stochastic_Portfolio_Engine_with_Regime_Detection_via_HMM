import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import het_white
import warnings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class FamaFrenchFactorModel:
    def __init__(self, risk_free_rate: Optional[pd.Series] = None):
        self.risk_free_rate = risk_free_rate
        self.factors = None
        self.factor_loadings = {}
        self.regression_results = {}
        
    def create_factors(
        self, 
        returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        book_to_market: Optional[pd.DataFrame] = None,
        profitability: Optional[pd.DataFrame] = None,
        investment: Optional[pd.DataFrame] = None,
        momentum: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        logger.info("Creating Fama-French factors")
        
        factors = pd.DataFrame(index=returns.index)
        
        factors['MKT'] = self._create_market_factor(returns, market_caps)
        
        if book_to_market is not None:
            smb, hml = self._create_size_value_factors(returns, market_caps, book_to_market)
            factors['SMB'] = smb
            factors['HML'] = hml
        
        if profitability is not None:
            factors['RMW'] = self._create_profitability_factor(returns, market_caps, profitability)
        
        if investment is not None:
            factors['CMA'] = self._create_investment_factor(returns, market_caps, investment)
        
        if momentum is not None:
            factors['MOM'] = self._create_momentum_factor(returns, market_caps, momentum)
        
        self.factors = factors.dropna()
        return self.factors
    
    def _create_market_factor(self, returns: pd.DataFrame, market_caps: pd.DataFrame) -> pd.Series:
        market_returns = []
        
        for date in returns.index:
            if date in market_caps.index:
                weights = market_caps.loc[date].fillna(0)
                weights = weights / weights.sum()
                
                daily_returns = returns.loc[date].reindex(weights.index).fillna(0)
                market_return = (weights * daily_returns).sum()
                market_returns.append(market_return)
            else:
                market_returns.append(np.nan)
        
        market_factor = pd.Series(market_returns, index=returns.index)
        
        if self.risk_free_rate is not None:
            aligned_rf = self.risk_free_rate.reindex(returns.index).fillna(0) / 252
            market_factor = market_factor - aligned_rf
        
        return market_factor
    
    def _create_size_value_factors(
        self, 
        returns: pd.DataFrame, 
        market_caps: pd.DataFrame,
        book_to_market: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        smb_returns = []
        hml_returns = []
        
        for date in returns.index:
            if date in market_caps.index and date in book_to_market.index:
                caps = market_caps.loc[date].dropna()
                btm = book_to_market.loc[date].dropna()
                rets = returns.loc[date]
                
                common_assets = caps.index.intersection(btm.index).intersection(rets.index)
                
                if len(common_assets) > 6:
                    caps_common = caps[common_assets]
                    btm_common = btm[common_assets]
                    rets_common = rets[common_assets]
                    
                    size_median = caps_common.median()
                    btm_30 = btm_common.quantile(0.3)
                    btm_70 = btm_common.quantile(0.7)
                    
                    small = caps_common <= size_median
                    big = caps_common > size_median
                    
                    low = btm_common <= btm_30
                    medium = (btm_common > btm_30) & (btm_common <= btm_70)
                    high = btm_common > btm_70
                    
                    portfolios = {
                        'SL': small & low,
                        'SM': small & medium,
                        'SH': small & high,
                        'BL': big & low,
                        'BM': big & medium,
                        'BH': big & high
                    }
                    
                    portfolio_returns = {}
                    for name, mask in portfolios.items():
                        if mask.sum() > 0:
                            weights = caps_common[mask] / caps_common[mask].sum()
                            portfolio_returns[name] = (weights * rets_common[mask]).sum()
                        else:
                            portfolio_returns[name] = 0
                    
                    smb = (portfolio_returns['SL'] + portfolio_returns['SM'] + portfolio_returns['SH']) / 3 - \
                          (portfolio_returns['BL'] + portfolio_returns['BM'] + portfolio_returns['BH']) / 3
                    
                    hml = (portfolio_returns['SH'] + portfolio_returns['BH']) / 2 - \
                          (portfolio_returns['SL'] + portfolio_returns['BL']) / 2
                    
                    smb_returns.append(smb)
                    hml_returns.append(hml)
                else:
                    smb_returns.append(np.nan)
                    hml_returns.append(np.nan)
            else:
                smb_returns.append(np.nan)
                hml_returns.append(np.nan)
        
        return pd.Series(smb_returns, index=returns.index), pd.Series(hml_returns, index=returns.index)
    
    def _create_profitability_factor(
        self, 
        returns: pd.DataFrame, 
        market_caps: pd.DataFrame,
        profitability: pd.DataFrame
    ) -> pd.Series:
        rmw_returns = []
        
        for date in returns.index:
            if date in market_caps.index and date in profitability.index:
                caps = market_caps.loc[date].dropna()
                prof = profitability.loc[date].dropna()
                rets = returns.loc[date]
                
                common_assets = caps.index.intersection(prof.index).intersection(rets.index)
                
                if len(common_assets) > 4:
                    caps_common = caps[common_assets]
                    prof_common = prof[common_assets]
                    rets_common = rets[common_assets]
                    
                    size_median = caps_common.median()
                    prof_30 = prof_common.quantile(0.3)
                    prof_70 = prof_common.quantile(0.7)
                    
                    small = caps_common <= size_median
                    big = caps_common > size_median
                    
                    weak = prof_common <= prof_30
                    robust = prof_common > prof_70
                    
                    portfolios = {
                        'SW': small & weak,
                        'SR': small & robust,
                        'BW': big & weak,
                        'BR': big & robust
                    }
                    
                    portfolio_returns = {}
                    for name, mask in portfolios.items():
                        if mask.sum() > 0:
                            weights = caps_common[mask] / caps_common[mask].sum()
                            portfolio_returns[name] = (weights * rets_common[mask]).sum()
                        else:
                            portfolio_returns[name] = 0
                    
                    rmw = (portfolio_returns['SR'] + portfolio_returns['BR']) / 2 - \
                          (portfolio_returns['SW'] + portfolio_returns['BW']) / 2
                    
                    rmw_returns.append(rmw)
                else:
                    rmw_returns.append(np.nan)
            else:
                rmw_returns.append(np.nan)
        
        return pd.Series(rmw_returns, index=returns.index)
    
    def _create_investment_factor(
        self, 
        returns: pd.DataFrame, 
        market_caps: pd.DataFrame,
        investment: pd.DataFrame
    ) -> pd.Series:
        cma_returns = []
        
        for date in returns.index:
            if date in market_caps.index and date in investment.index:
                caps = market_caps.loc[date].dropna()
                inv = investment.loc[date].dropna()
                rets = returns.loc[date]
                
                common_assets = caps.index.intersection(inv.index).intersection(rets.index)
                
                if len(common_assets) > 4:
                    caps_common = caps[common_assets]
                    inv_common = inv[common_assets]
                    rets_common = rets[common_assets]
                    
                    size_median = caps_common.median()
                    inv_30 = inv_common.quantile(0.3)
                    inv_70 = inv_common.quantile(0.7)
                    
                    small = caps_common <= size_median
                    big = caps_common > size_median
                    
                    conservative = inv_common <= inv_30
                    aggressive = inv_common > inv_70
                    
                    portfolios = {
                        'SC': small & conservative,
                        'SA': small & aggressive,
                        'BC': big & conservative,
                        'BA': big & aggressive
                    }
                    
                    portfolio_returns = {}
                    for name, mask in portfolios.items():
                        if mask.sum() > 0:
                            weights = caps_common[mask] / caps_common[mask].sum()
                            portfolio_returns[name] = (weights * rets_common[mask]).sum()
                        else:
                            portfolio_returns[name] = 0
                    
                    cma = (portfolio_returns['SC'] + portfolio_returns['BC']) / 2 - \
                          (portfolio_returns['SA'] + portfolio_returns['BA']) / 2
                    
                    cma_returns.append(cma)
                else:
                    cma_returns.append(np.nan)
            else:
                cma_returns.append(np.nan)
        
        return pd.Series(cma_returns, index=returns.index)
    
    def _create_momentum_factor(
        self, 
        returns: pd.DataFrame, 
        market_caps: pd.DataFrame,
        momentum: pd.DataFrame
    ) -> pd.Series:
        mom_returns = []
        
        for date in returns.index:
            if date in market_caps.index and date in momentum.index:
                caps = market_caps.loc[date].dropna()
                mom = momentum.loc[date].dropna()
                rets = returns.loc[date]
                
                common_assets = caps.index.intersection(mom.index).intersection(rets.index)
                
                if len(common_assets) > 4:
                    caps_common = caps[common_assets]
                    mom_common = mom[common_assets]
                    rets_common = rets[common_assets]
                    
                    size_median = caps_common.median()
                    mom_30 = mom_common.quantile(0.3)
                    mom_70 = mom_common.quantile(0.7)
                    
                    small = caps_common <= size_median
                    big = caps_common > size_median
                    
                    losers = mom_common <= mom_30
                    winners = mom_common > mom_70
                    
                    portfolios = {
                        'SL': small & losers,
                        'SW': small & winners,
                        'BL': big & losers,
                        'BW': big & winners
                    }
                    
                    portfolio_returns = {}
                    for name, mask in portfolios.items():
                        if mask.sum() > 0:
                            weights = caps_common[mask] / caps_common[mask].sum()
                            portfolio_returns[name] = (weights * rets_common[mask]).sum()
                        else:
                            portfolio_returns[name] = 0
                    
                    mom = (portfolio_returns['SW'] + portfolio_returns['BW']) / 2 - \
                          (portfolio_returns['SL'] + portfolio_returns['BL']) / 2
                    
                    mom_returns.append(mom)
                else:
                    mom_returns.append(np.nan)
            else:
                mom_returns.append(np.nan)
        
        return pd.Series(mom_returns, index=returns.index)
    
    def fit_factor_model(self, asset_returns: pd.DataFrame) -> Dict[str, Any]:
        if self.factors is None:
            raise ValueError("Factors must be created before fitting model")
        
        logger.info("Fitting Fama-French factor model")
        
        for asset in asset_returns.columns:
            asset_ret = asset_returns[asset].dropna()
            
            if self.risk_free_rate is not None:
                rf_aligned = self.risk_free_rate.reindex(asset_ret.index).fillna(0) / 252
                excess_returns = asset_ret - rf_aligned
            else:
                excess_returns = asset_ret
            
            factor_data = self.factors.reindex(excess_returns.index).dropna()
            common_dates = excess_returns.index.intersection(factor_data.index)
            
            if len(common_dates) > len(self.factors.columns) + 10:
                X = factor_data.loc[common_dates]
                y = excess_returns.loc[common_dates]
                
                model = LinearRegression()
                model.fit(X, y)
                
                y_pred = model.predict(X)
                residuals = y - y_pred
                
                r_squared = model.score(X, y)
                
                self.factor_loadings[asset] = {
                    'alpha': model.intercept_,
                    'betas': dict(zip(X.columns, model.coef_)),
                    'r_squared': r_squared,
                    'residual_std': np.std(residuals),
                    't_stats': self._calculate_t_stats(X, y, model),
                    'residuals': residuals
                }
        
        return self.factor_loadings
    
    def _calculate_t_stats(self, X: pd.DataFrame, y: pd.Series, model: LinearRegression) -> Dict[str, float]:
        n = len(y)
        k = X.shape[1]
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n - k - 1)
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        
        se_beta = np.sqrt(np.diag(var_beta))
        
        t_stats = {'alpha': model.intercept_ / se_beta[0]}
        for i, col in enumerate(X.columns):
            t_stats[col] = model.coef_[i] / se_beta[i + 1]
        
        return t_stats


class StatisticalFactorModel:
    def __init__(self, n_factors: int = 5, method: str = "pca"):
        self.n_factors = n_factors
        self.method = method
        self.factor_model = None
        self.factor_loadings = None
        self.factors = None
        
    def fit(self, returns: pd.DataFrame) -> 'StatisticalFactorModel':
        logger.info(f"Fitting statistical factor model with {self.n_factors} factors using {self.method}")
        
        returns_clean = returns.dropna()
        
        if self.method == "pca":
            self.factor_model = PCA(n_components=self.n_factors)
            self.factors = self.factor_model.fit_transform(returns_clean)
            self.factor_loadings = self.factor_model.components_.T
            
        elif self.method == "factor_analysis":
            self.factor_model = FactorAnalysis(n_components=self.n_factors, random_state=42)
            self.factors = self.factor_model.fit_transform(returns_clean)
            self.factor_loadings = self.factor_model.components_.T
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.factors = pd.DataFrame(
            self.factors,
            index=returns_clean.index,
            columns=[f"Factor_{i+1}" for i in range(self.n_factors)]
        )
        
        self.factor_loadings = pd.DataFrame(
            self.factor_loadings,
            index=returns_clean.columns,
            columns=[f"Factor_{i+1}" for i in range(self.n_factors)]
        )
        
        return self
    
    def get_factor_exposures(self) -> pd.DataFrame:
        return self.factor_loadings
    
    def get_factors(self) -> pd.DataFrame:
        return self.factors
    
    def calculate_specific_risk(self, returns: pd.DataFrame) -> pd.Series:
        if self.factor_model is None:
            raise ValueError("Model must be fitted first")
        
        explained_variance = np.sum(self.factor_loadings ** 2, axis=1)
        total_variance = returns.var()
        specific_variance = total_variance - explained_variance
        
        return np.sqrt(specific_variance.clip(lower=0))


class MacroeconomicFactorModel:
    def __init__(self):
        self.factor_loadings = {}
        self.macro_factors = None
        
    def create_macro_factors(
        self,
        gdp_growth: pd.Series,
        inflation: pd.Series,
        interest_rates: pd.Series,
        credit_spreads: pd.Series,
        exchange_rates: Optional[pd.Series] = None,
        commodity_prices: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        logger.info("Creating macroeconomic factors")
        
        factors = pd.DataFrame({
            'GDP_Growth': gdp_growth,
            'Inflation': inflation,
            'Interest_Rates': interest_rates,
            'Credit_Spreads': credit_spreads
        })
        
        if exchange_rates is not None:
            factors['Exchange_Rates'] = exchange_rates
        
        if commodity_prices is not None:
            factors['Commodity_Prices'] = commodity_prices
        
        factors['Term_Structure'] = factors['Interest_Rates'].diff()
        factors['Credit_Quality'] = factors['Credit_Spreads'].diff()
        
        self.macro_factors = factors.dropna()
        return self.macro_factors
    
    def fit_macro_model(self, asset_returns: pd.DataFrame) -> Dict[str, Any]:
        if self.macro_factors is None:
            raise ValueError("Macro factors must be created first")
        
        logger.info("Fitting macroeconomic factor model")
        
        for asset in asset_returns.columns:
            asset_ret = asset_returns[asset].dropna()
            factor_data = self.macro_factors.reindex(asset_ret.index).dropna()
            
            common_dates = asset_ret.index.intersection(factor_data.index)
            
            if len(common_dates) > len(self.macro_factors.columns) + 10:
                X = factor_data.loc[common_dates]
                y = asset_ret.loc[common_dates]
                
                model = LinearRegression()
                model.fit(X, y)
                
                self.factor_loadings[asset] = {
                    'alpha': model.intercept_,
                    'betas': dict(zip(X.columns, model.coef_)),
                    'r_squared': model.score(X, y)
                }
        
        return self.factor_loadings


class FactorModelEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        self.models[name] = model
        self.weights[name] = weight
        
    def combine_factor_loadings(self, asset: str) -> Dict[str, float]:
        if asset not in self.models:
            return {}
        
        combined_loadings = {}
        total_weight = sum(self.weights.values())
        
        for model_name, model in self.models.items():
            weight = self.weights[model_name] / total_weight
            
            if hasattr(model, 'factor_loadings') and asset in model.factor_loadings:
                loadings = model.factor_loadings[asset]
                
                if isinstance(loadings, dict) and 'betas' in loadings:
                    for factor, beta in loadings['betas'].items():
                        factor_key = f"{model_name}_{factor}"
                        combined_loadings[factor_key] = combined_loadings.get(factor_key, 0) + weight * beta
        
        return combined_loadings
    
    def get_model_comparison(self) -> pd.DataFrame:
        results = []
        
        all_assets = set()
        for model in self.models.values():
            if hasattr(model, 'factor_loadings'):
                all_assets.update(model.factor_loadings.keys())
        
        for asset in all_assets:
            for model_name, model in self.models.items():
                if hasattr(model, 'factor_loadings') and asset in model.factor_loadings:
                    loadings = model.factor_loadings[asset]
                    
                    if isinstance(loadings, dict):
                        results.append({
                            'asset': asset,
                            'model': model_name,
                            'r_squared': loadings.get('r_squared', np.nan),
                            'alpha': loadings.get('alpha', np.nan),
                            'residual_std': loadings.get('residual_std', np.nan)
                        })
        
        return pd.DataFrame(results) if results else pd.DataFrame()