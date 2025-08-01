import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from abc import ABC, abstractmethod
import requests
import yfinance as yf

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class FactorType(Enum):
    """Types of factor models"""
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    STATISTICAL = "statistical"
    MACROECONOMIC = "macroeconomic"
    FUNDAMENTAL = "fundamental"
    ESG = "esg"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    CUSTOM = "custom"


@dataclass
class FactorModel:
    """Factor model configuration and results"""
    name: str
    factor_type: FactorType
    factors: List[str]
    loadings: Optional[np.ndarray] = None
    factor_returns: Optional[pd.DataFrame] = None
    r_squared: Optional[float] = None
    alpha: Optional[np.ndarray] = None
    residual_risk: Optional[np.ndarray] = None
    factor_correlations: Optional[pd.DataFrame] = None
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class FactorExposure:
    """Factor exposure for a single asset or portfolio"""
    asset_name: str
    exposures: Dict[str, float]
    alpha: float
    r_squared: float
    residual_volatility: float
    t_statistics: Dict[str, float]
    p_values: Dict[str, float]


class BaseFactorModel(ABC):
    """Abstract base class for factor models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.factor_returns = None
        self.loadings = None
        
    @abstractmethod
    def fit(self, returns: pd.DataFrame, **kwargs) -> 'BaseFactorModel':
        """Fit the factor model to return data"""
        pass
    
    @abstractmethod
    def get_factor_exposures(self, asset_returns: pd.Series) -> FactorExposure:
        """Calculate factor exposures for an asset"""
        pass
    
    @abstractmethod
    def get_factor_names(self) -> List[str]:
        """Get list of factor names"""
        pass


class FamaFrenchModel(BaseFactorModel):
    """Fama-French factor models (3-factor and 5-factor)"""
    
    def __init__(self, model_type: str = "3_factor"):
        self.model_type = model_type
        super().__init__(f"Fama-French {model_type}")
        
        if model_type == "3_factor":
            self.factor_names = ["MKT-RF", "SMB", "HML"]
        elif model_type == "5_factor":
            self.factor_names = ["MKT-RF", "SMB", "HML", "RMW", "CMA"]
        else:
            raise ValueError("model_type must be '3_factor' or '5_factor'")
    
    def fit(self, returns: pd.DataFrame, **kwargs) -> 'FamaFrenchModel':
        """
        Fit Fama-French model using downloaded factor data
        """
        logger.info(f"Fitting {self.name} model")
        
        # Download Fama-French factors (this is a simplified implementation)
        # In practice, you would download from Ken French's data library
        factor_data = self._get_fama_french_factors()
        
        if factor_data is not None:
            # Align dates
            common_dates = returns.index.intersection(factor_data.index)
            
            if len(common_dates) > 0:
                self.factor_returns = factor_data.loc[common_dates, self.factor_names]
                aligned_returns = returns.loc[common_dates]
                
                # Calculate loadings using regression
                self.loadings = {}
                for asset in aligned_returns.columns:
                    asset_returns = aligned_returns[asset].dropna()
                    
                    # Align factor returns with asset returns
                    factor_subset = self.factor_returns.loc[asset_returns.index]
                    
                    if len(factor_subset) > 20:  # Minimum observations
                        # Subtract risk-free rate if available
                        if "RF" in factor_data.columns:
                            excess_returns = asset_returns - factor_data.loc[asset_returns.index, "RF"]
                        else:
                            excess_returns = asset_returns
                        
                        # Regression
                        X = factor_subset.values
                        y = excess_returns.values
                        
                        model = LinearRegression().fit(X, y)
                        self.loadings[asset] = {
                            'alpha': model.intercept_,
                            'betas': dict(zip(self.factor_names, model.coef_)),
                            'r_squared': r2_score(y, model.predict(X))
                        }
                
                self.is_fitted = True
                logger.info(f"{self.name} model fitted successfully")
            else:
                logger.warning("No common dates found between returns and factor data")
        else:
            logger.error("Failed to obtain Fama-French factor data")
        
        return self
    
    def _get_fama_french_factors(self) -> Optional[pd.DataFrame]:
        """
        Download Fama-French factors
        This is a simplified implementation - in practice you would download from:
        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
        """
        try:
            # Create synthetic factor data for demonstration
            # In real implementation, download from Ken French's website
            dates = pd.date_range('2020-01-01', periods=1000, freq='D')
            
            # Generate synthetic factor returns
            np.random.seed(42)
            n_obs = len(dates)
            
            factor_data = pd.DataFrame(index=dates)
            
            # Market factor (MKT-RF)
            factor_data['MKT-RF'] = np.random.normal(0.0005, 0.015, n_obs)
            
            # Size factor (SMB)
            factor_data['SMB'] = np.random.normal(0.0002, 0.008, n_obs)
            
            # Value factor (HML)
            factor_data['HML'] = np.random.normal(0.0001, 0.009, n_obs)
            
            if self.model_type == "5_factor":
                # Profitability factor (RMW)
                factor_data['RMW'] = np.random.normal(0.0001, 0.007, n_obs)
                
                # Investment factor (CMA)
                factor_data['CMA'] = np.random.normal(0.0001, 0.006, n_obs)
            
            # Risk-free rate
            factor_data['RF'] = np.random.normal(0.00005, 0.0001, n_obs)
            
            return factor_data
            
        except Exception as e:
            logger.error(f"Error creating factor data: {e}")
            return None
    
    def get_factor_exposures(self, asset_returns: pd.Series) -> FactorExposure:
        """Calculate factor exposures for an asset"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating exposures")
        
        asset_name = asset_returns.name or "Unknown"
        
        if asset_name in self.loadings:
            loadings = self.loadings[asset_name]
            
            # Calculate t-statistics and p-values
            common_dates = asset_returns.index.intersection(self.factor_returns.index)
            
            if len(common_dates) > 20:
                X = self.factor_returns.loc[common_dates]
                y = asset_returns.loc[common_dates]
                
                # Regression with statistics
                model = LinearRegression().fit(X, y)
                predictions = model.predict(X)
                residuals = y - predictions
                
                # Calculate standard errors
                n = len(y)
                k = X.shape[1]
                mse = np.sum(residuals**2) / (n - k - 1)
                
                X_with_intercept = np.column_stack([np.ones(n), X])
                var_coef = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                se_coef = np.sqrt(np.diag(var_coef))
                
                # t-statistics and p-values
                coef_with_intercept = np.concatenate([[model.intercept_], model.coef_])
                t_stats = coef_with_intercept / se_coef
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
                
                t_statistics = dict(zip(['alpha'] + self.factor_names, t_stats))
                p_val_dict = dict(zip(['alpha'] + self.factor_names, p_values))
                
                return FactorExposure(
                    asset_name=asset_name,
                    exposures=loadings['betas'],
                    alpha=loadings['alpha'],
                    r_squared=loadings['r_squared'],
                    residual_volatility=np.std(residuals),
                    t_statistics=t_statistics,
                    p_values=p_val_dict
                )
        
        # Return empty exposure if not found
        return FactorExposure(
            asset_name=asset_name,
            exposures={factor: 0.0 for factor in self.factor_names},
            alpha=0.0,
            r_squared=0.0,
            residual_volatility=asset_returns.std(),
            t_statistics={},
            p_values={}
        )
    
    def get_factor_names(self) -> List[str]:
        return self.factor_names


class StatisticalFactorModel(BaseFactorModel):
    """Statistical factor model using PCA or Factor Analysis"""
    
    def __init__(self, n_factors: int = 5, method: str = "pca"):
        self.n_factors = n_factors
        self.method = method
        super().__init__(f"Statistical-{method.upper()}-{n_factors}")
        
        if method == "pca":
            self.model = PCA(n_components=n_factors)
        elif method == "factor_analysis":
            self.model = FactorAnalysis(n_factors=n_factors)
        else:
            raise ValueError("method must be 'pca' or 'factor_analysis'")
        
        self.scaler = StandardScaler()
        self.factor_names = [f"Factor_{i+1}" for i in range(n_factors)]
    
    def fit(self, returns: pd.DataFrame, **kwargs) -> 'StatisticalFactorModel':
        """Fit statistical factor model"""
        logger.info(f"Fitting {self.name} model")
        
        # Remove assets with insufficient data
        min_observations = kwargs.get('min_observations', 100)
        valid_assets = returns.columns[returns.count() >= min_observations]
        clean_returns = returns[valid_assets].dropna()
        
        if clean_returns.empty:
            raise ValueError("No valid data for factor model fitting")
        
        # Standardize returns
        scaled_returns = self.scaler.fit_transform(clean_returns)
        
        # Fit factor model
        self.model.fit(scaled_returns)
        
        # Extract factor returns
        factor_scores = self.model.transform(scaled_returns)
        self.factor_returns = pd.DataFrame(
            factor_scores,
            index=clean_returns.index,
            columns=self.factor_names
        )
        
        # Calculate loadings
        if hasattr(self.model, 'components_'):
            # PCA case
            self.loadings = pd.DataFrame(
                self.model.components_.T,
                index=valid_assets,
                columns=self.factor_names
            )
        elif hasattr(self.model, 'components_'):
            # Factor Analysis case
            self.loadings = pd.DataFrame(
                self.model.components_.T,
                index=valid_assets,
                columns=self.factor_names
            )
        
        self.is_fitted = True
        logger.info(f"{self.name} model fitted successfully")
        
        return self
    
    def get_factor_exposures(self, asset_returns: pd.Series) -> FactorExposure:
        """Calculate factor exposures for an asset"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating exposures")
        
        asset_name = asset_returns.name or "Unknown"
        
        # Align data
        common_dates = asset_returns.index.intersection(self.factor_returns.index)
        
        if len(common_dates) < 20:
            return FactorExposure(
                asset_name=asset_name,
                exposures={factor: 0.0 for factor in self.factor_names},
                alpha=0.0,
                r_squared=0.0,
                residual_volatility=asset_returns.std(),
                t_statistics={},
                p_values={}
            )
        
        # Regression
        X = self.factor_returns.loc[common_dates]
        y = asset_returns.loc[common_dates]
        
        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Statistics
        n = len(y)
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k - 1)
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        var_coef = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_coef = np.sqrt(np.diag(var_coef))
        
        coef_with_intercept = np.concatenate([[model.intercept_], model.coef_])
        t_stats = coef_with_intercept / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        
        t_statistics = dict(zip(['alpha'] + self.factor_names, t_stats))
        p_val_dict = dict(zip(['alpha'] + self.factor_names, p_values))
        
        return FactorExposure(
            asset_name=asset_name,
            exposures=dict(zip(self.factor_names, model.coef_)),
            alpha=model.intercept_,
            r_squared=r2_score(y, predictions),
            residual_volatility=np.std(residuals),
            t_statistics=t_statistics,
            p_values=p_val_dict
        )
    
    def get_factor_names(self) -> List[str]:
        return self.factor_names


class MacroeconomicFactorModel(BaseFactorModel):
    """Macroeconomic factor model using economic indicators"""
    
    def __init__(self):
        super().__init__("Macroeconomic")
        self.factor_names = [
            "GDP_Growth",
            "Inflation",
            "Interest_Rates",
            "Industrial_Production",
            "Unemployment",
            "Credit_Spread",
            "Term_Spread",
            "Dollar_Index"
        ]
        self.macro_data = None
    
    def fit(self, returns: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None, **kwargs) -> 'MacroeconomicFactorModel':
        """Fit macroeconomic factor model"""
        logger.info("Fitting Macroeconomic factor model")
        
        if macro_data is not None:
            self.macro_data = macro_data
        else:
            # Generate synthetic macro data for demonstration
            self.macro_data = self._generate_synthetic_macro_data(returns.index)
        
        # Align data
        common_dates = returns.index.intersection(self.macro_data.index)
        
        if len(common_dates) < 50:
            raise ValueError("Insufficient overlapping data for macro factor model")
        
        self.factor_returns = self.macro_data.loc[common_dates]
        aligned_returns = returns.loc[common_dates]
        
        # Calculate factor loadings for each asset
        self.loadings = {}
        
        for asset in aligned_returns.columns:
            asset_returns = aligned_returns[asset].dropna()
            
            # Align macro factors
            factor_subset = self.factor_returns.loc[asset_returns.index]
            
            if len(factor_subset) > 20:
                # Ridge regression for stability
                model = Ridge(alpha=0.1)
                model.fit(factor_subset, asset_returns)
                
                self.loadings[asset] = {
                    'alpha': model.intercept_,
                    'betas': dict(zip(self.factor_names, model.coef_)),
                    'r_squared': r2_score(asset_returns, model.predict(factor_subset))
                }
        
        self.is_fitted = True
        logger.info("Macroeconomic factor model fitted successfully")
        
        return self
    
    def _generate_synthetic_macro_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic macroeconomic data for demonstration"""
        
        np.random.seed(42)
        n_obs = len(dates)
        
        macro_data = pd.DataFrame(index=dates)
        
        # GDP Growth (quarterly, interpolated to daily)
        macro_data['GDP_Growth'] = np.random.normal(0.0005, 0.002, n_obs)
        
        # Inflation (monthly changes)
        macro_data['Inflation'] = np.random.normal(0.0002, 0.001, n_obs)
        
        # Interest Rates (daily changes)
        macro_data['Interest_Rates'] = np.random.normal(0.00001, 0.0005, n_obs)
        
        # Industrial Production
        macro_data['Industrial_Production'] = np.random.normal(0.0001, 0.003, n_obs)
        
        # Unemployment (monthly changes)
        macro_data['Unemployment'] = np.random.normal(-0.00005, 0.0008, n_obs)
        
        # Credit Spread
        macro_data['Credit_Spread'] = np.random.normal(0.00002, 0.001, n_obs)
        
        # Term Spread
        macro_data['Term_Spread'] = np.random.normal(0.00001, 0.0008, n_obs)
        
        # Dollar Index
        macro_data['Dollar_Index'] = np.random.normal(0.0001, 0.005, n_obs)
        
        return macro_data
    
    def get_factor_exposures(self, asset_returns: pd.Series) -> FactorExposure:
        """Calculate factor exposures for an asset"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating exposures")
        
        asset_name = asset_returns.name or "Unknown"
        
        if asset_name in self.loadings:
            loadings = self.loadings[asset_name]
            
            # Calculate statistics
            common_dates = asset_returns.index.intersection(self.factor_returns.index)
            
            if len(common_dates) > 20:
                X = self.factor_returns.loc[common_dates]
                y = asset_returns.loc[common_dates]
                
                model = Ridge(alpha=0.1)
                model.fit(X, y)
                predictions = model.predict(X)
                residuals = y - predictions
                
                # Simple t-statistics (Ridge doesn't have direct inference)
                # This is approximate
                t_statistics = {factor: 0.0 for factor in ['alpha'] + self.factor_names}
                p_values = {factor: 0.5 for factor in ['alpha'] + self.factor_names}
                
                return FactorExposure(
                    asset_name=asset_name,
                    exposures=loadings['betas'],
                    alpha=loadings['alpha'],
                    r_squared=loadings['r_squared'],
                    residual_volatility=np.std(residuals),
                    t_statistics=t_statistics,
                    p_values=p_values
                )
        
        return FactorExposure(
            asset_name=asset_name,
            exposures={factor: 0.0 for factor in self.factor_names},
            alpha=0.0,
            r_squared=0.0,
            residual_volatility=asset_returns.std(),
            t_statistics={},
            p_values={}
        )
    
    def get_factor_names(self) -> List[str]:
        return self.factor_names


class CustomFactorModel(BaseFactorModel):
    """Custom factor model with user-defined factors"""
    
    def __init__(self, factor_definitions: Dict[str, Callable]):
        self.factor_definitions = factor_definitions
        self.factor_names = list(factor_definitions.keys())
        super().__init__(f"Custom-{len(self.factor_names)}-Factor")
    
    def fit(self, returns: pd.DataFrame, **kwargs) -> 'CustomFactorModel':
        """Fit custom factor model"""
        logger.info(f"Fitting {self.name} model")
        
        # Calculate custom factors
        factor_data = pd.DataFrame(index=returns.index)
        
        for factor_name, factor_func in self.factor_definitions.items():
            try:
                factor_data[factor_name] = factor_func(returns)
            except Exception as e:
                logger.warning(f"Error calculating factor {factor_name}: {e}")
                factor_data[factor_name] = 0.0
        
        self.factor_returns = factor_data.dropna()
        
        # Calculate loadings
        self.loadings = {}
        common_dates = returns.index.intersection(self.factor_returns.index)
        
        for asset in returns.columns:
            asset_returns = returns.loc[common_dates, asset].dropna()
            
            if len(asset_returns) > 20:
                factor_subset = self.factor_returns.loc[asset_returns.index]
                
                model = LinearRegression().fit(factor_subset, asset_returns)
                
                self.loadings[asset] = {
                    'alpha': model.intercept_,
                    'betas': dict(zip(self.factor_names, model.coef_)),
                    'r_squared': r2_score(asset_returns, model.predict(factor_subset))
                }
        
        self.is_fitted = True
        logger.info(f"{self.name} model fitted successfully")
        
        return self
    
    def get_factor_exposures(self, asset_returns: pd.Series) -> FactorExposure:
        """Calculate factor exposures for an asset"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating exposures")
        
        asset_name = asset_returns.name or "Unknown"
        
        if asset_name in self.loadings:
            loadings = self.loadings[asset_name]
            
            # Calculate statistics
            common_dates = asset_returns.index.intersection(self.factor_returns.index)
            
            if len(common_dates) > 20:
                X = self.factor_returns.loc[common_dates]
                y = asset_returns.loc[common_dates]
                
                model = LinearRegression().fit(X, y)
                predictions = model.predict(X)
                residuals = y - predictions
                
                # Calculate t-statistics
                n = len(y)
                k = X.shape[1]
                mse = np.sum(residuals**2) / (n - k - 1)
                
                X_with_intercept = np.column_stack([np.ones(n), X])
                var_coef = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                se_coef = np.sqrt(np.diag(var_coef))
                
                coef_with_intercept = np.concatenate([[model.intercept_], model.coef_])
                t_stats = coef_with_intercept / se_coef
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
                
                t_statistics = dict(zip(['alpha'] + self.factor_names, t_stats))
                p_val_dict = dict(zip(['alpha'] + self.factor_names, p_values))
                
                return FactorExposure(
                    asset_name=asset_name,
                    exposures=loadings['betas'],
                    alpha=loadings['alpha'],
                    r_squared=loadings['r_squared'],
                    residual_volatility=np.std(residuals),
                    t_statistics=t_statistics,
                    p_values=p_val_dict
                )
        
        return FactorExposure(
            asset_name=asset_name,
            exposures={factor: 0.0 for factor in self.factor_names},
            alpha=0.0,
            r_squared=0.0,
            residual_volatility=asset_returns.std(),
            t_statistics={},
            p_values={}
        )
    
    def get_factor_names(self) -> List[str]:
        return self.factor_names


class FactorModelManager:
    """Manager for multiple factor models with comparison and analysis"""
    
    def __init__(self):
        self.models: Dict[str, BaseFactorModel] = {}
        self.fitted_models: Dict[str, BaseFactorModel] = {}
        
    def add_model(self, name: str, model: BaseFactorModel):
        """Add a factor model"""
        self.models[name] = model
        logger.info(f"Added factor model: {name}")
    
    def fit_all_models(self, returns: pd.DataFrame, **kwargs) -> Dict[str, BaseFactorModel]:
        """Fit all registered models"""
        logger.info("Fitting all factor models")
        
        results = {}
        
        for name, model in self.models.items():
            try:
                fitted_model = model.fit(returns, **kwargs)
                self.fitted_models[name] = fitted_model
                results[name] = fitted_model
                logger.info(f"Successfully fitted model: {name}")
            except Exception as e:
                logger.error(f"Failed to fit model {name}: {e}")
        
        return results
    
    def compare_models(self, asset_returns: pd.Series) -> pd.DataFrame:
        """Compare factor models for a specific asset"""
        
        if not self.fitted_models:
            raise ValueError("No fitted models available for comparison")
        
        comparison_data = []
        
        for name, model in self.fitted_models.items():
            try:
                exposure = model.get_factor_exposures(asset_returns)
                
                comparison_data.append({
                    'model': name,
                    'alpha': exposure.alpha,
                    'r_squared': exposure.r_squared,
                    'residual_volatility': exposure.residual_volatility,
                    'n_factors': len(exposure.exposures),
                    'significant_factors': sum(1 for p in exposure.p_values.values() if p < 0.05)
                })
                
            except Exception as e:
                logger.warning(f"Error comparing model {name}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def get_factor_attribution(
        self,
        asset_returns: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """Calculate factor attribution for an asset using specified model"""
        
        if model_name not in self.fitted_models:
            raise ValueError(f"Model {model_name} is not fitted")
        
        model = self.fitted_models[model_name]
        exposure = model.get_factor_exposures(asset_returns)
        
        if not hasattr(model, 'factor_returns') or model.factor_returns is None:
            return {}
        
        # Calculate factor contributions
        common_dates = asset_returns.index.intersection(model.factor_returns.index)
        
        if len(common_dates) == 0:
            return {}
        
        factor_contributions = {}
        
        for factor_name, beta in exposure.exposures.items():
            if factor_name in model.factor_returns.columns:
                factor_return = model.factor_returns.loc[common_dates, factor_name].mean()
                contribution = beta * factor_return * len(common_dates)
                factor_contributions[factor_name] = contribution
        
        # Add alpha contribution
        factor_contributions['alpha'] = exposure.alpha * len(common_dates)
        
        return factor_contributions
    
    def generate_factor_report(
        self,
        asset_returns: pd.Series,
        model_names: Optional[List[str]] = None
    ) -> str:
        """Generate comprehensive factor analysis report"""
        
        if model_names is None:
            model_names = list(self.fitted_models.keys())
        
        asset_name = asset_returns.name or "Unknown Asset"
        
        report = f"FACTOR ANALYSIS REPORT - {asset_name.upper()}\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Analysis Period: {asset_returns.index[0].strftime('%Y-%m-%d')} to {asset_returns.index[-1].strftime('%Y-%m-%d')}\n"
        report += f"Total Observations: {len(asset_returns)}\n\n"
        
        # Model comparison
        if len(model_names) > 1:
            comparison = self.compare_models(asset_returns)
            
            report += "MODEL COMPARISON\n"
            report += "-" * 20 + "\n"
            
            for _, row in comparison.iterrows():
                report += f"{row['model']}:\n"
                report += f"  R-squared: {row['r_squared']:.3f}\n"
                report += f"  Alpha: {row['alpha']:.4f}\n"
                report += f"  Residual Vol: {row['residual_volatility']:.4f}\n"
                report += f"  Significant Factors: {row['significant_factors']}/{row['n_factors']}\n\n"
        
        # Detailed analysis for each model
        for model_name in model_names:
            if model_name in self.fitted_models:
                model = self.fitted_models[model_name]
                exposure = model.get_factor_exposures(asset_returns)
                
                report += f"{model_name.upper()} MODEL ANALYSIS\n"
                report += "-" * (len(model_name) + 15) + "\n"
                
                report += f"Alpha: {exposure.alpha:.4f}\n"
                report += f"R-squared: {exposure.r_squared:.3f}\n"
                report += f"Residual Volatility: {exposure.residual_volatility:.4f}\n\n"
                
                report += "Factor Exposures:\n"
                for factor, beta in exposure.exposures.items():
                    p_value = exposure.p_values.get(factor, 1.0)
                    significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
                    report += f"  {factor}: {beta:.3f} {significance}\n"
                
                # Factor attribution
                attribution = self.get_factor_attribution(asset_returns, model_name)
                if attribution:
                    report += "\nFactor Attribution:\n"
                    for factor, contribution in attribution.items():
                        report += f"  {factor}: {contribution:.4f}\n"
                
                report += "\n" + "=" * 60 + "\n\n"
        
        report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "*** p < 0.01, ** p < 0.05, * p < 0.10\n"
        
        return report


# Predefined factor functions for custom models
def momentum_factor(returns: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Calculate momentum factor (past returns)"""
    return returns.rolling(lookback).sum().mean(axis=1)


def size_factor(returns: pd.DataFrame, market_caps: Optional[pd.Series] = None) -> pd.Series:
    """Calculate size factor (small minus big)"""
    if market_caps is None:
        # Use return volatility as proxy for size (smaller = more volatile)
        volatilities = returns.rolling(252).std()
        return -volatilities.mean(axis=1)  # Negative because small cap = high vol
    else:
        # Use actual market caps if provided
        return -market_caps.rolling(30).mean()  # Small minus big


def quality_factor(returns: pd.DataFrame) -> pd.Series:
    """Calculate quality factor (stable earnings proxy)"""
    # Use return stability as proxy for quality
    return -returns.rolling(252).std().mean(axis=1)


def low_volatility_factor(returns: pd.DataFrame) -> pd.Series:
    """Calculate low volatility factor"""
    return -returns.rolling(60).std().mean(axis=1)


def create_default_factor_models() -> Dict[str, BaseFactorModel]:
    """Create a set of default factor models"""
    
    models = {}
    
    # Fama-French models
    models['FF3'] = FamaFrenchModel("3_factor")
    models['FF5'] = FamaFrenchModel("5_factor")
    
    # Statistical models
    models['PCA_5'] = StatisticalFactorModel(n_factors=5, method="pca")
    models['FA_3'] = StatisticalFactorModel(n_factors=3, method="factor_analysis")
    
    # Macroeconomic model
    models['Macro'] = MacroeconomicFactorModel()
    
    # Custom factor model with common factors
    custom_factors = {
        'momentum': momentum_factor,
        'size': size_factor,
        'quality': quality_factor,
        'low_volatility': low_volatility_factor
    }
    models['Custom'] = CustomFactorModel(custom_factors)
    
    return models