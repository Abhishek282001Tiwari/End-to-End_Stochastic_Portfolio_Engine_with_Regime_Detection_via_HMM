import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, scaling_method: str = "standard"):
        self.scaling_method = scaling_method
        self.scalers: Dict[str, object] = {}
        self.imputers: Dict[str, object] = {}
        
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        method: str = "knn"
    ) -> pd.DataFrame:
        logger.info(f"Handling missing values using {method} method")
        
        if method == "forward_fill":
            return data.fillna(method='ffill').fillna(method='bfill')
        
        elif method == "knn":
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(data)
            return pd.DataFrame(
                imputed_data, 
                index=data.index, 
                columns=data.columns
            )
        
        elif method == "interpolate":
            return data.interpolate(method='linear').fillna(method='bfill')
        
        else:
            raise ValueError(f"Unknown imputation method: {method}")
    
    def detect_outliers(
        self, 
        data: pd.DataFrame, 
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        logger.info(f"Detecting outliers using {method} method")
        
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = (
                (data < (Q1 - 1.5 * IQR)) | 
                (data > (Q3 + 1.5 * IQR))
            )
            
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_mask = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outlier_mask
    
    def treat_outliers(
        self, 
        data: pd.DataFrame, 
        outlier_mask: pd.DataFrame,
        method: str = "winsorize"
    ) -> pd.DataFrame:
        logger.info(f"Treating outliers using {method} method")
        
        if method == "winsorize":
            treated_data = data.copy()
            for col in data.columns:
                col_data = data[col]
                q01 = col_data.quantile(0.01)
                q99 = col_data.quantile(0.99)
                treated_data[col] = col_data.clip(lower=q01, upper=q99)
            
            return treated_data
        
        elif method == "remove":
            return data[~outlier_mask.any(axis=1)]
        
        elif method == "interpolate":
            treated_data = data.copy()
            treated_data[outlier_mask] = np.nan
            return self.handle_missing_values(treated_data, method="interpolate")
        
        else:
            raise ValueError(f"Unknown outlier treatment method: {method}")
    
    def scale_features(
        self, 
        data: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        logger.info(f"Scaling features using {self.scaling_method} scaler")
        
        scaled_data = data.copy()
        
        for col in data.columns:
            if col not in self.scalers:
                if self.scaling_method == "standard":
                    self.scalers[col] = StandardScaler()
                elif self.scaling_method == "robust":
                    self.scalers[col] = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            if fit:
                scaled_data[col] = self.scalers[col].fit_transform(
                    data[col].values.reshape(-1, 1)
                ).flatten()
            else:
                scaled_data[col] = self.scalers[col].transform(
                    data[col].values.reshape(-1, 1)
                ).flatten()
        
        return scaled_data
    
    def create_lagged_features(
        self, 
        data: pd.DataFrame, 
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        logger.info(f"Creating lagged features with lags: {lags}")
        
        lagged_data = data.copy()
        
        for col in data.columns:
            for lag in lags:
                lagged_data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        
        return lagged_data.dropna()
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating technical indicators")
        
        indicators = data.copy()
        
        for col in data.columns:
            if 'return' in col.lower():
                series = data[col]
                
                indicators[f"{col}_sma_5"] = series.rolling(5).mean()
                indicators[f"{col}_sma_20"] = series.rolling(20).mean()
                indicators[f"{col}_ema_12"] = series.ewm(span=12).mean()
                indicators[f"{col}_ema_26"] = series.ewm(span=26).mean()
                
                indicators[f"{col}_std_20"] = series.rolling(20).std()
                indicators[f"{col}_skew_20"] = series.rolling(20).skew()
                indicators[f"{col}_kurt_20"] = series.rolling(20).kurt()
                
                indicators[f"{col}_rsi"] = self._calculate_rsi(series)
                
                bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(series)
                indicators[f"{col}_bb_upper"] = bollinger_upper
                indicators[f"{col}_bb_lower"] = bollinger_lower
        
        return indicators.dropna()
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(
        self, 
        series: pd.Series, 
        window: int = 20, 
        num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series]:
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def process_data(
        self, 
        data: pd.DataFrame,
        handle_missing: bool = True,
        detect_outliers_flag: bool = True,
        scale_data: bool = True,
        create_lags: bool = True,
        create_technicals: bool = True
    ) -> pd.DataFrame:
        logger.info("Starting data processing pipeline")
        
        processed_data = data.copy()
        
        if handle_missing:
            processed_data = self.handle_missing_values(processed_data)
        
        if detect_outliers_flag:
            outlier_mask = self.detect_outliers(processed_data)
            processed_data = self.treat_outliers(processed_data, outlier_mask)
        
        if create_technicals:
            processed_data = self.create_technical_indicators(processed_data)
        
        if create_lags:
            processed_data = self.create_lagged_features(processed_data)
        
        if scale_data:
            processed_data = self.scale_features(processed_data)
        
        logger.info(f"Data processing complete. Final shape: {processed_data.shape}")
        return processed_data