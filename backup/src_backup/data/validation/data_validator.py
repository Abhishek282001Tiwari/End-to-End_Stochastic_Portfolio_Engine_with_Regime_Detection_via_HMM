import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataQualityCheck:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.errors: List[str] = []
        self.warnings: List[str] = []


class DataValidator:
    def __init__(self):
        self.checks: List[DataQualityCheck] = []
        
    def check_missing_values(
        self, 
        data: pd.DataFrame, 
        max_missing_pct: float = 0.05
    ) -> DataQualityCheck:
        check = DataQualityCheck(
            "missing_values",
            f"Check missing values are below {max_missing_pct*100}% threshold"
        )
        
        missing_pct = data.isnull().sum() / len(data)
        problematic_cols = missing_pct[missing_pct > max_missing_pct]
        
        if len(problematic_cols) > 0:
            check.errors.extend([
                f"Column '{col}' has {pct:.2%} missing values (>{max_missing_pct:.2%})"
                for col, pct in problematic_cols.items()
            ])
        else:
            check.passed = True
        
        self.checks.append(check)
        return check
    
    def check_data_freshness(
        self, 
        data: pd.DataFrame, 
        max_age_days: int = 1
    ) -> DataQualityCheck:
        check = DataQualityCheck(
            "data_freshness",
            f"Check data is no older than {max_age_days} days"
        )
        
        if not isinstance(data.index, pd.DatetimeIndex):
            check.errors.append("Data index is not a DatetimeIndex")
        else:
            latest_date = data.index.max()
            age_days = (datetime.now() - latest_date).days
            
            if age_days > max_age_days:
                check.errors.append(
                    f"Data is {age_days} days old (>{max_age_days} days)"
                )
            else:
                check.passed = True
        
        self.checks.append(check)
        return check
    
    def check_data_continuity(self, data: pd.DataFrame) -> DataQualityCheck:
        check = DataQualityCheck(
            "data_continuity",
            "Check for gaps in time series data"
        )
        
        if not isinstance(data.index, pd.DatetimeIndex):
            check.errors.append("Data index is not a DatetimeIndex")
        else:
            expected_freq = pd.infer_freq(data.index)
            if expected_freq is None:
                check.warnings.append("Could not infer data frequency")
            
            date_diffs = data.index.to_series().diff()[1:]
            median_diff = date_diffs.median()
            
            gaps = date_diffs[date_diffs > median_diff * 2]
            
            if len(gaps) > 0:
                check.warnings.extend([
                    f"Gap detected at {gap_date}: {gap_size}"
                    for gap_date, gap_size in gaps.items()
                ])
            
            check.passed = True
        
        self.checks.append(check)
        return check
    
    def check_outliers(
        self, 
        data: pd.DataFrame, 
        z_threshold: float = 4.0
    ) -> DataQualityCheck:
        check = DataQualityCheck(
            "outliers",
            f"Check for extreme outliers (|z-score| > {z_threshold})"
        )
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            extreme_outliers = z_scores > z_threshold
            
            if extreme_outliers.sum() > 0:
                outlier_count = extreme_outliers.sum()
                outlier_pct = outlier_count / len(data)
                
                if outlier_pct > 0.01:  # More than 1% outliers
                    check.errors.append(
                        f"Column '{col}' has {outlier_count} extreme outliers "
                        f"({outlier_pct:.2%} of data)"
                    )
                else:
                    check.warnings.append(
                        f"Column '{col}' has {outlier_count} extreme outliers "
                        f"({outlier_pct:.2%} of data)"
                    )
        
        if not check.errors:
            check.passed = True
        
        self.checks.append(check)
        return check
    
    def check_correlation_stability(
        self, 
        data: pd.DataFrame, 
        window: int = 252,
        correlation_change_threshold: float = 0.3
    ) -> DataQualityCheck:
        check = DataQualityCheck(
            "correlation_stability",
            "Check for sudden changes in correlation structure"
        )
        
        if len(data.columns) < 2:
            check.warnings.append("Need at least 2 columns to check correlations")
            check.passed = True
            self.checks.append(check)
            return check
        
        rolling_corr = data.rolling(window=window).corr()
        
        correlation_changes = []
        for i in range(window, len(data) - window):
            corr_before = data.iloc[i-window:i].corr()
            corr_after = data.iloc[i:i+window].corr()
            
            corr_diff = np.abs(corr_after - corr_before).max().max()
            
            if corr_diff > correlation_change_threshold:
                correlation_changes.append((data.index[i], corr_diff))
        
        if correlation_changes:
            check.warnings.extend([
                f"Large correlation change at {date}: {change:.3f}"
                for date, change in correlation_changes[:5]
            ])
        
        check.passed = True
        self.checks.append(check)
        return check
    
    def check_return_distributions(self, returns: pd.DataFrame) -> DataQualityCheck:
        check = DataQualityCheck(
            "return_distributions",
            "Check return distributions for anomalies"
        )
        
        for col in returns.columns:
            col_returns = returns[col].dropna()
            
            if len(col_returns) == 0:
                check.errors.append(f"Column '{col}' has no valid returns")
                continue
            
            skewness = col_returns.skew()
            kurtosis = col_returns.kurtosis()
            
            if abs(skewness) > 2:
                check.warnings.append(
                    f"Column '{col}' has high skewness: {skewness:.2f}"
                )
            
            if kurtosis > 10:
                check.warnings.append(
                    f"Column '{col}' has high kurtosis: {kurtosis:.2f}"
                )
            
            extreme_returns = col_returns[abs(col_returns) > 0.2]  # >20% daily return
            if len(extreme_returns) > 0:
                check.warnings.append(
                    f"Column '{col}' has {len(extreme_returns)} extreme returns (>20%)"
                )
        
        if not check.errors:
            check.passed = True
        
        self.checks.append(check)
        return check
    
    def validate_dataset(
        self, 
        data: pd.DataFrame,
        data_type: str = "general"
    ) -> Dict[str, Any]:
        logger.info(f"Starting data validation for {data_type} dataset")
        
        self.checks = []
        
        self.check_missing_values(data)
        self.check_data_freshness(data)
        self.check_data_continuity(data)
        self.check_outliers(data)
        
        if data_type == "returns":
            self.check_return_distributions(data)
        
        if len(data.columns) > 1:
            self.check_correlation_stability(data)
        
        total_checks = len(self.checks)
        passed_checks = sum(1 for check in self.checks if check.passed)
        
        validation_summary = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "checks": self.checks,
            "errors": [error for check in self.checks for error in check.errors],
            "warnings": [warning for check in self.checks for warning in check.warnings]
        }
        
        logger.info(
            f"Validation complete: {passed_checks}/{total_checks} checks passed "
            f"({validation_summary['success_rate']:.1%})"
        )
        
        if validation_summary["errors"]:
            logger.error(f"Found {len(validation_summary['errors'])} errors")
        
        if validation_summary["warnings"]:
            logger.warning(f"Found {len(validation_summary['warnings'])} warnings")
        
        return validation_summary
    
    def get_validation_report(self) -> str:
        report = "Data Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        for check in self.checks:
            status = "✓ PASSED" if check.passed else "✗ FAILED"
            report += f"{status}: {check.name}\n"
            report += f"Description: {check.description}\n"
            
            if check.errors:
                report += "Errors:\n"
                for error in check.errors:
                    report += f"  - {error}\n"
            
            if check.warnings:
                report += "Warnings:\n"
                for warning in check.warnings:
                    report += f"  - {warning}\n"
            
            report += "\n"
        
        return report