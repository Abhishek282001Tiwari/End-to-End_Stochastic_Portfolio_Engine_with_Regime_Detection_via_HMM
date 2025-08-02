import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
from src.optimization.objectives.risk_measures import RiskMeasures, PortfolioRiskCalculator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class AlertLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAlert:
    timestamp: datetime
    alert_type: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    suggested_action: str


@dataclass
class RiskLimits:
    max_portfolio_volatility: float = 0.20
    max_individual_weight: float = 0.15
    max_sector_weight: float = 0.25
    max_drawdown: float = 0.15
    min_liquidity_ratio: float = 0.95
    max_leverage: float = 1.0
    max_var_95: float = 0.05
    max_cvar_95: float = 0.07
    min_diversification_ratio: float = 0.3
    max_concentration_index: float = 0.4


class RealTimeRiskMonitor:
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.risk_calculator = PortfolioRiskCalculator()
        self.alert_history: List[RiskAlert] = []
        self.portfolio_history = {}
        
    def monitor_portfolio_risk(
        self,
        portfolio_weights: pd.Series,
        returns_data: pd.DataFrame,
        prices_data: Optional[pd.DataFrame] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        liquidity_scores: Optional[pd.Series] = None
    ) -> List[RiskAlert]:
        logger.info("Monitoring portfolio risk in real-time")
        
        current_alerts = []
        timestamp = datetime.now()
        
        try:
            current_alerts.extend(
                self._check_position_limits(portfolio_weights, timestamp)
            )
            
            if returns_data is not None and len(returns_data) > 30:
                current_alerts.extend(
                    self._check_portfolio_risk_metrics(
                        portfolio_weights, returns_data, timestamp
                    )
                )
            
            if sector_mapping is not None:
                current_alerts.extend(
                    self._check_sector_concentration(
                        portfolio_weights, sector_mapping, timestamp
                    )
                )
            
            if liquidity_scores is not None:
                current_alerts.extend(
                    self._check_liquidity_risk(
                        portfolio_weights, liquidity_scores, timestamp
                    )
                )
            
            if prices_data is not None:
                current_alerts.extend(
                    self._check_drawdown_risk(
                        portfolio_weights, prices_data, timestamp
                    )
                )
            
            current_alerts.extend(
                self._check_concentration_risk(portfolio_weights, timestamp)
            )
            
        except Exception as e:
            logger.error(f"Error during risk monitoring: {e}")
            current_alerts.append(
                RiskAlert(
                    timestamp=timestamp,
                    alert_type="system_error",
                    level=AlertLevel.HIGH,
                    message=f"Risk monitoring system error: {e}",
                    metric_name="system_health",
                    current_value=0,
                    threshold_value=1,
                    suggested_action="Check system logs and restart monitoring"
                )
            )
        
        self.alert_history.extend(current_alerts)
        
        if current_alerts:
            self._log_alerts(current_alerts)
        
        return current_alerts
    
    def _check_position_limits(
        self, 
        weights: pd.Series, 
        timestamp: datetime
    ) -> List[RiskAlert]:
        alerts = []
        
        max_weight = weights.max()
        if max_weight > self.risk_limits.max_individual_weight:
            alerts.append(
                RiskAlert(
                    timestamp=timestamp,
                    alert_type="position_limit",
                    level=AlertLevel.HIGH if max_weight > self.risk_limits.max_individual_weight * 1.2 else AlertLevel.MEDIUM,
                    message=f"Individual position weight ({max_weight:.2%}) exceeds limit ({self.risk_limits.max_individual_weight:.2%})",
                    metric_name="max_individual_weight",
                    current_value=max_weight,
                    threshold_value=self.risk_limits.max_individual_weight,
                    suggested_action="Reduce position size or rebalance portfolio"
                )
            )
        
        total_leverage = weights.abs().sum()
        if total_leverage > self.risk_limits.max_leverage:
            alerts.append(
                RiskAlert(
                    timestamp=timestamp,
                    alert_type="leverage_limit",
                    level=AlertLevel.CRITICAL if total_leverage > self.risk_limits.max_leverage * 1.1 else AlertLevel.HIGH,
                    message=f"Portfolio leverage ({total_leverage:.2f}) exceeds limit ({self.risk_limits.max_leverage:.2f})",
                    metric_name="total_leverage",
                    current_value=total_leverage,
                    threshold_value=self.risk_limits.max_leverage,
                    suggested_action="Reduce leverage by closing positions or adding cash"
                )
            )
        
        return alerts
    
    def _check_portfolio_risk_metrics(
        self, 
        weights: pd.Series, 
        returns: pd.DataFrame, 
        timestamp: datetime
    ) -> List[RiskAlert]:
        alerts = []
        
        try:
            portfolio_returns = (returns @ weights).dropna()
            
            if len(portfolio_returns) > 30:
                portfolio_vol = portfolio_returns.std() * np.sqrt(252)
                
                if portfolio_vol > self.risk_limits.max_portfolio_volatility:
                    alerts.append(
                        RiskAlert(
                            timestamp=timestamp,
                            alert_type="volatility_limit",
                            level=AlertLevel.HIGH if portfolio_vol > self.risk_limits.max_portfolio_volatility * 1.2 else AlertLevel.MEDIUM,
                            message=f"Portfolio volatility ({portfolio_vol:.2%}) exceeds limit ({self.risk_limits.max_portfolio_volatility:.2%})",
                            metric_name="portfolio_volatility",
                            current_value=portfolio_vol,
                            threshold_value=self.risk_limits.max_portfolio_volatility,
                            suggested_action="Reduce position sizes or add defensive assets"
                        )
                    )
                
                var_95 = RiskMeasures.value_at_risk(portfolio_returns, 0.05)
                if abs(var_95) > self.risk_limits.max_var_95:
                    alerts.append(
                        RiskAlert(
                            timestamp=timestamp,
                            alert_type="var_limit",
                            level=AlertLevel.HIGH,
                            message=f"95% VaR ({abs(var_95):.2%}) exceeds limit ({self.risk_limits.max_var_95:.2%})",
                            metric_name="var_95",
                            current_value=abs(var_95),
                            threshold_value=self.risk_limits.max_var_95,
                            suggested_action="Reduce risk exposure or hedge portfolio"
                        )
                    )
                
                cvar_95 = RiskMeasures.conditional_value_at_risk(portfolio_returns, 0.05)
                if abs(cvar_95) > self.risk_limits.max_cvar_95:
                    alerts.append(
                        RiskAlert(
                            timestamp=timestamp,
                            alert_type="cvar_limit",
                            level=AlertLevel.HIGH,
                            message=f"95% CVaR ({abs(cvar_95):.2%}) exceeds limit ({self.risk_limits.max_cvar_95:.2%})",
                            metric_name="cvar_95",
                            current_value=abs(cvar_95),
                            threshold_value=self.risk_limits.max_cvar_95,
                            suggested_action="Implement tail risk hedging strategies"
                        )
                    )
        
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
        
        return alerts
    
    def _check_sector_concentration(
        self, 
        weights: pd.Series, 
        sector_mapping: Dict[str, str], 
        timestamp: datetime
    ) -> List[RiskAlert]:
        alerts = []
        
        try:
            sector_weights = {}
            for asset, weight in weights.items():
                sector = sector_mapping.get(asset, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            max_sector_weight = max(sector_weights.values())
            max_sector = max(sector_weights, key=sector_weights.get)
            
            if max_sector_weight > self.risk_limits.max_sector_weight:
                alerts.append(
                    RiskAlert(
                        timestamp=timestamp,
                        alert_type="sector_concentration",
                        level=AlertLevel.MEDIUM if max_sector_weight < self.risk_limits.max_sector_weight * 1.2 else AlertLevel.HIGH,
                        message=f"Sector concentration in {max_sector} ({max_sector_weight:.2%}) exceeds limit ({self.risk_limits.max_sector_weight:.2%})",
                        metric_name="max_sector_weight",
                        current_value=max_sector_weight,
                        threshold_value=self.risk_limits.max_sector_weight,
                        suggested_action=f"Reduce exposure to {max_sector} or diversify across sectors"
                    )
                )
        
        except Exception as e:
            logger.error(f"Error checking sector concentration: {e}")
        
        return alerts
    
    def _check_liquidity_risk(
        self, 
        weights: pd.Series, 
        liquidity_scores: pd.Series, 
        timestamp: datetime
    ) -> List[RiskAlert]:
        alerts = []
        
        try:
            common_assets = weights.index.intersection(liquidity_scores.index)
            
            if len(common_assets) > 0:
                weighted_liquidity = (weights[common_assets] * liquidity_scores[common_assets]).sum()
                
                if weighted_liquidity < self.risk_limits.min_liquidity_ratio:
                    alerts.append(
                        RiskAlert(
                            timestamp=timestamp,
                            alert_type="liquidity_risk",
                            level=AlertLevel.MEDIUM if weighted_liquidity > self.risk_limits.min_liquidity_ratio * 0.9 else AlertLevel.HIGH,
                            message=f"Portfolio liquidity ratio ({weighted_liquidity:.2%}) below minimum ({self.risk_limits.min_liquidity_ratio:.2%})",
                            metric_name="liquidity_ratio",
                            current_value=weighted_liquidity,
                            threshold_value=self.risk_limits.min_liquidity_ratio,
                            suggested_action="Increase allocation to more liquid assets"
                        )
                    )
        
        except Exception as e:
            logger.error(f"Error checking liquidity risk: {e}")
        
        return alerts
    
    def _check_drawdown_risk(
        self, 
        weights: pd.Series, 
        prices: pd.DataFrame, 
        timestamp: datetime
    ) -> List[RiskAlert]:
        alerts = []
        
        try:
            common_assets = weights.index.intersection(prices.columns)
            
            if len(common_assets) > 0 and len(prices) > 30:
                portfolio_prices = (prices[common_assets] @ weights[common_assets])
                
                cumulative_returns = portfolio_prices / portfolio_prices.iloc[0] - 1
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = cumulative_returns - rolling_max
                
                current_drawdown = drawdowns.iloc[-1]
                
                if abs(current_drawdown) > self.risk_limits.max_drawdown:
                    alerts.append(
                        RiskAlert(
                            timestamp=timestamp,
                            alert_type="drawdown_limit",
                            level=AlertLevel.CRITICAL if abs(current_drawdown) > self.risk_limits.max_drawdown * 1.2 else AlertLevel.HIGH,
                            message=f"Current drawdown ({abs(current_drawdown):.2%}) exceeds limit ({self.risk_limits.max_drawdown:.2%})",
                            metric_name="current_drawdown",
                            current_value=abs(current_drawdown),
                            threshold_value=self.risk_limits.max_drawdown,
                            suggested_action="Implement stop-loss strategy or reduce risk exposure"
                        )
                    )
        
        except Exception as e:
            logger.error(f"Error checking drawdown risk: {e}")
        
        return alerts
    
    def _check_concentration_risk(
        self, 
        weights: pd.Series, 
        timestamp: datetime
    ) -> List[RiskAlert]:
        alerts = []
        
        try:
            herfindahl_index = np.sum(weights ** 2)
            
            if herfindahl_index > self.risk_limits.max_concentration_index:
                alerts.append(
                    RiskAlert(
                        timestamp=timestamp,
                        alert_type="concentration_risk",
                        level=AlertLevel.MEDIUM,
                        message=f"Portfolio concentration index ({herfindahl_index:.3f}) exceeds limit ({self.risk_limits.max_concentration_index:.3f})",
                        metric_name="concentration_index",
                        current_value=herfindahl_index,
                        threshold_value=self.risk_limits.max_concentration_index,
                        suggested_action="Increase diversification across holdings"
                    )
                )
            
            num_positions = (weights.abs() > 0.01).sum()
            diversification_ratio = 1 / np.sqrt(num_positions) if num_positions > 0 else 1
            
            if diversification_ratio < self.risk_limits.min_diversification_ratio:
                alerts.append(
                    RiskAlert(
                        timestamp=timestamp,
                        alert_type="diversification_risk",
                        level=AlertLevel.LOW,
                        message=f"Diversification ratio ({diversification_ratio:.3f}) below minimum ({self.risk_limits.min_diversification_ratio:.3f})",
                        metric_name="diversification_ratio",
                        current_value=diversification_ratio,
                        threshold_value=self.risk_limits.min_diversification_ratio,
                        suggested_action="Add more positions or rebalance existing ones"
                    )
                )
        
        except Exception as e:
            logger.error(f"Error checking concentration risk: {e}")
        
        return alerts
    
    def _log_alerts(self, alerts: List[RiskAlert]):
        for alert in alerts:
            log_message = f"RISK ALERT [{alert.level.value}]: {alert.message}"
            
            if alert.level == AlertLevel.CRITICAL:
                logger.critical(log_message)
            elif alert.level == AlertLevel.HIGH:
                logger.error(log_message)
            elif alert.level == AlertLevel.MEDIUM:
                logger.warning(log_message)
            else:
                logger.info(log_message)
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        alert_summary = {}
        for level in AlertLevel:
            alert_summary[level.value] = len([
                alert for alert in recent_alerts if alert.level == level
            ])
        
        alert_types = {}
        for alert in recent_alerts:
            alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1
        
        return {
            'total_alerts_24h': len(recent_alerts),
            'alert_summary': alert_summary,
            'alert_types': alert_types,
            'recent_alerts': recent_alerts[-10:],
            'system_status': 'HEALTHY' if alert_summary.get('CRITICAL', 0) == 0 else 'AT_RISK'
        }
    
    def update_risk_limits(self, new_limits: RiskLimits):
        self.risk_limits = new_limits
        logger.info("Risk limits updated")
    
    def generate_risk_report(self) -> str:
        dashboard = self.get_risk_dashboard()
        
        report = "REAL-TIME RISK MONITORING REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"System Status: {dashboard['system_status']}\n"
        report += f"Total Alerts (24h): {dashboard['total_alerts_24h']}\n\n"
        
        report += "ALERT SUMMARY (24h):\n"
        for level, count in dashboard['alert_summary'].items():
            report += f"  {level}: {count}\n"
        
        report += "\nALERT TYPES (24h):\n"
        for alert_type, count in dashboard['alert_types'].items():
            report += f"  {alert_type}: {count}\n"
        
        if dashboard['recent_alerts']:
            report += "\nRECENT ALERTS:\n"
            for alert in dashboard['recent_alerts']:
                report += f"  [{alert.level.value}] {alert.timestamp.strftime('%H:%M:%S')} - {alert.message}\n"
        
        return report