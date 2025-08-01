import time
import psutil
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from flask import Flask, Response
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PortfolioMetrics:
    """Comprehensive metrics collection for portfolio engine"""
    
    def __init__(self):
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Application metrics
        self.request_count = Counter(
            'portfolio_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'portfolio_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Portfolio performance metrics
        self.portfolio_value = Gauge(
            'portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.portfolio_return = Gauge(
            'portfolio_return_daily',
            'Daily portfolio return',
            registry=self.registry
        )
        
        self.portfolio_volatility = Gauge(
            'portfolio_volatility_annualized',
            'Annualized portfolio volatility',
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'portfolio_sharpe_ratio',
            'Portfolio Sharpe ratio',
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'portfolio_max_drawdown',
            'Maximum drawdown',
            registry=self.registry
        )
        
        # Risk metrics
        self.var_95 = Gauge(
            'portfolio_var_95',
            '95% Value at Risk',
            registry=self.registry
        )
        
        self.var_99 = Gauge(
            'portfolio_var_99',
            '99% Value at Risk',
            registry=self.registry
        )
        
        self.expected_shortfall = Gauge(
            'portfolio_expected_shortfall',
            'Expected Shortfall (CVaR)',
            registry=self.registry
        )
        
        # Regime detection metrics
        self.current_regime = Gauge(
            'regime_current',
            'Current market regime (0, 1, 2, etc.)',
            registry=self.registry
        )
        
        self.regime_confidence = Gauge(
            'regime_confidence',
            'Confidence in current regime detection',
            registry=self.registry
        )
        
        self.regime_switches = Counter(
            'regime_switches_total',
            'Total number of regime switches',
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model prediction accuracy',
            ['model_type'],
            registry=self.registry
        )
        
        self.model_training_time = Histogram(
            'model_training_duration_seconds',
            'Model training duration',
            ['model_type'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Model prediction latency',
            ['model_type'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            registry=self.registry
        )
        
        # Data quality metrics
        self.data_freshness = Gauge(
            'data_freshness_minutes',
            'Data freshness in minutes',
            ['data_source'],
            registry=self.registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['data_source'],
            registry=self.registry
        )
        
        self.missing_data_points = Gauge(
            'missing_data_points',
            'Number of missing data points',
            ['data_source'],
            registry=self.registry
        )
        
        # Alert metrics
        self.alerts_fired = Counter(
            'alerts_fired_total',
            'Total alerts fired',
            ['alert_type', 'severity'],
            registry=self.registry
        )
        
        self.alerts_active = Gauge(
            'alerts_active',
            'Currently active alerts',
            ['alert_type'],
            registry=self.registry
        )
        
        # Background monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring_thread is not None:
            logger.warning("Monitoring thread already running")
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started system monitoring thread")
    
    def stop_monitoring(self):
        """Stop background monitoring thread"""
        if self._monitoring_thread is None:
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5)
        self._monitoring_thread = None
        logger.info("Stopped system monitoring thread")
    
    def _monitor_system(self):
        """Background system monitoring"""
        while not self._stop_monitoring.wait(30):  # Update every 30 seconds
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.set(disk.used)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
    
    def update_portfolio_metrics(self, portfolio_data: Dict[str, Any]):
        """Update portfolio-related metrics"""
        try:
            # Portfolio value
            if 'portfolio_value' in portfolio_data:
                self.portfolio_value.set(portfolio_data['portfolio_value'])
            
            # Returns and risk metrics
            portfolio_returns = portfolio_data.get('portfolio_returns')
            if portfolio_returns is not None and len(portfolio_returns) > 0:
                # Daily return
                if len(portfolio_returns) > 0:
                    daily_return = portfolio_returns.iloc[-1]
                    self.portfolio_return.set(daily_return)
                
                # Volatility (annualized)
                if len(portfolio_returns) > 20:
                    volatility = portfolio_returns.std() * np.sqrt(252)
                    self.portfolio_volatility.set(volatility)
                
                # Sharpe ratio
                if len(portfolio_returns) > 20:
                    excess_returns = portfolio_returns - 0.02/252  # Assume 2% risk-free rate
                    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                    self.sharpe_ratio.set(sharpe)
                
                # Maximum drawdown
                cumulative = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_dd = drawdown.min()
                self.max_drawdown.set(abs(max_dd))
                
                # VaR calculations
                if len(portfolio_returns) > 20:
                    var_95 = np.percentile(portfolio_returns, 5)
                    var_99 = np.percentile(portfolio_returns, 1)
                    self.var_95.set(abs(var_95))
                    self.var_99.set(abs(var_99))
                    
                    # Expected Shortfall (CVaR)
                    es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
                    self.expected_shortfall.set(abs(es_95))
            
            # Risk metrics from risk monitor
            risk_metrics = portfolio_data.get('risk_metrics', {})
            for metric_name, value in risk_metrics.items():
                if metric_name == 'var_95':
                    self.var_95.set(abs(value))
                elif metric_name == 'var_99':
                    self.var_99.set(abs(value))
                elif metric_name == 'expected_shortfall':
                    self.expected_shortfall.set(abs(value))
                    
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def update_regime_metrics(self, regime_data: Dict[str, Any]):
        """Update regime detection metrics"""
        try:
            current_regime = regime_data.get('current_regime')
            if current_regime is not None:
                self.current_regime.set(current_regime)
            
            regime_confidence = regime_data.get('regime_confidence')
            if regime_confidence is not None:
                self.regime_confidence.set(regime_confidence)
                
        except Exception as e:
            logger.error(f"Error updating regime metrics: {e}")
    
    def record_regime_switch(self):
        """Record a regime switch event"""
        self.regime_switches.inc()
    
    def update_model_metrics(self, model_type: str, accuracy: float, training_time: float = None):
        """Update model performance metrics"""
        try:
            self.model_accuracy.labels(model_type=model_type).set(accuracy)
            
            if training_time is not None:
                self.model_training_time.labels(model_type=model_type).observe(training_time)
                
        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")
    
    def record_prediction_latency(self, model_type: str, latency: float):
        """Record model prediction latency"""
        self.prediction_latency.labels(model_type=model_type).observe(latency)
    
    def update_data_quality_metrics(self, data_source: str, freshness_minutes: float, 
                                   quality_score: float, missing_points: int):
        """Update data quality metrics"""
        try:
            self.data_freshness.labels(data_source=data_source).set(freshness_minutes)
            self.data_quality_score.labels(data_source=data_source).set(quality_score)
            self.missing_data_points.labels(data_source=data_source).set(missing_points)
            
        except Exception as e:
            logger.error(f"Error updating data quality metrics: {e}")
    
    def fire_alert(self, alert_type: str, severity: str):
        """Record an alert being fired"""
        self.alerts_fired.labels(alert_type=alert_type, severity=severity).inc()
        self.alerts_active.labels(alert_type=alert_type).inc()
    
    def resolve_alert(self, alert_type: str):
        """Record an alert being resolved"""
        current_value = self.alerts_active.labels(alert_type=alert_type)._value._value
        if current_value > 0:
            self.alerts_active.labels(alert_type=alert_type).dec()
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record HTTP request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')


class MetricsServer:
    """HTTP server for exposing metrics to Prometheus"""
    
    def __init__(self, metrics: PortfolioMetrics, port: int = 8080):
        self.metrics = metrics
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/metrics')
        def metrics():
            return Response(self.metrics.get_metrics(), mimetype='text/plain')
        
        @self.app.route('/health')
        def health():
            return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
    
    def start(self):
        """Start the metrics server"""
        logger.info(f"Starting metrics server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


class PerformanceProfiler:
    """Performance profiling and timing utilities"""
    
    def __init__(self, metrics: PortfolioMetrics):
        self.metrics = metrics
    
    def time_function(self, func_name: str):
        """Decorator for timing function execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    status = 'success'
                    return result
                except Exception as e:
                    status = 'error'
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics.request_duration.labels(
                        method='internal', 
                        endpoint=func_name
                    ).observe(duration)
                    
                    self.metrics.request_count.labels(
                        method='internal',
                        endpoint=func_name,
                        status=status
                    ).inc()
            return wrapper
        return decorator
    
    def time_model_prediction(self, model_type: str):
        """Decorator for timing model predictions"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.metrics.record_prediction_latency(model_type, duration)
                return result
            return wrapper
        return decorator


# Global metrics instance
portfolio_metrics = PortfolioMetrics()

def get_metrics_instance() -> PortfolioMetrics:
    """Get the global metrics instance"""
    return portfolio_metrics