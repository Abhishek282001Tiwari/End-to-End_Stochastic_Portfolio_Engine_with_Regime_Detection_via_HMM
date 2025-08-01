import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template, Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import time
import threading

from src.utils.performance_analytics import PerformanceAnalytics
from src.backtesting.attribution.performance_attribution import ComprehensiveAttribution
from src.models.hmm.regime_analyzer import RegimeAnalyzer
from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor
from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.performance_analytics = PerformanceAnalytics()
        self.config = get_config()
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default HTML templates for reports"""
        template_dir = Path(__file__).parent / "templates"
        
        # Daily report template
        daily_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Daily Portfolio Report - {{ report_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .metric-card { 
            display: inline-block; 
            margin: 10px; 
            padding: 15px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            width: 200px; 
            text-align: center; 
        }
        .positive { color: green; }
        .negative { color: red; }
        .section { margin: 30px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Daily Portfolio Report</h1>
        <h3>{{ report_date }}</h3>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metric-card">
            <h4>Portfolio Value</h4>
            <h2>${{ "%.2f"|format(portfolio_value) }}</h2>
        </div>
        <div class="metric-card">
            <h4>Daily Return</h4>
            <h2 class="{{ 'positive' if daily_return >= 0 else 'negative' }}">
                {{ "{:+.2%}".format(daily_return) }}
            </h2>
        </div>
        <div class="metric-card">
            <h4>YTD Return</h4>
            <h2 class="{{ 'positive' if ytd_return >= 0 else 'negative' }}">
                {{ "{:+.2%}".format(ytd_return) }}
            </h2>
        </div>
        <div class="metric-card">
            <h4>Current Regime</h4>
            <h2>{{ current_regime }}</h2>
            <small>{{ "{:.1%}".format(regime_confidence) }} confidence</small>
        </div>
    </div>
    
    <div class="section">
        <h2>Risk Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Limit</th><th>Status</th></tr>
            {% for metric in risk_metrics %}
            <tr>
                <td>{{ metric.name }}</td>
                <td>{{ metric.value }}</td>
                <td>{{ metric.limit }}</td>
                <td class="{{ 'negative' if metric.breach else 'positive' }}">
                    {{ 'BREACH' if metric.breach else 'OK' }}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Top Holdings</h2>
        <table>
            <tr><th>Asset</th><th>Weight</th><th>Value</th><th>Daily Return</th></tr>
            {% for holding in top_holdings %}
            <tr>
                <td>{{ holding.asset }}</td>
                <td>{{ "{:.1%}".format(holding.weight) }}</td>
                <td>${{ "%.0f"|format(holding.value) }}</td>
                <td class="{{ 'positive' if holding.daily_return >= 0 else 'negative' }}">
                    {{ "{:+.2%}".format(holding.daily_return) }}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    {% if risk_alerts %}
    <div class="section">
        <h2>Risk Alerts</h2>
        {% for alert in risk_alerts %}
        <div style="padding: 10px; margin: 5px 0; background-color: 
            {% if alert.level == 'CRITICAL' %}#ffebee{% elif alert.level == 'HIGH' %}#fff3e0{% else %}#e8f5e8{% endif %}; 
            border-left: 4px solid 
            {% if alert.level == 'CRITICAL' %}#f44336{% elif alert.level == 'HIGH' %}#ff9800{% else %}#4caf50{% endif %};">
            <strong>{{ alert.level }}</strong>: {{ alert.message }}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="section">
        <h2>Charts</h2>
        {% for chart in charts %}
        <div style="margin: 20px 0;">
            <h3>{{ chart.title }}</h3>
            {{ chart.html|safe }}
        </div>
        {% endfor %}
    </div>
    
    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        Generated on {{ generation_time }} | Stochastic Portfolio Engine v1.0
    </footer>
</body>
</html>
        """
        
        with open(template_dir / "daily_report.html", "w") as f:
            f.write(daily_template)
        
        # Weekly report template (simplified for brevity)
        weekly_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Portfolio Report - {{ report_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .section { margin: 30px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Weekly Portfolio Report</h1>
        <h3>Week ending {{ report_date }}</h3>
    </div>
    
    <div class="section">
        <h2>Weekly Performance Summary</h2>
        <p>Weekly Return: <span class="{{ 'positive' if weekly_return >= 0 else 'negative' }}">{{ "{:+.2%}".format(weekly_return) }}</span></p>
        <p>Portfolio Value: ${{ "%.2f"|format(portfolio_value) }}</p>
        <p>Sharpe Ratio (Weekly): {{ "{:.2f}".format(weekly_sharpe) }}</p>
    </div>
    
    <div class="section">
        <h2>Regime Analysis</h2>
        {{ regime_analysis|safe }}
    </div>
    
    <div class="section">
        <h2>Performance Attribution</h2>
        {{ attribution_analysis|safe }}
    </div>
    
    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        Generated on {{ generation_time }} | Stochastic Portfolio Engine v1.0
    </footer>
</body>
</html>
        """
        
        with open(template_dir / "weekly_report.html", "w") as f:
            f.write(weekly_template)
    
    def generate_daily_report(
        self,
        portfolio_data: Dict[str, Any],
        report_date: Optional[datetime] = None
    ) -> str:
        """Generate daily portfolio report"""
        if report_date is None:
            report_date = datetime.now()
        
        logger.info(f"Generating daily report for {report_date.strftime('%Y-%m-%d')}")
        
        # Process portfolio data
        portfolio_returns = portfolio_data.get('portfolio_returns', pd.Series())
        portfolio_weights = portfolio_data.get('portfolio_weights', pd.Series())
        benchmark_returns = portfolio_data.get('benchmark_returns', pd.Series())
        risk_metrics = portfolio_data.get('risk_metrics', {})
        regime_data = portfolio_data.get('regime_data', {})
        risk_alerts = portfolio_data.get('risk_alerts', [])
        
        # Calculate daily metrics
        if not portfolio_returns.empty and len(portfolio_returns) > 0:
            daily_return = portfolio_returns.iloc[-1]
            ytd_return = (1 + portfolio_returns[portfolio_returns.index >= f"{report_date.year}-01-01"]).prod() - 1
        else:
            daily_return = 0
            ytd_return = 0
        
        portfolio_value = portfolio_data.get('portfolio_value', 1000000)
        
        # Process top holdings
        top_holdings = []
        if not portfolio_weights.empty:
            top_weights = portfolio_weights.nlargest(10)
            for asset, weight in top_weights.items():
                holding_return = 0
                if not portfolio_returns.empty and asset in portfolio_returns:
                    holding_return = portfolio_returns.get(asset, 0)
                
                top_holdings.append({
                    'asset': asset,
                    'weight': weight,
                    'value': weight * portfolio_value,
                    'daily_return': holding_return
                })
        
        # Process risk metrics
        risk_metrics_list = []
        risk_limits = {
            'portfolio_volatility': 0.20,
            'var_95': 0.05,
            'maximum_drawdown': 0.15
        }
        
        for metric_name, value in risk_metrics.items():
            limit = risk_limits.get(metric_name, float('inf'))
            breach = abs(value) > limit if metric_name != 'sharpe_ratio' else value < 0
            
            risk_metrics_list.append({
                'name': metric_name.replace('_', ' ').title(),
                'value': f"{value:.2%}" if 'ratio' not in metric_name else f"{value:.2f}",
                'limit': f"{limit:.2%}" if 'ratio' not in metric_name else f"{limit:.2f}",
                'breach': breach
            })
        
        # Create charts
        charts = []
        
        # Performance chart
        if not portfolio_returns.empty:
            fig = go.Figure()
            cumulative = (1 + portfolio_returns.tail(30)).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name='Portfolio'
            ))
            
            charts.append({
                'title': 'Last 30 Days Performance',
                'html': fig.to_html(include_plotlyjs='cdn', div_id='perf_chart')
            })
        
        # Render template
        template = self.jinja_env.get_template('daily_report.html')
        
        html_content = template.render(
            report_date=report_date.strftime('%Y-%m-%d'),
            portfolio_value=portfolio_value,
            daily_return=daily_return,
            ytd_return=ytd_return,
            current_regime=regime_data.get('current_regime_name', 'Unknown'),
            regime_confidence=regime_data.get('regime_confidence', 0),
            risk_metrics=risk_metrics_list,
            top_holdings=top_holdings,
            risk_alerts=risk_alerts,
            charts=charts,
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Save report
        filename = f"daily_report_{report_date.strftime('%Y%m%d')}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Daily report saved to {filepath}")
        return str(filepath)
    
    def generate_weekly_report(
        self,
        portfolio_data: Dict[str, Any],
        report_date: Optional[datetime] = None
    ) -> str:
        """Generate weekly portfolio report"""
        if report_date is None:
            report_date = datetime.now()
        
        logger.info(f"Generating weekly report for week ending {report_date.strftime('%Y-%m-%d')}")
        
        portfolio_returns = portfolio_data.get('portfolio_returns', pd.Series())
        
        # Calculate weekly metrics
        week_start = report_date - timedelta(days=7)
        weekly_returns = portfolio_returns[portfolio_returns.index >= week_start]
        
        if not weekly_returns.empty:
            weekly_return = (1 + weekly_returns).prod() - 1
            weekly_sharpe = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(252) if weekly_returns.std() > 0 else 0
        else:
            weekly_return = 0
            weekly_sharpe = 0
        
        portfolio_value = portfolio_data.get('portfolio_value', 1000000)
        
        # Generate regime analysis content
        regime_analysis = self._generate_regime_analysis_content(portfolio_data)
        attribution_analysis = self._generate_attribution_analysis_content(portfolio_data)
        
        # Render template
        template = self.jinja_env.get_template('weekly_report.html')
        
        html_content = template.render(
            report_date=report_date.strftime('%Y-%m-%d'),
            weekly_return=weekly_return,
            portfolio_value=portfolio_value,
            weekly_sharpe=weekly_sharpe,
            regime_analysis=regime_analysis,
            attribution_analysis=attribution_analysis,
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Save report
        filename = f"weekly_report_{report_date.strftime('%Y%m%d')}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Weekly report saved to {filepath}")
        return str(filepath)
    
    def generate_monthly_report(
        self,
        portfolio_data: Dict[str, Any],
        report_date: Optional[datetime] = None
    ) -> str:
        """Generate comprehensive monthly report"""
        if report_date is None:
            report_date = datetime.now()
        
        logger.info(f"Generating monthly report for {report_date.strftime('%Y-%m')}")
        
        # Generate comprehensive analysis
        portfolio_returns = portfolio_data.get('portfolio_returns', pd.Series())
        
        if not portfolio_returns.empty:
            monthly_metrics = self.performance_analytics.calculate_comprehensive_metrics(
                portfolio_returns,
                portfolio_data.get('benchmark_returns'),
                risk_free_rate=0.02
            )
        else:
            monthly_metrics = {}
        
        # Create detailed charts and analysis
        charts_html = self._create_monthly_charts(portfolio_data)
        
        # Generate detailed regime analysis
        regime_analysis = self._generate_detailed_regime_analysis(portfolio_data)
        
        # Create comprehensive HTML report
        html_content = self._create_monthly_html_report(
            report_date, monthly_metrics, charts_html, regime_analysis, portfolio_data
        )
        
        # Save report
        filename = f"monthly_report_{report_date.strftime('%Y%m')}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Monthly report saved to {filepath}")
        return str(filepath)
    
    def _generate_regime_analysis_content(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate regime analysis HTML content"""
        regime_data = portfolio_data.get('regime_data', {})
        
        if not regime_data:
            return "<p>No regime data available</p>"
        
        html = "<table>"
        html += "<tr><th>Regime</th><th>Probability</th><th>Duration</th></tr>"
        
        for regime_id, prob in regime_data.get('regime_probabilities', {}).items():
            regime_name = regime_data.get('regime_names', {}).get(regime_id, f'Regime {regime_id}')
            duration = regime_data.get('regime_durations', {}).get(regime_id, 0)
            
            html += f"<tr><td>{regime_name}</td><td>{prob:.1%}</td><td>{duration} days</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_attribution_analysis_content(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate attribution analysis HTML content"""
        attribution_data = portfolio_data.get('attribution_data', {})
        
        if not attribution_data:
            return "<p>No attribution data available</p>"
        
        html = "<h3>Performance Attribution</h3>"
        html += "<table>"
        html += "<tr><th>Component</th><th>Contribution</th></tr>"
        
        for component, contribution in attribution_data.items():
            html += f"<tr><td>{component}</td><td>{contribution:.2%}</td></tr>"
        
        html += "</table>"
        return html
    
    def _create_monthly_charts(self, portfolio_data: Dict[str, Any]) -> str:
        """Create charts for monthly report"""
        charts_html = ""
        
        portfolio_returns = portfolio_data.get('portfolio_returns', pd.Series())
        
        if not portfolio_returns.empty:
            # Cumulative performance chart
            fig = go.Figure()
            cumulative = (1 + portfolio_returns).cumprod()
            
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name='Portfolio Performance'
            ))
            
            fig.update_layout(
                title='Monthly Cumulative Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Return'
            )
            
            charts_html += fig.to_html(include_plotlyjs='cdn', div_id='monthly_perf')
        
        return charts_html
    
    def _generate_detailed_regime_analysis(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate detailed regime analysis for monthly report"""
        return "<h3>Detailed Regime Analysis</h3><p>Coming soon...</p>"
    
    def _create_monthly_html_report(
        self,
        report_date: datetime,
        metrics: Dict[str, Any],
        charts_html: str,
        regime_analysis: str,
        portfolio_data: Dict[str, Any]
    ) -> str:
        """Create comprehensive monthly HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Monthly Portfolio Report - {report_date.strftime('%Y-%m')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Monthly Portfolio Report</h1>
                <h3>{report_date.strftime('%B %Y')}</h3>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in metric.lower():
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.2%}"
            else:
                formatted_value = str(value)
            
            html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        html += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Charts</h2>
                {charts_html}
            </div>
            
            <div class="section">
                {regime_analysis}
            </div>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Stochastic Portfolio Engine v1.0
            </footer>
        </body>
        </html>
        """
        
        return html


class AutomatedReportingSystem:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.report_generator = ReportGenerator()
        self.scheduler_thread = None
        self.running = False
        
        # Email configuration
        self.email_config = self.config.get('email', {})
        
    def setup_email_notifications(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        recipients: List[str]
    ):
        """Setup email notification configuration"""
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipients': recipients
        }
    
    def send_email_report(self, report_path: str, subject: str):
        """Send report via email"""
        if not self.email_config:
            logger.warning("Email configuration not set up")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = subject
            
            # Attach HTML report
            with open(report_path, 'r') as f:
                html_content = f.read()
            
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Report sent to {len(self.email_config['recipients'])} recipients")
            
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
    
    def schedule_reports(self, portfolio_data_source: callable):
        """Schedule automated report generation"""
        
        def generate_daily_report():
            try:
                portfolio_data = portfolio_data_source()
                report_path = self.report_generator.generate_daily_report(portfolio_data)
                
                if self.email_config:
                    self.send_email_report(
                        report_path,
                        f"Daily Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}"
                    )
                    
            except Exception as e:
                logger.error(f"Error generating daily report: {e}")
        
        def generate_weekly_report():
            try:
                portfolio_data = portfolio_data_source()
                report_path = self.report_generator.generate_weekly_report(portfolio_data)
                
                if self.email_config:
                    self.send_email_report(
                        report_path,
                        f"Weekly Portfolio Report - Week ending {datetime.now().strftime('%Y-%m-%d')}"
                    )
                    
            except Exception as e:
                logger.error(f"Error generating weekly report: {e}")
        
        def generate_monthly_report():
            try:
                portfolio_data = portfolio_data_source()
                report_path = self.report_generator.generate_monthly_report(portfolio_data)
                
                if self.email_config:
                    self.send_email_report(
                        report_path,
                        f"Monthly Portfolio Report - {datetime.now().strftime('%B %Y')}"
                    )
                    
            except Exception as e:
                logger.error(f"Error generating monthly report: {e}")
        
        # Schedule jobs
        schedule.every().day.at("18:00").do(generate_daily_report)
        schedule.every().friday.at("19:00").do(generate_weekly_report)
        schedule.every().month.do(generate_monthly_report)
        
        logger.info("Report scheduling configured")
    
    def start_scheduler(self, portfolio_data_source: callable):
        """Start the automated reporting scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.schedule_reports(portfolio_data_source)
        self.running = True
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Automated reporting scheduler started")
    
    def stop_scheduler(self):
        """Stop the automated reporting scheduler"""
        self.running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Automated reporting scheduler stopped")
    
    def generate_compliance_report(
        self,
        portfolio_data: Dict[str, Any],
        regulatory_requirements: Dict[str, Any]
    ) -> str:
        """Generate compliance report for regulatory requirements"""
        logger.info("Generating compliance report")
        
        # This would be customized based on specific regulatory requirements
        # For now, create a basic compliance structure
        
        compliance_checks = []
        
        # Example compliance checks
        portfolio_weights = portfolio_data.get('portfolio_weights', pd.Series())
        
        # Single position limit check
        max_position_limit = regulatory_requirements.get('max_single_position', 0.10)
        if not portfolio_weights.empty:
            max_weight = portfolio_weights.max()
            compliance_checks.append({
                'rule': 'Maximum Single Position',
                'limit': f"{max_position_limit:.1%}",
                'current': f"{max_weight:.1%}",
                'compliant': max_weight <= max_position_limit
            })
        
        # Leverage check
        leverage_limit = regulatory_requirements.get('max_leverage', 1.0)
        current_leverage = portfolio_weights.abs().sum() if not portfolio_weights.empty else 0
        compliance_checks.append({
            'rule': 'Maximum Leverage',
            'limit': f"{leverage_limit:.1f}x",
            'current': f"{current_leverage:.1f}x",
            'compliant': current_leverage <= leverage_limit
        })
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .compliant {{ color: green; }}
                .non-compliant {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Portfolio Compliance Report</h1>
            <h3>Report Date: {datetime.now().strftime('%Y-%m-%d')}</h3>
            
            <h2>Compliance Summary</h2>
            <table>
                <tr><th>Rule</th><th>Limit</th><th>Current</th><th>Status</th></tr>
        """
        
        for check in compliance_checks:
            status_class = "compliant" if check['compliant'] else "non-compliant"
            status_text = "COMPLIANT" if check['compliant'] else "BREACH"
            
            html_content += f"""
                <tr>
                    <td>{check['rule']}</td>
                    <td>{check['limit']}</td>
                    <td>{check['current']}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
                Generated by Stochastic Portfolio Engine
            </footer>
        </body>
        </html>
        """
        
        # Save compliance report
        filename = f"compliance_report_{datetime.now().strftime('%Y%m%d')}.html"
        filepath = self.report_generator.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Compliance report saved to {filepath}")
        return str(filepath)


def create_reporting_system(config: Optional[Dict[str, Any]] = None) -> AutomatedReportingSystem:
    """Factory function to create reporting system"""
    return AutomatedReportingSystem(config)