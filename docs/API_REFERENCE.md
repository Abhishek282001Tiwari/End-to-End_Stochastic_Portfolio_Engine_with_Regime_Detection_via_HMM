# API Reference

## Overview

The Portfolio Engine provides RESTful APIs for all major functionality including data access, model predictions, portfolio optimization, and risk monitoring.

## Base URL
```
Production: https://api.portfolio-engine.com/v1
Development: http://localhost:8000/v1
```

## Authentication

All API requests require authentication using JWT tokens:

```http
Authorization: Bearer <jwt_token>
```

### Get Authentication Token
```http
POST /auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## Data APIs

### Get Market Data
Retrieve historical market data for specified symbols.

```http
GET /data/market?symbols=AAPL,GOOGL&start_date=2023-01-01&end_date=2023-12-31
```

**Parameters:**
- `symbols` (string): Comma-separated list of stock symbols
- `start_date` (string): Start date in YYYY-MM-DD format
- `end_date` (string): End date in YYYY-MM-DD format
- `interval` (string, optional): Data interval (1d, 1h, 5m). Default: 1d

**Response:**
```json
{
  "data": {
    "AAPL": {
      "2023-01-01": {
        "open": 130.28,
        "high": 133.41,
        "low": 129.89,
        "close": 133.01,
        "volume": 112117471
      }
    }
  },
  "metadata": {
    "symbols": ["AAPL", "GOOGL"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "count": 252
  }
}
```

### Get Economic Indicators
```http
GET /data/economic?indicators=gdp,inflation,unemployment
```

**Response:**
```json
{
  "data": {
    "gdp": {
      "2023-Q1": 26.85,
      "2023-Q2": 27.02
    },
    "inflation": {
      "2023-01": 6.4,
      "2023-02": 6.0
    }
  }
}
```

## Model APIs

### Regime Detection

#### Get Current Regime
```http
GET /models/regime/current
```

**Response:**
```json
{
  "current_regime": 1,
  "regime_name": "High Volatility",
  "confidence": 0.87,
  "probabilities": [0.05, 0.87, 0.08],
  "timestamp": "2023-12-01T15:30:00Z"
}
```

#### Predict Regimes
```http
POST /models/regime/predict
Content-Type: application/json

{
  "data": {
    "features": [
      [0.02, 0.15, 1.2, 18, 0.01],
      [0.01, 0.12, 0.9, 16, 0.02]
    ],
    "feature_names": ["returns", "volatility", "volume_ratio", "vix", "yield_curve"]
  }
}
```

**Response:**
```json
{
  "predictions": [1, 0],
  "probabilities": [
    [0.05, 0.87, 0.08],
    [0.75, 0.20, 0.05]
  ],
  "model_info": {
    "model_type": "ensemble",
    "version": "1.2.0",
    "training_date": "2023-11-15T10:00:00Z"
  }
}
```

#### Train Regime Model
```http
POST /models/regime/train
Content-Type: application/json

{
  "data_source": "yahoo_finance",
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "start_date": "2020-01-01",
  "end_date": "2023-12-01",
  "model_config": {
    "n_components": 3,
    "ensemble_methods": ["hmm", "random_forest", "xgboost"]
  }
}
```

**Response:**
```json
{
  "task_id": "train_abc123",
  "status": "started",
  "estimated_duration": "15 minutes",
  "progress_url": "/tasks/train_abc123/progress"
}
```

### Model Validation

#### Validate Model
```http
POST /models/validate
Content-Type: application/json

{
  "model_type": "regime_detection",
  "validation_config": {
    "test_size": 0.3,
    "n_splits": 5,
    "metrics": ["accuracy", "precision", "recall"]
  }
}
```

**Response:**
```json
{
  "validation_id": "val_xyz789",
  "results": {
    "cross_validation": {
      "mean_score": 0.78,
      "std_score": 0.05
    },
    "test_accuracy": 0.82,
    "confusion_matrix": [[45, 3, 2], [5, 38, 7], [1, 6, 43]]
  }
}
```

## Portfolio APIs

### Portfolio Optimization

#### Optimize Portfolio
```http
POST /portfolio/optimize
Content-Type: application/json

{
  "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
  "method": "mean_variance",
  "constraints": {
    "max_weight": 0.4,
    "min_weight": 0.05,
    "target_return": 0.12
  },
  "risk_aversion": 1.0
}
```

**Response:**
```json
{
  "optimal_weights": {
    "AAPL": 0.35,
    "GOOGL": 0.28,
    "MSFT": 0.22,
    "AMZN": 0.15
  },
  "expected_return": 0.124,
  "expected_risk": 0.185,
  "sharpe_ratio": 1.67,
  "optimization_info": {
    "method": "mean_variance",
    "converged": true,
    "iterations": 23
  }
}
```

#### Get Portfolio Performance
```http
GET /portfolio/performance?start_date=2023-01-01&end_date=2023-12-01
```

**Response:**
```json
{
  "performance_metrics": {
    "total_return": 0.156,
    "annualized_return": 0.142,
    "volatility": 0.178,
    "sharpe_ratio": 1.45,
    "max_drawdown": 0.087,
    "var_95": 0.023,
    "var_99": 0.041
  },
  "portfolio_value": 1156000,
  "daily_returns": [...],
  "cumulative_returns": [...]
}
```

#### Rebalance Portfolio
```http
POST /portfolio/rebalance
Content-Type: application/json

{
  "current_weights": {
    "AAPL": 0.40,
    "GOOGL": 0.25,
    "MSFT": 0.20,
    "AMZN": 0.15
  },
  "target_weights": {
    "AAPL": 0.35,
    "GOOGL": 0.30,
    "MSFT": 0.20,
    "AMZN": 0.15
  },
  "portfolio_value": 1000000
}
```

**Response:**
```json
{
  "rebalancing_trades": {
    "AAPL": {
      "action": "sell",
      "shares": 543,
      "value": 50000
    },
    "GOOGL": {
      "action": "buy",
      "shares": 234,
      "value": 50000
    }
  },
  "transaction_costs": 125.50,
  "net_trades": 4
}
```

## Risk Management APIs

### Risk Monitoring

#### Get Risk Metrics
```http
GET /risk/metrics
```

**Response:**
```json
{
  "current_metrics": {
    "portfolio_var_95": 0.023,
    "portfolio_var_99": 0.041,
    "expected_shortfall": 0.032,
    "beta": 1.05,
    "tracking_error": 0.045,
    "information_ratio": 0.67
  },
  "risk_limits": {
    "var_95_limit": 0.030,
    "max_drawdown_limit": 0.100,
    "concentration_limit": 0.400
  },
  "breaches": [],
  "timestamp": "2023-12-01T15:30:00Z"
}
```

#### Set Risk Alerts
```http
POST /risk/alerts
Content-Type: application/json

{
  "alert_rules": [
    {
      "metric": "var_95",
      "threshold": 0.025,
      "condition": "greater_than",
      "severity": "high"
    },
    {
      "metric": "max_drawdown",
      "threshold": 0.10,
      "condition": "greater_than",
      "severity": "critical"
    }
  ]
}
```

**Response:**
```json
{
  "alert_id": "alert_123",
  "status": "active",
  "rules_created": 2
}
```

### Stress Testing

#### Run Stress Test
```http
POST /risk/stress-test
Content-Type: application/json

{
  "scenarios": [
    {
      "name": "Market Crash",
      "shocks": {
        "equity_return": -0.20,
        "volatility_multiplier": 2.0
      }
    },
    {
      "name": "Interest Rate Rise",
      "shocks": {
        "interest_rate_change": 0.02
      }
    }
  ]
}
```

**Response:**
```json
{
  "stress_test_id": "stress_456",
  "results": {
    "Market Crash": {
      "portfolio_return": -0.18,
      "var_95": 0.067,
      "worst_asset": "AAPL"
    },
    "Interest Rate Rise": {
      "portfolio_return": -0.03,
      "var_95": 0.028,
      "worst_asset": "bonds"
    }
  }
}
```

## Backtesting APIs

### Run Backtest
```http
POST /backtest/run
Content-Type: application/json

{
  "strategy": "regime_aware_portfolio",
  "start_date": "2020-01-01",
  "end_date": "2023-12-01",
  "initial_capital": 1000000,
  "benchmark": "SPY",
  "parameters": {
    "rebalance_frequency": "monthly",
    "transaction_costs": 0.001
  }
}
```

**Response:**
```json
{
  "backtest_id": "bt_789",
  "status": "running",
  "estimated_completion": "2023-12-01T16:00:00Z",
  "progress_url": "/backtest/bt_789/progress"
}
```

### Get Backtest Results
```http
GET /backtest/bt_789/results
```

**Response:**
```json
{
  "backtest_id": "bt_789",
  "performance": {
    "total_return": 0.234,
    "annualized_return": 0.089,
    "volatility": 0.156,
    "sharpe_ratio": 1.78,
    "max_drawdown": 0.067,
    "alpha": 0.023,
    "beta": 0.87
  },
  "benchmark_comparison": {
    "excess_return": 0.045,
    "tracking_error": 0.034,
    "information_ratio": 1.32
  },
  "attribution": {
    "security_selection": 0.018,
    "asset_allocation": 0.027
  }
}
```

## Reporting APIs

### Generate Report
```http
POST /reports/generate
Content-Type: application/json

{
  "report_type": "daily",
  "format": "html",
  "recipients": ["portfolio.manager@example.com"],
  "include_charts": true
}
```

**Response:**
```json
{
  "report_id": "rpt_abc",
  "status": "generated",
  "download_url": "/reports/rpt_abc/download",
  "email_sent": true
}
```

### List Reports
```http
GET /reports?limit=10&offset=0
```

**Response:**
```json
{
  "reports": [
    {
      "report_id": "rpt_abc",
      "type": "daily",
      "date": "2023-12-01",
      "status": "completed",
      "download_url": "/reports/rpt_abc/download"
    }
  ],
  "total": 45,
  "limit": 10,
  "offset": 0
}
```

## WebSocket APIs

### Real-time Data Stream
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

// Subscribe to specific data streams
ws.send(JSON.stringify({
  "action": "subscribe",
  "streams": ["prices", "regimes", "risk_alerts"]
}));
```

**Message Format:**
```json
{
  "stream": "regimes",
  "data": {
    "current_regime": 1,
    "confidence": 0.89,
    "timestamp": "2023-12-01T15:30:00Z"
  }
}
```

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid date format",
    "details": {
      "field": "start_date",
      "expected_format": "YYYY-MM-DD"
    }
  },
  "request_id": "req_12345"
}
```

### Common Error Codes
- `AUTHENTICATION_ERROR`: Invalid or expired token
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `VALIDATION_ERROR`: Invalid request parameters
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server-side error

## Rate Limiting

API endpoints are rate limited:
- Authentication: 5 requests per minute
- Data APIs: 100 requests per hour
- Model APIs: 20 requests per hour
- Portfolio APIs: 50 requests per hour

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1701432000
```

## Pagination

List endpoints support pagination:
```http
GET /data/market?limit=50&offset=100
```

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "limit": 50,
    "offset": 100,
    "total": 1250,
    "has_next": true,
    "has_prev": true
  }
}
```