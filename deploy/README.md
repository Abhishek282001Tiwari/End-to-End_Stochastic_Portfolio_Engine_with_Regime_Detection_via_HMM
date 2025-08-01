# Deployment Guide for Stochastic Portfolio Engine

This directory contains deployment configurations for various platforms.

## 🚀 Quick Deployment Options

### 1. Local Development
```bash
# Install dependencies
pip install -r streamlit_requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### 2. Docker Deployment
```bash
# Build and run with Docker
docker build -t stochastic-portfolio-engine .
docker run -p 8501:8501 stochastic-portfolio-engine

# Or use Docker Compose
docker-compose up -d
```

### 3. Kubernetes Deployment
```bash
# Apply Kubernetes configuration
kubectl apply -f deploy/kubernetes.yaml

# Check deployment status
kubectl get pods -l app=portfolio-engine
```

### 4. Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Configure secrets in the Streamlit Cloud dashboard
4. Deploy automatically

### 5. Heroku Deployment
```bash
# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set ALPHA_VANTAGE_API_KEY=your_key_here

# Deploy
git push heroku main
```

## 📋 Prerequisites

### Required Environment Variables
- `ALPHA_VANTAGE_API_KEY` (optional)
- `POLYGON_API_KEY` (optional)
- `QUANDL_API_KEY` (optional)
- `BLOOMBERG_API_KEY` (optional)

### System Requirements
- Python 3.11+
- 2GB+ RAM recommended
- 1GB+ disk space

## 🔧 Configuration

### Environment Variables
Create a `.env` file for local development:
```env
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
QUANDL_API_KEY=your_key_here
BLOOMBERG_API_KEY=your_key_here
```

### Streamlit Configuration
The application uses `.streamlit/config.toml` for Streamlit-specific settings:
- Theme colors
- Server configuration
- Performance settings

### Application Configuration
Main configuration is in `config/config.yaml`:
- Data sources
- Risk parameters
- Logging settings
- API endpoints

## 🏗️ Architecture

### Core Components
- **Streamlit Frontend**: Web-based dashboard
- **Portfolio Engine**: Core analytics and modeling
- **Data Pipeline**: Market data ingestion
- **Risk Monitor**: Real-time risk management
- **Monte Carlo Engine**: Stochastic simulation

### Optional Components
- **Redis**: Caching and session storage
- **PostgreSQL**: Persistent data storage
- **Docker**: Containerization
- **Kubernetes**: Orchestration and scaling

## 📊 Monitoring and Logging

### Health Checks
- Application health: `http://localhost:8501/_stcore/health`
- Custom health endpoint: `http://localhost:8501/health`

### Logging
- Application logs: `logs/streamlit_app.log`
- Error logs: `logs/error.log`
- Access logs: `logs/access.log`

### Metrics
- CPU and memory usage
- Request latency
- Error rates
- User sessions

## 🔒 Security

### API Key Management
- Store API keys as environment variables
- Use secrets management in production
- Rotate keys regularly
- Monitor API usage

### Network Security
- Enable HTTPS in production
- Configure CORS appropriately
- Use secure headers
- Implement rate limiting

### Data Security
- Encrypt sensitive data
- Use secure connections
- Implement access controls
- Regular security audits

## 🚨 Troubleshooting

### Common Issues
1. **Import Errors**: Check PYTHONPATH and dependencies
2. **Port Conflicts**: Ensure port 8501 is available
3. **Memory Issues**: Increase available RAM
4. **API Limits**: Check API key quotas

### Debug Mode
Enable debug mode by setting:
```env
STREAMLIT_LOGGER_LEVEL=debug
```

### Performance Optimization
- Use caching for expensive computations
- Optimize data loading
- Configure session state properly
- Monitor resource usage

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Heroku Documentation](https://devcenter.heroku.com/)

## 🤝 Support

For deployment issues:
1. Check the application logs
2. Verify environment variables
3. Test dependencies
4. Review configuration files
5. Contact support team

## 📝 License

This deployment configuration is part of the Stochastic Portfolio Engine project.
See the main LICENSE file for details.