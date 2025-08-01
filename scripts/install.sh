#!/bin/bash

# Installation script for Stochastic Portfolio Engine
# Usage: bash scripts/install.sh

set -e

echo "🚀 Installing Stochastic Portfolio Engine..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 9 ]]; then
        print_success "Python $PYTHON_VERSION found ✓"
    else
        print_error "Python 3.9+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "setup.py" ]]; then
    print_error "setup.py not found. Please run this script from the project root directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
print_status "Installing portfolio engine and dependencies..."
pip install -e .

# Install additional development dependencies
print_status "Installing development dependencies..."
pip install -e ".[dev]"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external
mkdir -p results
mkdir -p config

# Create default configuration
print_status "Creating default configuration..."
if [[ ! -f "config/config.yaml" ]]; then
    portfolio init-config -c config/config.yaml -t basic
    print_success "Default configuration created at config/config.yaml"
else
    print_warning "Configuration file already exists at config/config.yaml"
fi

# Make CLI script executable
print_status "Making CLI script executable..."
chmod +x src/cli/portfolio_cli.py

# Verify installation
print_status "Verifying installation..."
if portfolio version &> /dev/null; then
    print_success "Installation successful! 🎉"
    echo ""
    echo "📋 Quick Start Commands:"
    echo "  portfolio --help                    # Show all commands"
    echo "  portfolio version                   # Show version"
    echo "  portfolio init-config -t advanced   # Create advanced config"
    echo "  portfolio optimize -s 'AAPL,GOOGL,MSFT' # Optimize portfolio"
    echo "  portfolio simulate -s 'AAPL,GOOGL' --scenarios # Run simulation"
    echo "  portfolio analyze-regimes -s 'SPY' # Analyze market regimes"
    echo ""
    echo "📖 Documentation: See README.md for detailed usage instructions"
    echo "🔧 Configuration: Edit config/config.yaml to customize settings"
    echo ""
    echo "🎯 Example workflow:"
    echo "  1. Edit config/config.yaml with your preferences"
    echo "  2. Run: portfolio optimize -s 'AAPL,GOOGL,MSFT,AMZN' --monte-carlo"
    echo "  3. Run: portfolio backtest -s 'AAPL,GOOGL,MSFT,AMZN' --output results/"
    echo ""
else
    print_error "Installation verification failed"
    print_error "Please check the error messages above and try again"
    exit 1
fi

print_success "Setup complete! Activate the environment with: source venv/bin/activate"