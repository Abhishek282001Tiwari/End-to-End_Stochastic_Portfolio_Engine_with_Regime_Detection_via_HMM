#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stochastic-portfolio-engine",
    version="1.0.0",
    author="Portfolio Engine Team",
    author_email="team@portfolio.com",
    description="End-to-End Stochastic Portfolio Engine with Regime Detection via HMM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/End-to-End_Stochastic_Portfolio_Engine_with_Regime_Detection_via_HMM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=2.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "portfolio=src.cli.portfolio_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/End-to-End_Stochastic_Portfolio_Engine_with_Regime_Detection_via_HMM/issues",
        "Source": "https://github.com/your-username/End-to-End_Stochastic_Portfolio_Engine_with_Regime_Detection_via_HMM",
        "Documentation": "https://portfolio-engine.readthedocs.io/",
    },
)