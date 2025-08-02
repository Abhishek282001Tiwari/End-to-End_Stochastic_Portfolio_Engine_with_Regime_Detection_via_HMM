// Results page interactive charts and functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    initializeTabs();
});

// Tab functionality
function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => content.classList.remove('active'));
    
    // Remove active class from all buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => button.classList.remove('active'));
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.textContent.toLowerCase().replace(/\s+/g, '').replace('analysis', '').replace('metrics', '');
            showTab(tabName);
        });
    });
}

// Chart initialization
function initializeCharts() {
    initializePerformanceChart();
    initializeRegimeChart();
    initializeRiskReturnChart();
    initializeSharpeChart();
    initializeReturnsDistribution();
    initializeDrawdownChart();
    initializeSectorChart();
    initializeRegimeTransition();
    initializeMonteCarloChart();
}

// Performance Chart
function initializePerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    const dates = generateDateRange('2023-01-01', '2023-12-31');
    const portfolioData = generatePerformanceData(dates, 0.204, 0.142);
    const benchmarkData = generatePerformanceData(dates, 0.121, 0.168);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates.map(d => d.toISOString().split('T')[0]),
            datasets: [{
                label: 'Portfolio',
                data: portfolioData,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1
            }, {
                label: 'S&P 500 Benchmark',
                data: benchmarkData,
                borderColor: '#6b7280',
                backgroundColor: 'rgba(107, 114, 128, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Cumulative Return'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Regime Detection Chart
function initializeRegimeChart() {
    const ctx = document.getElementById('regimeChart');
    if (!ctx) return;
    
    const dates = generateDateRange('2023-01-01', '2023-12-31');
    const regimeData = generateRegimeData(dates);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates.map(d => d.toISOString().split('T')[0]),
            datasets: [{
                label: 'Market Regime',
                data: regimeData,
                backgroundColor: function(context) {
                    const value = context.parsed.y;
                    if (value === 0) return '#059669'; // Bull - Green
                    if (value === 1) return '#dc2626'; // Bear - Red
                    if (value === 2) return '#d97706'; // High Vol - Orange
                    return '#6b7280'; // Sideways - Gray
                },
                borderColor: 'transparent',
                pointRadius: 2,
                stepped: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Regime'
                    },
                    ticks: {
                        callback: function(value) {
                            const regimes = ['Bull', 'Bear', 'High Vol', 'Sideways'];
                            return regimes[value] || '';
                        }
                    },
                    min: 0,
                    max: 3
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Risk-Return Scatter Plot
function initializeRiskReturnChart() {
    const ctx = document.getElementById('riskReturnChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Portfolio',
                data: [{x: 14.2, y: 20.4}],
                backgroundColor: '#2563eb',
                borderColor: '#2563eb',
                pointRadius: 8
            }, {
                label: 'S&P 500',
                data: [{x: 16.8, y: 12.1}],
                backgroundColor: '#6b7280',
                borderColor: '#6b7280',
                pointRadius: 8
            }, {
                label: 'Efficient Frontier',
                data: generateEfficientFrontier(),
                backgroundColor: 'rgba(37, 99, 235, 0.3)',
                borderColor: '#2563eb',
                pointRadius: 2,
                showLine: true,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Volatility (%)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Return (%)'
                    }
                }
            }
        }
    });
}

// Rolling Sharpe Ratio Chart
function initializeSharpeChart() {
    const ctx = document.getElementById('sharpeChart');
    if (!ctx) return;
    
    const dates = generateDateRange('2023-01-01', '2023-12-31');
    const sharpeData = generateSharpeData(dates);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates.map(d => d.toISOString().split('T')[0]),
            datasets: [{
                label: 'Rolling Sharpe Ratio (60 days)',
                data: sharpeData,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Sharpe Ratio'
                    }
                }
            }
        }
    });
}

// Returns Distribution
function initializeReturnsDistribution() {
    const ctx = document.getElementById('returnsDistribution');
    if (!ctx) return;
    
    const returns = generateReturnsDistribution();
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: returns.labels,
            datasets: [{
                label: 'Frequency',
                data: returns.data,
                backgroundColor: 'rgba(37, 99, 235, 0.7)',
                borderColor: '#2563eb',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Monthly Return (%)'
                    }
                }
            }
        }
    });
}

// Drawdown Chart
function initializeDrawdownChart() {
    const ctx = document.getElementById('drawdownChart');
    if (!ctx) return;
    
    const dates = generateDateRange('2023-01-01', '2023-12-31');
    const drawdownData = generateDrawdownData(dates);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates.map(d => d.toISOString().split('T')[0]),
            datasets: [{
                label: 'Drawdown',
                data: drawdownData,
                borderColor: '#dc2626',
                backgroundColor: 'rgba(220, 38, 38, 0.3)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Drawdown (%)'
                    },
                    max: 0,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Sector Attribution Chart
function initializeSectorChart() {
    const ctx = document.getElementById('sectorChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 'Other'],
            datasets: [{
                data: [28.5, 18.2, 15.7, 12.3, 11.8, 13.5],
                backgroundColor: [
                    '#2563eb',
                    '#059669',
                    '#d97706',
                    '#dc2626',
                    '#7c3aed',
                    '#6b7280'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Regime Transition Chart
function initializeRegimeTransition() {
    const ctx = document.getElementById('regimeTransition');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Bull→Bear', 'Bull→Vol', 'Bear→Bull', 'Bear→Vol', 'Vol→Bull', 'Vol→Bear'],
            datasets: [{
                label: 'Transition Probability',
                data: [0.15, 0.25, 0.35, 0.45, 0.30, 0.20],
                backgroundColor: [
                    'rgba(220, 38, 38, 0.7)',
                    'rgba(217, 119, 6, 0.7)',
                    'rgba(5, 150, 105, 0.7)',
                    'rgba(217, 119, 6, 0.7)',
                    'rgba(5, 150, 105, 0.7)',
                    'rgba(220, 38, 38, 0.7)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Probability'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Monte Carlo Simulation Chart
function initializeMonteCarloChart() {
    const ctx = document.getElementById('monteCarloChart');
    if (!ctx) return;
    
    const distributionData = generateMonteCarloDistribution();
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: distributionData.labels,
            datasets: [{
                label: 'Frequency',
                data: distributionData.data,
                backgroundColor: 'rgba(37, 99, 235, 0.7)',
                borderColor: '#2563eb',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Annual Return (%)'
                    }
                }
            }
        }
    });
}

// Helper functions for data generation
function generateDateRange(startDate, endDate) {
    const dates = [];
    const currentDate = new Date(startDate);
    const lastDate = new Date(endDate);
    
    while (currentDate <= lastDate) {
        dates.push(new Date(currentDate));
        currentDate.setDate(currentDate.getDate() + 7); // Weekly data
    }
    
    return dates;
}

function generatePerformanceData(dates, annualReturn, volatility) {
    const data = [1.0]; // Start at 100%
    const dailyReturn = annualReturn / 52; // Weekly returns
    const dailyVol = volatility / Math.sqrt(52);
    
    for (let i = 1; i < dates.length; i++) {
        const randomReturn = (Math.random() - 0.5) * 2 * dailyVol + dailyReturn;
        data.push(data[i-1] * (1 + randomReturn));
    }
    
    return data.map(d => d - 1); // Convert to returns
}

function generateRegimeData(dates) {
    const data = [];
    let currentRegime = 0; // Start with Bull market
    
    for (let i = 0; i < dates.length; i++) {
        // Simulate regime changes
        if (Math.random() < 0.1) { // 10% chance of regime change each week
            currentRegime = Math.floor(Math.random() * 4);
        }
        data.push(currentRegime);
    }
    
    return data;
}

function generateSharpeData(dates) {
    const data = [];
    let baseSharpe = 1.47;
    
    for (let i = 0; i < dates.length; i++) {
        const noise = (Math.random() - 0.5) * 0.5;
        data.push(baseSharpe + noise);
    }
    
    return data;
}

function generateEfficientFrontier() {
    const points = [];
    for (let vol = 8; vol <= 25; vol += 0.5) {
        const ret = 2 + (vol - 8) * 1.2 - Math.pow(vol - 15, 2) * 0.01;
        points.push({x: vol, y: ret});
    }
    return points;
}

function generateReturnsDistribution() {
    const labels = [];
    const data = [];
    
    for (let i = -10; i <= 15; i += 2) {
        labels.push(i + '%');
        // Normal-ish distribution centered around positive returns
        const frequency = Math.exp(-Math.pow(i - 1.5, 2) / 18) * 100;
        data.push(frequency);
    }
    
    return {labels, data};
}

function generateDrawdownData(dates) {
    const data = [];
    let currentDrawdown = 0;
    
    for (let i = 0; i < dates.length; i++) {
        // Simulate recovery and new drawdowns
        if (Math.random() < 0.7) {
            currentDrawdown = Math.max(currentDrawdown * 0.95, 0); // Recovery
        } else {
            currentDrawdown = Math.min(currentDrawdown - Math.random() * 2, -8.2); // New drawdown
        }
        data.push(currentDrawdown);
    }
    
    return data;
}

function generateMonteCarloDistribution() {
    const labels = [];
    const data = [];
    
    for (let i = -10; i <= 40; i += 2) {
        labels.push(i + '%');
        // Log-normal-ish distribution for returns
        if (i >= 0) {
            const frequency = Math.exp(-Math.pow(i - 18, 2) / 80) * 1000;
            data.push(frequency);
        } else {
            const frequency = Math.exp(-Math.pow(i + 5, 2) / 20) * 200;
            data.push(frequency);
        }
    }
    
    return {labels, data};
}