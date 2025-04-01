# RiskManagement
Am interrested in becoming a quant so will be trying some new stuff
My bad for lack of comments lol, if i revisit it might add some

# Risk Management System

A lightweight Python library for financial and project risk modeling using Monte Carlo simulations.

## Features

- Create custom risk factors with various statistical distributions
- Run Monte Carlo simulations to model risk outcomes
- Calculate key risk metrics (VaR, CVaR)
- Generate visual distributions and summary reports
- Pre-built portfolio and project risk models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/risk-management-system.git

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy
```

## Quick Start

```python
from risk_management import create_example_portfolio_risk

# Create a portfolio risk model
risk_system = create_example_portfolio_risk()

# Run simulations
risk_system.run_all(n_simulations=10000)

# Generate report
report = risk_system.generate_report()
print(report)

# Plot distributions
risk_system.plot_all_distributions()
```

## Usage Examples

### Portfolio Risk Model

Models investment portfolio risk with factors like market returns, interest rates, and inflation.

### Project Risk Model

Models project risk with factors like schedule delays, cost overruns, and quality issues.

## License

MIT
