import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Callable

class RiskFactor:
    def __init__(self, name: str, distribution: Callable, **params):
        self.name = name
        self.distribution = distribution
        self.params = params
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        return self.distribution(**self.params, size=n_samples)

class RiskModel:
    def __init__(self, name: str, risk_factors: List[RiskFactor], impact_function: Callable):
        self.name = name
        self.risk_factors = risk_factors
        self.impact_function = impact_function
        self.results = None
        
    def run_simulation(self, n_simulations: int = 10000) -> np.ndarray:
        samples = {factor.name: factor.sample(n_simulations) for factor in self.risk_factors}
        self.results = self.impact_function(**samples)
        return self.results
    
    def get_var(self, confidence_level: float = 0.95) -> float:
        if self.results is None:
            raise ValueError("Run simulation first before calculating VaR")
        return np.percentile(self.results, (1 - confidence_level) * 100)
    
    def get_cvar(self, confidence_level: float = 0.95) -> float:
        if self.results is None:
            raise ValueError("Run simulation first before calculating CVaR")
        var = self.get_var(confidence_level)
        return np.mean(self.results[self.results <= var])
    
    def plot_distribution(self, bins: int = 50) -> None:
        if self.results is None:
            raise ValueError("Run simulation first before plotting distribution")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.results, bins=bins, kde=True)
        
        var_95 = self.get_var(0.95)
        cvar_95 = self.get_cvar(0.95)
        
        plt.axvline(var_95, color='r', linestyle='--', label=f'95% VaR: {var_95:.2f}')
        plt.axvline(cvar_95, color='g', linestyle='--', label=f'95% CVaR: {cvar_95:.2f}')
        plt.title(f'Risk Distribution for {self.name}')
        plt.xlabel('Impact')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def summary_statistics(self) -> Dict:
        if self.results is None:
            raise ValueError("Run simulation first before calculating statistics")
        
        return {
            'mean': np.mean(self.results),
            'median': np.median(self.results),
            'std_dev': np.std(self.results),
            'var_95': self.get_var(0.95),
            'cvar_95': self.get_cvar(0.95),
            'var_99': self.get_var(0.99),
            'cvar_99': self.get_cvar(0.99),
        }

class RiskManagerSystem:
    def __init__(self):
        self.models = {}
        
    def add_model(self, model: RiskModel) -> None:
        self.models[model.name] = model
    
    def run_all(self, n_simulations: int = 10000) -> Dict[str, np.ndarray]:
        results = {}
        for name, model in self.models.items():
            print(f"Running simulations for {name}...")
            results[name] = model.run_simulation(n_simulations)
            print(f"Completed simulations for {name}")
        return results
    
    def generate_report(self) -> pd.DataFrame:
        data = []
        for name, model in self.models.items():
            if model.results is None:
                print(f"Warning: Model '{name}' has not been run yet")
                continue
            stats = model.summary_statistics()
            stats['model'] = name
            data.append(stats)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df.set_index('model')
    
    def plot_all_distributions(self, bins: int = 50) -> None:
        n_models = len(self.models)
        if n_models == 0:
            print("No models to plot")
            return
        
        for name, model in self.models.items():
            if model.results is None:
                print(f"Warning: Model '{name}' has not been run yet")
                continue
            model.plot_distribution(bins)

def create_example_portfolio_risk():
    market_return = RiskFactor('market_return', np.random.normal, loc=0.05, scale=0.15)
    interest_rate = RiskFactor('interest_rate', np.random.normal, loc=0.02, scale=0.01)
    inflation = RiskFactor('inflation', np.random.normal, loc=0.025, scale=0.01)
    
    def portfolio_impact(market_return, interest_rate, inflation):
        equity_impact = 100000 * market_return
        bond_impact = -50000 * interest_rate
        inflation_impact = -75000 * inflation
        return equity_impact + bond_impact + inflation_impact
    
    portfolio_model = RiskModel(
        'Investment Portfolio',
        [market_return, interest_rate, inflation],
        portfolio_impact
    )
    
    risk_system = RiskManagerSystem()
    risk_system.add_model(portfolio_model)
    
    return risk_system

def create_example_project_risk():
    schedule_delay = RiskFactor('schedule_delay', np.random.triangular, left=0, mode=10, right=60)
    cost_overrun = RiskFactor('cost_overrun', np.random.beta, a=2, b=5)
    quality_issues = RiskFactor('quality_issues', np.random.poisson, lam=3)
    
    def project_impact(schedule_delay, cost_overrun, quality_issues):
        base_cost = 1000000
        delay_impact = schedule_delay * 5000
        cost_impact = base_cost * cost_overrun
        quality_impact = quality_issues * 25000
        return delay_impact + cost_impact + quality_impact
    
    project_model = RiskModel(
        'Project Risk',
        [schedule_delay, cost_overrun, quality_issues],
        project_impact
    )
    
    risk_system = RiskManagerSystem()
    risk_system.add_model(project_model)
    
    return risk_system

def main():
    print("=== Risk Management System ===")
    print("1. Run portfolio risk model")
    print("2. Run project risk model")
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        risk_system = create_example_portfolio_risk()
        print("Created portfolio risk model")
    else:
        risk_system = create_example_project_risk()
        print("Created project risk model")
    
    n_sim = 10000
    print(f"Running {n_sim} simulations...")
    risk_system.run_all(n_simulations=n_sim)
    
    print("\nRisk Report:")
    report = risk_system.generate_report()
    print(report)
    
    risk_system.plot_all_distributions()

if __name__ == "__main__":
    main()
    