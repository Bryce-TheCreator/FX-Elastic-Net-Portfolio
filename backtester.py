import numpy as np
import pandas as pd
from typing import Dict
from optimizers import BaselineFXOptimizer, ElasticNetFXOptimizer

class FXBacktester:
    """
    Simulates out-of-sample portfolio performance over time, explicitly 
    accounting for transaction costs (bid-ask spreads) caused by turnover.
    """
    
    def __init__(self, returns_data: pd.DataFrame, window_size: int = 252, transaction_cost_bps: float = 2.0):
        # We drop NaNs to ensure the rolling window calculation doesn't fail
        self.returns_data = returns_data.dropna()
        self.window_size = window_size
        # Convert basis points to decimal for fee calculation (e.g., 2 bps = 0.0002)
        self.tc_fee = transaction_cost_bps / 10000  
        self.num_assets = len(returns_data.columns)
        
    def calculate_turnover(self, current_weights: np.ndarray, target_weights: np.ndarray) -> float:
        """
        Calculates the two-way turnover required to rebalance the portfolio.
        Formula: sum(|w_target - w_current|)
        This measures the total percentage of the portfolio traded.
        """
        return np.sum(np.abs(target_weights - current_weights))

    def step_forward(self, daily_asset_returns: np.ndarray, target_weights: np.ndarray, turnover: float) -> float:
        """
        Calculates the net-of-fee return for the portfolio on a given day.
        """
        # 1. Calculate the gross return of the portfolio (Dot product of weights and returns)
        gross_return = target_weights @ daily_asset_returns
        
        # 2. Calculate the total fee paid based on the turnover
        fee_paid = turnover * self.tc_fee
        
        # 3. Return the net portfolio return (gross - fees)
        return gross_return - fee_paid

    def run_backtest(self) -> Dict[str, pd.Series]:
        """
        Executes the rolling window backtest for both the Baseline and Elastic Net models.
        """
        dates = self.returns_data.index[self.window_size:]
        
        # Trackers
        baseline_net_returns = []
        elastic_net_returns = []
        
        # Initialize holding weights (start with equal weight naive portfolio)
        w_hold_base = np.ones(self.num_assets) / self.num_assets
        w_hold_elastic = np.ones(self.num_assets) / self.num_assets
        
        # INSTANTIATE OPTIMIZERS ONCE (Before the loop for processing speed)
        opt_base = BaselineFXOptimizer(num_assets=self.num_assets)
        opt_elastic = ElasticNetFXOptimizer(num_assets=self.num_assets, lambda_1=0.05)
        
        print(f"Starting optimized backtest over {len(dates)} days...")
        
        for i in range(self.window_size, len(self.returns_data)):
            # 1. Get the rolling window of historical data (past N days)
            window_data = self.returns_data.iloc[i - self.window_size : i]
            
            # Calculate mu (mean) and Sigma (covariance)
            mu = window_data.mean().values * 252
            Sigma = window_data.cov().values * 252
            
            # 2. UPDATE AND SOLVE (Using parametrized optimizer)
            w_target_base = opt_base.update_and_solve(mu, Sigma)
            w_target_elastic = opt_elastic.update_and_solve(mu, Sigma)
            
            # 3. Calculate Turnover (How much did we change the weights?)
            turnover_base = self.calculate_turnover(w_hold_base, w_target_base)
            turnover_elastic = self.calculate_turnover(w_hold_elastic, w_target_elastic)
            
            # 4. Step Forward (Calculate actual market returns for today)
            todays_market_returns = self.returns_data.iloc[i].values
            
            # Calculate net-of-fee returns
            net_ret_base = self.step_forward(todays_market_returns, w_target_base, turnover_base)
            net_ret_elastic = self.step_forward(todays_market_returns, w_target_elastic, turnover_elastic)
            
            baseline_net_returns.append(net_ret_base)
            elastic_net_returns.append(net_ret_elastic)
            
            # 5. Update holding weights for the next iteration
            w_hold_base = w_target_base
            w_hold_elastic = w_target_elastic
            
        print("Backtest complete.")
        
        return {
            "Baseline (Ridge)": pd.Series(baseline_net_returns, index=dates),
            "Elastic Net (Sparse)": pd.Series(elastic_net_returns, index=dates)
        }