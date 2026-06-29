# download packages in terminal: pip install numpy cvxpy

import numpy as np
import cvxpy as cp

class BaselineFXOptimizer:
    """
    Solves the baseline Currency Allocation problem using an L2 (Ridge) Penalty.
    This acts as our control (Naive) group before we introduce the L1 (Lasso) transaction cost penalty.
    """
    def __init__(self, num_assets: int, gamma: float = 5.0, lambda_2: float = 0.1):
        self.num_assets = num_assets
        self.gamma = gamma # Risk aversion coefficient ---- To be tuned based on the investor's risk preference
        self.lambda_2 = lambda_2 # Ridge penalty parameter ---- To be tuned based on the desired level of regularization

        # 1. Define Variables and Parameters
        self.weights = cp.Variable(self.num_assets) # The variable we are solving for
        self.mu = cp.Parameter(self.num_assets) # Expected annualized returns
        self.Sigma = cp.Parameter((self.num_assets, self.num_assets), PSD=True) # Annualized Covariance Matrix. Positive Semi-Definite (PSD) ensures convexity of the optimization problem.

        # 2. Define Objective and Constraints ONCE

        # Portfolio Expected Return (w^T * mu)
        portfolio_return = self.weights.T @ self.mu # or cp.dot(self.weights, self.mu) for dot product

        # Portfolio Risk (gamma/2 * w^T * Sigma * w)
        portfolio_risk = (self.gamma / 2) * cp.quad_form(self.weights, self.Sigma)
            # cp.quad_form: https://www.cvxpy.org/examples/basic/quadratic_program.html

        # Ridge Penalty (lambda_2 * sum of squared weights)
        ridge_penalty = self.lambda_2 * cp.sum_squares(self.weights)
            # cp.sum_squares: https://www.cvxpy.org/tutorial/performance/index.html



        # Defining the objective function: Maximize expected return while minimizing risk and applying the ridge penalty
        # Stricly convex and non-negative, ensuring safe optimization.
        objective = cp.Maximize(portfolio_return - portfolio_risk - ridge_penalty)

        # Defining the constraints. The sum of the asset (currency) weights must equal 1 (100% invested)
        constraints = [cp.sum(self.weights) == 1]

        # 3. Compile Problem ONCE
        self.prob = cp.Problem(objective, constraints)
            # cp.Problem: https://www.cvxpy.org/tutorial/advanced/index.html#problems

    def update_and_solve(self, mu_val: np.ndarray, Sigma_val: np.ndarray) -> np.ndarray:
        # 4. Update parameters (Ensure strict symmetry for PSD requirement)
        self.mu.value = mu_val
        self.Sigma.value = (Sigma_val + Sigma_val.T) / 2
        
        # 5. Solve instantly
        self.prob.solve(solver=cp.OSQP)
        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback to equal weight if solver fails on bad data
            return np.ones(self.num_assets) / self.num_assets
            
        return self.weights.value
        
    


class ElasticNetFXOptimizer:
    """
        Formulates and solves the convex optimization problem using CVXPY.
        Incorporates both L1 (Lasso) and L2 (Ridge) penalties to achieve a balance between sparsity and regularization.
            This is known as the Elastic Net approach, which is particularly useful in scenarios where we want to select a subset of currencies while also controlling for multicollinearity.
    """
    def __init__(self, num_assets: int, gamma: float = 5.0, lambda_1: float = 0.05, lambda_2: float = 0.1):
        self.num_assets = num_assets
        self.gamma = gamma # Risk aversion coefficient ---- To be tuned based on the investor's risk preference
        self.lambda_1 = lambda_1 # Lasso penalty parameter ---- To be tuned based on the desired level of sparsity
        self.lambda_2 = lambda_2 # Ridge penalty parameter ---- To be tuned based on the desired level of regularization

        self.weights = cp.Variable(self.num_assets) # The variable we are solving for
        self.mu = cp.Parameter(self.num_assets) # Expected annualized returns
        self.Sigma = cp.Parameter((self.num_assets, self.num_assets), PSD=True) # Annualized Covariance Matrix. Positive Semi-Definite (PSD) ensures convexity of the optimization problem.

        # 2. Define Objective and Constraints ONCE

        # Portfolio Expected Return (w^T * mu)
        portfolio_return = self.weights.T @ self.mu

        # Portfolio Risk (gamma/2 * w^T * Sigma * w)
        portfolio_risk = (self.gamma / 2) * cp.quad_form(self.weights, self.Sigma)

        # Ridge Penalty (lambda_2 * sum of squared weights)
        ridge_penalty = self.lambda_2 * cp.sum_squares(self.weights)

        # Lasso Penalty (lambda_1 * sum of absolute weights)
        lasso_penalty = self.lambda_1 * cp.norm(self.weights, 1)


        # Defining the objective function: Maximize expected return while minimizing risk and applying both penalties
        # Stricly convex and non-negative, ensuring safe optimization.
        objective = cp.Maximize(portfolio_return - portfolio_risk - ridge_penalty - lasso_penalty)

        # Defining the constraints. The sum of the asset (currency) weights must equal 1 (100% invested)
        constraints = [cp.sum(self.weights) == 1]

        self.prob = cp.Problem(objective, constraints)

    def update_and_solve(self, mu_val: np.ndarray, Sigma_val: np.ndarray) -> np.ndarray:
        self.mu.value = mu_val
        self.Sigma.value = (Sigma_val + Sigma_val.T) / 2
        
        # Try OSQP first, fallback to ECOS if it fails
        try:
            self.prob.solve(solver=cp.OSQP)
        except:
            self.prob.solve(solver=cp.ECOS)
            
        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            # Final fallback: Equal Weight Portfolio (1/N)
            return np.ones(self.num_assets) / self.num_assets
            
        return self.weights.value
    
