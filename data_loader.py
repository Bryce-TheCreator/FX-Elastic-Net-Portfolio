# download packages in terminal: pip install pandas numpy yfinance

import pandas as pd 
import numpy as np
import yfinance as yf
from typing import List, Tuple

class ForeignExchangeDataLoader:

    """
    A production grade data pipeling for ingesting foreign exchange (FX) data and computing 
    covariance matrices and expected returns for portfolio optimization.

    Using Object Oriented Programming (OOP) principles, this class encapsulates the functionality
    for fetching FX spot prices, calculating returns, and preparing data for optimization.

    We use OOP because it allows us to create instances of the data loader for different sets of FX pairs,
    making the code more modular and reusable. Each instance can maintain its own state
    (like tickers, date range, and fetched data), which is beneficial for scenarios where we might want to
    analyze multiple self pairs or time periods simultaneously.

    Attributes:
        tickers (List[str]): A list of FX pair tickers to fetch data for.
        start_date (str): The start date for the data range.
        end_date (str): The end date for the data range.

    Methods:
        fetch_spot_data(): Fetches daily closing spot prices for the specified FX pairs.
        calculate_returns(): Calculates daily returns from the spot price data.
        get_optimization_inputs(): Computes the expected returns and covariance matrix for portfolio optimization.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.spot_data = None
        self.returns_data = None

    def fetch_spot_data(self) -> pd.DataFrame:
        """
        Fetches daily closing spot prices for the specified FX pairs.
        """
        print(f"Fetching spot data for {len(self.tickers)} pairs from {self.start_date} to {self.end_date}...")
        
        try:
            # If multiple tickers are provided, yfinance should return a multi-index DataFrame.
            data = yf.download(self.tickers, start = self.start_date, end = self.end_date)

            # We want the standard closing prices ('Close') because currencies do not issue dividends or 'split'.
            # If there is no closing price data for a given day, we will drop that row.
            self.spot_data = data['Close'].dropna()
            print(f"Successfully fetched spot data.")

            return self.spot_data
        
        except Exception as e:
            print(f"Error fetching spot data: {e}")
        

    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculates Unhedged Log Daily Returns from the spot close price data (self.fetch_spot_data()).

        Rt = ln(Pt / Pt-1) where Pt is the spot price at time t and Pt-1 is the spot price at time t-1.
        """
        if self.spot_data is None:
            raise ValueError("Spot data not available. Please fetch spot data first.")

        self.returns_data = np.log(self.spot_data).diff().dropna()
        # We can use .diff() here to calculate the difference in log prices from t to t + 1 (i.e., log(Pt) - log(Pt-1) = log(Pt / Pt-1)).
        print("Successfully calculated daily returns.")
    
        return self.returns_data
    
    def get_optimization_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the expected returns (mu) and the covariance matrix (Sigma)

        """

        if self.returns_data is None:
            self.returns_data = self.calculate_returns()

        # We will annualize the returns and covariance matrix assuming 252 trading days in a year.
        mu = self.returns_data.mean().values * 252
        Sigma = self.returns_data.cov().values * 252

        return mu, Sigma
    

# TEST EXECUTION

if __name__ == "__main__":
    # G10 currencies (EUR, JPY, GBP, AUD, CAD, CHF, NZD, SEK, NOK) against USD

    g10_tickers = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'CADUSD=X', 'CHFUSD=X', 'NZDUSD=X', 'SEKUSD=X', 'NOKUSD=X']

    loader = ForeignExchangeDataLoader(tickers=g10_tickers, start_date='2018-01-01', end_date='2023-01-01')
    loader.fetch_spot_data()


