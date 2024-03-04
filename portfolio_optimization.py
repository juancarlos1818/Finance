#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:49:34 2024

@author: diegoruiz
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize

tickers = ['SPY', 'BND', 'GLD', 'QQ Q', 'VTI']

end_date = datetime.today()

start_date = end_date - timedelta(days=5*365)

adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(tickers = ticker, start = start_date, end = end_date)
    adj_close_df[ticker] = data['Adj Close']
    
adj_close_df.shift(1)

log_returns = np.log(adj_close_df/adj_close_df.shift(1))
log_returns = log_returns.dropna()


cov_matrix = log_returns.cov()
cov_matrix = cov_matrix*252 #annualized

def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    
    return np.sqrt(variance)

def expected_return (weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252 #annualized

def sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return (weights, log_returns) - risk_free_rate)/standard_deviation(weights, cov_matrix)
    
    
risk_free_rate = 0.02

from fredapi import Fred

fred = Fred(api_key = '086b41bed50e226b3fe2cdb15c76da20')
#ten_year_treasury_rate = fred.get_series_latest_release('DGS10')/100 #diario
ten_year_treasury_rate = fred.get_series_latest_release('GS10')/100 #mensual
ten_year_treasury_rate = fred.get_series('GS10')/100


risk_free_rate = ten_year_treasury_rate.iloc[-1]
print(risk_free_rate)

def neg_sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

constraints = {'type':'eq','fun': lambda weights: np.sum(weights)-1}
bounds = [(0,0.4) for _ in range(len(tickers))]
initial_weights = np.array([1/len(tickers)]*len(tickers))

#SLSQP stands for Sequential Least Squares Quadratic Programming, which is a numerical optimization technique suitable for solving nonlinear optimization problems with constraints

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),method='SLSQP', constraints = constraints, bounds = bounds)

optimal_weights = optimized_results.x

print('Optimal Weights:')
for ticker, weight in zip(tickers, optimal_weights):
    print(f'{ticker}: {weight:.4f}')

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f'Expected Annual Return: {optimal_portfolio_return:.4f}')
print(f'Expected Volatility: {optimal_portfolio_volatility:.4f}')
print(f'Sharpe Ratio: {optimal_sharpe_ratio:.4f}')

import matplotlib.pyplot as plt

plt.figure(figsize = (10,6))
plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.show()

cov_matrix.head()

print(cov_matrix)

    






