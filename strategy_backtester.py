# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 21:23:53 2025

Reddit Ticker Backtesting and Performance Analysis

This script runs a backtest on weekly ticker selections, implements the specified
trading strategy, and calculates key performance metrics against benchmarks (SPY, QQQ).

@author: Zheng_Wang
"""

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from pathlib import Path

# === Configuration ===
# ❗ IMPORTANT: Update this with the path to your detailed weekly data file.
# This file should be inside the 'reddit_weekly_data' folder.
SELECTIONS_CSV_PATH = "reddit_weekly_data/weekly_top5_tickers_detailed_20250626_0458.csv" # <-- UPDATE THIS FILENAME

# --- Backtest Parameters ---
INITIAL_CAPITAL = 100_000
BENCHMARKS = ['SPY', 'QQQ']

# === Helper Functions ===

def get_trading_dates(week_start, week_end):
    """Gets the first and last trading day within a calendar week."""
    # Create a daily date range for the given week
    date_range = pd.to_datetime(pd.date_range(start=week_start, end=week_end))
    
    # Fetch SPY data for the week to identify actual trading days
    spy_data = yf.download('SPY', start=date_range.min() - pd.Timedelta(days=3), end=date_range.max() + pd.Timedelta(days=3), progress=False)
    
    # Filter for dates that were actual trading days
    trading_days_in_week = spy_data[spy_data.index.isin(date_range)]
    
    if not trading_days_in_week.empty:
        return trading_days_in_week.index.min(), trading_days_in_week.index.max()
    return None, None

def calculate_metrics(returns_df, risk_free_rate=0.0):
    """Calculates all requested performance metrics."""
    metrics = {}
    weeks_per_year = 52

    # Weekly Return: Mean and Volatility
    metrics['mean_weekly_return'] = returns_df.mean()
    metrics['weekly_volatility'] = returns_df.std()

    # Cumulative Return
    cumulative_returns = (1 + returns_df).cumprod()
    metrics['cumulative_return'] = cumulative_returns

    # CAGR (Compounded Annual Growth Rate)
    total_return = cumulative_returns.iloc[-1]
    years = len(returns_df) / weeks_per_year
    metrics['cagr'] = (total_return ** (1 / years)) - 1

    # Sharpe Ratio (Annualized)
    sharpe = (metrics['mean_weekly_return'] - (risk_free_rate / weeks_per_year)) / metrics['weekly_volatility']
    metrics['sharpe_ratio'] = sharpe * np.sqrt(weeks_per_year)

    # Max Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    metrics['max_drawdown'] = drawdown.min()

    return metrics

# === Main Execution ===

def run_backtest():
    """Main function to run the backtest and generate the analysis."""
    print("🚀 Starting Backtest and Performance Analysis...")
    
    # --- 1. Load Data ---
    selections_path = Path(SELECTIONS_CSV_PATH)
    if not selections_path.exists():
        print(f"❌ ERROR: File not found at '{SELECTIONS_CSV_PATH}'.")
        print("Please update the 'SELECTIONS_CSV_PATH' variable with the correct filename.")
        return

    df = pd.read_csv(selections_path)
    df['week_start'] = pd.to_datetime(df['week_start'])
    df['week_end'] = pd.to_datetime(df['week_end'])
    
    weekly_groups = df.groupby('week_start')
    
    print(f"📈 Found {len(weekly_groups)} weeks of ticker selections to process.")

    # --- 2. Run Trading Simulation ---
    portfolio_weekly_returns = []
    processed_weeks = []

    for week_start, group in weekly_groups:
        week_end = group['week_end'].iloc[0]
        tickers = group['ticker'].tolist()
        
        buy_date, sell_date = get_trading_dates(week_start, week_end)
        
        if not buy_date or not sell_date or buy_date == sell_date:
            continue # Skip weeks with no valid trading days
            
        # Fetch data for the tickers for the week
        data = yf.download(tickers, start=buy_date, end=sell_date + pd.Timedelta(days=1), progress=False)
        
        if data.empty or 'Open' not in data.columns or 'Close' not in data.columns:
            continue

        # Get prices for the trading rule
        open_prices = data['Open'].loc[buy_date]
        close_prices = data['Close'].loc[sell_date]
        
        # Handle cases where some tickers might not have data for the specific day
        valid_tickers = open_prices.dropna().index.intersection(close_prices.dropna().index)
        if valid_tickers.empty:
            continue
            
        open_prices = open_prices[valid_tickers]
        close_prices = close_prices[valid_tickers]
        
        # Calculate weekly return for each stock and average for the portfolio
        weekly_ticker_returns = (close_prices - open_prices) / open_prices
        week_portfolio_return = weekly_ticker_returns.mean()
        
        portfolio_weekly_returns.append(week_portfolio_return)
        processed_weeks.append(week_start)

    portfolio_returns = pd.Series(portfolio_weekly_returns, index=pd.to_datetime(processed_weeks)).rename("Portfolio")
    
    print("✅ Trading simulation complete.")

    # --- 3. Fetch Benchmark Data ---
    start_date = portfolio_returns.index.min()
    end_date = portfolio_returns.index.max() + pd.Timedelta(days=7) # Add buffer
    
    benchmark_data = yf.download(BENCHMARKS, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    benchmark_returns = benchmark_data.resample('W-MON').ffill().pct_change().dropna()
    
    # Align data
    analysis_df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    
    print("✅ Benchmark data fetched and aligned.")
    
    # --- 4. Calculate Performance Metrics ---
    portfolio_metrics = calculate_metrics(analysis_df['Portfolio'])
    spy_metrics = calculate_metrics(analysis_df['SPY'])
    qqq_metrics = calculate_metrics(analysis_df['QQQ'])

    # Win Rate vs. Benchmarks
    win_rate_spy = (analysis_df['Portfolio'] > analysis_df['SPY']).mean()
    win_rate_qqq = (analysis_df['Portfolio'] > analysis_df['QQQ']).mean()
    
    print("✅ Performance metrics calculated.")

    # --- 5. Display Results ---
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS RESULTS")
    print("="*50)
    
    results_summary = pd.DataFrame({
        'Portfolio': {
            'CAGR': f"{portfolio_metrics['cagr']:.2%}",
            'Sharpe Ratio': f"{portfolio_metrics['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{portfolio_metrics['max_drawdown']:.2%}",
            'Weekly Volatility': f"{portfolio_metrics['weekly_volatility']:.2%}",
            'Win Rate vs SPY': f"{win_rate_spy:.2%}",
            'Win Rate vs QQQ': f"{win_rate_qqq:.2%}",
        },
        'SPY': {
            'CAGR': f"{spy_metrics['cagr']:.2%}",
            'Sharpe Ratio': f"{spy_metrics['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{spy_metrics['max_drawdown']:.2%}",
            'Weekly Volatility': f"{spy_metrics['weekly_volatility']:.2%}",
            'Win Rate vs SPY': '---',
            'Win Rate vs QQQ': '---',
        },
        'QQQ': {
            'CAGR': f"{qqq_metrics['cagr']:.2%}",
            'Sharpe Ratio': f"{qqq_metrics['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{qqq_metrics['max_drawdown']:.2%}",
            'Weekly Volatility': f"{qqq_metrics['weekly_volatility']:.2%}",
            'Win Rate vs SPY': '---',
            'Win Rate vs QQQ': '---',
        }
    })
    
    print(results_summary.to_string())
    print("\n" + "="*50)

    # --- 6. Plot Cumulative Returns ($100 Growth) ---
    style.use('seaborn-v0_8-darkgrid')
    growth_df = 100 * portfolio_metrics['cumulative_return']
    spy_growth = 100 * spy_metrics['cumulative_return']
    qqq_growth = 100 * qqq_metrics['cumulative_return']

    plt.figure(figsize=(14, 8))
    plt.plot(growth_df.index, growth_df, label='Reddit Portfolio', linewidth=2.5)
    plt.plot(spy_growth.index, spy_growth, label='SPY (S&P 500)', linestyle='--')
    plt.plot(qqq_growth.index, qqq_growth, label='QQQ (Nasdaq-100)', linestyle='--')
    
    plt.title('Growth of $100 Investment', fontsize=18)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(fontsize=12)
    plt.figtext(0.1, 0.02, f"Analysis based on selections from '{selections_path.name}'", ha="left", fontsize=8, color='gray')
    plt.show()

    # --- 7. Conclusion ---
    print("\nCONCLUSION:")
    portfolio_cagr = portfolio_metrics['cagr']
    spy_cagr = spy_metrics['cagr']
    qqq_cagr = qqq_metrics['cagr']
    
    portfolio_sharpe = portfolio_metrics['sharpe_ratio']
    spy_sharpe = spy_metrics['sharpe_ratio']
    qqq_sharpe = qqq_metrics['sharpe_ratio']
    
    conclusion_parts = []
    
    # Compare by CAGR
    if portfolio_cagr > spy_cagr and portfolio_cagr > qqq_cagr:
        conclusion_parts.append(f"The portfolio's CAGR of {portfolio_cagr:.2%} outperformed both SPY ({spy_cagr:.2%}) and QQQ ({qqq_cagr:.2%}).")
    elif portfolio_cagr > spy_cagr:
        conclusion_parts.append(f"The portfolio's CAGR of {portfolio_cagr:.2%} outperformed SPY ({spy_cagr:.2%}) but underperformed QQQ ({qqq_cagr:.2%}).")
    elif portfolio_cagr > qqq_cagr:
        conclusion_parts.append(f"The portfolio's CAGR of {portfolio_cagr:.2%} outperformed QQQ ({qqq_cagr:.2%}) but underperformed SPY ({spy_cagr:.2%}).")
    else:
        conclusion_parts.append(f"The portfolio's CAGR of {portfolio_cagr:.2%} underperformed both SPY ({spy_cagr:.2%}) and QQQ ({qqq_cagr:.2%}).")

    # Compare by Sharpe Ratio
    if portfolio_sharpe > spy_sharpe and portfolio_sharpe > qqq_sharpe:
        conclusion_parts.append(f"On a risk-adjusted basis, its Sharpe Ratio of {portfolio_sharpe:.2f} was superior to SPY ({spy_sharpe:.2f}) and QQQ ({qqq_sharpe:.2f}).")
    else:
        conclusion_parts.append(f"However, its risk-adjusted return (Sharpe Ratio: {portfolio_sharpe:.2f}) did not beat both benchmarks.")
        
    print(" ".join(conclusion_parts))
    print("="*50 + "\n")


if __name__ == "__main__":
    run_backtest()