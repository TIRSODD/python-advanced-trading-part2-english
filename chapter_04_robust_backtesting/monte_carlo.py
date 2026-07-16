"""
Monte Carlo Simulations for Strategy Robustness
Chapter 4: Robust Backtesting and Anti-Overfitting
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def monte_carlo_bootstrap(df_operations, num_simulations=1000, initial_capital=10000):
    """
    Monte Carlo simulation using bootstrap (sampling with replacement of real returns).
    
    This method captures fat tails and real asymmetries, unlike parametric 
    Monte Carlo which assumes normality.
    
    Parameters:
    -----------
    df_operations : pandas DataFrame
        Must contain a 'pnl_usd' column with the profit/loss of each trade.
    num_simulations : int, default=1000
        Number of random scenarios to generate.
    initial_capital : float, default=10000
        Starting capital for the equity curve.
        
    Returns:
    --------
    pandas DataFrame : Summary statistics of the simulations.
    """
    print("=== MONTE CARLO WITH BOOTSTRAP ===\n")
    
    # Extract real historical returns
    real_returns = df_operations['pnl_usd'].values
    num_operations = len(real_returns)
    
    if num_operations == 0:
        print("Error: No operations found in the DataFrame.")
        return None
        
    simulation_results = []
    
    # Run simulations
    for sim in range(num_simulations):
        # Sampling with replacement of actual returns
        simulated_returns = np.random.choice(real_returns, size=num_operations, replace=True)
        
        # Calculate equity curve
        equity = initial_capital + np.cumsum(simulated_returns)
        
        # Calculate metrics for this simulation
        final_return = equity[-1]
        max_equity = np.maximum.accumulate(equity)
        drawdowns = (equity - max_equity) / max_equity * 100
        max_drawdown = drawdowns.min()
        
        simulation_results.append({
            'simulation_id': sim + 1,
            'final_return': final_return,
            'max_drawdown': max_drawdown
        })
        
    df_results = pd.DataFrame(simulation_results)
    
    # Print Summary Statistics
    print(f"Number of simulations: {num_simulations}")
    print(f"Number of operations per simulation: {num_operations}\n")
    
    print("FINAL RETURN STATISTICS:")
    print(f"  Mean: ${df_results['final_return'].mean():,.2f}")
    print(f"  Median: ${df_results['final_return'].median():,.2f}")
    print(f"  Percentile 5 (Worst 5%): ${df_results['final_return'].quantile(0.05):,.2f}")
    print(f"  Percentile 95 (Best 5%): ${df_results['final_return'].quantile(0.95):,.2f}\n")
    
    print("MAX DRAWDOWN STATISTICS:")
    print(f"  Mean: {df_results['max_drawdown'].mean():.2f}%")
    print(f"  Median: {df_results['max_drawdown'].median():.2f}%")
    print(f"  Percentile 5 (Worst 5%): {df_results['max_drawdown'].quantile(0.05):.2f}%")
    print(f"  Percentile 95 (Best 5%): {df_results['max_drawdown'].quantile(0.95):.2f}%\n")
    
    # Probability of Ruin (Losing money)
    prob_ruin = len(df_results[df_results['final_return'] < initial_capital]) / num_simulations * 100
    print(f"Probability of Ruin (Final Capital < Initial): {prob_ruin:.2f}%")
    
    return df_results


def analyze_confidence_intervals(df_results, confidence_level=0.95):
    """
    Analyze the confidence intervals for the final return and max drawdown.
    
    Parameters:
    -----------
    df_results : pandas DataFrame
        Output from monte_carlo_bootstrap.
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95%).
        
    Returns:
    --------
    dict : Confidence intervals for return and drawdown.
    """
    alpha = 1 - confidence_level
    
    lower_return = df_results['final_return'].quantile(alpha / 2)
    upper_return = df_results['final_return'].quantile(1 - alpha / 2)
    
    lower_dd = df_results['max_drawdown'].quantile(1 - alpha / 2) # Note: DD is negative
    upper_dd = df_results['max_drawdown'].quantile(alpha / 2)
    
    print(f"\n=== {int(confidence_level*100)}% CONFIDENCE INTERVALS ===\n")
    print(f"Final Return: [${lower_return:,.2f}, ${upper_return:,.2f}]")
    print(f"Max Drawdown: [{lower_dd:.2f}%, {upper_dd:.2f}%]")
    
    return {
        'return_ci': (lower_return, upper_return),
        'drawdown_ci': (lower_dd, upper_dd)
    }


# Example usage
if __name__ == "__main__":
    # Create sample data for testing (simulating a profitable strategy)
    np.random.seed(42)
    n_trades = 150
    
    # Generate a sequence of trades with a positive edge
    # 60% win rate, average win $150, average loss $100
    trades = []
    for _ in range(n_trades):
        if np.random.random() < 0.60:
            pnl = np.random.normal(150, 50) # Win
        else:
            pnl = np.random.normal(-100, 40) # Loss
        trades.append(pnl)
        
    df_sample = pd.DataFrame({
        'trade_id': range(1, n_trades + 1),
        'pnl_usd': trades
    })
    
    # Run Monte Carlo Simulation
    df_mc_results = monte_carlo_bootstrap(
        df_sample, 
        num_simulations=1000, 
        initial_capital=10000
    )
    
    # Analyze Confidence Intervals
    if df_mc_results is not None:
        analyze_confidence_intervals(df_mc_results, confidence_level=0.95)
