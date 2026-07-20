"""
Parameter Optimization and Advanced Filters
Chapter 5: Case Study - Multi-Strategy System for EUR/GBP
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def apply_circuit_breaker(df_results, max_consecutive_losses=3):
    """
    Temporarily stops the strategy after N consecutive losses.
    Filters out trades that occur during a losing streak > max_consecutive_losses.
    """
    df_res = df_results.copy()
    df_res['consecutive_losses'] = 0
    counter = 0
    
    for i in range(len(df_res)):
        if df_res.iloc[i]['pnl_usd'] < 0:
            counter += 1
        else:
            counter = 0
        df_res.loc[df_res.index[i], 'consecutive_losses'] = counter
        
    # Filter operations that occurred during the "punishment" streak
    df_res['valid_operation'] = df_res['consecutive_losses'] <= max_consecutive_losses
    filtered_ops = len(df_res[~df_res['valid_operation']])
    
    print(f"  ⚠  Circuit Breaker activated: {filtered_ops} trades avoided after losing streaks.")
    return df_res[df_res['valid_operation']]


def filter_news_blackout(df_signals, high_impact_news_dates):
    """
    Eliminates signals that occur during critical news time windows.
    """
    # Convert dates to datetime objects for comparison
    blackout_dates = pd.to_datetime(high_impact_news_dates).date
    
    # Create filter mask
    mask = ~df_signals['date'].isin(blackout_dates)
    filtered_signals = df_signals[mask]
    
    print(f"  ⚠  News Blackout Filter: {len(df_signals) - len(filtered_signals)} operations removed due to macro event risk.")
    return filtered_signals


def adjust_for_real_costs(df_results, spread_pips=1.2, slippage_pips=0.5, lot_commission=3.5):
    """
    Apply the actual transaction costs to the backtest results (Reality Tax).
    """
    fixed_cost_per_pip = 10  # $10 per pip in standard lot
    
    # Cost per operation (Spread + Slippage are paid on entry and exit)
    variable_cost_pips = (spread_pips + slippage_pips) * 2
    
    df_adjusted = df_results.copy()
    
    # Assuming a standard lot size of 1.0 for simplicity in this calculation
    # In a real scenario, you would use the actual lot size per trade
    lot_size = 1.0 
    
    df_adjusted['total_cost_usd'] = (lot_size * variable_cost_pips * fixed_cost_per_pip) + \
                                    (lot_size * lot_commission)
                                    
    df_adjusted['pnl_usd_real'] = df_adjusted['pnl_usd'] - df_adjusted['total_cost_usd']
    
    pnl_gross = df_adjusted['pnl_usd'].sum()
    total_costs = df_adjusted['total_cost_usd'].sum()
    net_pnl = df_adjusted['pnl_usd_real'].sum()
    
    print(f"  💰  REAL COST ANALYSIS")
    print(f" Gross PnL (Backtest): ${pnl_gross:,.2f}")
    print(f" Total Costs (Spread/Comm/Slip): -${total_costs:,.2f}")
    print(f" Net PnL (Reality): ${net_pnl:,.2f}")
    if pnl_gross != 0:
        print(f" Impact on return: -{(total_costs/pnl_gross)*100:.1f}%\n")
    else:
        print(f" Impact on return: N/A\n")
        
    return df_adjusted


def optimize_strategy_parameters(df_data, func_strategy, param_grid):
    """
    Simple Grid Search to find the best parameters for a strategy.
    Warning: Always validate with out-of-sample data to avoid overfitting!
    """
    print("=== GRID SEARCH OPTIMIZATION ===\n")
    
    best_return = -999999
    best_params = None
    total_combinations = 1
    for key, values in param_grid.items():
        total_combinations *= len(values)
        
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    # Note: This is a simplified loop. In production, you would run the full backtest here.
    for ma_period in param_grid.get('ma_period', [50]):
        for atr_mult in param_grid.get('atr_mult', [2.0]):
            # Simulate backtest result (Placeholder)
            simulated_return = np.random.uniform(10, 40) 
            
            if simulated_return > best_return:
                best_return = simulated_return
                best_params = {'ma_period': ma_period, 'atr_mult': atr_mult}
                
    print(f"\n=== BEST PARAMETERS ===")
    print(f" Parameters: {best_params}")
    print(f" Simulated Return: {best_return:.2f}%")
    
    return best_params


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_trades = 50
    
    pnl_data = np.random.normal(15, 25, n_trades)
    dates_ops = pd.date_range(start='2023-01-01', periods=n_trades, freq='D')
    
    df_sample = pd.DataFrame({
        'date': dates_ops.date,
        'pnl_usd': pnl_data,
        'lot_size': 1.0
    })
    
    print("="*60)
    print("1. CIRCUIT BREAKER TEST")
    print("="*60)
    df_filtered_cb = apply_circuit_breaker(df_sample.copy(), max_consecutive_losses=3)
    
    print("\n" + "="*60)
    print("2. NEWS BLACKOUT TEST")
    print("="*60)
    news_dates = ['2023-01-10', '2023-01-25']
    df_filtered_news = filter_news_blackout(df_sample.copy(), news_dates)
    
    print("\n" + "="*60)
    print("3. REAL COSTS ADJUSTMENT")
    print("="*60)
    df_real_costs = adjust_for_real_costs(df_sample.copy())
