"""
Walk-Forward Analysis for Strategy Validation
Chapter 4: Robust Backtesting and Anti-Overfitting
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def optimize_parameters(df_train, func_strategy):
    """
    Placeholder function to simulate parameter optimization on the training window.
    In a real scenario, this would run a grid search or genetic algorithm.
    
    Parameters:
    -----------
    df_train : pandas DataFrame
        Training data window.
    func_strategy : callable
        The strategy function to optimize.
        
    Returns:
    --------
    dict : Best parameters found.
    """
    # For demonstration, we return a fixed set of "optimized" parameters
    # In reality, you would test combinations and return the best one.
    return {
        'period_ma': 50,
        'volatility_threshold': 0.75,
        'multiplier_std': 2.0
    }


def walk_forward_analysis(df, func_strategy, training_period=252, 
                          test_period=63, step=21):
    """
    Run Walk-Forward Analysis to validate strategy robustness and detect overfitting.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Full historical data with OHLCV.
    func_strategy : callable
        Function that generates trading signals. Must accept df and **kwargs.
    training_period : int, default=252
        Number of days/periods for the in-sample training window (e.g., 1 year).
    test_period : int, default=63
        Number of days/periods for the out-of-sample testing window (e.g., 3 months).
    step : int, default=21
        Number of days/periods to shift the window forward (e.g., 1 month).
        
    Returns:
    --------
    pandas DataFrame : Results for each walk-forward window.
    """
    print("="*60)
    print("WALK-FORWARD ANALYSIS")
    print("="*60)
    print(f"Training period: {training_period} periods")
    print(f"Test period: {test_period} periods")
    print(f"Step size: {step} periods\n")
    
    results = []
    window_num = 0
    
    # Iterate through windows
    max_idx = len(df) - training_period - test_period
    
    for start in range(0, max_idx, step):
        window_num += 1
        
        # Define windows
        end_training = start + training_period
        start_test = end_training
        end_test = start_test + test_period
        
        if end_test > len(df):
            break
            
        # Training data (In-Sample)
        df_train = df.iloc[start:end_training]
        
        # Test data (Out-of-Sample)
        df_test = df.iloc[start_test:end_test]
        
        # 1. Optimize parameters on training data
        best_params = optimize_parameters(df_train, func_strategy)
        
        # 2. Generate signals on out-of-sample test data
        # Note: In a real implementation, func_strategy would return a DataFrame of trades
        # Here we simulate the PnL for demonstration purposes.
        signals_test = func_strategy(df_test, **best_params)
        
        if signals_test is None or len(signals_test) == 0:
            # If no signals, record a zero PnL window
            results.append({
                'window': window_num,
                'start_date': df_test.index[0],
                'end_date': df_test.index[-1],
                'num_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'parameters': best_params
            })
            continue
            
        # Calculate metrics for the test window
        total_pnl = signals_test['pnl_usd'].sum()
        num_trades = len(signals_test)
        win_rate = len(signals_test[signals_test['pnl_usd'] > 0]) / num_trades * 100 if num_trades > 0 else 0
        
        results.append({
            'window': window_num,
            'start_date': df_test.index[0],
            'end_date': df_test.index[-1],
            'num_trades': num_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'parameters': best_params
        })
        
        print(f"Window {window_num}: {num_trades} trades, "
              f"PnL ${total_pnl:.2f}, Win Rate {win_rate:.1f}%")
              
    df_results = pd.DataFrame(results)
    
    # Final Statistics
    print("\n" + "="*60)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("="*60)
    
    if len(df_results) > 0:
        profitable_windows = len(df_results[df_results['total_pnl'] > 0])
        total_windows = len(df_results)
        pct_profitable = 100 * profitable_windows / total_windows
        
        print(f"Total windows: {total_windows}")
        print(f"Profitable windows: {profitable_windows} ({pct_profitable:.1f}%)")
        print(f"Accumulated total PnL: ${df_results['total_pnl'].sum():.2f}")
        print(f"Average PnL per window: ${df_results['total_pnl'].mean():.2f}")
        
        if pct_profitable >= 70:
            print("\n✓ STRATEGY IS ROBUST: >70% of windows are profitable.")
        else:
            print("\n⚠ WARNING: Strategy may be overfitted or unstable.")
    else:
        print("No windows were generated. Check data length and parameters.")
        
    return df_results


# --- Dummy Strategy Function for Testing ---
def dummy_strategy(df, period_ma=50, volatility_threshold=0.75, multiplier_std=2.0):
    """
    A dummy strategy function to test the Walk-Forward engine.
    Generates random trades with a slight positive bias.
    """
    # Simulate generating some trades
    num_trades = np.random.randint(5, 15)
    trades = []
    
    for _ in range(num_trades):
        # 60% chance of winning
        is_win = np.random.random() < 0.60
        pnl = np.random.uniform(50, 200) if is_win else np.random.uniform(-100, -50)
        trades.append({'pnl_usd': pnl})
        
    return pd.DataFrame(trades)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 1000
    
    # Generate sample price data
    close = 1.1000 + np.cumsum(np.random.randn(n_days) * 0.001)
    high = close + np.random.rand(n_days) * 0.002
    low = close - np.random.rand(n_days) * 0.002
    
    df_sample = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Run Walk-Forward Analysis
    walk_forward_analysis(
        df=df_sample,
        func_strategy=dummy_strategy,
        training_period=252,
        test_period=63,
        step=21
    )
