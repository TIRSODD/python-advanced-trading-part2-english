"""
Dynamic Trailing Stops and Stop-Loss Management
Chapter 3: Dynamic Risk Management
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def calculate_dynamic_stop_atr(df, signals, multiplier_atr=2.0):
    """
    Calculate initial stop-loss based on current ATR.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain 'atr' column
    signals : pandas DataFrame
        Must contain 'timestamp', 'direction', 'entry_price'
    multiplier_atr : float, default=2.0
        Multiplier for ATR to determine stop distance
        
    Returns:
    --------
    pandas DataFrame with added 'stop_loss' and 'used_atr' columns
    """
    signals = signals.copy()
    stop_losses = []
    
    for _, signal in signals.iterrows():
        timestamp = signal['timestamp']
        # Find ATR at the time of the signal
        current_atr = df.loc[df.index == timestamp, 'atr'].iloc[0] if timestamp in df.index else df['atr'].iloc[-1]
        
        if pd.isna(current_atr):
            continue
            
        stop_distance = current_atr * multiplier_atr
        
        if signal['direction'] == 'long':
            stop_loss = signal['entry_price'] - stop_distance
        else:
            stop_loss = signal['entry_price'] + stop_distance
            
        adjusted_signal = signal.copy()
        adjusted_signal['stop_loss'] = stop_loss
        adjusted_signal['used_atr'] = current_atr
        stop_losses.append(adjusted_signal)
        
    return pd.DataFrame(stop_losses)


def implement_trailing_stop_dinamico(df, signal, multiplier_trailing=1.5):
    """
    Simulate a dynamic trailing stop (Chandelier Exit style) for a single trade.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Full historical data with 'high', 'low', 'close', 'atr'
    signal : dict or Series
        Must contain 'timestamp', 'entry_price', 'direction'
    multiplier_trailing : float, default=1.5
        Multiplier for ATR to trail the stop
        
    Returns:
    --------
    dict : Exit details (exit_type, exit_price, pnl_pips)
    """
    start_idx = df.index.get_loc(signal['timestamp'])
    data = df.iloc[start_idx:].copy()
    
    entry_price = signal['entry_price']
    direction = signal['direction']
    
    # Initial stop based on ATR at entry
    initial_atr = data['atr'].iloc[0]
    
    if direction == 'long':
        current_stop = entry_price - (initial_atr * multiplier_trailing)
        max_price = entry_price
    else:
        current_stop = entry_price + (initial_atr * multiplier_trailing)
        min_price = entry_price
        
    for i in range(1, len(data)):
        row = data.iloc[i]
        current_atr = row['atr']
        
        if direction == 'long':
            if row['high'] > max_price:
                max_price = row['high']
                new_stop = max_price - (current_atr * multiplier_trailing)
                current_stop = max(current_stop, new_stop) # Stop only moves up
                
            if row['low'] <= current_stop:
                pnl_pips = (current_stop - entry_price) * 10000
                return {
                    'exit_type': 'trailing_stop',
                    'exit_price': current_stop,
                    'pnl_pips': pnl_pips
                }
                
        else:  # short
            if row['low'] < min_price:
                min_price = row['low']
                new_stop = min_price + (current_atr * multiplier_trailing)
                current_stop = min(current_stop, new_stop) # Stop only moves down
                
            if row['high'] >= current_stop:
                pnl_pips = (entry_price - current_stop) * 10000
                return {
                    'exit_type': 'trailing_stop',
                    'exit_price': current_stop,
                    'pnl_pips': pnl_pips
                }
                
    # If the trade never hit the trailing stop, close at the last available price
    final_price = data['close'].iloc[-1]
    pnl_pips = (final_price - entry_price) * 10000 if direction == 'long' else (entry_price - final_price) * 10000
    
    return {
        'exit_type': 'open_expired',
        'exit_price': final_price,
        'pnl_pips': pnl_pips
    }


def compare_fixed_vs_trailing(df, num_simulations=100, fixed_stop_pips=20, multiplier_trailing=1.5):
    """
    Compare fixed stop-loss vs dynamic trailing stop using random trade simulations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Historical data with 'high', 'low', 'close', 'atr'
    num_simulations : int, default=100
        Number of random trades to simulate
    fixed_stop_pips : int, default=20
        Fixed stop loss in pips
    multiplier_trailing : float, default=1.5
        ATR multiplier for trailing stop
        
    Returns:
    --------
    dict : Comparison metrics
    """
    np.random.seed(42)
    fixed_results = []
    trailing_results = []
    
    # Randomly pick starting points for simulations
    start_indices = np.random.choice(range(50, len(df) - 50), num_simulations, replace=False)
    
    for idx in start_indices:
        entry_price = df['close'].iloc[idx]
        direction = 'long' if np.random.random() > 0.5 else 'short'
        
        # 1. Fixed Stop Simulation (Simplified: random walk until hit)
        # For demonstration, we simulate a random PnL bounded by the fixed stop
        is_winner = np.random.random() < 0.55 # 55% win rate baseline
        if is_winner:
            pnl_fixed = np.random.uniform(15, 45) # Random win in pips
        else:
            pnl_fixed = -fixed_stop_pips # Fixed loss
        fixed_results.append(pnl_fixed)
        
        # 2. Trailing Stop Simulation
        # We use the actual historical data from that point forward
        signal = {
            'timestamp': df.index[idx],
            'entry_price': entry_price,
            'direction': direction
        }
        exit_result = implement_trailing_stop_dinamic(df, signal, multiplier_trailing)
        trailing_results.append(exit_result['pnl_pips'])
        
    # Calculate metrics
    avg_fixed = np.mean(fixed_results)
    avg_trailing = np.mean(trailing_results)
    win_rate_fixed = len([x for x in fixed_results if x > 0]) / num_simulations * 100
    win_rate_trailing = len([x for x in trailing_results if x > 0]) / num_simulations * 100
    
    print("=== FIXED STOP vs DYNAMIC TRAILING STOP ===\n")
    print(f"Simulated trades: {num_simulations}")
    print(f"\n1. Fixed Stop ({fixed_stop_pips} pips):")
    print(f"   Average PnL: {avg_fixed:.2f} pips")
    print(f"   Win Rate: {win_rate_fixed:.1f}%")
    
    print(f"\n2. Dynamic Trailing Stop ({multiplier_trailing}x ATR):")
    print(f"   Average PnL: {avg_trailing:.2f} pips")
    print(f"   Win Rate: {win_rate_trailing:.1f}%")
    
    if avg_fixed != 0:
        improvement = 100 * (avg_trailing - avg_fixed) / abs(avg_fixed)
        print(f"\n   → Improvement with Trailing Stop: {improvement:.1f}% in average PnL")
    else:
        print("\n   → Improvement with Trailing Stop: N/A")
        
    return {
        'avg_fixed': avg_fixed,
        'avg_trailing': avg_trailing,
        'win_rate_fixed': win_rate_fixed,
        'win_rate_trailing': win_rate_trailing
    }


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 500
    
    # Generate sample OHLC data
    close = 1.1000 + np.cumsum(np.random.randn(n_days) * 0.001)
    high = close + np.random.rand(n_days) * 0.002
    low = close - np.random.rand(n_days) * 0.002
    
    df_sample = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Calculate ATR for the sample data
    df_sample['tr'] = np.maximum(
        df_sample['high'] - df_sample['low'],
        np.maximum(
            np.abs(df_sample['high'] - df_sample['close'].shift(1)),
            np.abs(df_sample['low'] - df_sample['close'].shift(1))
        )
    )
    df_sample['atr'] = df_sample['tr'].rolling(14).mean()
    df_sample = df_sample.dropna()
    
    print("="*60)
    print("1. DYNAMIC TRAILING STOP SIMULATION")
    print("="*60)
    compare_fixed_vs_trailing(df_sample, num_simulations=100, fixed_stop_pips=20, multiplier_trailing=1.5)
