"""
Survival Filters: Circuit Breakers and Security Filters
Chapter 3: Dynamic Risk Management
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def extreme_volatility_filter(df, period_atr=14, low_threshold_pct=10, high_threshold_pct=90):
    """
    Filter 1: Extreme Volatility.
    Identifies periods where the market is either dead (very low volatility) 
    or in panic (very high volatility), which are usually bad for trading.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain 'high', 'low', 'close' columns.
    period_atr : int, default=14
        Period for ATR calculation.
    low_threshold_pct : float, default=10
        Lower percentile threshold (e.g., 10th percentile).
    high_threshold_pct : float, default=90
        Upper percentile threshold (e.g., 90th percentile).
        
    Returns:
    --------
    pandas DataFrame with added 'operable' boolean column.
    """
    df = df.copy()
    
    # Calculate ATR and normalized ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(period_atr).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Calculate thresholds
    low_threshold = df['atr_pct'].quantile(low_threshold_pct / 100)
    high_threshold = df['atr_pct'].quantile(high_threshold_pct / 100)
    
    # Mark operable periods (between the two thresholds)
    df['operable'] = (df['atr_pct'] >= low_threshold) & (df['atr_pct'] <= high_threshold)
    
    # Statistics
    total = len(df[df['atr_pct'].notna()])
    operables = df['operable'].sum()
    
    print("=== EXTREME VOLATILITY FILTER ===\n")
    print(f"Total periods analyzed: {total}")
    print(f"Operable periods: {operables} ({100*operables/total:.1f}%)")
    print(f"Filtered periods: {total - operables} ({100*(total-operables)/total:.1f}%)")
    print(f"\nLow Threshold (P{low_threshold_pct}): {low_threshold:.3f}%")
    print(f"High Threshold (P{high_threshold_pct}): {high_threshold:.3f}%")
    
    return df


def filter_news_days(df, news_dates):
    """
    Filter 2: Macro News Days.
    Eliminates or flags days with high-impact macroeconomic events 
    (like NFP, CPI, FOMC) where price behavior is unpredictable.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Historical price data with a datetime index.
    news_dates : list
        List of dates (strings 'YYYY-MM-DD' or datetime objects) with high-impact news.
        
    Returns:
    --------
    pandas DataFrame with added 'is_news_day' boolean column.
    """
    df = df.copy()
    
    # Convert news dates to datetime.date objects for easy comparison
    news_dates_obj = [pd.to_datetime(d).date() for d in news_dates]
    
    # Create the filter mask
    df['is_news_day'] = df.index.date.isin(news_dates_obj)
    
    # Statistics
    total_days = len(df.index.date.unique())
    news_days_count = df['is_news_day'].sum()
    
    print("=== MACRO NEWS DAYS FILTER ===\n")
    print(f"Total trading days: {total_days}")
    print(f"News days filtered: {news_days_count} ({100*news_days_count/total_days:.1f}%)")
    print(f"Clean trading days: {total_days - news_days_count}")
    
    return df


def filter_consecutive_losses(df_operations, max_losses=3):
    """
    Filter 3: Consecutive Losses (Circuit Breaker).
    If the strategy experiences N consecutive losses, it indicates that 
    market conditions may have changed. This filter flags or stops trading 
    until a winning trade resets the counter.
    
    Parameters:
    -----------
    df_operations : pandas DataFrame
        Must contain 'pnl_usd' and 'timestamp' columns.
    max_losses : int, default=3
        Maximum number of consecutive losses allowed before triggering the circuit breaker.
        
    Returns:
    --------
    pandas DataFrame with added 'consecutive_losses' and 'circuit_breaker_active' columns.
    """
    df_ops = df_operations.copy().sort_values('timestamp')
    
    # Identify losing trades
    df_ops['is_loser'] = df_ops['pnl_usd'] < 0
    
    consecutive_losses = 0
    circuit_breaker_active = False
    cb_flags = []
    loss_counts = []
    
    for _, row in df_ops.iterrows():
        if row['is_loser']:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
            circuit_breaker_active = False  # Reset circuit breaker on a win
            
        # Activate circuit breaker if max losses reached
        if consecutive_losses >= max_losses:
            circuit_breaker_active = True
            
        loss_counts.append(consecutive_losses)
        cb_flags.append(circuit_breaker_active)
        
    df_ops['consecutive_losses'] = loss_counts
    df_ops['circuit_breaker_active'] = cb_flags
    
    # Statistics
    total_ops = len(df_ops)
    filtered_ops = df_ops['circuit_breaker_active'].sum()
    
    print("=== CONSECUTIVE LOSSES FILTER (CIRCUIT BREAKER) ===\n")
    print(f"Total operations: {total_ops}")
    print(f"Operations filtered by Circuit Breaker: {filtered_ops} ({100*filtered_ops/total_ops:.1f}%)")
    print(f"Operations executed: {total_ops - filtered_ops}")
    print(f"Max consecutive losses allowed: {max_losses}")
    
    return df_ops


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
    
    # 1. Test Extreme Volatility Filter
    print("="*60)
    print("1. EXTREME VOLATILITY FILTER")
    print("="*60)
    df_vol = extreme_volatility_filter(df_sample, period_atr=14)
    
    # 2. Test News Days Filter
    print("\n" + "="*60)
    print("2. MACRO NEWS DAYS FILTER")
    print("="*60)
    # Simulate 10 random news days
    sample_news_dates = pd.date_range(start='2020-02-01', periods=10, freq='15D').strftime('%Y-%m-%d').tolist()
    df_news = filter_news_days(df_sample, sample_news_dates)
    
    # 3. Test Consecutive Losses Filter
    print("\n" + "="*60)
    print("3. CONSECUTIVE LOSSES FILTER")
    print("="*60)
    # Simulate a sequence of operations with some losing streaks
    pnl_data = np.random.choice([100, -80, 150, -120, -90, -110, 200, -50], size=50)
    dates_ops = pd.date_range(start='2020-01-01', periods=50, freq='D')
    
    df_ops_sample = pd.DataFrame({
        'timestamp': dates_ops,
        'pnl_usd': pnl_data
    })
    
    filter_consecutive_losses(df_ops_sample, max_losses=3)
