"""
ATR (Average True Range) Calculation and Analysis
Chapter 1: Volatility Measurement and Modeling
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def calculate_true_range(df):
    """
    Calculate the True Range for each candle.
    
    True Range is the maximum of:
    1. High - Low (current candle range)
    2. |High - Previous Close| (bullish gap)
    3. |Low - Previous Close| (bearish gap)
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'high', 'low', 'close'
    
    Returns:
    --------
    pandas DataFrame with added 'true_range' column
    """
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Calculate the three components of True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    # True Range is the maximum of the three
    df['true_range'] = np.maximum(high_low, np.maximum(high_close, low_close))
    
    return df


def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR).
    
    The ATR is the moving average of the True Range,
    typically using 14 periods.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'high', 'low', 'close'
    period : int, default=14
        Period for the moving average
    
    Returns:
    --------
    pandas DataFrame with added 'atr' and 'atr_pct' columns
    """
    # First calculate True Range
    df = calculate_true_range(df)
    
    # Calculate ATR as simple moving average of True Range
    df['atr'] = df['true_range'].rolling(window=period).mean()
    
    # ATR as percentage of price (for comparison across assets)
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    return df


def analyze_normalized_atr(df, period=14):
    """
    Analyze the normalized ATR to compare volatility across assets.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'high', 'low', 'close'
    period : int, default=14
        Period for ATR calculation
    
    Returns:
    --------
    pandas DataFrame with ATR analysis
    """
    # Calculate ATR
    df = calculate_atr(df, period)
    
    # Get ATR percentage (drop NaN values)
    atr_pct = df['atr_pct'].dropna()
    
    print("=== NORMALIZED ATR ANALYSIS ===\n")
    print(f"ATR% mean: {atr_pct.mean():.3f}%")
    print(f"ATR% median: {atr_pct.median():.3f}%")
    print(f"ATR% minimum: {atr_pct.min():.3f}%")
    print(f"ATR% maximum: {atr_pct.max():.3f}%")
    print(f"ATR% standard deviation: {atr_pct.std():.3f}%\n")
    
    # Calculate percentiles
    print("ATR% percentiles:")
    for p in [10, 25, 50, 75, 90]:
        value = atr_pct.quantile(p / 100)
        print(f"  P{p}: {value:.3f}%")
    
    # Classify volatility regimes based on percentiles
    p25 = atr_pct.quantile(0.25)
    p75 = atr_pct.quantile(0.75)
    
    print(f"\n=== VOLATILITY REGIMES ===")
    print(f"Low volatility (ATR% < {p25:.3f}%): {len(atr_pct[atr_pct < p25])} periods")
    print(f"Normal volatility (P25-P75): {len(atr_pct[(atr_pct >= p25) & (atr_pct <= p75)])} periods")
    print(f"High volatility (ATR% > {p75:.3f}%): {len(atr_pct[atr_pct > p75])} periods")
    
    return df


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLC data
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.random.rand(100) * 2
    low = close - np.random.rand(100) * 2
    open_price = close.shift(1).fillna(close)
    
    df_sample = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }, index=dates)
    
    # Run analysis
    df_result = analyze_normalized_atr(df_sample, period=14)
    
    print("\n=== First 10 rows of ATR data ===")
    print(df_result[['close', 'true_range', 'atr', 'atr_pct']].head(10))
