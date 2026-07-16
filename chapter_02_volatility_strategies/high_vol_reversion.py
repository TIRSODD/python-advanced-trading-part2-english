"""
High Volatility Mean Reversion Strategy
Chapter 2: Volatility-Based Strategies
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR).
    
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
    df = df.copy()
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate ATR
    df['atr'] = df['true_range'].rolling(window=period).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    return df


def high_volatility_reversal_strategy(df, period_ma=50, volatility_threshold=0.75,
                                       multiplier_std=2.0, capital=10000, risk_pct=1.0):
    """
    Mean reversion strategy in a high volatility environment.
    
    This strategy uses Bollinger Bands to detect extreme deviations,
    filtering signals only when the normalized ATR confirms that we are
    in a high volatility regime.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'high', 'low', 'close'
    period_ma : int, default=50
        Period for the moving average and Bollinger Bands
    volatility_threshold : float, default=0.75
        Percentile threshold for high volatility (P75 = high volatility)
    multiplier_std : float, default=2.0
        Standard deviation multiplier for Bollinger Bands
    capital : float, default=10000
        Initial capital for position sizing
    risk_pct : float, default=1.0
        Risk percentage per trade
    
    Returns:
    --------
    pandas DataFrame with trading signals
    """
    df = df.copy()
    
    # Calculate volatility (normalized ATR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Calculate moving average and standard deviation
    df['ma'] = df['close'].rolling(period_ma).mean()
    df['std'] = df['close'].rolling(period_ma).std()
    
    # Calculate Bollinger Bands
    df['upper_band'] = df['ma'] + (multiplier_std * df['std'])
    df['lower_band'] = df['ma'] - (multiplier_std * df['std'])
    
    # Volatility thresholds (P75 = high volatility)
    high_threshold = df['atr_pct'].quantile(volatility_threshold)
    
    signals = []
    
    for i in range(period_ma, len(df)):
        # Only trade in high volatility
        if df['atr_pct'].iloc[i] < high_threshold:
            continue
        
        price = df['close'].iloc[i]
        ma = df['ma'].iloc[i]
        upper_band = df['upper_band'].iloc[i]
        lower_band = df['lower_band'].iloc[i]
        
        # Sell signal (overbought)
        if price > upper_band:
            # Entry: when the price crosses back below the upper band
            if i > 0 and df['close'].iloc[i-1] > upper_band and price <= upper_band:
                stop_loss = upper_band + (df['atr'].iloc[i] * 1.5)
                take_profit = ma  # Objective: return to the average
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'short',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': df['atr'].iloc[i],
                    'size_lots': round(lot_size, 2)
                })
        
        # Buy signal (oversold)
        elif price < lower_band:
            # Entry: when the price crosses back above the lower band
            if i > 0 and df['close'].iloc[i-1] < lower_band and price >= lower_band:
                stop_loss = lower_band - (df['atr'].iloc[i] * 1.5)
                take_profit = ma  # Objective: return to the average
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'long',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': df['atr'].iloc[i],
                    'size_lots': round(lot_size, 2)
                })
    
    df_signals = pd.DataFrame(signals)
    
    # Print summary
    print("=== HIGH VOLATILITY REVERSAL STRATEGY ===\n")
    print(f"Total signals generated: {len(df_signals)}")
    if len(df_signals) > 0:
        long_signals = len(df_signals[df_signals['direction'] == 'long'])
        short_signals = len(df_signals[df_signals['direction'] == 'short'])
        print(f"Long signals: {long_signals}")
        print(f"Short signals: {short_signals}")
        print(f"High volatility threshold (P{int(volatility_threshold*100)}): {high_threshold:.3f}%")
    
    return df_signals


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 500
    
    # Generate sample price data with volatility clustering
    returns = np.random.randn(n_days) * 0.01
    returns[0:100] *= 1.5  # High volatility period
    returns[200:300] *= 0.5  # Low volatility period
    
    prices = 100 * np.cumprod(1 + returns)
    high = prices * (1 + np.random.rand(n_days) * 0.01)
    low = prices * (1 - np.random.rand(n_days) * 0.01)
    
    df_sample = pd.DataFrame({
        'open': prices,
        'high': high,
        'low': low,
        'close': prices
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Run strategy
    signals = high_volatility_reversal_strategy(
        df_sample,
        period_ma=50,
        volatility_threshold=0.75,
        multiplier_std=2.0,
        capital=10000,
        risk_pct=1.0
    )
    
    if len(signals) > 0:
        print("\n=== First 5 signals ===")
        print(signals.head())
