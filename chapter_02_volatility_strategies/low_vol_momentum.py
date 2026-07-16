"""
Low Volatility Momentum Strategy
Chapter 2: Volatility-Based Strategies
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def low_volatility_momentum_strategy(df, period_ma=50, volatility_threshold=0.25,
                                     breakdown_period=20, capital=10000, risk_pct=1.0):
    """
    Momentum strategy in a low volatility environment.
    
    Detects consolidation ranges using moving highs and lows, 
    and trades breakouts filtered by a trend moving average.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'high', 'low', 'close'
    period_ma : int, default=50
        Period for the moving average
    volatility_threshold : float, default=0.25
        Percentile threshold for low volatility (P25 = low volatility)
    breakdown_period : int, default=20
        Period for calculating the consolidation range
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
    
    # Calculate moving average for trend filter
    df['ma'] = df['close'].rolling(period_ma).mean()
    
    # Calculate consolidation range (rolling high and low)
    df['max_consolidation'] = df['high'].rolling(breakdown_period).max()
    df['min_consolidation'] = df['low'].rolling(breakdown_period).min()
    
    # Volatility thresholds (P25 = low volatility)
    low_threshold = df['atr_pct'].quantile(volatility_threshold)
    
    signals = []
    
    for i in range(max(period_ma, breakdown_period), len(df)):
        # Only trade in low volatility
        if df['atr_pct'].iloc[i] > low_threshold:
            continue
            
        price = df['close'].iloc[i]
        max_cons = df['max_consolidation'].iloc[i-1]
        min_cons = df['min_consolidation'].iloc[i-1]
        ma = df['ma'].iloc[i]
        
        # Bullish breakout from consolidation range
        if price > max_cons:
            # Confirmation: price above the moving average
            if price > ma:
                stop_loss = min_cons  # Stop at the minimum of the range
                take_profit = price + (price - min_cons) * 2  # Ratio 2:1
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
                
        # Bearish breakout from consolidation range
        elif price < min_cons:
            # Confirmation: price below the moving average
            if price < ma:
                stop_loss = max_cons  # Stop at the maximum of the range
                take_profit = price - (max_cons - price) * 2  # Ratio 2:1
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
                
    df_signals = pd.DataFrame(signals)
    
    # Print summary
    print("=== LOW VOLATILITY MOMENTUM STRATEGY ===\n")
    print(f"Total signals generated: {len(df_signals)}")
    if len(df_signals) > 0:
        long_signals = len(df_signals[df_signals['direction'] == 'long'])
        short_signals = len(df_signals[df_signals['direction'] == 'short'])
        print(f"Long signals: {long_signals}")
        print(f"Short signals: {short_signals}")
        print(f"Low volatility threshold (P{int(volatility_threshold*100)}): {low_threshold:.3f}%")
        
    return df_signals


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 500
    
    # Generate sample price data with periods of consolidation
    returns = np.random.randn(n_days) * 0.01
    returns[100:200] *= 0.3  # Low volatility period (consolidation)
    returns[300:400] *= 0.3  # Another low volatility period
    
    prices = 100 * np.cumprod(1 + returns)
    high = prices * (1 + np.random.rand(n_days) * 0.005)
    low = prices * (1 - np.random.rand(n_days) * 0.005)
    
    df_sample = pd.DataFrame({
        'open': prices,
        'high': high,
        'low': low,
        'close': prices
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Run strategy
    signals = low_volatility_momentum_strategy(
        df_sample,
        period_ma=50,
        volatility_threshold=0.25,
        breakdown_period=20,
        capital=10000,
        risk_pct=1.0
    )
    
    if len(signals) > 0:
        print("\n=== First 5 signals ===")
        print(signals.head())
