"""
Volatility Squeeze Strategy (TTM Squeeze)
Chapter 2: Volatility-Based Strategies
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def volatility_squeeze_strategy(df, period_bb=20, multiplier_bb=2.0,
                                compression_threshold=0.10, capital=10000, risk_pct=1.0):
    """
    Volatility squeeze strategy.
    
    Detects periods of extreme compression (narrow Bollinger Bands)
    and trades the explosive breakout when the bands expand.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'high', 'low', 'close'
    period_bb : int, default=20
        Period for Bollinger Bands
    multiplier_bb : float, default=2.0
        Standard deviation multiplier for Bollinger Bands
    compression_threshold : float, default=0.10
        Percentile threshold for compression (P10 = extreme compression)
    capital : float, default=10000
        Initial capital for position sizing
    risk_pct : float, default=1.0
        Risk percentage per trade
        
    Returns:
    --------
    pandas DataFrame with trading signals
    """
    df = df.copy()
    
    # Calculate Bollinger Bands
    df['ma'] = df['close'].rolling(period_bb).mean()
    df['std'] = df['close'].rolling(period_bb).std()
    df['bb_upper'] = df['ma'] + (multiplier_bb * df['std'])
    df['bb_lower'] = df['ma'] - (multiplier_bb * df['std'])
    
    # Calculate band width (compression measurement)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma']
    
    # Calculate ATR for stop loss / position sizing
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Compression threshold (10th percentile = very narrow bands)
    threshold = df['bb_width'].quantile(compression_threshold)
    
    signals = []
    in_compression = False
    
    for i in range(period_bb, len(df)):
        bb_width = df['bb_width'].iloc[i]
        price = df['close'].iloc[i]
        bb_sup = df['bb_upper'].iloc[i]
        bb_inf = df['bb_lower'].iloc[i]
        
        # Detect start of compression
        if bb_width < threshold:
            in_compression = True
            continue
            
        # If we were in compression and now the bands are expanding
        if in_compression and bb_width >= threshold:
            # Bullish breakout
            if price > bb_sup:
                stop_loss = bb_inf # Stop at the lower band
                take_profit = price + (price - bb_inf) * 2 # Ratio 2:1
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'long',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'bb_width': bb_width,
                    'atr': df['atr'].iloc[i],
                    'size_lots': round(lot_size, 2)
                })
                in_compression = False
                
            # Bearish breakout
            elif price < bb_inf:
                stop_loss = bb_sup # Stop on the upper band
                take_profit = price - (bb_sup - price) * 2 # Ratio 2:1
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'short',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'bb_width': bb_width,
                    'atr': df['atr'].iloc[i],
                    'size_lots': round(lot_size, 2)
                })
                in_compression = False
                
    df_signals = pd.DataFrame(signals)
    
    # Print summary
    print("=== VOLATILITY SQUEEZE STRATEGY ===\n")
    print(f"Total signals generated: {len(df_signals)}")
    if len(df_signals) > 0:
        long_signals = len(df_signals[df_signals['direction'] == 'long'])
        short_signals = len(df_signals[df_signals['direction'] == 'short'])
        print(f"Long signals: {long_signals}")
        print(f"Short signals: {short_signals}")
        print(f"Compression threshold (P{int(compression_threshold*100)}): {threshold:.5f}")
        
    return df_signals


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 1000
    
    # Generate sample price data with periods of extreme compression
    returns = np.random.randn(n_days) * 0.01
    returns[200:250] *= 0.1  # Extreme compression period
    returns[600:650] *= 0.1  # Another compression period
    
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
    signals = volatility_squeeze_strategy(
        df_sample,
        period_bb=20,
        multiplier_bb=2.0,
        compression_threshold=0.10,
        capital=10000,
        risk_pct=1.0
    )
    
    if len(signals) > 0:
        print("\n=== First 5 signals ===")
        print(signals.head())
