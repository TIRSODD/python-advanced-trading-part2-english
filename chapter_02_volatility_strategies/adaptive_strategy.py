"""
Adaptive Volatility Strategy
Chapter 2: Volatility-Based Strategies
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def adaptive_volatility_strategy(df, period_ma=50, period_bb=20, multiplier_bb=2.0,
                                 capital=10000, risk_pct=1.0):
    """
    Adaptive strategy that changes logic according to the detected volatility regime.
    
    - High Volatility (ATR > P75): Mean Reversion
    - Low Volatility (ATR < P25): Momentum / Breakout
    - Volatility Squeeze (BB Width < P10): Compression Breakout
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'high', 'low', 'close'
    period_ma : int, default=50
        Period for the moving average
    period_bb : int, default=20
        Period for Bollinger Bands
    multiplier_bb : float, default=2.0
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
    
    # 1. Calculate Volatility (ATR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # 2. Calculate Bollinger Bands
    df['ma'] = df['close'].rolling(period_ma).mean()
    df['std'] = df['close'].rolling(period_bb).std()
    df['bb_upper'] = df['ma'] + (multiplier_bb * df['std'])
    df['bb_lower'] = df['ma'] - (multiplier_bb * df['std'])
    
    # 3. Calculate Band Width (for Squeeze detection)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma']
    
    # 4. Calculate Regime Thresholds
    high_threshold = df['atr_pct'].quantile(0.75)
    low_threshold = df['atr_pct'].quantile(0.25)
    squeeze_threshold = df['bb_width'].quantile(0.10)
    
    signals = []
    in_squeeze = False
    
    # Start loop after the maximum period to ensure all indicators are ready
    start_idx = max(period_ma, period_bb, 14)
    
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        atr_pct = df['atr_pct'].iloc[i]
        bb_width = df['bb_width'].iloc[i]
        ma = df['ma'].iloc[i]
        bb_sup = df['bb_upper'].iloc[i]
        bb_inf = df['bb_lower'].iloc[i]
        atr = df['atr'].iloc[i]
        
        # --- REGIME 1: VOLATILITY SQUEEZE (Priority) ---
        if bb_width < squeeze_threshold:
            in_squeeze = True
            continue
            
        if in_squeeze and bb_width >= squeeze_threshold:
            # Bullish breakout from squeeze
            if price > bb_sup:
                stop_loss = bb_inf
                take_profit = price + (price - bb_inf) * 2
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'long',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'regime': 'squeeze',
                    'size_lots': round(lot_size, 2)
                })
                in_squeeze = False
                continue
                
            # Bearish breakout from squeeze
            elif price < bb_inf:
                stop_loss = bb_sup
                take_profit = price - (bb_sup - price) * 2
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'short',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'regime': 'squeeze',
                    'size_lots': round(lot_size, 2)
                })
                in_squeeze = False
                continue
        
        # --- REGIME 2: HIGH VOLATILITY (Mean Reversion) ---
        if atr_pct > high_threshold:
            # Sell signal (overbought)
            if price > bb_sup:
                if i > 0 and df['close'].iloc[i-1] > bb_sup and price <= bb_sup:
                    stop_loss = bb_sup + (atr * 1.5)
                    take_profit = ma
                    risk_pips = abs(stop_loss - price) * 10000
                    monetary_risk = capital * (risk_pct / 100)
                    lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                    
                    signals.append({
                        'timestamp': df.index[i],
                        'direction': 'short',
                        'entry_price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'regime': 'high_vol',
                        'size_lots': round(lot_size, 2)
                    })
            
            # Buy signal (oversold)
            elif price < bb_inf:
                if i > 0 and df['close'].iloc[i-1] < bb_inf and price >= bb_inf:
                    stop_loss = bb_inf - (atr * 1.5)
                    take_profit = ma
                    risk_pips = abs(stop_loss - price) * 10000
                    monetary_risk = capital * (risk_pct / 100)
                    lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                    
                    signals.append({
                        'timestamp': df.index[i],
                        'direction': 'long',
                        'entry_price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'regime': 'high_vol',
                        'size_lots': round(lot_size, 2)
                    })
                    
        # --- REGIME 3: LOW VOLATILITY (Momentum) ---
        elif atr_pct < low_threshold:
            # Calculate consolidation range (last 20 periods)
            max_cons = df['high'].iloc[i-20:i].max()
            min_cons = df['low'].iloc[i-20:i].min()
            
            # Bullish breakout
            if price > max_cons and price > ma:
                stop_loss = min_cons
                take_profit = price + (price - min_cons) * 2
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'long',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'regime': 'low_vol',
                    'size_lots': round(lot_size, 2)
                })
                
            # Bearish breakout
            elif price < min_cons and price < ma:
                stop_loss = max_cons
                take_profit = price - (max_cons - price) * 2
                risk_pips = abs(stop_loss - price) * 10000
                monetary_risk = capital * (risk_pct / 100)
                lot_size = monetary_risk / (risk_pips * 10) if risk_pips > 0 else 0
                
                signals.append({
                    'timestamp': df.index[i],
                    'direction': 'short',
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'regime': 'low_vol',
                    'size_lots': round(lot_size, 2)
                })
                
    df_signals = pd.DataFrame(signals)
    
    # Print summary
    print("=== ADAPTIVE VOLATILITY STRATEGY ===\n")
    print(f"Total signals generated: {len(df_signals)}")
    if len(df_signals) > 0:
        regime_counts = df_signals['regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} signals ({100*count/len(df_signals):.1f}%)")
        print(f"\nHigh Volatility Threshold (P75): {high_threshold:.3f}%")
        print(f"Low Volatility Threshold (P25): {low_threshold:.3f}%")
        print(f"Squeeze Threshold (P10): {squeeze_threshold:.5f}")
        
    return df_signals


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 1000
    
    # Generate sample price data with different volatility regimes
    returns = np.random.randn(n_days) * 0.01
    returns[0:100] *= 1.5   # High volatility period
    returns[200:300] *= 0.3 # Low volatility period (consolidation)
    returns[500:550] *= 0.1 # Extreme compression (squeeze)
    
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
    signals = adaptive_volatility_strategy(
        df_sample,
        period_ma=50,
        period_bb=20,
        multiplier_bb=2.0,
        capital=10000,
        risk_pct=1.0
    )
    
    if len(signals) > 0:
        print("\n=== First 5 signals ===")
        print(signals.head())
