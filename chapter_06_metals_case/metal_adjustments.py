"""
Contract Specifications, PnL Calculation, and Correlation Analysis between Metals
Chapter 6: Case Study - Gold and Silver
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def comparative_analysis_of_metals(df_eurgbp, df_xauusd, df_xagusd):
    """
    Comparative analysis between EUR/GBP, Gold and Silver.
    """
    print("=== COMPARATIVE ANALYSIS: FOREX vs METALS ===\n")
    assets = {
        'EUR/GBP': df_eurgbp,
        'Gold (XAU/USD)': df_xauusd,
        'Silver (XAG/USD)': df_xagusd
    }
    
    results = {}
    for name, df in assets.items():
        # 1. Annualized volatility
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        annual_volatility = df['log_return'].std() * np.sqrt(252 * 24 * 4) * 100
        
        # 2. Standardized ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        atr_average = df['atr_pct'].mean()
        
        results[name] = {
            'annual_volatility': annual_volatility,
            'atr_average': atr_average
        }

    # Display results in table format
    print(f"{'Asset':<20} {'Volatility':>12} {'ATR%':>10}")
    print("-" * 45)
    for name, data in results.items():
        print(f"{name:<20} {data['annual_volatility']:>11.2f}% {data['atr_average']:>9.4f}%")
        
    print("\n=== INTERPRETATION ===")
    print("• Gold has 2-3x more volatility than EUR/GBP")
    print("• Silver is even more volatile than gold")
    print("• ATR% is 3-4x higher in metals than in Forex")
    return results


def adjust_metal_gaps_thresholds(df, asset='Gold'):
    """
    Adjust the gap thresholds for precious metals.
    """
    # Simulate gap calculation for the example
    df['gap'] = df['open'] - df['close'].shift(1)
    df['absolute_gap'] = df['gap'].abs()
    
    if asset == 'Gold':
        minimum_threshold = 5.0  # $5 minimum
        maximum_threshold = 25.0 # $25 maximum
        unit = 'USD'
    else:  # Silver
        minimum_threshold = 0.30 # $0.30 minimum
        maximum_threshold = 1.50 # $1.50 maximum
        unit = 'USD'

    # Filter significant gaps
    df_gaps_sig = df[df['absolute_gap'].between(minimum_threshold, maximum_threshold)]
    
    print(f"\n=== THRESHOLD ADJUSTMENT - {asset} ===\n")
    print(f"Minimum threshold: {minimum_threshold} {unit}")
    print(f"Maximum threshold: {maximum_threshold} {unit}")
    print(f"\nSignificant gaps: {len(df_gaps_sig)}")
    if len(df) > 0:
        print(f"Percentage of total: {100*len(df_gaps_sig)/len(df):.1f}%")
    print(f"Average size: {df_gaps_sig['absolute_gap'].mean():.2f} {unit}")
    
    return df_gaps_sig


def analyze_orb_timeframes(df, asset='Gold'):
    """
    Analyze which ORB timeframe works best for metals.
    """
    print(f"\n=== ORB TIMEFRAME ANALYSIS - {asset} ===\n")
    
    minutes_ranges = [15, 30, 45, 60]
    results = {}
    
    # Simulated success rates for demonstration
    if asset == 'Gold':
        simulated_rates = {15: 0.523, 30: 0.647, 45: 0.612, 60: 0.558}
    else:
        simulated_rates = {15: 0.480, 30: 0.590, 45: 0.570, 60: 0.510}
        
    print(f"{'Minutes':<10} {'Success Rate':<12}")
    print("-" * 25)
    for minutes in minutes_ranges:
        rate = simulated_rates[minutes]
        results[minutes] = rate
        print(f"{minutes:<10} {rate*100:.1f}%")
        
    best = max(results.items(), key=lambda x: x[1])
    print(f"\n  ✓  Recommended timeframe: {best[0]} minutes ({best[1]*100:.1f}%)")
    return results


def adjust_stop_multipliers(df, asset='Gold'):
    """
    Determine the optimal stop multipliers for metals.
    """
    print(f"\n=== STOP MULTIPLIERS SETTING - {asset} ===\n")
    
    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    multipliers = [1.5, 2.0, 2.5, 3.0]
    results = {}
    
    # Simulated stop-out rates for demonstration
    if asset == 'Gold':
        simulated_rates = {1.5: 0.684, 2.0: 0.452, 2.5: 0.327, 3.0: 0.241}
    else:
        simulated_rates = {1.5: 0.750, 2.0: 0.550, 2.5: 0.420, 3.0: 0.310}
        
    print(f"{'Multiplier':<15} {'% Stops Taken':<18}")
    print("-" * 35)
    for mult in multipliers:
        rate = simulated_rates[mult]
        results[mult] = rate
        print(f"{mult:<15} {rate*100:.1f}%")
        
    # Looking for a noise-induced stop loss rate of between 30-40%
    best_mult = min(results.items(), key=lambda x: abs(x[1] - 0.35))
    print(f"\n  ✓  Recommended multiplier: {best_mult[0]}x ATR")
    print(f" (Stop rate caused by noise: {best_mult[1]*100:.1f}%)")
    return results


def adjust_for_metal_costs(df_operations, asset='Gold'):
    """
    Adjust the results for transaction costs in metals.
    """
    if asset == 'Gold':
        average_spread = 0.50  # $0.50 per ounce
        commission_per_lot = 5.0 # $5 per lot
    else:  # Silver
        average_spread = 0.03  # $0.03 per ounce
        commission_per_lot = 3.0
        
    df_adjusted = df_operations.copy()
    
    # Simulate lot size for cost calculation
    df_adjusted['lot_size'] = 1.0 
    
    # Cost per operation (round trip)
    cost_spread = df_adjusted['lot_size'] * average_spread * 2 * 100
    cost_commission = df_adjusted['lot_size'] * commission_per_lot
    total_cost = cost_spread + cost_commission
    
    df_adjusted['pnl_usd_adjusted'] = df_adjusted['pnl_usd'] - total_cost
    
    pnl_total_unadjusted = df_adjusted['pnl_usd'].sum()
    total_costs = total_cost.sum()
    pnl_total_adjusted = df_adjusted['pnl_usd_adjusted'].sum()
    
    print(f"\n=== COST ADJUSTMENT - {asset} ===\n")
    print(f"Total unadjusted PnL: ${pnl_total_unadjusted:,.2f}")
    print(f"Total costs: ${total_costs:,.2f}")
    print(f"Total adjusted PnL: ${pnl_total_adjusted:,.2f}")
    if pnl_total_unadjusted != 0:
        print(f"\nImpact: {100*total_costs/pnl_total_unadjusted:.1f}% of PnL")
    return df_adjusted


def analyze_gold_silver_correlation(df_ops_gold, df_ops_silver):
    """
    Analyze the correlation between Gold and Silver results.
    """
    # Group by date to get daily PnL
    df_gold_daily = df_ops_gold.groupby(df_ops_gold['date'].dt.date)['pnl_usd'].sum()
    df_silver_daily = df_ops_silver.groupby(df_ops_silver['date'].dt.date)['pnl_usd'].sum()
    
    df_combined = pd.DataFrame({
        'gold': df_gold_daily,
        'silver': df_silver_daily
    }).fillna(0)
    
    if len(df_combined) > 1:
        correlation = df_combined.corr().loc['gold', 'silver']
    else:
        correlation = 0.85 # Default for demonstration
        
    print("\n=== GOLD-SILVER CORRELATION ===\n")
    print(f"Daily PnL correlation: {correlation:.3f}")
    if correlation > 0.7:
        print("⚠  High correlation: They do not diversify risk")
        print("Consider operating only one of the two")
    else:
        print("✓  Low correlation: Good for diversification")
        
    return correlation


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 500
    
    # EUR/GBP data
    close_eur = 0.8500 + np.cumsum(np.random.randn(n_days) * 0.0005)
    df_eurgbp = pd.DataFrame({
        'open': close_eur, 'high': close_eur + 0.001, 
        'low': close_eur - 0.001, 'close': close_eur
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Gold data
    close_gold = 1800.00 + np.cumsum(np.random.randn(n_days) * 2.5)
    df_xauusd = pd.DataFrame({
        'open': close_gold, 'high': close_gold + 5.0, 
        'low': close_gold - 5.0, 'close': close_gold
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Silver data
    close_silver = 25.00 + np.cumsum(np.random.randn(n_days) * 0.05)
    df_xagusd = pd.DataFrame({
        'open': close_silver, 'high': close_silver + 0.15, 
        'low': close_silver - 0.15, 'close': close_silver
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # 1. Comparative Analysis
    comparative_analysis_of_metals(df_eurgbp, df_xauusd, df_xagusd)
    
    # 2. Gap Thresholds
    adjust_metal_gaps_thresholds(df_xauusd, asset='Gold')
    
    # 3. ORB Timeframes
    analyze_orb_timeframes(df_xauusd, asset='Gold')
    
    # 4. Stop Multipliers
    adjust_stop_multipliers(df_xauusd, asset='Gold')
    
    # 5. Cost Adjustment
    # Create dummy operations for cost testing
    ops_gold = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
        'pnl_usd': np.random.normal(20, 30, 100)
    })
    adjust_for_metal_costs(ops_gold, asset='Gold')
    
    # 6. Correlation Analysis
    ops_silver = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
        'pnl_usd': np.random.normal(15, 25, 100)
    })
    analyze_gold_silver_correlation(ops_gold, ops_silver)
