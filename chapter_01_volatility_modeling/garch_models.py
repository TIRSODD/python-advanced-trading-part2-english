"""
GARCH Models for Volatility Prediction
Chapter 1: Volatility Measurement and Modeling
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd
from arch import arch_model


def basic_garch_model(df, period=14):
    """
    Fits a basic GARCH(1,1) model to the returns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain 'close' column
    period : int
        Period for returns calculation
    
    Returns:
    --------
    arch.model.ARCHModelResult : Fitted GARCH model
    """
    # Calculate log-returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df_returns = df['log_return'].dropna() * 100  # Scale for better convergence
    
    # Fit GARCH(1,1) model
    model = arch_model(df_returns, vol='Garch', p=1, q=1, dist='normal')
    result = model.fit(disp='off')
    
    print("=== GARCH(1,1) MODEL ===\n")
    print(result.summary())
    
    # Parameter Interpretation
    params = result.params
    print("\n=== INTERPRETATION ===\n")
    print(f"Omega (ω): {params['omega']:.6f}")
    print(" → Base volatility (constant)")
    print(f"\nAlpha (α): {params['alpha[1]']:.6f}")
    print(" → Impact of recent shocks (news)")
    print(" → If α is high, volatility reacts strongly to news")
    print(f"\nBeta (β): {params['beta[1]']:.6f}")
    print(" → Persistence of volatility")
    print(" → If β is high (close to 1), volatility persists for a long time")
    print(f"\nα + β = {params['alpha[1]'] + params['beta[1]']:.6f}")
    if params['alpha[1]'] + params['beta[1]'] > 0.9:
        print(" → High persistence: volatility shocks last a long time")
    else:
        print(" → Low persistence: volatility quickly returns to normal")
    
    return result


def predict_volatility_garch(garch_result, horizon=5):
    """
    Predict the volatility for the next N periods.
    
    Parameters:
    -----------
    garch_result : arch.model.ARCHModelResult
        Fitted GARCH model
    horizon : int
        Number of periods to forecast
    
    Returns:
    --------
    numpy.array : Predicted volatility for each horizon
    """
    # Predict volatility
    predictions = garch_result.forecast(horizon=horizon)
    
    # Predicted variance
    predicted_variance = predictions.variance.iloc[-1]
    
    # Predicted volatility (square root of variance)
    predicted_volatility = np.sqrt(predicted_variance)
    
    print(f"=== VOLATILITY PREDICTION ({horizon} periods) ===\n")
    for i, vol in enumerate(predicted_volatility, 1):
        print(f"Day +{i}: Predicted volatility = {vol:.4f}%")
    
    # Annualized volatility
    annual_vol = predicted_volatility * np.sqrt(252)
    print(f"\nPredicted annualized volatility:")
    for i, vol in enumerate(annual_vol, 1):
        print(f"  Day +{i}: {vol:.2f}%")
    
    return predicted_volatility


def compare_garch_vs_historical(df, period_hv=20):
    """
    Compare GARCH's prediction with simple historical volatility.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain 'close' column
    period_hv : int
        Period for historical volatility calculation
    
    Returns:
    --------
    tuple : (GARCH error, Historical Volatility error)
    """
    # Calculate historical volatility
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['hv'] = df['log_return'].rolling(period_hv).std()
    
    # GARCH in rolling window (last 500 days)
    rolling_window = 500
    errors_garch = []
    errors_hv = []
    
    for i in range(rolling_window, len(df) - 1):
        # Training Window
        df_train = df.iloc[i-rolling_window:i]
        returns_train = df_train['log_return'].dropna() * 100
        
        if len(returns_train) < 100:
            continue
        
        try:
            # Fit GARCH
            model = arch_model(returns_train, vol='Garch', p=1, q=1, dist='normal')
            result = model.fit(disp='off')
            
            # Prediction for tomorrow
            pred = result.forecast(horizon=1)
            var_pred = pred.variance.iloc[-1].values[0]
            vol_garch_pred = np.sqrt(var_pred) / 100  # Rescale
            
            # Historical volatility
            vol_hv_pred = df.iloc[i]['hv']
            
            # Real volatility tomorrow
            real_vol = abs(df.iloc[i+1]['log_return'])
            
            # Errors
            errors_garch.append(abs(vol_garch_pred - real_vol))
            errors_hv.append(abs(vol_hv_pred - real_vol))
            
        except:
            continue
    
    # Error statistics
    mean_error_garch = np.mean(errors_garch)
    mean_error_hv = np.mean(errors_hv)
    
    print("=== COMPARISON: GARCH vs. HISTORICAL VOLATILITY ===\n")
    print(f"GARCH average error: {mean_error_garch:.6f}")
    print(f"Historical Volatility average error: {mean_error_hv:.6f}")
    print(f"\nRelative improvement: {100*(mean_error_hv - mean_error_garch)/mean_error_hv:.1f}%")
    
    if mean_error_garch < mean_error_hv:
        print("  ✓  GARCH predicts better than historical volatility")
    else:
        print("  ✗  Historical volatility predicts as well as or better than GARCH")
    
    return mean_error_garch, mean_error_hv


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 1000
    
    # Generate sample price data with volatility clustering
    returns = np.random.randn(n_days) * 0.01
    returns[0:100] *= 1.5  # High volatility period
    returns[500:600] *= 0.5  # Low volatility period
    
    prices = 100 * np.cumprod(1 + returns)
    
    df_sample = pd.DataFrame({
        'close': prices
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Run GARCH analysis
    print("="*70)
    print("GARCH MODEL ANALYSIS")
    print("="*70)
    print()
    
    # Fit model
    garch_result = basic_garch_model(df_sample)
    
    # Predict
    print()
    predict_volatility_garch(garch_result, horizon=5)
    
    # Compare with historical volatility
    print()
    print("="*70)
    compare_garch_vs_historical(df_sample, period_hv=20)
