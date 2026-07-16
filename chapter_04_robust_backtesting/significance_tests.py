"""
Statistical Significance Tests for Strategy Validation
Chapter 4: Robust Backtesting and Anti-Overfitting
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd
from scipy import stats


def test_statistical_significance(df_operations):
    """
    Test if the average return is statistically different from zero.
    
    Parameters:
    -----------
    df_operations : pandas DataFrame
        Must contain 'pnl_usd' column with profit/loss of each trade.
        
    Returns:
    --------
    dict : T-test results and confidence intervals.
    """
    print("="*60)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("="*60)
    
    returns = df_operations['pnl_usd'].values
    
    # 1. T-Test: Is the average return different from zero?
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    
    print("\n1. T-Test (Is average return ≠ 0?):")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("   ✓  Average return is statistically significant (p < 0.05)")
    else:
        print("   ✗  We cannot claim that the return is different from zero")
    
    # 2. Confidence interval of the mean return
    mean = returns.mean()
    std_error = returns.std(ddof=1) / np.sqrt(len(returns))
    ci_95 = stats.t.interval(0.95, len(returns)-1, loc=mean, scale=std_error)
    
    print(f"\n2. 95% Confidence Interval of the mean return:")
    print(f"   Mean: ${mean:.2f}")
    print(f"   Standard Error: ${std_error:.2f}")
    print(f"   95% CI: [${ci_95[0]:.2f}, ${ci_95[1]:.2f}]")
    
    if ci_95[0] > 0:
        print("   ✓  Lower limit is positive → profitable strategy with 95% confidence")
    else:
        print("   ✗  Interval includes zero → we cannot guarantee profitability")
    
    # 3. Effect Size (Cohen's d)
    cohens_d = mean / returns.std(ddof=1)
    
    print(f"\n3. Effect Size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) >= 0.8:
        print("   ✓  Large effect size")
    elif abs(cohens_d) >= 0.5:
        print("   ✓  Medium effect size")
    elif abs(cohens_d) >= 0.2:
        print("     Small effect size")
    else:
        print("   ✗  Negligible effect size")
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_return': mean,
        'ci_95': ci_95,
        'cohens_d': cohens_d,
        'is_significant': p_value < 0.05
    }


def test_win_rate_significance(df_operations, baseline_win_rate=0.50):
    """
    Test if the win rate is statistically different from a baseline (e.g., 50%).
    
    Parameters:
    -----------
    df_operations : pandas DataFrame
        Must contain 'pnl_usd' column.
    baseline_win_rate : float, default=0.50
        Baseline win rate to test against (random = 50%).
        
    Returns:
    --------
    dict : Binomial test results.
    """
    print("\n" + "="*60)
    print("WIN RATE SIGNIFICANCE TEST")
    print("="*60)
    
    n_trades = len(df_operations)
    n_wins = len(df_operations[df_operations['pnl_usd'] > 0])
    observed_win_rate = n_wins / n_trades
    
    print(f"\nTotal trades: {n_trades}")
    print(f"Wins: {n_wins}")
    print(f"Observed win rate: {observed_win_rate*100:.2f}%")
    print(f"Baseline win rate: {baseline_win_rate*100:.2f}%")
    
    # Binomial test
    p_value = stats.binomtest(n_wins, n_trades, p=baseline_win_rate, alternative='greater').pvalue
    
    print(f"\nBinomial Test (one-tailed):")
    print(f"   P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"   ✓  Win rate is significantly greater than {baseline_win_rate*100:.0f}%")
    else:
        print(f"   ✗  Win rate is not significantly different from {baseline_win_rate*100:.0f}%")
    
    # Confidence interval for win rate
    ci_low, ci_high = stats.binomtest(n_wins, n_trades).proportion_ci(confidence_level=0.95)
    
    print(f"\n95% Confidence Interval for win rate:")
    print(f"   [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    
    return {
        'observed_win_rate': observed_win_rate,
        'n_wins': n_wins,
        'n_trades': n_trades,
        'p_value': p_value,
        'ci_95': (ci_low, ci_high),
        'is_significant': p_value < 0.05
    }


def test_normality_of_returns(df_operations):
    """
    Test if returns follow a normal distribution (Shapiro-Wilk test).
    
    Parameters:
    -----------
    df_operations : pandas DataFrame
        Must contain 'pnl_usd' column.
        
    Returns:
    --------
    dict : Normality test results.
    """
    print("\n" + "="*60)
    print("NORMALITY TEST OF RETURNS")
    print("="*60)
    
    returns = df_operations['pnl_usd'].values
    
    # Shapiro-Wilk test
    stat, p_value = stats.shapiro(returns)
    
    print(f"\nShapiro-Wilk Test:")
    print(f"   Test statistic: {stat:.4f}")
    print(f"   P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("   ✗  Returns are NOT normally distributed (p < 0.05)")
        print("   → This is normal in trading: fat tails are common")
    else:
        print("   ✓  Returns appear to be normally distributed")
    
    # Skewness and Kurtosis
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis
    
    print(f"\nDistribution characteristics:")
    print(f"   Skewness: {skewness:.4f}")
    print(f"   Excess Kurtosis: {kurtosis:.4f}")
    
    if skewness > 0:
        print("   → Positive skew: more large positive returns")
    elif skewness < 0:
        print("   → Negative skew: more large negative returns")
    else:
        print("   → Symmetric distribution")
    
    if kurtosis > 0:
        print(f"   → Fat tails (leptokurtic): extreme events {kurtosis:.1f}x more likely than normal")
    elif kurtosis < 0:
        print("   → Thin tails (platykurtic): fewer extreme events than normal")
    else:
        print("   → Normal kurtosis")
    
    return {
        'shapiro_stat': stat,
        'p_value': p_value,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'is_normal': p_value >= 0.05
    }


def test_drawdown_significance(df_operations, initial_capital=10000):
    """
    Analyze if the maximum drawdown is within expected statistical limits.
    
    Parameters:
    -----------
    df_operations : pandas DataFrame
        Must contain 'pnl_usd' and 'timestamp' columns.
    initial_capital : float, default=10000
        Starting capital for equity curve calculation.
        
    Returns:
    --------
    dict : Drawdown analysis.
    """
    print("\n" + "="*60)
    print("DRAWDOWN ANALYSIS")
    print("="*60)
    
    # Calculate equity curve
    df_sorted = df_operations.sort_values('timestamp').copy()
    equity = initial_capital + df_sorted['pnl_usd'].cumsum()
    
    # Calculate drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()
    
    print(f"\nMaximum Drawdown: {max_dd:.2f}%")
    
    # Calculate expected drawdown based on returns distribution
    returns = df_sorted['pnl_usd'].values
    mean_return = returns.mean()
    std_return = returns.std()
    n_trades = len(returns)
    
    # Approximate expected max drawdown (simplified formula)
    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
    expected_dd = -abs(sharpe_ratio) * np.sqrt(n_trades) * 2  # Rough approximation
    
    print(f"Expected drawdown (approximate): {expected_dd:.2f}%")
    
    if abs(max_dd) > abs(expected_dd) * 1.5:
        print("   ⚠  Drawdown is larger than expected")
        print("   → May indicate regime change or overfitting")
    else:
        print("   ✓  Drawdown is within expected statistical limits")
    
    # Drawdown duration
    dd_series = drawdown
    in_drawdown = dd_series < 0
    
    # Count consecutive periods in drawdown
    dd_periods = in_drawdown.groupby((~in_drawdown).cumsum()).cumsum()
    max_dd_duration = dd_periods.max()
    
    print(f"\nMaximum drawdown duration: {max_dd_duration} trades")
    
    return {
        'max_drawdown': max_dd,
        'expected_drawdown': expected_dd,
        'max_duration': max_dd_duration,
        'is_within_limits': abs(max_dd) <= abs(expected_dd) * 1.5
    }


def comprehensive_significance_analysis(df_operations, initial_capital=10000):
    """
    Run all significance tests and provide a final verdict.
    
    Parameters:
    -----------
    df_operations : pandas DataFrame
        Must contain 'pnl_usd' and 'timestamp' columns.
    initial_capital : float, default=10000
        Starting capital.
        
    Returns:
    --------
    dict : Comprehensive analysis results.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*70)
    
    # Run all tests
    t_test_results = test_statistical_significance(df_operations)
    win_rate_results = test_win_rate_significance(df_operations)
    normality_results = test_normality_of_returns(df_operations)
    drawdown_results = test_drawdown_significance(df_operations, initial_capital)
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    checks_passed = 0
    total_checks = 4
    
    if t_test_results['is_significant']:
        print("✓ Statistical significance: PASSED")
        checks_passed += 1
    else:
        print("✗ Statistical significance: FAILED")
    
    if win_rate_results['is_significant']:
        print("✓ Win rate significance: PASSED")
        checks_passed += 1
    else:
        print("✗ Win rate significance: FAILED")
    
    if drawdown_results['is_within_limits']:
        print("✓ Drawdown within limits: PASSED")
        checks_passed += 1
    else:
        print("✗ Drawdown within limits: FAILED")
    
    # Sample size check
    if len(df_operations) >= 100:
        print("✓ Sample size (≥100 trades): PASSED")
        checks_passed += 1
    else:
        print(f"✗ Sample size: FAILED ({len(df_operations)} trades, need ≥100)")
    
    print("\n" + "="*70)
    if checks_passed >= 3:
        print(f"✓ STRATEGY PASSED ({checks_passed}/{total_checks} checks)")
        print("The results appear to be statistically robust")
    elif checks_passed >= 2:
        print(f"⚠ STRATEGY CONDITIONAL ({checks_passed}/{total_checks} checks)")
        print("Requires more data or validation")
    else:
        print(f"✗ STRATEGY FAILED ({checks_passed}/{total_checks} checks)")
        print("Results may be due to chance")
    print("="*70)
    
    return {
        't_test': t_test_results,
        'win_rate': win_rate_results,
        'normality': normality_results,
        'drawdown': drawdown_results,
        'checks_passed': checks_passed,
        'total_checks': total_checks,
        'is_robust': checks_passed >= 3
    }


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_trades = 150
    
    # Generate a sequence of trades with a positive edge
    # 60% win rate, average win $150, average loss $100
    trades = []
    for i in range(n_trades):
        if np.random.random() < 0.60:
            pnl = np.random.normal(150, 50)  # Win
        else:
            pnl = np.random.normal(-100, 40)  # Loss
        trades.append(pnl)
    
    df_sample = pd.DataFrame({
        'trade_id': range(1, n_trades + 1),
        'timestamp': pd.date_range(start='2020-01-01', periods=n_trades, freq='D'),
        'pnl_usd': trades
    })
    
    # Run comprehensive analysis
    comprehensive_significance_analysis(df_sample, initial_capital=10000)
