"""
Final Validation Checklist for Strategy Production
Chapter 4: Robust Backtesting and Anti-Overfitting
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd
from scipy import stats


def check_sample_size(df_operations, min_trades=100):
    """
    Check if the number of trades is statistically significant.
    """
    n_trades = len(df_operations)
    passed = n_trades >= min_trades
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Sample Size: {n_trades} trades (Minimum required: {min_trades})")
    return passed


def check_statistical_significance(df_operations, alpha=0.05):
    """
    Check if the average return is statistically different from zero.
    """
    returns = df_operations['pnl_usd'].values
    if len(returns) < 2:
        print("[FAIL] Statistical Significance: Not enough data to calculate.")
        return False
        
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    passed = p_value < alpha
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Statistical Significance: p-value = {p_value:.4f} (Threshold: {alpha})")
    return passed


def check_overfitting_signals(df_operations):
    """
    Detect warning signs of overfitting (too good to be true).
    """
    n_trades = len(df_operations)
    if n_trades == 0:
        return False
        
    wins = df_operations[df_operations['pnl_usd'] > 0]
    losses = df_operations[df_operations['pnl_usd'] < 0]
    
    win_rate = len(wins) / n_trades
    avg_win = wins['pnl_usd'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl_usd'].mean()) if len(losses) > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
    
    # Realistic ranges
    wr_ok = win_rate <= 0.75  # Win rate > 75% is suspicious
    pf_ok = profit_factor <= 3.0  # PF > 3.0 is suspicious
    
    passed = wr_ok and pf_ok
    status = "PASS" if passed else "FAIL"
    
    print(f"[{status}] Overfitting Check: Win Rate = {win_rate*100:.1f}%, Profit Factor = {profit_factor:.2f}")
    if not wr_ok:
        print(f"   -> Warning: Win rate is suspiciously high.")
    if not pf_ok:
        print(f"   -> Warning: Profit factor is suspiciously high.")
        
    return passed


def check_drawdown_limits(df_operations, initial_capital=10000, max_dd_limit=-0.20):
    """
    Check if the maximum drawdown is within acceptable psychological limits.
    """
    equity = initial_capital + df_operations['pnl_usd'].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    
    passed = max_dd >= max_dd_limit
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Drawdown Limit: Max DD = {max_dd*100:.2f}% (Limit: {max_dd_limit*100:.0f}%)")
    return passed


def run_final_checklist(df_operations, initial_capital=10000):
    """
    Run all validation checks and provide a final verdict.
    """
    print("="*60)
    print("FINAL PRODUCTION CHECKLIST")
    print("="*60)
    
    checks = [
        check_sample_size(df_operations),
        check_statistical_significance(df_operations),
        check_overfitting_signals(df_operations),
        check_drawdown_limits(df_operations, initial_capital)
    ]
    
    passed_count = sum(checks)
    total_count = len(checks)
    
    print("\n" + "="*60)
    print(f"FINAL VERDICT: {passed_count}/{total_count} checks passed.")
    print("="*60)
    
    if passed_count == total_count:
        print("RESULT: STRATEGY APPROVED FOR PAPER TRADING.")
        print("The strategy has passed all robust validation metrics.")
    elif passed_count >= total_count - 1:
        print("RESULT: CONDITIONAL APPROVAL.")
        print("The strategy is mostly solid but requires minor review.")
    else:
        print("RESULT: STRATEGY REJECTED.")
        print("Do not trade this strategy with real money. Review parameters.")
        
    return passed_count == total_count


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_trades = 150
    
    trades = []
    for _ in range(n_trades):
        if np.random.random() < 0.58:
            pnl = np.random.normal(120, 40)
        else:
            pnl = np.random.normal(-80, 30)
        trades.append(pnl)
        
    df_sample = pd.DataFrame({
        'trade_id': range(1, n_trades + 1),
        'pnl_usd': trades
    })
    
    # Run the checklist
    run_final_checklist(df_sample, initial_capital=10000)
