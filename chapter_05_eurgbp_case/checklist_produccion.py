"""
Pre-Production Checklist and Reality Check
Chapter 5: Case Study - Multi-Strategy System for EUR/GBP
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


class ProductionChecklist:
    """
    Evaluates if a strategy is ready to move from backtesting to paper trading 
    and eventually to live production.
    """
    def __init__(self, df_operations, initial_capital=10000):
        self.df_ops = df_operations.copy()
        self.initial_capital = initial_capital
        self.checks_passed = 0
        self.total_checks = 5

    def check_walk_forward(self, profitable_windows_pct=70.0):
        """
        Check 1: Walk-Forward Stability.
        """
        print("[CHECK 1] Walk-Forward Stability")
        # In a real scenario, this would take the actual WFA results.
        # Here we simulate a robust result for the example.
        simulated_wfa_profitable = 80.0 
        
        passed = simulated_wfa_profitable >= profitable_windows_pct
        if passed:
            print(f"  ✓ PASSED: {simulated_wfa_profitable}% of windows were profitable (Target: >{profitable_windows_pct}%)\n")
            self.checks_passed += 1
        else:
            print(f"  ✗ FAILED: Only {simulated_wfa_profitable}% of windows were profitable.\n")
        return passed

    def check_monte_carlo(self, max_ruin_probability=5.0):
        """
        Check 2: Monte Carlo Ruin Probability.
        """
        print("[CHECK 2] Monte Carlo Ruin Probability")
        # Simulated ruin probability
        simulated_ruin_prob = 2.5 
        
        passed = simulated_ruin_prob <= max_ruin_probability
        if passed:
            print(f"  ✓ PASSED: Probability of ruin is {simulated_ruin_prob}% (Target: <{max_ruin_probability}%)\n")
            self.checks_passed += 1
        else:
            print(f"  ✗ FAILED: Probability of ruin is too high ({simulated_ruin_prob}%).\n")
        return passed

    def check_real_costs(self, spread_pips=1.2, slippage_pips=0.5, commission=3.5):
        """
        Check 3: Impact of Real Costs (Reality Tax).
        """
        print("[CHECK 3] Real Costs Impact (Reality Tax)")
        
        # Calculate gross PnL
        gross_pnl = self.df_ops['pnl_usd'].sum()
        
        # Simulate costs (assuming average 1 lot size for simplicity)
        lot_size = 1.0
        cost_per_trade = (spread_pips + slippage_pips) * 2 * 10 + commission
        total_costs = cost_per_trade * len(self.df_ops)
        
        net_pnl = gross_pnl - total_costs
        cost_impact_pct = (total_costs / gross_pnl) * 100 if gross_pnl > 0 else 0
        
        # Pass if costs don't eat more than 30% of gross profits
        passed = cost_impact_pct < 30.0 
        
        print(f"  Gross PnL: ${gross_pnl:,.2f}")
        print(f"  Estimated Costs: -${total_costs:,.2f}")
        print(f"  Net PnL: ${net_pnl:,.2f}")
        
        if passed:
            print(f"  ✓ PASSED: Costs impact is {cost_impact_pct:.1f}% (Target: <30%)\n")
            self.checks_passed += 1
        else:
            print(f"   FAILED: Costs impact is too high ({cost_impact_pct:.1f}%).\n")
        return passed

    def check_drawdown_limits(self, max_allowed_dd=-15.0):
        """
        Check 4: Maximum Drawdown within psychological limits.
        """
        print("[CHECK 4] Maximum Drawdown Limits")
        
        equity = self.initial_capital + self.df_ops['pnl_usd'].cumsum()
        peak = equity.cummax()
        drawdown = ((equity - peak) / peak) * 100
        max_dd = drawdown.min()
        
        passed = max_dd >= max_allowed_dd
        
        print(f"  Max Drawdown: {max_dd:.2f}% (Limit: {max_allowed_dd}%)")
        if passed:
            print(f"  ✓ PASSED: Drawdown is within acceptable limits.\n")
            self.checks_passed += 1
        else:
            print(f"  ✗ FAILED: Drawdown exceeds psychological limits.\n")
        return passed

    def check_infrastructure(self):
        """
        Check 5: Infrastructure and Disaster Plan.
        """
        print("[CHECK 5] Infrastructure & Disaster Plan")
        print("  Requirements:")
        print("  - [ ] VPS located in London/NY (Low latency)")
        print("  - [ ] API Error Handling (Try/Except blocks)")
        print("  - [ ] Telegram/Email Alerts for disconnections or DD > 10%")
        print("  - [ ] Written Disaster Plan (Broker margin changes, Flash crashes)")
        # This check is always "passed" if the user acknowledges it.
        self.checks_passed += 1
        print("  ✓ PASSED: Infrastructure requirements acknowledged.\n")
        return True

    def run_full_checklist(self):
        """
        Run all checks and print final verdict.
        """
        print("="*60)
        print("PRE-PRODUCTION CHECKLIST: EUR/GBP SYSTEM")
        print("="*60 + "\n")
        
        self.check_walk_forward()
        self.check_monte_carlo()
        self.check_real_costs()
        self.check_drawdown_limits()
        self.check_infrastructure()
        
        print("="*60)
        print(f"FINAL VERDICT: {self.checks_passed}/{self.total_checks} checks passed.")
        print("="*60)
        
        if self.checks_passed == self.total_checks:
            print("RESULT: SYSTEM APPROVED FOR PAPER TRADING.")
            print("Next step: Run on a demo account for at least 3 months.")
        elif self.checks_passed >= self.total_checks - 1:
            print("RESULT: CONDITIONAL APPROVAL.")
            print("Review the failed check before going live.")
        else:
            print("RESULT: SYSTEM REJECTED FOR PRODUCTION.")
            print("Do not trade with real money. Return to optimization phase.")
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_trades = 150
    
    trades = []
    for _ in range(n_trades):
        if np.random.random() < 0.62:
            pnl = np.random.normal(140, 45)
        else:
            pnl = np.random.normal(-90, 35)
        trades.append(pnl)
        
    df_sample = pd.DataFrame({
        'trade_id': range(1, n_trades + 1),
        'pnl_usd': trades
    })
    
    # Run the checklist
    checklist = ProductionChecklist(df_sample, initial_capital=10000)
    checklist.run_full_checklist()
