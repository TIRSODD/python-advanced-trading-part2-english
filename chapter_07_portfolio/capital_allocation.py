"""
Capital Allocation Methods for Multi-Strategy Portfolios
Chapter 7: Building a Multi-Strategy Portfolio
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def equal_weight_allocation(num_strategies, total_capital):
    """
    Method 1: Equal Weight Allocation.
    Distributes capital equally among all strategies, regardless of their risk/return profile.
    
    Parameters:
    -----------
    num_strategies : int
        Number of strategies in the portfolio.
    total_capital : float
        Total capital to allocate.
        
    Returns:
    --------
    dict : Allocation per strategy.
    """
    allocation_per_strategy = total_capital / num_strategies
    
    print("=== METHOD 1: EQUAL WEIGHT ALLOCATION ===\n")
    print(f"Total Capital: ${total_capital:,.2f}")
    print(f"Number of Strategies: {num_strategies}")
    print(f"Allocation per Strategy: ${allocation_per_strategy:,.2f} ({100/num_strategies:.1f}%)\n")
    
    print("Advantages:")
    print("  ✓ Simple to implement and understand")
    print("  ✓ No dependency on historical performance")
    print("  ✓ Avoids overfitting to past data\n")
    
    print("Disadvantages:")
    print("   Ignores differences in volatility between strategies")
    print("  ✗ High-volatility strategies may dominate portfolio risk")
    print("  ✗ Not optimal for risk-adjusted returns\n")
    
    return {f'strategy_{i+1}': allocation_per_strategy for i in range(num_strategies)}


def risk_parity_allocation(strategy_returns, total_capital, target_risk=0.10):
    """
    Method 2: Risk Parity Allocation.
    Allocates capital inversely proportional to each strategy's volatility,
    so each strategy contributes equally to the portfolio's total risk.
    
    Parameters:
    -----------
    strategy_returns : dict
        Dictionary with strategy names as keys and arrays of returns as values.
    total_capital : float
        Total capital to allocate.
    target_risk : float, default=0.10
        Target annualized portfolio volatility (10%).
        
    Returns:
    --------
    dict : Capital allocation per strategy.
    """
    print("=== METHOD 2: RISK PARITY ALLOCATION ===\n")
    
    # Calculate volatility for each strategy
    volatilities = {}
    for name, returns in strategy_returns.items():
        # Annualized volatility (assuming daily returns)
        vol = np.std(returns) * np.sqrt(252)
        volatilities[name] = vol
        
    # Calculate inverse volatility weights
    inverse_vols = {name: 1/vol for name, vol in volatilities.items()}
    total_inverse_vol = sum(inverse_vols.values())
    
    # Normalize weights
    weights = {name: inv_vol / total_inverse_vol for name, inv_vol in inverse_vols.items()}
    
    # Calculate allocations
    allocations = {name: weight * total_capital for name, weight in weights.items()}
    
    # Display results
    print(f"Total Capital: ${total_capital:,.2f}")
    print(f"Target Portfolio Risk: {target_risk*100:.1f}%\n")
    
    print(f"{'Strategy':<20} {'Volatility':>12} {'Weight':>10} {'Allocation':>15}")
    print("-" * 60)
    for name in volatilities.keys():
        print(f"{name:<20} {volatilities[name]*100:>10.2f}% {weights[name]*100:>8.1f}% ${allocations[name]:>12,.2f}")
        
    # Portfolio volatility (simplified, assuming zero correlation)
    portfolio_vol = np.sqrt(sum((weights[name] * volatilities[name])**2 for name in volatilities))
    print(f"\nEstimated Portfolio Volatility: {portfolio_vol*100:.2f}%")
    
    print("\nAdvantages:")
    print("  ✓ Balances risk contribution across strategies")
    print("  ✓ More stable equity curve")
    print("  ✓ Reduces dominance of high-volatility strategies\n")
    
    print("Disadvantages:")
    print("  ✗ Requires reliable volatility estimates")
    print("  ✗ May underallocate to high-return strategies")
    print("  ✗ Assumes correlations are stable\n")
    
    return allocations


def kelly_criterion_allocation(strategy_results, total_capital, fractional_kelly=0.5):
    """
    Method 3: Kelly Criterion (Fractional).
    Allocates capital based on the Kelly formula, which maximizes long-term growth.
    Uses a fractional Kelly (typically 0.5) to reduce volatility and estimation error.
    
    Parameters:
    -----------
    strategy_results : dict
        Dictionary with strategy names as keys and dicts with 'win_rate' and 'avg_win_loss_ratio'.
    total_capital : float
        Total capital to allocate.
    fractional_kelly : float, default=0.5
        Fraction of the full Kelly to use (0.5 = half Kelly).
        
    Returns:
    --------
    dict : Capital allocation per strategy.
    """
    print("=== METHOD 3: KELLY CRITERION (FRACTIONAL) ===\n")
    
    kelly_allocations = {}
    
    for name, results in strategy_results.items():
        win_rate = results['win_rate']
        win_loss_ratio = results['avg_win_loss_ratio']
        
        # Kelly formula: f* = (p*b - q) / b
        # where p = win rate, q = 1 - p, b = win/loss ratio
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        
        full_kelly = (p * b - q) / b
        
        # Apply fractional Kelly
        fractional_kelly_value = full_kelly * fractional_kelly
        
        # Ensure non-negative allocation
        kelly_allocations[name] = max(0, fractional_kelly_value)
        
    # Normalize allocations to sum to 1 (if all are positive)
    total_kelly = sum(kelly_allocations.values())
    if total_kelly > 0:
        normalized_weights = {name: kelly / total_kelly for name, kelly in kelly_allocations.items()}
    else:
        # If all Kelly values are negative or zero, use equal weight
        num_strategies = len(strategy_results)
        normalized_weights = {name: 1/num_strategies for name in strategy_results}
        
    # Calculate final allocations
    allocations = {name: weight * total_capital for name, weight in normalized_weights.items()}
    
    # Display results
    print(f"Total Capital: ${total_capital:,.2f}")
    print(f"Fractional Kelly: {fractional_kelly}\n")
    
    print(f"{'Strategy':<20} {'Win Rate':>10} {'W/L Ratio':>12} {'Full Kelly':>12} {'Allocation':>15}")
    print("-" * 75)
    for name, results in strategy_results.items():
        full_kelly = kelly_allocations[name] / fractional_kelly if fractional_kelly > 0 else 0
        print(f"{name:<20} {results['win_rate']*100:>8.1f}% {results['avg_win_loss_ratio']:>10.2f} {full_kelly*100:>10.1f}% ${allocations[name]:>12,.2f}")
        
    print("\nAdvantages:")
    print("  ✓ Maximizes long-term geometric growth")
    print("  ✓ Theoretically optimal for repeated bets")
    print("  ✓ Accounts for both win rate and payoff ratio\n")
    
    print("Disadvantages:")
    print("  ✗ Highly sensitive to parameter estimation errors")
    print("  ✗ Can lead to very aggressive allocations")
    print("  ✗ Assumes stationary win rates and payoffs\n")
    
    return allocations


def compare_allocation_methods(strategy_data, total_capital=10000):
    """
    Compare all three allocation methods side by side.
    
    Parameters:
    -----------
    strategy_data : dict
        Contains 'returns' (for risk parity) and 'results' (for Kelly).
    total_capital : float, default=10000
        Total capital to allocate.
        
    Returns:
    --------
    dict : Allocations from each method.
    """
    print("="*80)
    print("COMPARISON OF CAPITAL ALLOCATION METHODS")
    print("="*80 + "\n")
    
    num_strategies = len(strategy_data['returns'])
    
    # Method 1: Equal Weight
    equal_alloc = equal_weight_allocation(num_strategies, total_capital)
    
    print("\n" + "="*80 + "\n")
    
    # Method 2: Risk Parity
    risk_parity_alloc = risk_parity_allocation(strategy_data['returns'], total_capital)
    
    print("\n" + "="*80 + "\n")
    
    # Method 3: Kelly Criterion
    kelly_alloc = kelly_criterion_allocation(strategy_data['results'], total_capital, fractional_kelly=0.5)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    strategies = list(strategy_data['returns'].keys())
    
    print(f"\n{'Strategy':<20} {'Equal Weight':>15} {'Risk Parity':>15} {'Kelly (Half)':>15}")
    print("-" * 70)
    
    for strategy in strategies:
        equal_val = equal_alloc.get(strategy, 0)
        risk_val = risk_parity_alloc.get(strategy, 0)
        kelly_val = kelly_alloc.get(strategy, 0)
        
        print(f"{strategy:<20} ${equal_val:>12,.2f} ${risk_val:>12,.2f} ${kelly_val:>12,.2f}")
        
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nFor most traders, we recommend:")
    print("  1. Start with EQUAL WEIGHT for simplicity and robustness")
    print("  2. Transition to RISK PARITY once you have 6+ months of data")
    print("  3. Use KELLY CRITERION only if you have high confidence in your estimates")
    print("     and always use fractional Kelly (0.25 to 0.5)\n")
    
    return {
        'equal_weight': equal_alloc,
        'risk_parity': risk_parity_alloc,
        'kelly': kelly_alloc
    }


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Strategy 1: EUR/GBP Gap Fade (low volatility, moderate returns)
    returns_1 = np.random.normal(0.001, 0.008, 252)  # Daily returns
    
    # Strategy 2: Gold ORB (high volatility, higher returns)
    returns_2 = np.random.normal(0.0015, 0.015, 252)
    
    # Strategy 3: Silver Momentum (very high volatility, variable returns)
    returns_3 = np.random.normal(0.001, 0.020, 252)
    
    strategy_returns = {
        'EUR/GBP Gap Fade': returns_1,
        'Gold ORB 30min': returns_2,
        'Silver Momentum': returns_3
    }
    
    # Kelly results (estimated from historical performance)
    strategy_results = {
        'EUR/GBP Gap Fade': {
            'win_rate': 0.62,
            'avg_win_loss_ratio': 1.2
        },
        'Gold ORB 30min': {
            'win_rate': 0.58,
            'avg_win_loss_ratio': 1.5
        },
        'Silver Momentum': {
            'win_rate': 0.55,
            'avg_win_loss_ratio': 1.3
        }
    }
    
    strategy_data = {
        'returns': strategy_returns,
        'results': strategy_results
    }
    
    # Run comparison
    compare_allocation_methods(strategy_data, total_capital=10000)
