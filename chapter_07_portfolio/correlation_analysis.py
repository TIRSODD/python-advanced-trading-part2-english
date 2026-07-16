"""
Correlation Analysis and Diversification Metrics
Chapter 7: Building a Multi-Strategy Portfolio
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


def calculate_correlation_matrix(strategy_returns):
    """
    Calculate the correlation matrix between the returns of different strategies.
    
    Parameters:
    -----------
    strategy_returns : pandas DataFrame
        DataFrame where each column represents a strategy's daily returns,
        and the index is the date.
        
    Returns:
    --------
    pandas DataFrame : Correlation matrix.
    """
    print("=== CORRELATION MATRIX ===\n")
    
    # Calculate correlation matrix
    corr_matrix = strategy_returns.corr()
    
    # Print the matrix
    print(corr_matrix.round(3))
    
    return corr_matrix


def identify_highly_correlated_pairs(corr_matrix, threshold=0.70):
    """
    Identify pairs of strategies that are highly correlated (redundant).
    
    Parameters:
    -----------
    corr_matrix : pandas DataFrame
        Correlation matrix from calculate_correlation_matrix.
    threshold : float, default=0.70
        Correlation threshold above which pairs are considered highly correlated.
        
    Returns:
    --------
    list : List of tuples containing highly correlated strategy pairs.
    """
    print(f"\n=== HIGHLY CORRELATED PAIRS (Threshold > {threshold}) ===\n")
    
    highly_correlated = []
    strategies = corr_matrix.columns
    
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                pair = (strategies[i], strategies[j], corr_value)
                highly_correlated.append(pair)
                print(f"  ⚠ {strategies[i]} & {strategies[j]}: {corr_value:.3f}")
                
    if not highly_correlated:
        print("  ✓ No highly correlated pairs found. Excellent diversification!")
    else:
        print(f"\n  ⚠ Warning: {len(highly_correlated)} pair(s) are highly correlated.")
        print("  Consider removing one strategy from each pair to improve diversification.")
        
    return highly_correlated


def calculate_diversification_ratio(strategy_returns, weights):
    """
    Calculate the Diversification Ratio (DR) of the portfolio.
    DR = Weighted average of individual volatilities / Portfolio volatility.
    A DR > 1 indicates diversification benefits.
    
    Parameters:
    -----------
    strategy_returns : pandas DataFrame
        Daily returns for each strategy.
    weights : dict or pandas Series
        Portfolio weights for each strategy.
        
    Returns:
    --------
    float : Diversification Ratio.
    """
    print("\n=== DIVERSIFICATION RATIO ===\n")
    
    # Convert weights to array aligned with DataFrame columns
    w = np.array([weights.get(col, 0) for col in strategy_returns.columns])
    
    # Individual volatilities (annualized)
    vols = strategy_returns.std() * np.sqrt(252)
    
    # Portfolio volatility
    cov_matrix = strategy_returns.cov() * 252
    portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    # Weighted average of individual volatilities
    weighted_avg_vol = np.dot(w, vols)
    
    # Diversification Ratio
    dr = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
    
    print(f"Weighted Average Individual Volatility: {weighted_avg_vol*100:.2f}%")
    print(f"Portfolio Volatility: {portfolio_vol*100:.2f}%")
    print(f"Diversification Ratio (DR): {dr:.3f}")
    
    if dr > 1.0:
        print("  ✓ DR > 1.0: The portfolio benefits from diversification.")
    else:
        print("  ✗ DR <= 1.0: No diversification benefit. Strategies are likely highly correlated.")
        
    return dr


def comprehensive_portfolio_analysis(strategy_returns, weights):
    """
    Run a complete correlation and diversification analysis.
    """
    print("="*70)
    print("COMPREHENSIVE PORTFOLIO DIVERSIFICATION ANALYSIS")
    print("="*70 + "\n")
    
    # 1. Correlation Matrix
    corr_matrix = calculate_correlation_matrix(strategy_returns)
    
    # 2. Highly Correlated Pairs
    identify_highly_correlated_pairs(corr_matrix, threshold=0.70)
    
    # 3. Diversification Ratio
    calculate_diversification_ratio(strategy_returns, weights)
    
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70)
    print("\n1. Aim for correlations < 0.50 between strategies.")
    print("2. A Diversification Ratio > 1.2 is considered excellent.")
    print("3. If two strategies are highly correlated, keep the one with")
    print("   the better Sharpe Ratio or lower maximum drawdown.\n")


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 252  # 1 year of daily data
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # Generate correlated returns
    # Strategy 1 and 2 are somewhat correlated (0.6)
    # Strategy 3 is independent
    base_1 = np.random.normal(0.001, 0.010, n_days)
    base_2 = np.random.normal(0.001, 0.012, n_days)
    base_3 = np.random.normal(0.0008, 0.015, n_days)
    
    # Introduce correlation between 1 and 2
    returns_1 = base_1
    returns_2 = 0.6 * base_1 + 0.4 * base_2 
    returns_3 = base_3  # Independent
    
    strategy_returns = pd.DataFrame({
        'EUR/GBP Gap Fade': returns_1,
        'Gold ORB 30min': returns_2,
        'Silver Momentum': returns_3
    }, index=dates)
    
    # Define portfolio weights
    weights = {
        'EUR/GBP Gap Fade': 0.40,
        'Gold ORB 30min': 0.35,
        'Silver Momentum': 0.25
    }
    
    # Run analysis
    comprehensive_portfolio_analysis(strategy_returns, weights)
