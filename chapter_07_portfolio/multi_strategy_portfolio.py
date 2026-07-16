"""
Multi-Strategy Portfolio Construction and Management
Chapter 7: Building a Multi-Strategy Portfolio
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


class MultiStrategyPortfolio:
    """
    Manages a portfolio of multiple trading strategies across different assets.
    Implements capital allocation, correlation analysis, and risk management.
    """
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies = {}
        self.portfolio_results = []
        
        print("=== MULTI-STRATEGY PORTFOLIO INITIALIZED ===\n")
        print(f"Initial Capital: ${initial_capital:,.2f}\n")

    def add_strategy(self, name, df_results, capital_allocation, risk_per_trade=1.0):
        """
        Add a strategy to the portfolio with its historical results.
        
        Parameters:
        -----------
        name : str
            Strategy name
        df_results : pandas DataFrame
            Must contain 'pnl_usd' and 'timestamp' columns
        capital_allocation : float
            Capital allocated to this strategy (in USD)
        risk_per_trade : float
            Risk percentage per trade
        """
        self.strategies[name] = {
            'df_results': df_results.copy(),
            'capital': capital_allocation,
            'risk_per_trade': risk_per_trade,
            'active': True
        }
        print(f"  ✓ Added strategy: {name}")
        print(f"    Capital: ${capital_allocation:,.2f} ({capital_allocation/self.initial_capital*100:.0f}%)")
        print(f"    Risk per trade: {risk_per_trade}%\n")

    def calculate_portfolio_metrics(self):
        """
        Calculate key metrics for the entire portfolio.
        """
        if not self.strategies:
            print("No strategies in portfolio.")
            return None
            
        print("="*70)
        print("PORTFOLIO METRICS ANALYSIS")
        print("="*70)
        
        # Combine all strategy results
        all_results = []
        for name, strategy in self.strategies.items():
            if not strategy['active']:
                continue
            df = strategy['df_results'].copy()
            df['strategy_name'] = name
            df['capital_allocated'] = strategy['capital']
            all_results.append(df)
            
        if not all_results:
            print("No active strategies to analyze.")
            return None
            
        df_combined = pd.concat(all_results, ignore_index=True)
        
        # Calculate portfolio-level metrics
        total_pnl = df_combined['pnl_usd'].sum()
        total_return = (total_pnl / self.initial_capital) * 100
        
        # Equity curve
        df_combined = df_combined.sort_values('timestamp')
        equity = self.initial_capital + df_combined['pnl_usd'].cumsum()
        peak = equity.cummax()
        drawdown = ((equity - peak) / peak) * 100
        max_dd = drawdown.min()
        
        # Win rate
        total_trades = len(df_combined)
        winning_trades = len(df_combined[df_combined['pnl_usd'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = df_combined[df_combined['pnl_usd'] > 0]['pnl_usd'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df_combined[df_combined['pnl_usd'] < 0]['pnl_usd'].mean()) if (total_trades - winning_trades) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Calmar Ratio
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        
        print(f"\n--- PORTFOLIO SUMMARY ---")
        print(f"Total Capital: ${self.initial_capital:,.2f}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Calmar Ratio: {calmar:.2f}")
        
        return {
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar
        }

    def analyze_strategy_contribution(self):
        """
        Analyze how each strategy contributes to the portfolio.
        """
        print("\n" + "="*70)
        print("STRATEGY CONTRIBUTION ANALYSIS")
        print("="*70)
        
        contributions = []
        
        for name, strategy in self.strategies.items():
            if not strategy['active']:
                continue
                
            df = strategy['df_results']
            pnl = df['pnl_usd'].sum()
            return_pct = (pnl / strategy['capital']) * 100
            num_trades = len(df)
            win_rate = len(df[df['pnl_usd'] > 0]) / num_trades * 100 if num_trades > 0 else 0
            
            contributions.append({
                'strategy': name,
                'pnl': pnl,
                'return_pct': return_pct,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'capital': strategy['capital']
            })
            
        df_contrib = pd.DataFrame(contributions)
        
        if df_contrib.empty:
            print("No active strategies to analyze.")
            return
            
        # Sort by PnL
        df_contrib = df_contrib.sort_values('pnl', ascending=False)
        
        print(f"\n{'Strategy':<25} {'PnL':>12} {'Return':>10} {'Trades':>8} {'Win Rate':>10}")
        print("-" * 70)
        
        for _, row in df_contrib.iterrows():
            print(f"{row['strategy']:<25} ${row['pnl']:>10,.2f} {row['return_pct']:>8.2f}% {row['num_trades']:>7} {row['win_rate']:>8.1f}%")
            
        # Portfolio concentration
        total_pnl = df_contrib['pnl'].sum()
        if total_pnl != 0:
            print(f"\n--- PORTFOLIO CONCENTRATION ---")
            for _, row in df_contrib.iterrows():
                concentration = (row['pnl'] / total_pnl) * 100
                print(f"  {row['strategy']}: {concentration:.1f}% of total PnL")
                
        return df_contrib

    def generate_equity_curve(self):
        """
        Generate and display the portfolio equity curve.
        """
        print("\n" + "="*70)
        print("PORTFOLIO EQUITY CURVE")
        print("="*70)
        
        # Combine all results chronologically
        all_results = []
        for name, strategy in self.strategies.items():
            if not strategy['active']:
                continue
            df = strategy['df_results'].copy()
            all_results.append(df)
            
        if not all_results:
            print("No active strategies.")
            return None
            
        df_combined = pd.concat(all_results, ignore_index=True)
        df_combined = df_combined.sort_values('timestamp')
        
        # Calculate equity
        equity = self.initial_capital + df_combined['pnl_usd'].cumsum()
        
        # Display key points
        print(f"\nInitial Equity: ${self.initial_capital:,.2f}")
        print(f"Final Equity: ${equity.iloc[-1]:,.2f}")
        print(f"Peak Equity: ${equity.max():,.2f}")
        print(f"Lowest Equity: ${equity.min():,.2f}")
        
        # Drawdown periods
        peak = equity.cummax()
        drawdown = ((equity - peak) / peak) * 100
        in_drawdown = drawdown < -5  # Periods with >5% drawdown
        
        dd_periods = in_drawdown.sum()
        total_periods = len(equity)
        pct_in_drawdown = (dd_periods / total_periods) * 100 if total_periods > 0 else 0
        
        print(f"\nTime in Drawdown (>5%): {pct_in_drawdown:.1f}%")
        
        return equity

    def rebalance_portfolio(self, new_allocations):
        """
        Rebalance capital allocations across strategies.
        
        Parameters:
        -----------
        new_allocations : dict
            Dictionary with strategy names and new capital allocations
        """
        print("\n" + "="*70)
        print("PORTFOLIO REBALANCING")
        print("="*70)
        
        total_new = sum(new_allocations.values())
        if total_new != self.initial_capital:
            print(f"⚠  Warning: New allocations (${total_new:,.2f}) don't match initial capital (${self.initial_capital:,.2f})")
            print("Adjusting proportionally...")
            adjustment_factor = self.initial_capital / total_new
            new_allocations = {k: v * adjustment_factor for k, v in new_allocations.items()}
            
        print(f"\nPrevious Allocations:")
        for name, strategy in self.strategies.items():
            print(f"  {name}: ${strategy['capital']:,.2f} ({strategy['capital']/self.initial_capital*100:.1f}%)")
            
        print(f"\nNew Allocations:")
        for name, allocation in new_allocations.items():
            if name in self.strategies:
                self.strategies[name]['capital'] = allocation
                print(f"  {name}: ${allocation:,.2f} ({allocation/self.initial_capital*100:.1f}%)")
            else:
                print(f"  ⚠  Strategy '{name}' not found in portfolio")
                
        print("\n✓ Portfolio rebalanced successfully.")


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Strategy 1: EUR/GBP Gap Fade
    n_trades_1 = 150
    pnl_1 = np.random.normal(15, 25, n_trades_1)
    dates_1 = pd.date_range(start='2020-01-01', periods=n_trades_1, freq='D')
    df_strategy_1 = pd.DataFrame({
        'timestamp': dates_1,
        'pnl_usd': pnl_1
    })
    
    # Strategy 2: Gold ORB
    n_trades_2 = 120
    pnl_2 = np.random.normal(20, 35, n_trades_2)
    dates_2 = pd.date_range(start='2020-01-01', periods=n_trades_2, freq='D')
    df_strategy_2 = pd.DataFrame({
        'timestamp': dates_2,
        'pnl_usd': pnl_2
    })
    
    # Strategy 3: Silver Momentum
    n_trades_3 = 90
    pnl_3 = np.random.normal(14, 28, n_trades_3)
    dates_3 = pd.date_range(start='2020-01-01', periods=n_trades_3, freq='D')
    df_strategy_3 = pd.DataFrame({
        'timestamp': dates_3,
        'pnl_usd': pnl_3
    })
    
    # Initialize Portfolio
    portfolio = MultiStrategyPortfolio(initial_capital=10000)
    
    # Add Strategies
    portfolio.add_strategy('EUR/GBP Gap Fade', df_strategy_1, capital_allocation=4000, risk_per_trade=1.0)
    portfolio.add_strategy('Gold ORB 30min', df_strategy_2, capital_allocation=3500, risk_per_trade=0.8)
    portfolio.add_strategy('Silver Momentum', df_strategy_3, capital_allocation=2500, risk_per_trade=0.6)
    
    # Calculate Metrics
    portfolio.calculate_portfolio_metrics()
    
    # Analyze Contribution
    portfolio.analyze_strategy_contribution()
    
    # Generate Equity Curve
    portfolio.generate_equity_curve()
    
    # Rebalance Example
    new_allocations = {
        'EUR/GBP Gap Fade': 3500,
        'Gold ORB 30min': 4000,
        'Silver Momentum': 2500
    }
    portfolio.rebalance_portfolio(new_allocations)
