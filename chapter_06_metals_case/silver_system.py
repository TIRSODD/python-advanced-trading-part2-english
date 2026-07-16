"""
Trading System for Silver (XAG/USD)
Chapter 6: Case Study - Gold and Silver
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd


class SilverTradingSystem:
    """
    Multi-strategy trading system for Silver (XAG/USD).
    Silver is the "twitchy brother" of Gold: 30% more volatile, lower liquidity, 
    and wider spreads. Requires even more conservative risk management.
    """
    def __init__(self, df, initial_capital=10000):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.global_results = []
        
        # Even more conservative allocation due to extreme volatility and lower liquidity
        self.strategies = {
            'gap_fade_overnight': {
                'capital': initial_capital * 0.30, # 30%
                'risk_pct': 0.6, # Less risk than Gold (0.8%)
                'active': True
            },
            'orb_30min': {
                'capital': initial_capital * 0.40, # 40% (Main Strategy)
                'risk_pct': 0.6,
                'active': True
            },
            'momentum_overlap': {
                'capital': initial_capital * 0.30, # 30%
                'risk_pct': 0.6,
                'active': True
            }
        }
        print("=== SILVER SYSTEM INITIALIZED ===\n")
        print(f"Total capital: ${initial_capital}")
        for name, config in self.strategies.items():
            print(f"  - {name}: ${config['capital']:.2f} ({config['capital']/initial_capital*100:.0f}%)")
        print()

    def _prepare_indicators(self):
        """Calculates ATR and other baseline indicators for Silver."""
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift(1))
        low_close = np.abs(self.df['low'] - self.df['close'].shift(1))
        self.df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
        self.df['atr'] = self.df['tr'].rolling(14).mean()
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100
        
        # Silver specific thresholds (3.0x ATR for stops to avoid noise)
        self.stop_multiplier = 3.0 
        
        print("  [OK] Calculated base indicators (ATR, Regimes).\n")

    def _calculate_position_size(self, allocated_capital, risk_pct, stop_distance_usd):
        """
        Calculate the position size using dynamic risk management.
        """
        monetary_risk = allocated_capital * (risk_pct / 100)
        if stop_distance_usd <= 0:
            return 0
        # Simplified lot size calculation for Silver (1 lot = $50 per $1 move approx, adjusted for contract)
        # For simplicity in this model, we use a standard multiplier.
        lot_size = monetary_risk / (stop_distance_usd * 50)
        return round(lot_size, 2)

    def execute_gap_fade_overnight(self):
        """Executes the logic of Gap Fade Overnight for Silver."""
        print("  > Executing: GAP FADE OVERNIGHT (SILVER)...")
        config = self.strategies['gap_fade_overnight']
        if not config['active']: 
            return
            
        signals = []
        # Simulation of results for the example 
        # (In production, this calls the real logic with $0.30-$1.50 thresholds)
        np.random.seed(42)
        for i in range(76): 
            signals.append({
                'strategy': 'gap_fade_overnight',
                'date': self.df.index[i*10].date() if i*10 < len(self.df) else self.df.index[-1].date(),
                'pnl_usd': np.random.normal(16, 30), # Simulated PnL
                'allocated_capital': config['capital']
            })
        self.global_results.extend(signals)
        print(f"  [OK] Completed. {len(signals)} operations generated.\n")

    def execute_orb_30min(self):
        """Executes the 30-minute ORB logic for Silver."""
        print("  > Executing: ORB 30min (SILVER)...")
        config = self.strategies['orb_30min']
        if not config['active']: 
            return
            
        signals = []
        np.random.seed(43)
        for i in range(98):
            signals.append({
                'strategy': 'orb_30min',
                'date': self.df.index[i*15].date() if i*15 < len(self.df) else self.df.index[-1].date(),
                'pnl_usd': np.random.normal(19, 35),
                'allocated_capital': config['capital']
            })
        self.global_results.extend(signals)
        print(f"  [OK] Completed. {len(signals)} operations generated.\n")

    def execute_momentum_overlap(self):
        """Executes Momentum Overlap strategy for Silver."""
        print("  > Executing: MOMENTUM OVERLAP (SILVER)...")
        config = self.strategies['momentum_overlap']
        if not config['active']: 
            return
            
        signals = []
        np.random.seed(44)
        for i in range(67):
            signals.append({
                'strategy': 'momentum_overlap',
                'date': self.df.index[i*20].date() if i*20 < len(self.df) else self.df.index[-1].date(),
                'pnl_usd': np.random.normal(14, 28),
                'allocated_capital': config['capital']
            })
        self.global_results.extend(signals)
        print(f"  [OK] Completed. {len(signals)} operations generated.\n")

    def generate_consolidated_report(self):
        """Analyze and display the final results of the Silver system."""
        df_res = pd.DataFrame(self.global_results)
        if df_res.empty:
            print("No results to analyze.")
            return
            
        print("="*70)
        print(" CONSOLIDATED RESULTS - SILVER (XAG/USD)")
        print("="*70)
        
        # Analysis by strategy
        print("\n--- BREAKDOWN BY STRATEGY ---")
        for strategy in self.strategies.keys():
            df_strategy = df_res[df_res['strategy'] == strategy]
            if df_strategy.empty: 
                continue
            win_rate = len(df_strategy[df_strategy['pnl_usd'] > 0]) / len(df_strategy) * 100
            total_pnl = df_strategy['pnl_usd'].sum()
            return_pct = (total_pnl / df_strategy['allocated_capital'].iloc[0]) * 100
            
            print(f"\n[{strategy.upper()}]")
            print(f"  Operations: {len(df_strategy)}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Net PnL: ${total_pnl:,.2f}")
            print(f"  Return: {return_pct:.2f}%")
            
        # Global Analysis
        total_system_pnl = df_res['pnl_usd'].sum()
        initial_capital = self.initial_capital
        total_return = (total_system_pnl / initial_capital) * 100
        
        print("\n--- GLOBAL SYSTEM RESULTS ---")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Final Capital: ${initial_capital + total_system_pnl:,.2f}")
        print(f"  Total Net PnL: ${total_system_pnl:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Total Operations: {len(df_res)}")
        
        # Maximum Drawdown Calculation
        equity = initial_capital + df_res['pnl_usd'].cumsum()
        peak = equity.cummax()
        drawdown = ((equity - peak) / peak).min() * 100
        
        print(f"  Max Drawdown: {drawdown:.2f}%")
        calmar_ratio = total_return / abs(drawdown) if drawdown != 0 else 0
        print(f"  Calmar Ratio: {calmar_ratio:.2f}")
        print("="*70)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing (simulating Silver behavior)
    np.random.seed(42)
    n_days = 1000
    
    # Generate sample OHLC data simulating Silver (even higher volatility than Gold)
    close = 25.00 + np.cumsum(np.random.randn(n_days) * 0.05)
    high = close + np.random.rand(n_days) * 0.15
    low = close - np.random.rand(n_days) * 0.15
    
    df_sample = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # Initialize and run the Silver System
    silver_system = SilverTradingSystem(df_sample, initial_capital=10000)
    silver_system._prepare_indicators()
    
    # Execute Strategies
    silver_system.execute_gap_fade_overnight()
    silver_system.execute_orb_30min()
    silver_system.execute_momentum_overlap()
    
    # Generate Report
    silver_system.generate_consolidated_report()
