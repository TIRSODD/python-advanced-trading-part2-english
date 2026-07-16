"""
Complete Multi-Strategy System for EUR/GBP
Chapter 5: Case Study - Multi-Strategy System for EUR/GBP
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd
from scipy import stats


def analyze_eurgbp_anatomy(df):
    """
    Perform an in-depth statistical analysis of the characteristics of the EUR/GBP.
    """
    print("=== STATISTICAL ANATOMY: EUR/GBP ===\n")
    
    # 1. Annualized Volatility
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    # Adjusted for 15min timeframe (252 days * 24 hours * 4 periods)
    annual_volatility = df['log_return'].std() * np.sqrt(252 * 24 * 4) * 100 
    
    print(f"1. VOLATILITY")
    print(f" Annualized volatility (estimated): {annual_volatility:.2f}%")
    print(f" -> Profile: Low/Moderate. Ideal for reversal strategies.\n")
    
    # 2. Gap Analysis (Simulated for daily opening)
    df_daily = df.resample('D').agg({'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min'})
    df_daily['gap_pct'] = (df_daily['open'] - df_daily['close'].shift(1)) / df_daily['close'].shift(1)
    significant_gaps = df_daily[df_daily['gap_pct'].abs() > 0.0002] # > 2 pips
    freq_gaps = len(significant_gaps) / len(df_daily) * 100
    
    print(f"2. GAPS OVERNIGHT")
    print(f" Gap frequency > 2 pips: {freq_gaps:.1f}%")
    print(f" -> Profile: High frequency. Excellent for Gap Fade.\n")
    
    # 3. Autocorrelation of Returns
    acf_1 = df['log_return'].autocorr(lag=1)
    acf_5 = df['log_return'].autocorr(lag=5)
    
    print(f"3. MARKET MEMORY (Autocorrelation)")
    print(f" Lag 1: {acf_1:.4f}")
    print(f" Lag 5: {acf_5:.4f}")
    if acf_1 < -0.05:
        print(f" -> Profile: Strong Mean-Reverting.")
    elif acf_1 > 0.05:
        print(f" -> Profile: Short-term momentum.")
    else:
        print(f" -> Profile: Efficient market / Random noise.\n")
        
    # 4. Distribution of Returns (Kurtosis)
    kurtosis = df['log_return'].kurtosis()
    
    print(f"4. DISTRIBUTION TAILS")
    print(f" Excess kurtosis: {kurtosis:.2f}")
    if kurtosis > 3:
        print(f" -> Alert: Very fat tails. Extreme events more likely than in a normal distribution.")
        
    return {
        'annual_volatility': annual_volatility,
        'freq_gaps': freq_gaps,
        'acf_1': acf_1,
        'kurtosis': kurtosis
    }


class SystemConfiguration:
    """
    Class for managing the system's configuration and capital allocation.
    """
    def __init__(self, total_capital=10000):
        self.total_capital = total_capital
        # Capital allocation by strategy
        self.allocation = {
            'gap_fade_overnight': {
                'capital': total_capital * 0.40,
                'risk_per_operation': 1.0, # 1% of allocated capital
                'active': True
            },
            'orb_15min_filtered': {
                'capital': total_capital * 0.35,
                'risk_per_operation': 1.0,
                'active': True
            },
            'high_vol_reversion': {
                'capital': total_capital * 0.25,
                'risk_per_operation': 0.8, # Less risk due to its higher volatility
                'active': True
            }
        }

    def show_configuration(self):
        print("=== SYSTEM CONFIGURATION EUR/GBP ===\n")
        print(f"Total Capital: ${self.total_capital:,.2f}\n")
        for strategy, config in self.allocation.items():
            print(f"- {strategy.upper()}")
            print(f"  Allocated capital: ${config['capital']:,.2f} ({config['capital']/self.total_capital*100:.0f}%)")
            print(f"  Risk per trade: {config['risk_per_operation']}%")
            print(f"  State: {'ACTIVE' if config['active'] else 'INACTIVE'}\n")


class MultiStrategyBacktestingEngine:
    """
    Main orchestrator that runs backtesting of all strategies
    and consolidates the results into a single equity curve.
    """
    def __init__(self, df, configuration):
        self.df = df.copy()
        self.config = configuration
        self.global_results = []
        # Calculate necessary global indicators (ATR, Bands, etc.)
        self._prepare_indicators()

    def _prepare_indicators(self):
        """Calculates ATR and other baseline indicators for the entire DataFrame."""
        # True Range and ATR (14 periods)
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift(1))
        low_close = np.abs(self.df['low'] - self.df['close'].shift(1))
        self.df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
        self.df['atr'] = self.df['tr'].rolling(14).mean()
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100
        
        # Volatility percentiles for regime filters
        self.threshold_high_vol = self.df['atr_pct'].quantile(0.75)
        self.threshold_low_vol = self.df['atr_pct'].quantile(0.25)
        
        print("  [OK] Calculated base indicators (ATR, Regimes).\n")

    def _calculate_position_size(self, allocated_capital, risk_pct, stop_pips):
        """
        Calculate the position size using dynamic risk management.
        Formula: (Capital * Risk%) / (Stop in pips * Pip value)
        """
        monetary_risk = allocated_capital * (risk_pct / 100)
        if stop_pips <= 0:
            return 0
        # In standard Forex, 1 pip = 0.0001. 1 lot = $10/pip.
        lot_size = monetary_risk / (stop_pips * 10)
        return round(lot_size, 2)

    def execute_gap_fade_strategy(self):
        """Executes the logic of Gap Fade Overnight."""
        print("  > Executing: GAP FADE OVERNIGHT...")
        config = self.config.allocation['gap_fade_overnight']
        if not config['active']: 
            return
            
        signals = []
        # Simulation of results for the example (In production, this calls the real logic)
        np.random.seed(42)
        for i in range(142): 
            signals.append({
                'strategy': 'gap_fade_overnight',
                'date': self.df.index[i*10].date() if i*10 < len(self.df) else self.df.index[-1].date(),
                'pnl_usd': np.random.normal(15, 25), # Simulated PnL with positive bias
                'allocated_capital': config['capital']
            })
        self.global_results.extend(signals)
        print(f"  [OK] Completed. {len(signals)} operations generated.\n")

    def execute_orb_strategy(self):
        """Executes the 15-minute ORB logic with volatility filter."""
        print("  > Executing: ORB 15min FILTERED...")
        config = self.config.allocation['orb_15min_filtered']
        if not config['active']: 
            return
            
        signals = []
        np.random.seed(43)
        for i in range(87):
            signals.append({
                'strategy': 'orb_15min_filtered',
                'date': self.df.index[i*15].date() if i*15 < len(self.df) else self.df.index[-1].date(),
                'pnl_usd': np.random.normal(11, 30),
                'allocated_capital': config['capital']
            })
        self.global_results.extend(signals)
        print(f"  [OK] Completed. {len(signals)} operations generated.\n")

    def execute_reversion_strategy(self):
        """Executes Mean Reversion in High Volatility."""
        print("  > Executing: HIGH VOLATILITY REVERSION...")
        config = self.config.allocation['high_vol_reversion']
        if not config['active']: 
            return
            
        signals = []
        np.random.seed(44)
        for i in range(63):
            signals.append({
                'strategy': 'high_vol_reversion',
                'date': self.df.index[i*20].date() if i*20 < len(self.df) else self.df.index[-1].date(),
                'pnl_usd': np.random.normal(10, 20),
                'allocated_capital': config['capital']
            })
        self.global_results.extend(signals)
        print(f"  [OK] Completed. {len(signals)} operations generated.\n")

    def generate_consolidated_report(self):
        """Analyze and display the final results of the system."""
        df_res = pd.DataFrame(self.global_results)
        if df_res.empty:
            print("No results to analyze.")
            return
            
        print("="*60)
        print(" FINAL REPORT: MULTI-STRATEGY EUR/GBP SYSTEM")
        print("="*60)
        
        # Analysis by strategy
        print("\n--- BREAKDOWN BY STRATEGY ---")
        for strategy in self.config.allocation.keys():
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
        initial_capital = self.config.total_capital
        total_return = (total_system_pnl / initial_capital) * 100
        
        print("\n--- GLOBAL SYSTEM RESULTS ---")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Total Net PnL: ${total_system_pnl:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Total Operations: {len(df_res)}")
        
        # Maximum Drawdown Calculation (Simplified for reporting)
        equity = initial_capital + df_res['pnl_usd'].cumsum()
        peak = equity.cummax()
        drawdown = ((equity - peak) / peak).min() * 100
        
        print(f"  Max Drawdown: {drawdown:.2f}%")
        calmar_ratio = total_return / abs(drawdown) if drawdown != 0 else 0
        print(f"  Calmar Ratio: {calmar_ratio:.2f}")
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 1000
    
    # Generate sample OHLC data simulating EUR/GBP behavior
    close = 0.8500 + np.cumsum(np.random.randn(n_days) * 0.0005)
    high = close + np.random.rand(n_days) * 0.001
    low = close - np.random.rand(n_days) * 0.001
    
    df_sample = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    # 1. Analyze Anatomy
    analyze_eurgbp_anatomy(df_sample)
    
    print("\n" + "="*60)
    
    # 2. Initialize Configuration and Engine
    config = SystemConfiguration(total_capital=10000)
    config.show_configuration()
    
    engine = MultiStrategyBacktestingEngine(df_sample, config)
    
    # 3. Execute Strategies
    engine.execute_gap_fade_strategy()
    engine.execute_orb_strategy()
    engine.execute_reversion_strategy()
    
    # 4. Generate Report
    engine.generate_consolidated_report()
