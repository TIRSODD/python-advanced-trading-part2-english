"""
Market Regime Detection using Volatility
Chapter 1: Volatility Measurement and Modeling
Book 2: Python for Advanced Algorithmic Trading
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calculate_atr_helper(df, period=14):
    """
    Helper function to calculate ATR if not already present.
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = df['true_range'].rolling(window=period).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    return df


def detect_regimes_atr(df, period=14):
    """
    Method 1: Detects volatility regimes using fixed percentile thresholds.
    """
    df = calculate_atr_helper(df, period)
    atr_pct = df['atr_pct'].dropna()
    
    # Thresholds based on percentiles
    low_threshold = atr_pct.quantile(0.25)
    high_threshold = atr_pct.quantile(0.75)
    
    # Classify each period
    def classify_regime(atr_value):
        if atr_value < low_threshold:
            return 'low'
        elif atr_value > high_threshold:
            return 'high'
        else:
            return 'normal'
            
    df['volatility_regime'] = df['atr_pct'].apply(classify_regime)
    
    # Statistics
    print("=== DISTRIBUTION OF REGIMES (PERCENTILES) ===\n")
    counts = df['volatility_regime'].value_counts()
    total = len(df[df['volatility_regime'].notna()])
    for regime in ['low', 'normal', 'high']:
        count = counts.get(regime, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"{regime.capitalize()}: {count} periods ({pct:.1f}%)")
        
    return df


def detect_bollinger_atr_regimes(df, period_atr=14, period_bollinger=50, num_std=2):
    """
    Method 2: Detects regimes using Bollinger Bands applied to the ATR.
    """
    df = calculate_atr_helper(df, period_atr)
    
    # Bollinger Bands over ATR
    df['atr_ma'] = df['atr'].rolling(period_bollinger).mean()
    df['atr_std'] = df['atr'].rolling(period_bollinger).std()
    df['atr_upper'] = df['atr_ma'] + (num_std * df['atr_std'])
    df['atr_lower'] = df['atr_ma'] - (num_std * df['atr_std'])
    
    # Classify regimes
    def classify_regime(row):
        if pd.isna(row['atr_upper']):
            return 'unknown'
        elif row['atr'] > row['atr_upper']:
            return 'high'
        elif row['atr'] < row['atr_lower']:
            return 'low'
        else:
            return 'normal'
            
    df['volatility_regime_bb'] = df.apply(classify_regime, axis=1)
    
    # Statistics
    print("=== REGIMES WITH BOLLINGER ATR ===\n")
    counts = df['volatility_regime_bb'].value_counts()
    total = len(df[df['volatility_regime_bb'] != 'unknown'])
    for regime in ['low', 'normal', 'high']:
        count = counts.get(regime, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"{regime.capitalize()}: {count} periods ({pct:.1f}%)")
        
    return df


def detect_clustering_regimes(df, period_atr=14, num_clusters=3):
    """
    Method 3: Detects regimes using K-Means clustering on multiple volatility features.
    """
    df = calculate_atr_helper(df, period_atr)
    
    # Features for clustering
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_5'] = df['log_return'].rolling(5).std()
    df['vol_20'] = df['log_return'].rolling(20).std()
    df['vol_50'] = df['log_return'].rolling(50).std()
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # Select features and remove NaNs
    features = ['atr', 'vol_5', 'vol_20', 'vol_50', 'range']
    df_features = df[features].dropna().copy()
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(df_features)
    
    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_features['cluster'] = kmeans.fit_predict(normalized_features)
    
    # Analyze clusters
    print("=== VOLATILITY CLUSTER ANALYSIS ===\n")
    for cluster in range(num_clusters):
        subset = df_features[df_features['cluster'] == cluster]
        print(f"Cluster {cluster + 1}:")
        print(f"  Periods: {len(subset)} ({100*len(subset)/len(df_features):.1f}%)")
        print(f"  ATR mean: {subset['atr'].mean():.5f}")
        print(f"  Average volatility 5: {subset['vol_5'].mean():.5f}")
        print(f"  Average volatility 20: {subset['vol_20'].mean():.5f}")
        print(f"  Mean range: {subset['range'].mean()*100:.3f}%\n")
    
    # Assign qualitative labels based on ATR mean
    cluster_stats = []
    for cluster in range(num_clusters):
        subset = df_features[df_features['cluster'] == cluster]
        cluster_stats.append({
            'cluster': cluster,
            'atr_mean': subset['atr'].mean()
        })
    
    cluster_stats = sorted(cluster_stats, key=lambda x: x['atr_mean'])
    labels = {
        cluster_stats[0]['cluster']: 'low',
        cluster_stats[1]['cluster']: 'normal',
        cluster_stats[2]['cluster']: 'high'
    }
    
    df_features['regime'] = df_features['cluster'].map(labels)
    
    # Merge with original df
    df = df.join(df_features['regime'], how='left')
    
    return df


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
    high = prices * (1 + np.random.rand(n_days) * 0.01)
    low = prices * (1 - np.random.rand(n_days) * 0.01)
    
    df_sample = pd.DataFrame({
        'open': prices,
        'high': high,
        'low': low,
        'close': prices
    }, index=pd.date_range(start='2020-01-01', periods=n_days, freq='D'))
    
    print("="*70)
    print("REGIME DETECTION METHODS")
    print("="*70)
    print()
    
    # Method 1
    df_res1 = detect_regimes_atr(df_sample.copy(), period=14)
    print()
    
    # Method 2
    df_res2 = detect_bollinger_atr_regimes(df_sample.copy(), period_atr=14, period_bollinger=50)
    print()
    
    # Method 3
    df_res3 = detect_clustering_regimes(df_sample.copy(), period_atr=14, num_clusters=3)
