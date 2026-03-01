import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def correlation_distance(returns):
    #Convert correlation matrix into distance matrix
    corr = returns.corr()
    dist = ((1- corr)/2) ** 0.5
    return corr, dist

def get_quasi_design(link):
    #Reorder matrix based on hierarchical clustering.
    return list(leaves_list(link))

def get_cluster_var(cov, cluster_items):
    #Calculate variance of a cluster using inverse-variance weights.
    cov_slice = cov.iloc[cluster_items, cluster_items]
    ivp = 1 / np.diag(cov_slice)
    ivp /= ivp.sum()
    cluster_var = np.dot(ivp, np.dot(cov_slice, ivp))
    return cluster_var

def get_hrp_weights(cov, sorted_ix):
    #Calculate HRP weights using recursive bisection.
    weights = pd.Series(1.0, index=sorted_ix)
    clusters = [sorted_ix]
    
    while len(clusters) > 0:
        clusters = [c[start:end] for c in clusters
                    for start, end in ((0, len(c)//2), (len(c)//2, len(c)))
                    if len(c) > 1]
        
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c0 = clusters[i]
            c1 = clusters[i + 1]
            
            var0 = get_cluster_var(cov, c0)
            var1 = get_cluster_var(cov, c1)
            
            alpha = 1 - var0 / (var0 + var1)
            weights[c0] *= alpha
            weights[c1] *= (1 - alpha) 

    return weights

def hrp(returns):
    #Full HRP pipeline: cluster, reorder , allocate
    corr , dist = correlation_distance(returns)
    dist_condensed = squareform(dist.values, checks=False)
    link = linkage(dist_condensed, method='single')
    sorted_ix = get_quasi_design(link)

    cov = returns.cov() * 252
    weights = get_hrp_weights(cov, sorted_ix)

    #Original order
    result = pd.Series(index=returns.columns)
    for i, col_idx in enumerate(sorted_ix):
        result.iloc[col_idx] = weights.iloc[i]
    
    return result

if __name__ == "__main__":
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    sys.path.insert(0, data_dir)
    
    from data_loader import download_data, clean_data, calculate_daily_returns
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    
    weights = hrp(returns)
    
    print("=== HRP Weights ===")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight*100:.1f}%")