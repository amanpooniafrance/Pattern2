
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.spatial.distance import jensenshannon, pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATA ACQUISITION
# ==============================================================================
# Mathematical objective: Retrieve daily closing prices.
# Statistical interpretation: Prices form a continuous random walk, but we will model returns.
# Why: We need raw price data for EURUSD to compute returns and Markov states.
# Output interpretation: A pandas Series or DataFrame of raw closing prices.
def get_data():
    print("Downloading data...")
    data = yf.download('EURUSD=X', start='2003-12-01', end='2025-12-31')
    if data.empty:
        raise RuntimeError("Failed to download data.")
    close = data['Close'].copy()
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return close

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================
# Mathematical objective: Transform prices into log returns, and map to 3 discrete states {-1, 0, 1}.
# Statistical interpretation: Log returns approximate continuous percentage changes. By using the 30th
#                             and 70th percentiles, we assign the lowest 30% of returns to -1 (Bearish),
#                             the middle 40% to 0 (Neutral), and the highest 30% to +1 (Bullish).
# Why: This normalizes volatility and converts the continuous return into a discrete Markov state.
# Output interpretation: A pandas Series of integers (-1, 0, 1) representing the market regime for each period.
def prepare_data(close):
    print("Preparing data...")
    log_ret = np.log(close / close.shift(1))
    log_ret = log_ret.dropna()

    q30 = log_ret.quantile(0.3)
    q70 = log_ret.quantile(0.7)

    states = pd.Series(index=log_ret.index, dtype=int)
    states[log_ret < q30] = -1
    states[(log_ret >= q30) & (log_ret <= q70)] = 0
    states[log_ret > q70] = 1
    return states

# ==============================================================================
# 3. STATE CONSTRUCTION & ORDERING
# ==============================================================================
# Mathematical objective: Construct an M5 Markov transition space where X = [s1, s2, s3, s4, s5] and Y = state(t).
# Statistical interpretation: The current state Y depends on the sequence of the 5 previous states.
# Why: We want to capture path-dependent dynamics over a 5-step lookback window.
# Output interpretation: A DataFrame of historical M5 patterns mapped to their forward 1-step outcomes.
def construct_states(states, lookback=5):
    df = pd.DataFrame({'State': states})
    df['s1'] = df['State'].shift(5)
    df['s2'] = df['State'].shift(4)
    df['s3'] = df['State'].shift(3)
    df['s4'] = df['State'].shift(2)
    df['s5'] = df['State'].shift(1)
    df['Y'] = df['State']

    df = df.dropna().astype(int)

    order = []
    for s5 in [-1,0,1]:
        for s4 in [-1,0,1]:
            for s3 in [-1,0,1]:
                for s2 in [-1,0,1]:
                    for s1 in [-1,0,1]:
                        order.append((s1,s2,s3,s4,s5))
    return df, order

# ==============================================================================
# 4. TRANSITION TABLE & STATISTICS
# ==============================================================================
# Mathematical objective: Compute raw counts and empirical probabilities P(-1), P(0), P(+1) for each M5 state.
# Statistical interpretation: Calculate MLE (Maximum Likelihood Estimates) of transition probabilities.
#                             Compute Direction, Error, Gini, Cross-Entropy, Confidence using ISLR definitions.
# Why: To understand the natural unconditional drift of each M5 configuration.
# Output interpretation: A CSV file mapping all 243 possible M5 states to their transition probabilities and metrics.
def compute_transition_table(df, order):
    print("Computing transition table...")
    grouped = df.groupby(['s1', 's2', 's3', 's4', 's5'])['Y'].value_counts().unstack(fill_value=0)
    for col in [-1, 0, 1]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped = grouped[[-1, 0, 1]]

    grouped = grouped.reindex(order, fill_value=0)

    N_Obs = grouped.sum(axis=1)
    P_neg1 = grouped[-1] / N_Obs
    P_0 = grouped[0] / N_Obs
    P_pos1 = grouped[1] / N_Obs

    P_neg1 = P_neg1.fillna(0)
    P_0 = P_0.fillna(0)
    P_pos1 = P_pos1.fillna(0)

    Direction = P_pos1 - P_neg1
    Confidence = np.maximum(np.maximum(P_neg1, P_0), P_pos1)
    Class_Error = 1 - Confidence
    Gini = P_neg1*(1-P_neg1) + P_0*(1-P_0) + P_pos1*(1-P_pos1)

    def safe_log(p):
        return np.log(p) if p > 0 else 0

    Cross_Entropy = -(P_neg1.apply(safe_log)*P_neg1 + P_0.apply(safe_log)*P_0 + P_pos1.apply(safe_log)*P_pos1)

    transition_table = pd.DataFrame({
        'N_Observations': N_Obs,
        'P(-1)': P_neg1,
        'P(0)': P_0,
        'P(+1)': P_pos1,
        'Direction': Direction,
        'Classification_Error': Class_Error,
        'Gini': Gini,
        'Cross_Entropy': Cross_Entropy,
        'Confidence': Confidence
    })
    transition_table.to_csv('state_transition_table.csv')
    return transition_table

# ==============================================================================
# 5. METHODS 1 & 2: DECISION TREES
# ==============================================================================
# Mathematical objective: Partition the M5 state space using recursive binary splits.
# Statistical interpretation: Gini measures variance impurity. Cross-Entropy measures information gain.
#                             Trees group states that result in similar outcomes Y.
# Why: To aggregate sparsely populated M5 states into dense clusters with higher statistical significance.
# Output interpretation: Optimal tree models for Gini and Entropy criteria, chosen via CV accuracy.
def tune_and_fit_trees(df):
    X = df[['s1', 's2', 's3', 's4', 's5']]
    Y = df['Y']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def tune_tree(criterion):
        best_depth = 1
        best_score = 0
        for depth in range(1, 11):
            tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
            score = cross_val_score(tree, X, Y, cv=cv, scoring='accuracy').mean()
            if score > best_score:
                best_score = score
                best_depth = depth
        return best_depth, best_score

    print("Tuning Gini Tree...")
    gini_depth, gini_acc = tune_tree('gini')
    gini_tree = DecisionTreeClassifier(criterion='gini', max_depth=gini_depth, random_state=42)
    gini_tree.fit(X, Y)

    print("Tuning Entropy Tree...")
    entropy_depth, entropy_acc = tune_tree('entropy')
    entropy_tree = DecisionTreeClassifier(criterion='entropy', max_depth=entropy_depth, random_state=42)
    entropy_tree.fit(X, Y)
    
    return gini_tree, gini_depth, gini_acc, entropy_tree, entropy_depth, entropy_acc, X

# ==============================================================================
# 6. TREE CLUSTERS RE-ESTIMATION
# ==============================================================================
# Mathematical objective: Re-estimate transition probabilities P(-1), P(0), P(+1) by pooling observations in leaf nodes.
# Statistical interpretation: Avoids Simpson's Paradox by combining the raw counts rather than averaging probabilities.
# Why: Leaf nodes represent groups of M5 states with similar dynamics. Pooling improves estimation robustness.
# Output interpretation: Summary table of the clustered state space ordered by Direction (Most Bearish to Most Bullish).
def process_tree_clusters(tree, prefix, df, X, order):
    leaf_ids = tree.apply(X)
    df_cluster = df.copy()
    df_cluster['Cluster'] = leaf_ids
    
    clust_grouped = df_cluster.groupby('Cluster')['Y'].value_counts().unstack(fill_value=0)
    for col in [-1, 0, 1]:
        if col not in clust_grouped.columns:
            clust_grouped[col] = 0
    clust_grouped = clust_grouped[[-1, 0, 1]]
    
    N_Obs = clust_grouped.sum(axis=1)
    P_neg1 = clust_grouped[-1] / N_Obs
    P_0 = clust_grouped[0] / N_Obs
    P_pos1 = clust_grouped[1] / N_Obs
    
    Direction = P_pos1 - P_neg1
    Confidence = np.maximum(np.maximum(P_neg1, P_0), P_pos1)
    Class_Error = 1 - Confidence
    Gini = P_neg1*(1-P_neg1) + P_0*(1-P_0) + P_pos1*(1-P_pos1)
    
    def safe_log(p):
        return np.log(p) if p > 0 else 0
    Cross_Entropy = -(P_neg1.apply(safe_log)*P_neg1 + P_0.apply(safe_log)*P_0 + P_pos1.apply(safe_log)*P_pos1)
    
    state_cols = ['s1', 's2', 's3', 's4', 's5']
    unique_states = df_cluster.groupby('Cluster')[state_cols].apply(lambda x: len(x.drop_duplicates()))
    
    summary = pd.DataFrame({
        'N_States': unique_states,
        'N_Observations': N_Obs,
        'P(-1)': P_neg1,
        'P(0)': P_0,
        'P(+1)': P_pos1,
        'Direction': Direction,
        'Classification_Error': Class_Error,
        'Gini': Gini,
        'Cross_Entropy': Cross_Entropy,
        'Confidence': Confidence
    })
    
    summary = summary.sort_values('Direction')
    summary.to_csv(f'{prefix}_tree_cluster_summary.csv')
    
    members = df_cluster[state_cols + ['Cluster']].drop_duplicates()
    order_map = {st: i for i, st in enumerate(order)}
    members['order_idx'] = members.apply(lambda row: order_map[tuple(row[state_cols])], axis=1)
    members = members.sort_values('order_idx').drop(columns=['order_idx'])
    members.to_csv(f'{prefix}_tree_cluster_members.csv', index=False)
    
    return summary

# ==============================================================================
# 7. METHODS 3 & 4: STATE CLUSTERING
# ==============================================================================
# Mathematical objective: Cluster states using pairwise distances between their probability vectors [P(-1), P(0), P(+1)].
# Statistical interpretation: Jensen-Shannon measures divergence between distributions; SSE measures Euclidean geometry.
# Why: Hierarchical clustering discovers natural groups of states with identical probability profiles without a target Y.
# Output interpretation: Cluster assignments for every M5 state, aggregated to re-estimate probabilities.
def state_clustering(transition_table, df, order):
    valid_states = transition_table[transition_table['N_Observations'] > 0].copy()
    state_probs = valid_states[['P(-1)', 'P(0)', 'P(+1)']].values

    print("Jensen-Shannon Clustering...")
    js_dist = pdist(state_probs, metric=jensenshannon)
    js_dist = np.nan_to_num(js_dist, nan=0.0)
    js_square = squareform(js_dist)

    best_js_k = 2
    best_js_score = -1
    for k in range(2, min(21, len(state_probs))):
        clusterer = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = clusterer.fit_predict(js_square)
        score = silhouette_score(js_square, labels, metric='precomputed')
        if score > best_js_score:
            best_js_score = score
            best_js_k = k

    js_clusterer = AgglomerativeClustering(n_clusters=best_js_k, metric='precomputed', linkage='average')
    js_labels = js_clusterer.fit_predict(js_square)
    valid_states['JS_Cluster'] = js_labels

    print("SSE Clustering...")
    best_sse_k = 2
    best_sse_score = -1
    for k in range(2, min(21, len(state_probs))):
        clusterer = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='average')
        labels = clusterer.fit_predict(state_probs)
        score = silhouette_score(state_probs, labels, metric='euclidean')
        if score > best_sse_score:
            best_sse_score = score
            best_sse_k = k

    sse_clusterer = AgglomerativeClustering(n_clusters=best_sse_k, metric='euclidean', linkage='average')
    sse_labels = sse_clusterer.fit_predict(state_probs)
    valid_states['SSE_Cluster'] = sse_labels

    def process_state_clusters(cluster_col, prefix):
        df_cluster = df.copy()
        df_cluster = df_cluster.join(valid_states[[cluster_col]], on=['s1','s2','s3','s4','s5'], how='inner')
        df_cluster = df_cluster.rename(columns={cluster_col: 'Cluster'})
        
        clust_grouped = df_cluster.groupby('Cluster')['Y'].value_counts().unstack(fill_value=0)
        for col in [-1, 0, 1]:
            if col not in clust_grouped.columns:
                clust_grouped[col] = 0
        clust_grouped = clust_grouped[[-1, 0, 1]]
        
        N_Obs = clust_grouped.sum(axis=1)
        P_neg1 = clust_grouped[-1] / N_Obs
        P_0 = clust_grouped[0] / N_Obs
        P_pos1 = clust_grouped[1] / N_Obs
        
        Direction = P_pos1 - P_neg1
        Confidence = np.maximum(np.maximum(P_neg1, P_0), P_pos1)
        Class_Error = 1 - Confidence
        Gini = P_neg1*(1-P_neg1) + P_0*(1-P_0) + P_pos1*(1-P_pos1)
        
        def safe_log(p):
            return np.log(p) if p > 0 else 0
        Cross_Entropy = -(P_neg1.apply(safe_log)*P_neg1 + P_0.apply(safe_log)*P_0 + P_pos1.apply(safe_log)*P_pos1)
        
        state_cols = ['s1', 's2', 's3', 's4', 's5']
        unique_states = df_cluster.groupby('Cluster')[state_cols].apply(lambda x: len(x.drop_duplicates()))
        
        summary = pd.DataFrame({
            'N_States': unique_states,
            'N_Observations': N_Obs,
            'P(-1)': P_neg1,
            'P(0)': P_0,
            'P(+1)': P_pos1,
            'Direction': Direction,
            'Classification_Error': Class_Error,
            'Gini': Gini,
            'Cross_Entropy': Cross_Entropy,
            'Confidence': Confidence
        })
        summary = summary.sort_values('Direction')
        summary.to_csv(f'{prefix}_cluster_summary.csv')
        
        members = df_cluster[state_cols + ['Cluster']].drop_duplicates()
        order_map = {st: i for i, st in enumerate(order)}
        members['order_idx'] = members.apply(lambda row: order_map[tuple(row[state_cols])], axis=1)
        members = members.sort_values('order_idx').drop(columns=['order_idx'])
        members.to_csv(f'{prefix}_cluster_members.csv', index=False)
        
        return summary

    js_summary = process_state_clusters('JS_Cluster', 'js')
    sse_summary = process_state_clusters('SSE_Cluster', 'sse')
    
    return js_summary, sse_summary, best_js_k, best_sse_k, best_js_score, best_sse_score, js_dist, state_probs

# ==============================================================================
# 8. COMPARISON REPORT & VISUALIZATIONS
# ==============================================================================
# Mathematical objective: Provide visual and tabular summaries of the clustering and tree performance.
# Statistical interpretation: Compare Gini, Entropy, JS, and SSE directly.
# Why: Allow research comparison between the four distinct families of models.
# Output interpretation: Dendrograms, Trees, Heatmaps, and a consolidated CSV report.
def generate_reports_and_visuals(gini_tree, entropy_tree, gini_depth, entropy_depth,
                                 gini_acc, entropy_acc, gini_summary, entropy_summary,
                                 js_summary, sse_summary, best_js_k, best_sse_k, best_js_score, best_sse_score,
                                 js_dist, state_probs):
    comparison = pd.DataFrame({
        'Method': ['Gini Tree', 'Cross-Entropy Tree', 'Jensen-Shannon', 'SSE'],
        'Best_Depth': [gini_depth, entropy_depth, np.nan, np.nan],
        'CV_Accuracy': [gini_acc, entropy_acc, np.nan, np.nan],
        'Number_of_Clusters': [len(gini_summary), len(entropy_summary), best_js_k, best_sse_k],
        'Average_Cluster_Size': [gini_summary['N_States'].mean(), entropy_summary['N_States'].mean(), js_summary['N_States'].mean(), sse_summary['N_States'].mean()],
        'Average_CrossEntropy': [gini_summary['Cross_Entropy'].mean(), entropy_summary['Cross_Entropy'].mean(), js_summary['Cross_Entropy'].mean(), sse_summary['Cross_Entropy'].mean()],
        'Average_Confidence': [gini_summary['Confidence'].mean(), entropy_summary['Confidence'].mean(), js_summary['Confidence'].mean(), sse_summary['Confidence'].mean()],
        'Average_Direction': [gini_summary['Direction'].mean(), entropy_summary['Direction'].mean(), js_summary['Direction'].mean(), sse_summary['Direction'].mean()],
        'Silhouette_Score': [np.nan, np.nan, best_js_score, best_sse_score]
    })
    comparison.to_csv('comparison_report.csv', index=False)

    print("Generating visualizations...")
    plt.figure(figsize=(20,10))
    plot_tree(gini_tree, feature_names=['s1','s2','s3','s4','s5'], class_names=['-1','0','1'], filled=True)
    plt.title("Gini Decision Tree")
    plt.savefig('gini_tree_diagram.png')
    plt.close()

    plt.figure(figsize=(20,10))
    plot_tree(entropy_tree, feature_names=['s1','s2','s3','s4','s5'], class_names=['-1','0','1'], filled=True)
    plt.title("Cross-Entropy Decision Tree")
    plt.savefig('entropy_tree_diagram.png')
    plt.close()

    from scipy.cluster.hierarchy import dendrogram, linkage

    plt.figure(figsize=(15, 8))
    Z_js = linkage(js_dist, method='average')
    dendrogram(Z_js)
    plt.title("Jensen-Shannon Dendrogram")
    plt.savefig('js_dendrogram.png')
    plt.close()

    plt.figure(figsize=(15, 8))
    Z_sse = linkage(state_probs, method='average', metric='euclidean')
    dendrogram(Z_sse)
    plt.title("SSE Dendrogram")
    plt.savefig('sse_dendrogram.png')
    plt.close()

    def plot_heatmap(summary, title, filename):
        plt.figure(figsize=(8, 6))
        sns.heatmap(summary[['P(-1)','P(0)','P(+1)']], annot=True, cmap='RdYlGn', center=0.33)
        plt.title(title)
        plt.savefig(filename)
        plt.close()

    plot_heatmap(gini_summary, "Gini Tree Cluster Transition Probabilities", 'gini_heatmap.png')
    plot_heatmap(entropy_summary, "Entropy Tree Cluster Transition Probabilities", 'entropy_heatmap.png')
    plot_heatmap(js_summary, "JS Cluster Transition Probabilities", 'js_heatmap.png')
    plot_heatmap(sse_summary, "SSE Cluster Transition Probabilities", 'sse_heatmap.png')

    def plot_size(summary, title, filename):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=summary.index, y=summary['N_States'], palette='viridis')
        plt.title(title)
        plt.xlabel('Cluster (Ordered by Direction)')
        plt.ylabel('Number of States')
        plt.savefig(filename)
        plt.close()

    plot_size(gini_summary, "Gini Tree Cluster Sizes", 'gini_size.png')
    plot_size(entropy_summary, "Entropy Tree Cluster Sizes", 'entropy_size.png')
    plot_size(js_summary, "JS Cluster Sizes", 'js_size.png')
    plot_size(sse_summary, "SSE Cluster Sizes", 'sse_size.png')

    def plot_direction(summary, title, filename):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=summary.index, y=summary['Direction'], palette='coolwarm')
        plt.title(title)
        plt.xlabel('Cluster')
        plt.ylabel('Direction (P(+1) - P(-1))')
        plt.savefig(filename)
        plt.close()

    plot_direction(gini_summary, "Gini Tree Cluster Direction", 'gini_direction.png')
    plot_direction(entropy_summary, "Entropy Tree Cluster Direction", 'entropy_direction.png')
    plot_direction(js_summary, "JS Cluster Direction", 'js_direction.png')
    plot_direction(sse_summary, "SSE Cluster Direction", 'sse_direction.png')

def main():
    close = get_data()
    states = prepare_data(close)
    df, order = construct_states(states, lookback=5)
    transition_table = compute_transition_table(df, order)
    gini_tree, gini_depth, gini_acc, entropy_tree, entropy_depth, entropy_acc, X = tune_and_fit_trees(df)
    
    gini_summary = process_tree_clusters(gini_tree, 'gini', df, X, order)
    entropy_summary = process_tree_clusters(entropy_tree, 'entropy', df, X, order)
    
    js_summary, sse_summary, best_js_k, best_sse_k, best_js_score, best_sse_score, js_dist, state_probs = state_clustering(transition_table, df, order)
    
    generate_reports_and_visuals(gini_tree, entropy_tree, gini_depth, entropy_depth,
                                 gini_acc, entropy_acc, gini_summary, entropy_summary,
                                 js_summary, sse_summary, best_js_k, best_sse_k, best_js_score, best_sse_score,
                                 js_dist, state_probs)
    print("All done!")

if __name__ == '__main__':
    main()
