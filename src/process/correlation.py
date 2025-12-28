import random
import itertools
from datasets import Dataset
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
from tqdm import tqdm  # Progress bar

# ---------------------------------------------------------------------------
# 1. Data Loading & Filtering (Polymarket Data)
# ---------------------------------------------------------------------------
DIR = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet"
PARQUETS = glob.glob(f"{DIR}/*.parquet")

# Fast-path parquet -> Arrow Dataset
ds = Dataset.from_parquet(PARQUETS)
pattern = re.compile(r"\b(bitcoin|ethereum|solana|xrp|btc|eth)\b", re.IGNORECASE)

def crypto_filter_batch(batch):
    qs = batch["question"]
    return [bool(pattern.search(q)) if q else False for q in qs]

ds_crypto = ds.filter(
    crypto_filter_batch,
    batched=True,
    batch_size=50_000,
    num_proc=32,
)
start = pd.Timestamp("2025-10-02", tz="UTC")
end = start + pd.Timedelta(days=2)

def in_range_batch(batch):
    ts = pd.to_datetime(batch["timestamp"], utc=True, errors="coerce")
    return (ts >= start) & (ts <= end)

ds_range = ds_crypto.filter(
    in_range_batch,
    batched=True,
    batch_size=50_000,
    num_proc=32,
    load_from_cache_file=True,
)

# ---------------------------------------------------------------------------
# 2. Feature Engineering: Belief Updates (Log-Odds)
# ---------------------------------------------------------------------------
df = ds_range.to_pandas()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
group_key = "market_id"

# Calculate Log-Odds
eps = 1e-6
p = df["price"].astype("float64").clip(eps, 1 - eps)
df["log_odds"] = np.log(p / (1 - p))

# Calculate Belief Update (Returns)
df = df.sort_values([group_key, "timestamp"])
df["belief_update"] = df.groupby(group_key)["log_odds"].diff().fillna(0.0)

# Filter active markets
belief_movement = df.groupby(group_key)["belief_update"].apply(lambda x: np.abs(x).sum())
active_markets = belief_movement[belief_movement > 0].index
df = df[df[group_key].isin(active_markets)].copy()

print(f"Active Markets: {df[group_key].nunique()} | Rows: {len(df)}")

# ---------------------------------------------------------------------------
# 3. Pivot Data for Correlation Analysis
# ---------------------------------------------------------------------------
# Floor to nearest minute to align timestamps
df['timestamp_min'] = df['timestamp'].dt.floor('1min')

# Pivot: Rows=Time, Columns=Market, Values=Belief Update
# Using 'question' as column name for readability, you might prefer market_id for uniqueness
pivot_df = df.pivot_table(
    index='timestamp_min',
    columns='question',
    values='belief_update',
    aggfunc='last'
).fillna(0.0)
pivot_df.to_csv("polymarket_belief_updates.csv")
print("Saved belief updates to 'polymarket_belief_updates.csv'")
print(f"Matrix Shape: {pivot_df.shape} (Time x Markets)")
# ---- PRICE PIVOT (for copula fair pricing) ----
price_pivot = df.pivot_table(
    index="timestamp_min",
    columns="question",
    values="price",
    aggfunc="last"
).ffill().dropna(how="all")  # forward fill for missing ticks
price_pivot.to_csv("polymarket_price_pivot.csv")
print("Saved price pivot to 'polymarket_price_pivot.csv'")
print("Price Pivot Shape:", price_pivot.shape)

# ---------------------------------------------------------------------------
# 4. Optimized Lead-Lag Correlation (NumPy + Vectorization)
# ---------------------------------------------------------------------------
def compute_lead_lag_fast(df_pivot, max_lag=60):
    """
    Vectorized computation of lead-lag correlations.
    Returns:
        corr_df: Max absolute correlation found for each pair.
        lag_df: The lag (in minutes) where the max correlation occurred.
    """
    # Convert to NumPy for speed
    # Shape: (Time, Markets)
    data = df_pivot.values
    markets = df_pivot.columns
    n_samples, n_features = data.shape

    # 1. Global Standardization (Z-score)
    # We standardize globally to enable fast matrix multiplication per lag
    # This is a standard approximation for sliding window correlations on stationary data
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)

    # Avoid division by zero for constant columns
    valid_cols = stds > 1e-9
    norm_data = np.zeros_like(data)
    norm_data[:, valid_cols] = (data[:, valid_cols] - means[valid_cols]) / stds[valid_cols]

    # 2. Initialize Output Matrices
    # best_corr: stores the signed correlation value with the highest magnitude
    # best_lag: stores the lag corresponding to that correlation
    best_corr = np.zeros((n_features, n_features))
    best_lag = np.zeros((n_features, n_features), dtype=int)
    max_abs_corr = -np.ones((n_features, n_features)) # Track max magnitude

    # 3. Iterate over lags
    lags = range(-max_lag, max_lag + 1)

    print(f"Computing correlations across {len(lags)} lags...")
    for lag in tqdm(lags, desc="Vectorized Lag Sweep"):

        # Define time slices for lag
        if lag == 0:
            # Simple Correlation Matrix: A^T * A
            # N-1 for sample correlation
            cov = (norm_data.T @ norm_data) / (n_samples - 1)

        elif lag > 0:
            # Corr(X_i(t), X_j(t+lag)) => X_i leads X_j
            # X_i uses [0 : N-lag], X_j uses [lag : N]
            n_overlap = n_samples - lag
            if n_overlap < 2: continue

            slice_early = norm_data[:-lag]
            slice_late  = norm_data[lag:]

            # Matrix Mult: (Markets x Overlap) @ (Overlap x Markets) -> (Markets x Markets)
            cov = (slice_early.T @ slice_late) / (n_overlap - 1)

        else: # lag < 0
            # Corr(X_i(t), X_j(t-|lag|)) => X_j leads X_i
            k = -lag
            n_overlap = n_samples - k
            if n_overlap < 2: continue

            # X_i uses [k : N], X_j uses [0 : N-k]
            slice_late = norm_data[k:]
            slice_early = norm_data[:-k]

            cov = (slice_late.T @ slice_early) / (n_overlap - 1)

        # 4. Update Best Matrices (Element-wise)
        # Check where current lag produces a stronger correlation than previously found
        abs_cov = np.abs(cov)
        is_better = abs_cov > max_abs_corr

        # Update records
        best_corr[is_better] = cov[is_better]
        best_lag[is_better]  = lag
        max_abs_corr[is_better] = abs_cov[is_better]

    # Wrap in Pandas
    corr_df = pd.DataFrame(best_corr, index=markets, columns=markets)
    lag_df_out = pd.DataFrame(best_lag, index=markets, columns=markets)

    return corr_df, lag_df_out

# Execute Optimized Function
max_lag_minutes = 15
lead_lag_corr_df, optimal_lags_df = compute_lead_lag_fast(pivot_df, max_lag=max_lag_minutes)

# ---------------------------------------------------------------------------
# 5. Save & Display Results
# ---------------------------------------------------------------------------

# Save Matrix
output_file = "market_lead_lag_correlation_matrix.csv"
lead_lag_corr_df.to_csv(output_file)
print(f"\nCorrelation matrix saved to: {output_file}")

# Analyze Top Pairs
# Use Upper Triangle to avoid duplicates (A-B vs B-A) and self-correlation (diagonal)
upper_tri_mask = np.triu(np.ones(lead_lag_corr_df.shape), k=1).astype(bool)
top_corrs = lead_lag_corr_df.where(upper_tri_mask).stack().sort_values(ascending=False, key=abs)
top_n_counts = 10
print("\n" + "="*50)
print(f"TOP {top_n_counts} LEAD-LAG RELATIONSHIPS")
print("="*50)

count = 0
for (mkt_b, mkt_a), correlation in top_corrs.items():
    if count >= top_n_counts: break

    lag = optimal_lags_df.loc[mkt_b, mkt_a]
    if lag == 0:
        continue  # Skip synchronous for this analysis
    print(f"\nPair #{count+1}")
    print(f"   Market A: {mkt_a}")
    print(f"   Market B: {mkt_b}")
    print(f"   Max Correlation: {correlation:.4f}")

    # Interpret Lag
    # Our func: Corr(A, B_shifted_by_lag)
    # If lag > 0: A(t) matches B(t+lag) -> A leads B
    # If lag < 0: A(t) matches B(t-|lag|) -> B leads A
    if lag > 0:
        print(f"   DYNAMICS: '{mkt_a}' leads by {abs(lag)} min")
    elif lag < 0:
        print(f"   DYNAMICS: '{mkt_b}' leads by {abs(lag)} min")
    else:
        continue

    count += 1

# ---------------------------------------------------------------------------
# 6. Plotting the Top Pairs
# ---------------------------------------------------------------------------

def plot_lead_lag(df_pivot, market_a, market_b, lag, correlation, rank):
    """
    Plots the cumulative belief state of two markets to visualize their relationship.
    """
    # 1. Get the raw series
    series_a = df_pivot[market_a].cumsum() # Cumulative to see the "Trend"
    series_b = df_pivot[market_b].cumsum()

    # 2. Normalize (Z-Score) for visual comparison on the same axis
    # This helps if one market is much more volatile than the other
    series_a_norm = (series_a - series_a.mean()) / series_a.std()
    series_b_norm = (series_b - series_b.mean()) / series_b.std()

    fig = go.Figure()

    # Trace A (Leader or Follower)
    fig.add_trace(go.Scatter(
        x=series_a_norm.index,
        y=series_a_norm,
        mode='lines',
        name=f"{market_a} (Normalized)",
        line=dict(width=1.5),
        opacity=0.8
    ))

    # Trace B
    fig.add_trace(go.Scatter(
        x=series_b_norm.index,
        y=series_b_norm,
        mode='lines',
        name=f"{market_b} (Normalized)",
        line=dict(width=1.5),
        opacity=0.8
    ))

    # Add title details
    lag_text = f"{market_a} leads by {lag} min" if lag > 0 else f"{market_b} leads by {abs(lag)} min"
    if lag == 0: lag_text = "Synchronous"

    fig.update_layout(
        title=f"<b>Pair #{rank}: {market_a} vs {market_b}</b><br>" +
              f"Correlation: {correlation:.4f} | Optimal Lag: {lag_text}",
        xaxis_title="Time (UTC)",
        yaxis_title="Normalized Log-Odds (Cumulative Belief)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.show()

latest_log_odds = df.sort_values('timestamp').groupby('question')['log_odds'].last()

# 2. Save to CSV
latest_log_odds.to_csv("latest_market_state.csv")
print("✅ Saved latest market states to 'latest_market_state.csv'")
import networkx as nx

# ---------------------------------------------------------------------------
# 7. Network Graph Visualization
# ---------------------------------------------------------------------------

def plot_correlation_network(corr_df, lag_df, threshold=0.8):
    """
    Creates an interactive network graph of markets.
    Nodes = Markets
    Edges = Correlations > threshold
    """
    print(f"\nBuilding network graph (Threshold: {threshold})...")

    # 1. Initialize Graph
    G = nx.Graph()

    # Add nodes
    markets = corr_df.columns.tolist()
    G.add_nodes_from(markets)

    # Add edges based on threshold
    # We iterate upper triangle to avoid duplicates
    count_edges = 0
    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            mkt_a = markets[i]
            mkt_b = markets[j]

            corr = corr_df.iloc[i, j]
            lag = lag_df.iloc[i, j]

            if abs(corr) > threshold:
                # Add edge with attributes
                G.add_edge(mkt_a, mkt_b, weight=abs(corr), correlation=corr, lag=lag)
                count_edges += 1

    print(f"Graph stats: {len(G.nodes)} Nodes, {count_edges} Edges")

    # Remove isolated nodes (markets with no correlations > 0.8) to clean up the view
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    print(f"Removed {len(isolated)} isolated markets. Active nodes: {len(G.nodes)}")

    if len(G.nodes) == 0:
        print("No correlations found above threshold.")
        return

    # 2. Generate Layout (Positioning)
    # k controls the distance between nodes (higher = spread out)
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # 3. Create Plotly Traces

    # -- Edge Trace --
    edge_x = []
    edge_y = []
    edge_text = [] # For hover info on edges (requires a little hack in plotly, usually simpler on nodes)

    # We will color edges by correlation sign (Green positive, Red negative)
    # Since Plotly Line traces are single-color, we separate them or use a colorscale hack.
    # For simplicity here, we use a single grey line, but vary opacity/width.

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    # -- Node Trace --
    node_x = []
    node_y = []
    node_text = []
    node_adjacencies = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Calculate number of connections (degree)
        adj = len(G[node])
        node_adjacencies.append(adj)

        # Hover text
        node_text.append(f"<b>{node}</b><br>Connections: {adj}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Node Connections',  # Title text goes here
                    side='right'            # 'titleside' is replaced by this
                ),
                xanchor='left',
            ),
            line_width=2))

    # Color nodes by number of connections
    node_trace.marker.color = node_adjacencies

    # 4. Draw Figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'<b>Polymarket Crypto Correlations (> {threshold})</b><br>'
                              f'Lines represent strong statistical coupling',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        template="plotly_white",
                        height=800
                    ))

    fig.show()

    # Optional: Save as HTML
    fig.write_html("polymarket_network_graph.html")
    print("Graph saved to 'polymarket_network_graph.html'")

# Execute the function with your existing dataframes
# You can adjust the threshold (e.g., 0.8, 0.9, 0.95)
plot_correlation_network(lead_lag_corr_df, optimal_lags_df, threshold=0.8)