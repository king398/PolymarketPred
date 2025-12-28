import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
INPUT_FILE = "polymarket_belief_updates.csv"
STATE_FILE = "latest_market_state.csv"
SIMULATION_HORIZON_MINUTES = 60 * 24   # Forecast 5 Days
NUM_SIMULATIONS = 2000                    # Higher count for smooth fair price
SEED = 42
NUM_SAMPLE_PATHS = 10

def load_data():
    """Loads history (for covariance) and current state (for start points)."""
    try:
        # Load History (The changes)
        history_df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
        # Load Latest State (The absolute values)
        state_df = pd.read_csv(STATE_FILE, index_col=0)

        print(f"✅ Loaded history: {history_df.shape} | Loaded current states: {state_df.shape}")
        return history_df, state_df
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure you ran the Data Processing script first to generate CSVs.")
        return None, None

# ---------------------------------------------------------------------------
# 2. Monte Carlo Engine
# ---------------------------------------------------------------------------
class CorrelatedMonteCarlo:
    def __init__(self, history_df, current_state_df, seed=None):
        self.markets = history_df.columns.tolist()
        self.n_markets = len(self.markets)
        self.last_timestamp = history_df.index.max()

        # Stats
        # CRITICAL: We set drift to ZERO (Martingale assumption).
        # Prediction markets are efficient; best guess for tomorrow is today's price.
        self.drift = np.zeros(self.n_markets)
        self.cov_matrix = history_df.cov().values

        # Set Start Values
        try:
            aligned_state = current_state_df.reindex(self.markets)
            if aligned_state.isnull().values.any():
                print("⚠️ Warning: Filling missing start states with 0 (50% prob).")
                aligned_state = aligned_state.fillna(0)
            self.last_log_odds = aligned_state.values.flatten()
        except Exception as e:
            print(f"❌ Error aligning start states: {e}")
            self.last_log_odds = history_df.sum().values

        if seed:
            np.random.seed(seed)

    def generate_paths(self, horizon_steps, n_sims):
        print(f"   -> Computing Cholesky decomposition for {self.n_markets} markets...")

        # Cholesky Decomposition (Covariance -> Correlated Noise)
        try:
            L = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            # Regularization for non-positive definite matrices
            L = np.linalg.cholesky(self.cov_matrix + np.eye(self.n_markets) * 1e-6)

        # 1. Generate Random Shocks (Standard Normal)
        uncorrelated_shocks = np.random.normal(size=(horizon_steps, n_sims, self.n_markets))

        # 2. Apply Correlation
        shocks_flat = uncorrelated_shocks.reshape(-1, self.n_markets)
        correlated_shocks = (shocks_flat @ L.T).reshape(horizon_steps, n_sims, self.n_markets)

        # 3. Add Drift (Zero) & Accumulate
        drift_reshaped = self.drift.reshape(1, 1, self.n_markets)
        path_updates = correlated_shocks + drift_reshaped
        cumulative_paths = np.cumsum(path_updates, axis=0)

        # 4. Add Start Values
        start_vals = self.last_log_odds.reshape(1, 1, self.n_markets)
        final_log_odds = start_vals + cumulative_paths

        # 5. Convert Log-Odds -> Probability (Price)
        price_paths = 1 / (1 + np.exp(-final_log_odds))

        return price_paths

# ---------------------------------------------------------------------------
# 3. Visualization & Analysis
# ---------------------------------------------------------------------------
def plot_forecast_cone(sim_data, markets, market_name, start_time, horizon_min, n_samples=5):
    """
    Plots the forecast cone, sample paths, and fair price line.
    """
    if market_name not in markets:
        print(f"❌ Market '{market_name}' not found.")
        return

    mkt_idx = markets.index(market_name)
    paths = sim_data[:, :, mkt_idx] # Shape: (Steps, Sims)

    # 1. Generate Date Range for X-Axis
    date_range = pd.date_range(start=start_time, periods=horizon_min, freq="min")

    # 2. Statistics
    p05, p25, p75, p95 = np.percentile(paths, [5, 25, 75, 95], axis=1)

    # Calculate Fair Price (Mean of all simulations)
    fair_price_mean = np.mean(paths, axis=1)

    fig = go.Figure()

    # Fan Chart (Confidence Intervals)
    fig.add_trace(go.Scatter(
        x=np.concatenate([date_range, date_range[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself', fillcolor='rgba(0,100,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'), name='90% Range',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([date_range, date_range[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'), name='50% Range',
        hoverinfo='skip'
    ))

    # Sample Paths (Grey Lines)
    for i in range(n_samples):
        fig.add_trace(go.Scatter(
            x=date_range, y=paths[:, i], mode='lines',
            line=dict(color='rgba(50, 50, 50, 0.3)', width=1),
            name=f'Sim Path {i}', hoverinfo='y'
        ))

    # Fair Price Line (Green Dashed)
    fig.add_trace(go.Scatter(
        x=date_range, y=fair_price_mean, mode='lines',
        line=dict(color='green', width=3, dash='dash'),
        name='Fair Price (Mean)'
    ))

    start_price = paths[0,0]

    # 3. Layout Fixes
    end_time = date_range[-1]

    fig.update_layout(
        title=f"<b>Monte Carlo Forecast: {market_name}</b><br>Start: {start_time.strftime('%Y-%m-%d %H:%M')}",
        xaxis_title="Date (UTC)",
        yaxis_title="Implied Probability (Price)",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(range=[start_time, end_time]) # Force X-axis to start at forecast time
    )

    # Vertical Line Fix (Pandas Timestamp -> Numeric MS)
    start_time_ms = start_time.timestamp() * 1000
    fig.add_vline(
        x=start_time_ms,
        line_dash="dot",
        line_color="black",
        annotation_text="Forecast Start"
    )

    # Add Start Price Annotation
    fig.add_hline(y=start_price, line_dash="dot", line_color="grey", annotation_text=f"Start: {start_price:.2f}")

    fig.show()

def display_fair_prices(sim_data, markets, current_state_df):
    """
    Computes the implied Fair Price (Mean of Simulations) and identifies
    markets where volatility skew creates a theoretical edge.
    """
    # 1. Get final prices (Last step of simulation)
    final_prices = sim_data[-1, :, :]

    # 2. Calculate Mean (Fair Price)
    fair_prices = np.mean(final_prices, axis=0)

    # 3. Build Summary Table
    df_res = pd.DataFrame({
        'Market': markets,
        'Fair_Price_Sim': fair_prices
    }).set_index('Market')

    # 4. Join with Actual Current Prices
    try:
        # Convert Log-Odds back to Price for comparison
        current_prices = 1 / (1 + np.exp(-current_state_df))

        # Align data
        if isinstance(current_prices, pd.DataFrame):
            current_vals = current_prices.reindex(markets).values.flatten()
        else:
            current_vals = current_prices.reindex(markets).values

        df_res['Current_Price'] = current_vals
        df_res['Diff'] = df_res['Fair_Price_Sim'] - df_res['Current_Price']

        # Sort by absolute difference
        df_res['Abs_Diff'] = df_res['Diff'].abs()

        print("\n" + "="*50)
        print("📢 FAIR PRICE ANALYSIS (Volatility Skew)")
        print("="*50)
        print("Markets where Simulation Mean != Current Price")
        print("(Caused by Jensen's Inequality on high-volatility pairs)")
        print("-" * 50)
        print(df_res.sort_values('Abs_Diff', ascending=False).head(5)[['Fair_Price_Sim', 'Current_Price', 'Diff']])

        return df_res

    except Exception as e:
        print(f"⚠️ Could not align prices for table: {e}")
        return df_res

# ---------------------------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Load Data
    history_df, state_df = load_data()

    if history_df is not None and state_df is not None:
        print(f"🚀 Simulating {NUM_SIMULATIONS} universes...")

        # 2. Run Simulation
        mc = CorrelatedMonteCarlo(history_df, state_df, seed=SEED)
        sim_results = mc.generate_paths(SIMULATION_HORIZON_MINUTES, NUM_SIMULATIONS)

        print(f"✅ Simulation Complete. Start Time: {mc.last_timestamp}")

        # 3. Analyze Fair Prices
        fair_price_df = display_fair_prices(sim_results, mc.markets, state_df)

        # 4. Plot Specific Market
        # Try to plot a specific market, or default to the first one
        target_name = "Will Ethereum dip to $2,800 in November?" # Example name

        # If example doesn't exist, pick the first available market
        if target_name not in mc.markets:
            target_name = mc.markets[0]

        print(f"\n📊 Plotting: {target_name}")

        plot_forecast_cone(
            sim_results,
            mc.markets,
            target_name,
            mc.last_timestamp,
            SIMULATION_HORIZON_MINUTES,
            n_samples=NUM_SAMPLE_PATHS
        )