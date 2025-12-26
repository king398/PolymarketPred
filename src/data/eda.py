import pandas as pd
import plotly.graph_objects as go

# Path to ONE token parquet file
parquet_path = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet/1423e104-e850-4e7a-b07b-020f3624a38f.parquet"

# Load data
df = pd.read_parquet(parquet_path)

# Ensure correct types
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp")

# Create Plotly figure
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["price"],
        mode="lines",
        name="Price",
        hovertemplate="Time: %{x}<br>Price: %{y:.4f}<extra></extra>",
    )
)

fig.update_layout(
    title=f"Polymarket Price Time Series<br>{df['question'].iloc[0]}",
    xaxis_title="Time (UTC)",
    yaxis_title="Price",
    template="plotly_dark",  # remove or change if you want light theme
    height=500,
)

fig.show()
