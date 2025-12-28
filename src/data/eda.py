import pandas as pd
import plotly.graph_objects as go

p1 = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet/6a15f560-b2bb-48f0-bbf5-d47f7fab97f5.parquet"
p2 = "/home/mithil/PycharmProjects/PolymarketPred/data/polymarket_minute_parquet/1a453af6-1475-4f6d-8239-1b28bb401464.parquet"

def load_last_quarter(path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    start = int(n * 0.95)  # last 25%
    df = df.iloc[start:].copy()

    label = str(df["question"].iloc[0]) if "question" in df.columns and len(df) else path
    return df, label

(d1, l1) = load_last_quarter(p1)
(d2, l2) = load_last_quarter(p2)

fig = go.Figure()

for df, label in [(d1, l1), (d2, l2)]:
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["price"],
            mode="lines",
            name=label[:70],
            line=dict(width=2.5),
            hovertemplate="Time: %{x}<br>Price: %{y:.4f}<extra></extra>",
        )
    )

fig.update_layout(
    title=dict(
        text="Polymarket Price Time Series (Last 25%)",
        x=0.5,
        font=dict(size=20),
    ),
    template="plotly_dark",  # good baseline; we’ll polish via layout tweaks below
    height=600,
    margin=dict(l=60, r=30, t=80, b=60),
    hovermode="x unified",
    legend=dict(
        title="Market",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.0,
        bgcolor="rgba(0,0,0,0.25)",
        borderwidth=0,
    ),
    font=dict(size=13),
)

fig.update_xaxes(
    title="Time (UTC)",
    showgrid=True,
    gridcolor="rgba(255,255,255,0.08)",
    zeroline=False,
    rangeslider=dict(visible=True),
)

fig.update_yaxes(
    title="Price",
    showgrid=True,
    gridcolor="rgba(255,255,255,0.08)",
    zeroline=False,
)

fig.show()
