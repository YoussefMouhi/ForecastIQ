"""
Retail Sales Forecasting Dashboard
Run: streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import os

st.set_page_config(
    page_title="ForecastIQ | Retail Sales Intelligence",
    page_icon="assets/forecastiq_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 16px 20px;
        border: 1px solid #3a3f5c; margin-bottom: 8px;
    }
    .metric-label { color: #8b9dc3; font-size: 13px; font-weight: 500; }
    .metric-value { color: #ffffff; font-size: 26px; font-weight: 700; margin-top: 4px; }
    .metric-delta { font-size: 13px; margin-top: 2px; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #e0e4f0;
        border-left: 4px solid #4c72b0; padding-left: 12px;
        margin: 20px 0 12px;
    }
    .alert-green { background: #1a3a2a; border: 1px solid #2ecc71;
                   border-radius: 8px; padding: 10px 14px; color: #2ecc71; }
    .alert-red   { background: #3a1a1a; border: 1px solid #e74c3c;
                   border-radius: 8px; padding: 10px 14px; color: #e74c3c; }
    .alert-yellow{ background: #3a3a1a; border: 1px solid #f39c12;
                   border-radius: 8px; padding: 10px 14px; color: #f39c12; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv", parse_dates=["date"])
    stores = pd.read_csv("data/stores.csv")
    return df.merge(stores, on="store_nbr", how="left")

@st.cache_resource
def load_models():
    m = {}
    v2_dir = Path("validation/models_v2")
    v2_files = [
        v2_dir / "xgboost_model.joblib",
        v2_dir / "encoders.joblib",
        v2_dir / "features.joblib",
    ]

    if all(p.exists() for p in v2_files):
        m["xgb"] = joblib.load(v2_files[0])
        encoders = joblib.load(v2_files[1])
        m["le_store"] = encoders["store_nbr"]
        m["le_item"] = encoders["item_nbr"]
        m["xgb_feats"] = joblib.load(v2_files[2])
        m["xgb_version"] = "V2"

    elif (p := Path("models/xgb_model.pkl")).exists():
        m["xgb"] = joblib.load(p)
        m["le_store"] = joblib.load("models/le_store.pkl")
        m["le_item"] = joblib.load("models/le_item.pkl")
        m["xgb_feats"] = joblib.load("models/xgb_features.pkl")
        m["xgb_version"] = "Original"
    if (p := Path("models/results.csv")).exists():
        m["results"] = pd.read_csv(p)
    return m


def make_forecast_xgb(df_ser, models, horizon=30, onpromo=0):
    """Recursive XGBoost forecast without future-sales leakage."""

    from utils.features_v2 import build_features

    history = df_ser.sort_values("date").copy().reset_index(drop=True)

    store = history["store_nbr"].iloc[0]
    item = history["item_nbr"].iloc[0]

    xgb = models["xgb"]
    le_store = models["le_store"]
    le_item = models["le_item"]
    feature_names = models["xgb_feats"]

    future_dates = pd.date_range(
        history["date"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    predictions = []

    for future_date in future_dates:

        future_row = pd.DataFrame({
            "date": [future_date],
            "store_nbr": [store],
            "item_nbr": [item],
            "family": [history["family"].iloc[0]],
            "sales": [np.nan],
            "onpromotion": [int(onpromo)]
        })

        combined = pd.concat(
            [history, future_row],
            ignore_index=True
        )

        features = build_features(combined)

        features["store_enc"] = le_store.transform(
            features["store_nbr"]
        )

        features["item_enc"] = le_item.transform(
            features["item_nbr"]
        )

        row = features.iloc[[-1]][feature_names]

        if row.isna().any().any():
            raise ValueError(
                "Missing XGBoost features during recursive forecasting."
            )

        prediction = max(
            0.0,
            float(xgb.predict(row)[0])
        )

        predictions.append(prediction)

        future_row["sales"] = prediction

        history = pd.concat(
            [history, future_row],
            ignore_index=True
        )

    predictions = np.asarray(predictions)

    return pd.DataFrame({
        "date": future_dates,
        "forecast": predictions,
        "lower": predictions,
        "upper": predictions
    })


@st.cache_resource
def load_sarima_model(store, item):
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults

    model_path = (
        f"models/multiseries/sarima/store_{store}_{item}.pkl"
    )

    return SARIMAXResults.load(model_path)


def make_forecast_sarima(df_ser, horizon=30, store=None, item=None):
    """Forecast future demand using saved SARIMA parameters."""

    model = load_sarima_model(store, item)

    history = (
        df_ser.sort_values("date")["sales"]
        .to_numpy(dtype=float)
    )

    if len(history) < model.nobs:
        raise ValueError("Insufficient historical observations.")

    if len(history) > model.nobs:
        updated = model.append(
            history[model.nobs:],
            refit=False
        )
    else:
        updated = model

    prediction = updated.get_forecast(steps=horizon)

    values = np.maximum(
        np.asarray(prediction.predicted_mean),
        0
    )

    intervals = np.asarray(
        prediction.conf_int(alpha=0.05)
    )

    future_dates = pd.date_range(
        df_ser["date"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    return pd.DataFrame({
        "date": future_dates,
        "forecast": values,
        "lower": np.maximum(intervals[:, 0], 0),
        "upper": np.maximum(intervals[:, 1], 0),
    })


@st.cache_resource
def load_prophet_model(store, item):
    return joblib.load(
        f"models/multiseries/prophet/store_{store}_{item}.pkl"
    )


def make_forecast_prophet(df_ser, horizon=30, store=None, item=None):
    """Forecast after last historical date with the fitted Prophet model."""
    model = load_prophet_model(store, item)
    dates = pd.date_range(
        df_ser["date"].max() + pd.Timedelta(days=1),
        periods=horizon, freq="D"
    )
    prediction = model.predict(pd.DataFrame({"ds": dates}))
    return pd.DataFrame({
        "date": dates,
        "forecast": np.maximum(prediction["yhat"].to_numpy(), 0),
        "lower": np.maximum(prediction["yhat_lower"].to_numpy(), 0),
        "upper": np.maximum(prediction["yhat_upper"].to_numpy(), 0),
    })


@st.cache_resource
def load_lstm_model():
    import torch
    from torch import nn

    torch.set_num_threads(2)

    class GlobalLSTM(nn.Module):
        def __init__(self, n_stores, n_items):
            super().__init__()

            self.store_embedding = nn.Embedding(n_stores, 4)
            self.item_embedding = nn.Embedding(n_items, 6)

            self.lstm = nn.LSTM(
                1,
                64,
                batch_first=True
            )

            self.fc = nn.Sequential(
                nn.Linear(64 + 4 + 6, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x, store, item):
            out, _ = self.lstm(x)

            last = out[:, -1, :]
            store_emb = self.store_embedding(store)
            item_emb = self.item_embedding(item)

            combined = torch.cat(
                [last, store_emb, item_emb],
                dim=1
            )

            return self.fc(combined)

    config = joblib.load(
        "models/multiseries/lstm/global_lstm_config.pkl"
    )

    model = GlobalLSTM(
        config["n_stores"],
        config["n_items"]
    )

    state = torch.load(
        "models/multiseries/lstm/global_lstm.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model, config


def make_forecast_lstm(df_ser, horizon=30, store=None, item=None):
    import torch

    model, config = load_lstm_model()

    store = int(store)
    item = str(item)

    key = (store, item)

    if key not in config["scales"]:
        raise ValueError(
            f"No LSTM scaling configuration for Store {store} / {item}"
        )

    if store not in config["store_map"]:
        raise ValueError(f"Unknown store: {store}")

    if item not in config["item_map"]:
        raise ValueError(f"Unknown item: {item}")

    scale = config["scales"][key]
    seq_len = config["seq_len"]

    values = (
        df_ser.sort_values("date")["sales"]
        .to_numpy(dtype=float)
    )

    if len(values) < seq_len:
        raise ValueError(
            f"LSTM requires at least {seq_len} observations."
        )

    context = (values / scale).tolist()

    store_tensor = torch.tensor(
        [config["store_map"][store]],
        dtype=torch.long
    )

    item_tensor = torch.tensor(
        [config["item_map"][item]],
        dtype=torch.long
    )

    predictions = []

    with torch.no_grad():
        for _ in range(horizon):
            batch = torch.tensor(
                context[-seq_len:],
                dtype=torch.float32
            ).reshape(1, seq_len, 1)

            prediction = float(
                model(
                    batch,
                    store_tensor,
                    item_tensor
                ).item()
            )

            prediction = max(prediction, 0.0)

            predictions.append(prediction * scale)
            context.append(prediction)

    dates = pd.date_range(
        df_ser["date"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    predictions = np.asarray(predictions)

    return pd.DataFrame({
        "date": dates,
        "forecast": predictions,
        "lower": predictions,
        "upper": predictions
    })


def make_simple_forecast(df_ser, horizon=30):
    """Fallback: rolling average + trend extrapolation."""
    recent = df_ser.tail(60)["sales"].values
    avg = np.convolve(recent, np.ones(7)/7, mode="valid")[-1]
    trend = (recent[-7:].mean() - recent[-30:-23].mean()) / 23 if len(recent) >= 30 else 0
    preds = [max(0, avg + trend * d + np.random.normal(0, avg * 0.08)) for d in range(horizon)]
    lows  = [max(0, p * 0.85) for p in preds]
    highs = [p * 1.15 for p in preds]
    future = pd.date_range(df_ser["date"].max() + pd.Timedelta(1, "D"), periods=horizon)
    return pd.DataFrame({"date": future, "forecast": preds, "lower": lows, "upper": highs})


def plot_forecast(historical, forecast_df, title):
    fig = make_subplots(rows=1, cols=1)
    # Historical
    fig.add_trace(go.Scatter(
        x=historical["date"], y=historical["sales"],
        mode="lines", name="Historical", line=dict(color="#4c72b0", width=1.5),
        opacity=0.8,
    ))
    # Rolling mean
    roll = historical["sales"].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=historical["date"], y=roll,
        mode="lines", name="7-day MA", line=dict(color="#dd8452", width=2, dash="dot"),
    ))
    # CI
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
        y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
        fill="toself", fillcolor="rgba(46,204,113,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Forecast uncertainty band", showlegend=True,
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#2ecc71", width=2.5),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e4f0")),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Sales"),
        legend=dict(bgcolor="#1e2130", bordercolor="#3a3f5c"),
        height=420,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# Check data exists
if not Path("data/train.csv").exists():
    st.error(" Data not found! Run `python data/generate_data.py` first.")
    st.stop()

df     = load_data()
models = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/forecastiq_logo.png", width=85)
    st.title("ForecastIQ")
    st.markdown("---")

    page = st.radio(" Navigation", [
        " Overview",
        " Forecast",
        " Inventory",
        " Model Comparison",
        " EDA",
    ])

    st.markdown("---")
    st.markdown("### Filters")

    with st.form("forecast_filters"):

        stores_list = sorted(df["store_nbr"].unique())
        items_list = sorted(df["item_nbr"].unique())

        sel_store = st.selectbox(
            "Store",
            stores_list
        )

        sel_item = st.selectbox(
            "Product",
            items_list
        )

        selected_model = st.selectbox(
            "Forecasting Model",
            ["XGBoost", "SARIMA", "Prophet", "LSTM", "LSTM V2"],
            index=0,
            help="All four forecasting models support all 5 stores and 10 items."
        )

        horizon = st.slider(
            "Forecast horizon (days)",
            7,
            90,
            30
        )

        onpromo = st.checkbox(
            "On Promotion during forecast?",
            False
        )

        apply_filters = st.form_submit_button(
            "Apply Filters",
            type="primary",
            use_container_width=True
        )

    st.markdown("---")
    st.caption("Built with  Python · XGBoost · Prophet · SARIMA · LSTM")


# ── Filter series ─────────────────────────────────────────────────────────────
ser = df[(df.store_nbr == sel_store) & (df.item_nbr == sel_item)].copy().sort_values("date")
store_info = df[df.store_nbr == sel_store][["type"]].iloc[0]

# ── Generate forecast ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_forecast(store, item, h, promo, has_model):
    s = df[(df.store_nbr == store) & (df.item_nbr == item)].copy().sort_values("date")
    if has_model:
        try:
            return make_forecast_xgb(s, load_models(), horizon=h, onpromo=int(promo))
        except Exception:
            pass
    return make_simple_forecast(s, horizon=h)

has_model = "xgb" in models
if selected_model == "SARIMA":

    try:
        forecast_df = make_forecast_sarima(
            ser,
            horizon=horizon,
            store=sel_store,
            item=sel_item
        )
        active_model = "SARIMA"

    except Exception as e:
        st.error(f"SARIMA forecasting failed: {e}")
        st.stop()

elif selected_model == "XGBoost":

    forecast_df = get_forecast(
        sel_store,
        sel_item,
        horizon,
        onpromo,
        has_model
    )

    active_model = "XGBoost" if has_model else "Trend MA"

elif selected_model == "Prophet":

    try:
        forecast_df = make_forecast_prophet(
            ser,
            horizon=horizon,
            store=sel_store,
            item=sel_item
        )
        active_model = "Prophet"

    except Exception as e:
        st.error(f"Prophet forecasting failed: {e}")
        st.stop()


elif selected_model == "LSTM V2":
    try:
        from models.lstm_v2_inference import load_v2, forecast_v2

        @st.cache_resource
        def cached_lstm_v2():
            return load_v2()

        forecast_df = forecast_v2(
            ser,
            horizon=horizon,
            store=sel_store,
            item=sel_item,
            loaded=cached_lstm_v2(),
        )
        active_model = "LSTM V2"
    except Exception as e:
        st.error(f"LSTM V2 forecasting failed: {e}")
        st.stop()

elif selected_model == "LSTM":

    try:
        forecast_df = make_forecast_lstm(
            ser,
            horizon=horizon,
            store=sel_store,
            item=sel_item
        )
        active_model = "LSTM"

    except Exception as e:
        st.error(f"LSTM forecasting failed: {e}")
        st.stop()

else:
    st.error(f"Unknown forecasting model: {selected_model}")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════════════════════
if page == " Overview":
    st.title(" Sales Overview Dashboard")

    total_sales  = df["sales"].sum()
    avg_daily    = df.groupby("date")["sales"].sum().mean()
    promo_lift   = df.groupby("onpromotion")["sales"].mean()
    promo_pct    = (promo_lift[1] / promo_lift[0] - 1) * 100 if len(promo_lift) > 1 else 0
    n_stores     = df["store_nbr"].nunique()
    n_items      = df["item_nbr"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta in zip(
        [c1, c2, c3, c4],
        ["Total Sales", "Avg Daily Sales", "Promo Lift", "Coverage"],
        [f"{total_sales:,.0f}", f"{avg_daily:,.0f}", f"+{promo_pct:.1f}%", f"{n_stores} stores"],
        ["All time", "Per day", "vs non-promo", f"{n_items} products"],
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-delta" style="color:#8b9dc3">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Overall Sales Trend</div>', unsafe_allow_html=True)
    daily = df.groupby("date")["sales"].sum().reset_index()
    daily["roll7"]  = daily["sales"].rolling(7).mean()
    daily["roll30"] = daily["sales"].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["sales"],
        mode="lines", name="Daily", line=dict(color="#4c72b0", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["roll7"],
        mode="lines", name="7-day MA", line=dict(color="#dd8452", width=2)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["roll30"],
        mode="lines", name="30-day MA", line=dict(color="#e74c3c", width=2.5)))
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=350,
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        legend=dict(bgcolor="#1e2130"))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Sales by Store Type</div>', unsafe_allow_html=True)
        type_sales = df.groupby("type")["sales"].sum().reset_index()
        fig2 = px.pie(type_sales, values="sales", names="type",
                      color_discrete_sequence=["#4c72b0","#dd8452","#2ecc71"])
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           font=dict(color="#8b9dc3"), height=300,
                           legend=dict(bgcolor="#1e2130"))
        st.plotly_chart(fig2, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Sales by Product Family</div>', unsafe_allow_html=True)
        fam = df.groupby("family")["sales"].sum().sort_values(ascending=True).reset_index()
        fig3 = px.bar(fam, x="sales", y="family", orientation="h",
                      color="sales", color_continuous_scale="Blues")
        fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           font=dict(color="#8b9dc3"), height=300,
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig3, width="stretch")

    st.markdown('<div class="section-header">Day-of-Week Patterns</div>', unsafe_allow_html=True)
    df["dow"] = df["date"].dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_avg = df.groupby("dow")["sales"].mean().reindex(dow_order).reset_index()
    colors = ["#4c72b0"]*5 + ["#2ecc71"]*2
    fig4 = go.Figure(go.Bar(x=dow_avg["dow"], y=dow_avg["sales"],
                             marker_color=colors))
    fig4.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font=dict(color="#8b9dc3"), height=300,
                       xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2130"))
    st.plotly_chart(fig4, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Forecast
# ════════════════════════════════════════════════════════════════════════════
elif page == " Forecast":
    st.title(f" Forecast — Store {sel_store} · {sel_item}")

    # KPIs
    recent_7  = ser.tail(7)["sales"].mean()
    next_7    = forecast_df.head(7)["forecast"].mean()
    delta_7   = (next_7 - recent_7) / recent_7 * 100
    total_fc  = forecast_df["forecast"].sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta_str, color in [
        (c1, "Avg Sales (last 7d)", f"{recent_7:.1f}", "", "#8b9dc3"),
        (c2, f"Avg Forecast (next 7d)", f"{next_7:.1f}",
             f"{'▲' if delta_7>0 else '▼'} {abs(delta_7):.1f}%", "#2ecc71" if delta_7>0 else "#e74c3c"),
        (c3, f"Total Forecast ({horizon}d)", f"{total_fc:,.0f}", "units", "#8b9dc3"),
        (c4, "Model", active_model, "active", "#4c72b0"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-delta" style="color:{color}">{delta_str}</div>
        </div>""", unsafe_allow_html=True)

    # Main chart
    historical = ser.tail(120)
    fig = plot_forecast(historical, forecast_df,
                        f"Sales Forecast — Store {sel_store}, {sel_item} ({horizon} days)")
    st.plotly_chart(fig, width="stretch")

    # Forecast table
    with st.expander(" View Forecast Table"):
        disp = forecast_df.copy()
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        disp = disp.rename(columns={"forecast":"Forecast","lower":"Lower CI","upper":"Upper CI"})
        disp["Forecast"]  = disp["Forecast"].round(1)
        disp["Lower CI"]  = disp["Lower CI"].round(1)
        disp["Upper CI"]  = disp["Upper CI"].round(1)
        st.dataframe(disp, width="stretch")

    # Promotion analysis
    st.markdown('<div class="section-header"> Historical Promotion Impact Analysis</div>', unsafe_allow_html=True)
    promo = ser.groupby("onpromotion")["sales"].agg(["mean","std","count"]).reset_index()
    if len(promo) > 1:
        base_avg  = promo[promo.onpromotion == 0]["mean"].values[0]
        promo_avg = promo[promo.onpromotion == 1]["mean"].values[0]
        lift = (promo_avg / base_avg - 1) * 100
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Historical Promotion Lift</div>
              <div class="metric-value" style="color:#2ecc71">+{lift:.1f}%</div>
              <div class="metric-delta" style="color:#8b9dc3">historical association, not causal impact</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Non-Promo Avg</div>
              <div class="metric-value">{base_avg:.1f}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Promo Avg</div>
              <div class="metric-value">{promo_avg:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            fig5 = go.Figure(go.Bar(
                x=["No Promotion", "On Promotion"],
                y=[base_avg, promo_avg],
                marker_color=["#4c72b0", "#2ecc71"],
                text=[f"{base_avg:.1f}", f"{promo_avg:.1f}"],
                textposition="outside",
            ))
            fig5.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                               font=dict(color="#8b9dc3"), height=280,
                               yaxis=dict(showgrid=True, gridcolor="#1e2130"))
            st.plotly_chart(fig5, width="stretch")


    st.markdown(
        '<div class="section-header"> Forecasted Promotion Impact</div>',
        unsafe_allow_html=True
    )

    if active_model == "XGBoost":
        try:
            baseline = make_forecast_xgb(
                ser, models, horizon=horizon, onpromo=0
            )
            promoted = make_forecast_xgb(
                ser, models, horizon=horizon, onpromo=1
            )

            baseline_total = baseline["forecast"].sum()
            promoted_total = promoted["forecast"].sum()

            expected_lift = (
                (promoted_total / baseline_total - 1) * 100
                if baseline_total > 0 else 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Baseline Forecast", f"{baseline_total:,.0f} units")
            c2.metric("Promotion Forecast", f"{promoted_total:,.0f} units")
            c3.metric("Estimated Promotion Difference", f"{expected_lift:+.1f}%")

            st.caption(
                "Model-based scenario comparison, not a causal estimate. "
                "Both scenarios assume the promotion setting applies "
                "throughout the forecast horizon."
            )

        except Exception as e:
            st.warning(f"Promotion scenario comparison unavailable: {e}")

    else:
        st.info(
            f"{active_model} does not use promotion as an explicit "
            "forecasting input. Promotion-specific scenario comparisons "
            "are currently available only for XGBoost."
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Inventory
# ════════════════════════════════════════════════════════════════════════════
elif page == " Inventory":
    st.title(f" Inventory Simulation — Store {sel_store} · {sel_item}")
    from utils.inventory import compute_inventory_metrics, simulate_inventory_trajectory

    col1, col2, col3 = st.columns(3)
    with col1:
        current_stock = st.number_input("Current Stock (units)", min_value=0.0, value=500.0, step=50.0)
    with col2:
        lead_time = st.slider("Supplier Lead Time (days)", 1, 30, 7)
    with col3:
        svc_level = st.selectbox("Service Level", [0.90, 0.95, 0.99], index=1,
                                  format_func=lambda x: f"{int(x*100)}%")

    forecast_vals = forecast_df["forecast"].values
    metrics = compute_inventory_metrics(
        forecast_vals,
        lead_time_days=lead_time,
        service_level=svc_level,
        current_stock=current_stock,
    )

    # Alert
    if metrics["needs_reorder"]:
        st.markdown(f'<div class="alert-red"> <b>REORDER ALERT</b> — Current stock ({current_stock:.0f}) is below reorder point ({metrics["reorder_point"]:.0f}). Place order immediately!</div>', unsafe_allow_html=True)
    elif metrics["stockout_prob"] and metrics["stockout_prob"] > 30:
        st.markdown(f'<div class="alert-yellow"> <b>LOW STOCK WARNING</b> — {metrics["stockout_prob"]:.1f}% stockout risk detected.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-green"> <b>STOCK HEALTHY</b> — Inventory levels are adequate for the forecast period.</div>', unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Safety Stock", f"{metrics['safety_stock']:.0f} units", "#4c72b0"),
        (c2, "Reorder Point", f"{metrics['reorder_point']:.0f} units", "#dd8452"),
        (c3, "Days of Stock", f"{metrics['days_of_stock']} days" if metrics['days_of_stock'] else "—", "#2ecc71"),
        (c4, "Stockout Risk", f"{metrics['stockout_prob']}%" if metrics['stockout_prob'] else "—",
             "#e74c3c" if (metrics['stockout_prob'] or 0) > 30 else "#2ecc71"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    # Simulation trajectory
    st.markdown('<div class="section-header">Inventory Trajectory Simulation</div>', unsafe_allow_html=True)
    order_qty = max(metrics["reorder_point"] * 1.5, 100)
    sim = simulate_inventory_trajectory(
        forecast_vals, current_stock,
        metrics["reorder_point"], order_qty, lead_time,
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Stock Level", "Daily Demand vs Sales"),
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=sim["day"], y=sim["stock_level"],
        mode="lines+markers", name="Stock", line=dict(color="#4c72b0", width=2)), row=1, col=1)
    fig.add_hline(y=metrics["reorder_point"], line_dash="dash", line_color="#e74c3c",
                  annotation_text="Reorder Point", row=1, col=1)
    fig.add_hline(y=metrics["safety_stock"], line_dash="dot", line_color="#f39c12",
                  annotation_text="Safety Stock", row=1, col=1)

    # Mark reorder days
    reorder_days = sim[sim["reorder"]]["day"]
    if len(reorder_days):
        fig.add_trace(go.Scatter(
            x=reorder_days, y=sim[sim["reorder"]]["stock_level"],
            mode="markers", name="Order Placed",
            marker=dict(color="#2ecc71", size=12, symbol="triangle-up"),
        ), row=1, col=1)

    fig.add_trace(go.Bar(x=sim["day"], y=sim["forecast_demand"],
        name="Forecast Demand", marker_color="#dd8452", opacity=0.6), row=2, col=1)
    fig.add_trace(go.Bar(x=sim["day"], y=sim["lost_sales"],
        name="Lost Sales", marker_color="#e74c3c", opacity=0.8), row=2, col=1)

    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=550,
        xaxis2=dict(title="Day", showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis2=dict(showgrid=True, gridcolor="#1e2130"),
        legend=dict(bgcolor="#1e2130"), barmode="overlay")
    st.plotly_chart(fig, width="stretch")

    fill_rate = (sim["actual_sales"].sum() / sim["forecast_demand"].sum() * 100) if sim["forecast_demand"].sum() > 0 else 100
    lost = sim["lost_sales"].sum()
    col1, col2 = st.columns(2)
    col1.markdown(f"""<div class="metric-card">
        <div class="metric-label">Fill Rate</div>
        <div class="metric-value" style="color:{'#2ecc71' if fill_rate>95 else '#e74c3c'}">{fill_rate:.1f}%</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Lost Sales</div>
        <div class="metric-value" style="color:{'#2ecc71' if lost<1 else '#e74c3c'}">{lost:.1f} units</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ════════════════════════════════════════════════════════════════════════════
elif page == " Model Comparison":

    st.title("Model Performance Comparison")

    st.info(f"Currently selected forecasting model: {active_model}")

    st.caption(
        "Multi-series evaluation across 5 stores and 10 items. "
        "Forecasts start on October 1, 2020. "
        "Metrics are pooled across all 50 store-item combinations."
    )

    results_path = Path(
        "models/multiseries/evaluation/horizon_comparison_v2.csv"
    )

    if results_path.exists():

        results_all = pd.read_csv(results_path)

        required = {
            "model", "horizon", "n_series",
            "MAE", "RMSE", "WAPE", "n_predictions"
        }

        missing = required - set(results_all.columns)

        if missing:
            st.error(f"Missing evaluation columns: {sorted(missing)}")
            st.stop()

        horizons = sorted(
            results_all["horizon"].dropna().astype(int).unique().tolist()
        )

        selected_horizon = st.select_slider(
            "Evaluation horizon (days)",
            options=horizons,
            value=30 if 30 in horizons else horizons[0],
            key="comparison_horizon"
        )

        results = (
            results_all[
                results_all["horizon"] == selected_horizon
            ]
            .copy()
            .sort_values("MAE")
            .reset_index(drop=True)
        )

        st.markdown(
            '<div class="section-header">Model Leaderboard</div>',
            unsafe_allow_html=True
        )

        best = results.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Best Model", best["model"])
        col2.metric("Best MAE", f'{best["MAE"]:.2f}')
        col3.metric("Best WAPE", f'{best["WAPE"]:.2f}%')

        display = results[
            ["model", "MAE", "RMSE", "WAPE",
             "n_series", "n_predictions"]
        ].rename(columns={
            "model": "Model",
            "n_series": "Series",
            "n_predictions": "Predictions",
            "WAPE": "WAPE (%)"
        })

        st.dataframe(
            display.style.highlight_min(
                subset=["MAE", "RMSE", "WAPE (%)"],
                color="#1a3a2a"
            ),
            width="stretch",
            hide_index=True
        )

        st.markdown(
            '<div class="section-header">Performance Metrics</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(3)

        for col, metric in zip(cols, ["MAE", "RMSE", "WAPE"]):

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=results["model"],
                    y=results[metric],
                    marker_color=[
                        "#2ecc71" if value == results[metric].min()
                        else "#4c72b0"
                        for value in results[metric]
                    ],
                    text=results[metric].round(2),
                    textposition="outside"
                )
            )

            fig.update_layout(
                title=metric if metric != "WAPE" else "WAPE (%)",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#8b9dc3"),
                height=320,
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#1e2130"
                ),
                margin=dict(t=60, b=40)
            )

            col.plotly_chart(fig, width="stretch")

        st.markdown(
            '<div class="section-header">Accuracy by Forecast Horizon</div>',
            unsafe_allow_html=True
        )

        metric_choice = st.selectbox(
            "Metric",
            ["MAE", "RMSE", "WAPE"],
            key="comparison_metric"
        )

        fig = go.Figure()

        colors = {
            "XGBoost": "#2ecc71",
            "Prophet": "#3498db",
            "SARIMA": "#e67e22",
            "LSTM": "#9b59b6"
        }

        for model_name in ["XGBoost", "Prophet", "SARIMA", "LSTM"]:

            model_data = (
                results_all[
                    results_all["model"] == model_name
                ]
                .sort_values("horizon")
            )

            fig.add_trace(
                go.Scatter(
                    x=model_data["horizon"],
                    y=model_data[metric_choice],
                    mode="lines+markers",
                    name=model_name,
                    line=dict(
                        color=colors.get(model_name, "#4c72b0"),
                        width=3
                    )
                )
            )

        fig.update_layout(
            xaxis_title="Forecast Horizon (Days)",
            yaxis_title=(
                metric_choice + " (%)"
                if metric_choice == "WAPE"
                else metric_choice
            ),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#8b9dc3"),
            height=400,
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(
                showgrid=True,
                gridcolor="#1e2130"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#1e2130"
            )
        )

        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Benchmark limitations: Models were not trained using "
            "identical cutoffs. XGBoost evaluation used the observed "
            "future promotion schedule. These results are preliminary "
            "and should not be interpreted as guaranteed future accuracy."
        )

    else:
        st.warning(
            "Multi-series evaluation results not found. "
            "Run models/evaluate_multiseries.py first."
        )

    st.markdown(
        '<div class="section-header">Model Architecture Overview</div>',
        unsafe_allow_html=True
    )

    arch = pd.DataFrame([
        {
            "Model": "LSTM V2",
            "Type": "Deep Learning — Direct Multi-Horizon",
            "Strengths": "60-day history, calendar features, direct 90-day predictions",
            "When to use": "Medium- and long-horizon retail forecasting"
        },

        {
            "Model": "SARIMA",
            "Type": "Statistical",
            "Strengths": "Trend and weekly seasonality",
            "When to use": "Interpretable series-specific forecasts"
        },
        {
            "Model": "Prophet",
            "Type": "Decomposition",
            "Strengths": "Trend and multiple seasonal patterns",
            "When to use": "Business-oriented seasonal forecasting"
        },
        {
            "Model": "XGBoost",
            "Type": "Gradient Boosting",
            "Strengths": "Multi-series learning and custom features",
            "When to use": "Retail forecasting across stores and items"
        },
        {
            "Model": "LSTM V1",
            "Type": "Deep Learning",
            "Strengths": "Temporal patterns and learned embeddings",
            "When to use": "Sequence-based demand forecasting"
        }
    ])

    st.dataframe(
        arch,
        width="stretch",
        hide_index=True
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == " EDA":
    st.title(" Exploratory Data Analysis")

    # Selected series
    st.markdown(f'<div class="section-header">Store {sel_store} · {sel_item} — Historical Sales</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ser["date"], y=ser["sales"],
        mode="lines", name="Sales", line=dict(color="#4c72b0", width=1.2)))
    fig.add_trace(go.Scatter(x=ser["date"], y=ser["sales"].rolling(30).mean(),
        mode="lines", name="30-day MA", line=dict(color="#e74c3c", width=2)))
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=350,
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Sales"),
        legend=dict(bgcolor="#1e2130"))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Monthly Seasonality</div>', unsafe_allow_html=True)
        monthly = ser.copy()
        monthly["month"] = monthly["date"].dt.month
        mon_avg = monthly.groupby("month")["sales"].mean().reset_index()
        mon_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        mon_avg["month_name"] = mon_avg["month"].map(mon_names)
        fig6 = go.Figure(go.Bar(x=mon_avg["month_name"], y=mon_avg["sales"],
            marker_color=px.colors.sequential.Blues[-len(mon_avg):]))
        fig6.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#8b9dc3"), height=320,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2130"))
        st.plotly_chart(fig6, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Sales Distribution</div>', unsafe_allow_html=True)
        fig7 = go.Figure()
        for label, mask, color in [
            ("No Promo", ser.onpromotion == 0, "#4c72b0"),
            ("On Promo", ser.onpromotion == 1, "#2ecc71"),
        ]:
            subset = ser[mask]["sales"]
            if len(subset) > 0:
                fig7.add_trace(go.Histogram(x=subset, name=label,
                    marker_color=color, opacity=0.7, nbinsx=40))
        fig7.update_layout(barmode="overlay", plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117", font=dict(color="#8b9dc3"),
            height=320, xaxis=dict(title="Sales"),
            legend=dict(bgcolor="#1e2130"))
        st.plotly_chart(fig7, width="stretch")

    # Lag correlation
    st.markdown('<div class="section-header">Autocorrelation (Lag Features)</div>', unsafe_allow_html=True)
    acf_vals = [ser["sales"].autocorr(lag=i) for i in range(1, 31)]
    fig8 = go.Figure(go.Bar(x=list(range(1, 31)), y=acf_vals,
        marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in acf_vals]))
    fig8.add_hline(y=0, line_color="#ffffff", line_width=1)
    fig8.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=300,
        xaxis=dict(title="Lag (days)", showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(title="Autocorrelation", showgrid=True, gridcolor="#1e2130"))
    st.plotly_chart(fig8, width="stretch")

    # Raw data
    with st.expander(" Raw Data Sample"):
        st.dataframe(ser.tail(50), width="stretch")

# ─────────────────────────────────────────────
# ForecastIQ Copyright Footer
# ─────────────────────────────────────────────

st.markdown("""
<style>
.forecastiq-footer {
    text-align: center;
    padding: 25px 10px;
    margin-top: 50px;
    border-top: 1px solid #334155;
    color: #CBD5E1;
    font-size: 13px;
}
.forecastiq-footer strong {
    color: #22D3EE;
}
</style>

<div class="forecastiq-footer">
    © 2026 <strong>Youssef MOUHI</strong>. All rights reserved.
    <br>
    ForecastIQ | AI-Powered Retail Demand Intelligence
</div>
""", unsafe_allow_html=True)
