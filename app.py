"""
Adversarial Sales Forecaster – Railway Web App
Wraps the preprocessing pipeline in a Flask interface.
"""

import os, io, base64, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from flask import Flask, render_template_string, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline functions (same logic as your local script)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sales_data(n_days=730):
    dates      = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    trend      = np.linspace(200, 350, n_days)
    day_of_week = np.array([d.dayofweek for d in dates])
    weekly     = np.where(day_of_week >= 4, 1.30, 1.0)
    t          = np.arange(n_days)
    seasonal   = (40 * np.sin(2 * np.pi * t / 365 + np.pi / 6)
                  + 60 * np.exp(-((t % 365 - 355) ** 2) / (2 * 15 ** 2)))
    promo      = np.random.choice([0, 1], size=n_days, p=[0.90, 0.10])
    promo_eff  = promo * np.random.uniform(30, 80, n_days)
    noise      = np.random.normal(0, 15, n_days)
    sales      = np.clip(trend * weekly + seasonal + promo_eff + noise, 50, None)

    anomaly_idx = np.random.choice(n_days, size=18, replace=False)
    for i, idx in enumerate(anomaly_idx):
        sales[idx] *= np.random.uniform(3.0, 5.0) if i % 2 == 0 else np.random.uniform(0.05, 0.15)

    missing_idx = np.random.choice(n_days, size=int(n_days * 0.03), replace=False)
    sales_m     = sales.copy().astype(float)
    sales_m[missing_idx] = np.nan

    df = pd.DataFrame({
        "date": dates, "sales": sales_m, "promotion": promo,
        "day_of_week": day_of_week, "month": [d.month for d in dates],
        "is_weekend": (day_of_week >= 5).astype(int),
    })
    df["is_anomaly_true"] = 0
    df.loc[anomaly_idx, "is_anomaly_true"] = 1
    return df, int(df["sales"].isna().sum()), len(anomaly_idx)


def run_pipeline():
    df, n_missing, n_injected = generate_sales_data()

    # KNN imputation
    features = ["sales", "day_of_week", "month", "is_weekend", "promotion"]
    df[features] = KNNImputer(n_neighbors=5).fit_transform(df[features])

    # IQR
    Q1, Q3 = df["sales"].quantile(0.25), df["sales"].quantile(0.75)
    IQR    = Q3 - Q1
    lower, upper = Q1 - 2.5 * IQR, Q3 + 2.5 * IQR
    df["iqr_anomaly"] = ((df["sales"] < lower) | (df["sales"] > upper)).astype(int)

    # Isolation Forest
    df["rolling_mean_7"]  = df["sales"].rolling(7,  min_periods=1).mean()
    df["rolling_std_7"]   = df["sales"].rolling(7,  min_periods=1).std().fillna(0)
    df["rolling_mean_30"] = df["sales"].rolling(30, min_periods=1).mean()
    df["sales_lag_1"]     = df["sales"].shift(1).fillna(df["sales"].mean())
    df["sales_lag_7"]     = df["sales"].shift(7).fillna(df["sales"].mean())
    feat_cols = ["sales","rolling_mean_7","rolling_std_7","rolling_mean_30",
                 "sales_lag_1","sales_lag_7","day_of_week","month","promotion"]
    X = StandardScaler().fit_transform(df[feat_cols])
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    preds = iso.fit_predict(X)
    df["iso_anomaly"] = (preds == -1).astype(int)
    df["iso_score"]   = iso.score_samples(X)

    # Ensemble
    df["is_anomaly"] = ((df["iqr_anomaly"] == 1) & (df["iso_anomaly"] == 1)).astype(int)

    # Evaluation
    tp = ((df["is_anomaly"]==1) & (df["is_anomaly_true"]==1)).sum()
    fp = ((df["is_anomaly"]==1) & (df["is_anomaly_true"]==0)).sum()
    fn = ((df["is_anomaly"]==0) & (df["is_anomaly_true"]==1)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # Correct anomalies
    df["sales_raw"] = df["sales"].copy()
    mask = df["is_anomaly"] == 1
    df.loc[mask, "sales"] = df["sales"].rolling(7, center=True, min_periods=1).median()[mask]

    stats = {
        "total_rows":       len(df),
        "missing_imputed":  n_missing,
        "injected_anomalies": n_injected,
        "iqr_flagged":      int(df["iqr_anomaly"].sum()),
        "iso_flagged":      int(df["iso_anomaly"].sum()),
        "ensemble_flagged": int(df["is_anomaly"].sum()),
        "precision":        round(float(precision), 2),
        "recall":           round(float(recall),    2),
        "f1":               round(float(f1),        2),
        "lower_bound":      round(float(lower), 1),
        "upper_bound":      round(float(upper), 1),
    }
    return df, stats


def make_chart(df):
    fig = plt.figure(figsize=(16, 12), facecolor="#0f1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    PANEL_BG = "#1a1d27"
    BLUE, RED, GREEN, ORANGE, GREY = "#4fc3f7","#ef5350","#66bb6a","#ffa726","#607d8b"

    def style(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for s in ax.spines.values(): s.set_edgecolor("#2d3142")
        ax.grid(axis="y", color="#2d3142", linewidth=0.5, linestyle="--")

    style(ax1, "Raw Sales with Anomalies")
    ax1.plot(df["date"], df["sales_raw"], color=BLUE, lw=0.8, alpha=0.85, label="Sales")
    ax1.scatter(df.loc[df["is_anomaly_true"]==1,"date"],
                df.loc[df["is_anomaly_true"]==1,"sales_raw"],
                color=ORANGE, s=55, zorder=5, label="True anomaly", marker="D")
    ax1.scatter(df.loc[df["is_anomaly"]==1,"date"],
                df.loc[df["is_anomaly"]==1,"sales_raw"],
                color=RED, s=75, zorder=6, label="Detected", marker="x", linewidths=2)
    ax1.set_ylabel("Units", color="#aaaaaa")
    ax1.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

    style(ax2, "IQR Distribution")
    Q1, Q3 = df["sales_raw"].quantile(0.25), df["sales_raw"].quantile(0.75)
    IQR = Q3 - Q1
    ax2.hist(df["sales_raw"].dropna(), bins=55, color=BLUE, alpha=0.6, edgecolor="none")
    ax2.axvline(Q1-2.5*IQR, color=RED,    ls="--", lw=1.5, label=f"Lower")
    ax2.axvline(Q3+2.5*IQR, color=RED,    ls="--", lw=1.5, label=f"Upper")
    ax2.axvline(Q1,          color=ORANGE, ls=":",  lw=1,   label="Q1/Q3")
    ax2.axvline(Q3,          color=ORANGE, ls=":",  lw=1)
    ax2.set_xlabel("Sales", color="#aaaaaa"); ax2.set_ylabel("Freq", color="#aaaaaa")
    ax2.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

    style(ax3, "Isolation Forest Scores")
    nm = df["iso_anomaly"]==0
    ax3.scatter(df.loc[nm,"date"],  df.loc[nm,"iso_score"],  color=BLUE, s=3,  alpha=0.5, label="Normal")
    ax3.scatter(df.loc[~nm,"date"], df.loc[~nm,"iso_score"], color=RED,  s=18, alpha=0.9, label="Anomaly", marker="x")
    ax3.set_ylabel("Score", color="#aaaaaa")
    ax3.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

    style(ax4, "Cleaned vs Raw Sales")
    ax4.plot(df["date"], df["sales_raw"], color=GREY,  lw=0.7, alpha=0.6, label="Raw")
    ax4.plot(df["date"], df["sales"],     color=GREEN, lw=1.0, alpha=0.9, label="Cleaned")
    ax4.set_ylabel("Units", color="#aaaaaa")
    ax4.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

    fig.suptitle("Sales Forecaster – Preprocessing & Anomaly Detection",
                 color="white", fontsize=13, fontweight="bold", y=0.99)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sales Forecaster</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0f1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif;min-height:100vh}
  header{background:linear-gradient(135deg,#1a237e,#283593);padding:24px 40px;
         border-bottom:1px solid #283593}
  header h1{font-size:1.6rem;font-weight:700;color:#fff}
  header p{color:#90caf9;font-size:.9rem;margin-top:4px}
  .container{max-width:1100px;margin:0 auto;padding:32px 24px}
  .run-btn{display:inline-block;background:linear-gradient(135deg,#1976d2,#1565c0);
           color:#fff;border:none;padding:14px 36px;border-radius:8px;font-size:1rem;
           font-weight:600;cursor:pointer;transition:.2s;text-decoration:none}
  .run-btn:hover{background:linear-gradient(135deg,#1565c0,#0d47a1);transform:translateY(-1px)}
  .run-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
  .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:28px 0}
  .stat-card{background:#1a1d27;border:1px solid #2d3142;border-radius:10px;padding:18px 20px}
  .stat-card .label{color:#90caf9;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}
  .stat-card .value{color:#fff;font-size:1.6rem;font-weight:700;margin-top:4px}
  .stat-card .sub{color:#607d8b;font-size:.75rem;margin-top:2px}
  .chart-wrap{background:#1a1d27;border:1px solid #2d3142;border-radius:10px;
              padding:20px;margin-top:8px;text-align:center}
  .chart-wrap img{max-width:100%;border-radius:6px}
  #status{margin:16px 0;color:#90caf9;font-size:.9rem;min-height:22px}
  .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
  .badge-green{background:#1b5e20;color:#a5d6a7}
  .badge-blue {background:#0d47a1;color:#90caf9}
  .badge-orange{background:#e65100;color:#ffcc80}
  .section-title{font-size:1.05rem;font-weight:600;color:#90caf9;
                 margin:28px 0 12px;border-left:3px solid #1976d2;padding-left:10px}
  footer{text-align:center;color:#37474f;font-size:.78rem;padding:24px;border-top:1px solid #1a1d27}
</style>
</head>
<body>
<header>
  <h1>⚡ Adversarial Sales Forecaster</h1>
  <p>Module 1 · Data Preprocessing &amp; Anomaly Detection Pipeline</p>
</header>

<div class="container">
  <p style="color:#b0bec5;margin-bottom:20px;line-height:1.6">
    This pipeline generates 2 years of dummy sales data, imputes missing values with
    <strong style="color:#90caf9">KNN</strong>, detects anomalies using
    <strong style="color:#90caf9">Isolation Forest</strong> +
    <strong style="color:#90caf9">IQR</strong> ensemble, corrects them,
    and visualises every step below.
  </p>

  <button class="run-btn" id="runBtn" onclick="runPipeline()">▶ Run Pipeline</button>
  <div id="status"></div>

  <div id="results" style="display:none">
    <div class="section-title">Pipeline Statistics</div>
    <div class="stats-grid" id="statsGrid"></div>

    <div class="section-title">Anomaly Detection Chart</div>
    <div class="chart-wrap">
      <img id="chartImg" src="" alt="Results chart">
    </div>
  </div>
</div>

<footer>Adversarial Sales Forecaster · Built with Flask + Scikit-learn · Deployed on Railway</footer>

<script>
async function runPipeline() {
  const btn = document.getElementById("runBtn");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "⏳ Running pipeline… this takes ~5 seconds";

  try {
    const res  = await fetch("/run");
    const data = await res.json();
    if (data.error) { status.textContent = "❌ Error: " + data.error; btn.disabled=false; return; }

    status.textContent = "✅ Pipeline complete!";

    const s = data.stats;
    const cards = [
      {label:"Total Rows",       value: s.total_rows,          sub:"2 years daily data",      badge:null},
      {label:"Missing Imputed",  value: s.missing_imputed,     sub:"via KNN (k=5)",            badge:"blue"},
      {label:"Injected Anomalies",value:s.injected_anomalies,  sub:"ground truth",             badge:"orange"},
      {label:"IQR Flagged",      value: s.iqr_flagged,         sub:`bounds [${s.lower_bound}, ${s.upper_bound}]`, badge:null},
      {label:"IF Flagged",       value: s.iso_flagged,         sub:"Isolation Forest",        badge:null},
      {label:"Ensemble Detected",value: s.ensemble_flagged,    sub:"IQR ∩ Isolation Forest",  badge:"green"},
      {label:"Precision",        value: s.precision,           sub:"no false positives ideal", badge:"green"},
      {label:"Recall",           value: s.recall,              sub:"anomalies caught",         badge:null},
      {label:"F1 Score",         value: s.f1,                  sub:"harmonic mean",            badge:null},
    ];

    document.getElementById("statsGrid").innerHTML = cards.map(c => `
      <div class="stat-card">
        <div class="label">${c.label}</div>
        <div class="value">${c.value}</div>
        <div class="sub">${c.sub}</div>
      </div>`).join("");

    document.getElementById("chartImg").src = "data:image/png;base64," + data.chart;
    document.getElementById("results").style.display = "block";
    btn.disabled = false;
    btn.textContent = "↺ Run Again";
  } catch(e) {
    status.textContent = "❌ Request failed: " + e;
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/run")
def run():
    try:
        df, stats = run_pipeline()
        chart     = make_chart(df)
        return jsonify({"stats": stats, "chart": chart})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
