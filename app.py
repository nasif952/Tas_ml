import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, accuracy_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tasmania Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Push content below Streamlit's sticky top bar */
    .block-container { padding-top: 4.5rem !important; }

    /* Ensure the tab bar itself isn't clipped */
    div[data-testid="stTabs"] { margin-top: 0.5rem; }
    div[data-baseweb="tab-list"] { gap: 6px; flex-wrap: nowrap; overflow-x: auto; }

    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg,#1E88E5,#42A5F5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title { color: #90CAF9; font-size: 1rem; margin-top: 0; }
    .section-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #90CAF9;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ─── Synthetic data generators ───────────────────────────────────────────────

@st.cache_data
def make_hobart_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    lat = rng.uniform(-42.96, -42.78, n)
    lon = rng.uniform(147.18, 147.46, n)

    # Elevation: Mt Wellington in NW, valley near river
    mw_lat, mw_lon = -42.896, 147.234
    dist_mw = np.sqrt((lat - mw_lat) ** 2 + (lon - mw_lon) ** 2)
    elevation = 1200 * np.exp(-dist_mw ** 2 / 0.006) + rng.normal(0, 20, n)
    elevation = np.clip(elevation, 0, 1270)

    slope = np.clip(rng.gamma(2, 3, n) + elevation * 0.014, 0, 45)

    # Derwent River approximate path
    river_lon_at_lat = 147.335 + (lat + 42.88) * 0.5
    dist_river = np.sqrt((lon - river_lon_at_lat) ** 2) * 82  # km

    twi = np.clip(12 - slope * 0.15 - elevation * 0.003 + rng.normal(0, 1.5, n), 1, 18)
    rainfall = np.clip(750 + elevation * 0.25 + rng.normal(0, 60, n), 400, 1800)
    flow_acc = np.exp(-dist_river / 3) * 2000 + rng.exponential(100, n)
    land_cover = rng.choice([0, 1, 2, 3], n, p=[0.35, 0.35, 0.25, 0.05])

    sar_base = np.array([-8.0, -14.0, -11.0, -20.0])[land_cover]
    sar_pre = sar_base + rng.normal(0, 2, n)

    logit = -elevation * 0.008 - dist_river * 0.35 + twi * 0.28 + flow_acc * 0.00015 - slope * 0.06 + 1.8
    flood_prob = 1 / (1 + np.exp(-logit))
    flooded = (rng.random(n) < flood_prob).astype(int)

    sar_change = flooded * rng.uniform(-8, -3, n) + rng.normal(0, 1.2, n)
    sar_flood = sar_pre + sar_change

    return pd.DataFrame({
        "lat": lat, "lon": lon,
        "elevation_m": elevation,
        "slope_deg": slope,
        "dist_river_km": dist_river,
        "rainfall_mm": rainfall,
        "twi": twi,
        "flow_acc": flow_acc,
        "land_cover": land_cover,
        "sar_pre_db": sar_pre,
        "sar_flood_db": sar_flood,
        "sar_change_db": sar_change,
        "flood_prob": flood_prob,
        "flooded": flooded,
    })


@st.cache_data
def make_time_series() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    days = 730  # 2 years starting Jan 2017
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    doy = np.arange(days)

    seasonal = 0.28 * np.sin(2 * np.pi * (doy - 120) / 365)
    spike = np.zeros(days)
    flood_day = 487  # ~May 2018
    spike[flood_day : flood_day + 9] = rng.uniform(0.35, 0.72, 9)

    risk = np.clip(0.30 + seasonal + spike + rng.normal(0, 0.045, days), 0, 1)
    sar = -12 - risk * 4 + rng.normal(0, 0.8, days)
    rain = np.clip(seasonal * 10 + rng.exponential(2.5, days), 0, 90)

    return pd.DataFrame({"date": dates, "risk": risk, "sar_db": sar, "rain_mm": rain})


# ─── Sidebar ─────────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "elevation_m":    "Elevation (m)",
    "slope_deg":      "Slope (°)",
    "dist_river_km":  "Distance to River (km)",
    "rainfall_mm":    "Annual Rainfall (mm)",
    "twi":            "Topographic Wetness Index",
    "flow_acc":       "Flow Accumulation",
    "sar_pre_db":     "SAR Pre-flood (dB)",
}
ALL_FEATURES = list(FEATURE_LABELS.keys())

with st.sidebar:
    st.markdown("## 🌊 Tas Flood Intelligence")
    st.markdown("**Sentinel-1 SAR · May 2018 Hobart Flood**")
    st.divider()

    st.markdown('<p class="section-label">Simulation Settings</p>', unsafe_allow_html=True)
    n_pts   = st.slider("Data Points", 300, 1200, 600, 100)
    r_seed  = st.number_input("Random Seed", 0, 9999, 42)

    st.divider()
    st.markdown("**Study Area:** Hobart, Tasmania")
    st.markdown("**Satellite:** Sentinel-1 GRD")
    st.markdown("**Platform:** Google Earth Engine")
    st.markdown("**Polarisation:** VV / VH")

df  = make_hobart_dataset(n_pts, r_seed)
ts  = make_time_series()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

t1, t2, t3, t4, t5 = st.tabs([
    "🏠 Dashboard",
    "🛰️ SAR Flood Mapping",
    "🤖 Susceptibility Model",
    "📈 Monitoring",
    "📊 Data Explorer",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with t1:
    st.markdown('<p class="main-title">Tasmania Flood Intelligence System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Sentinel-1 SAR-based flood mapping & risk intelligence · May 2018 Hobart case study</p>', unsafe_allow_html=True)
    st.divider()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    flood_pct   = df["flooded"].mean() * 100
    mean_risk   = df["flood_prob"].mean()
    median_chng = df.loc[df["flooded"] == 1, "sar_change_db"].median()
    high_risk   = (df["flood_prob"] > 0.6).sum()

    c1.metric("Total Points",      f"{len(df):,}")
    c2.metric("Flooded Points",    f"{df['flooded'].sum():,}", f"{flood_pct:.1f}%")
    c3.metric("Mean Flood Risk",   f"{mean_risk:.3f}")
    c4.metric("High-Risk Points",  f"{high_risk:,}", "> 0.6 probability")
    c5.metric("Median SAR Change", f"{median_chng:.1f} dB", "flooded areas")

    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Research Phase Progress")
        phases = {
            "Phase 1 · Retrospective Flood Mapping": 70,
            "Phase 2 · Validation & Accuracy":       25,
            "Phase 3 · Susceptibility Modelling":    40,
            "Phase 4 · Automation Pipeline":          10,
            "Phase 5 · Dashboard Prototype":          30,
            "Phase 6 · Commercial Pilot":              5,
        }
        for phase, pct in phases.items():
            st.markdown(f"**{phase}**")
            st.progress(pct / 100, text=f"{pct}%")

    with col_r:
        fig_risk = px.histogram(
            df, x="flood_prob", nbins=40,
            color_discrete_sequence=["#1E88E5"],
            labels={"flood_prob": "Flood Probability"},
            title="Flood Susceptibility Distribution",
        )
        fig_risk.update_layout(showlegend=False, height=340, margin=dict(t=40))
        st.plotly_chart(fig_risk, use_container_width=True)

    # Overview map
    st.subheader("Study Area — Hobart, Tasmania")
    m_ov = folium.Map(location=[-42.88, 147.33], zoom_start=11, tiles="CartoDB positron")
    sample150 = df.sample(min(150, len(df)), random_state=0)
    for _, row in sample150.iterrows():
        clr = "#E53935" if row["flooded"] else "#43A047"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4, color=clr, fill=True, fill_opacity=0.65,
            popup=(
                f"Flooded: {'Yes' if row['flooded'] else 'No'}<br>"
                f"Elevation: {row['elevation_m']:.0f} m<br>"
                f"Risk: {row['flood_prob']:.2f}"
            ),
        ).add_to(m_ov)
    st_folium(m_ov, height=370, use_container_width=True)
    st.caption("🔴 Flooded  🟢 Not flooded — 150 sampled points")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SAR FLOOD MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

with t2:
    st.header("🛰️ Sentinel-1 SAR Flood Detection")
    st.markdown("Adjust detection parameters to explore how threshold choices affect flood extent classification.")

    col_ctrl, col_map = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Detection Parameters")
        threshold = st.slider(
            "Backscatter Change Threshold (dB)",
            min_value=-10.0, max_value=-0.5, value=-3.0, step=0.25,
            help="Pixels with SAR change below this are classified flooded",
        )
        polarisation = st.selectbox("Polarisation", ["VV", "VH", "VV+VH Combined"])
        orbit_dir    = st.selectbox("Orbit Direction", ["Ascending", "Descending"])

        st.divider()
        st.subheader("Masking Filters")
        rm_perm  = st.checkbox("Remove Permanent Water", value=True)
        rm_slope = st.checkbox("Mask Steep Slopes", value=True)
        slope_thresh = st.slider("Slope Threshold (°)", 5, 40, 15, disabled=not rm_slope)

        # Apply detection logic
        mask = df["sar_change_db"] < threshold
        if rm_perm:
            mask &= df["land_cover"] != 3
        if rm_slope:
            mask &= df["slope_deg"] < slope_thresh

        tp = int(( mask & (df["flooded"] == 1)).sum())
        fp = int(( mask & (df["flooded"] == 0)).sum())
        fn = int((~mask & (df["flooded"] == 1)).sum())
        tn = int((~mask & (df["flooded"] == 0)).sum())
        prec   = tp / (tp + fp + 1e-9)
        rec    = tp / (tp + fn + 1e-9)
        f1_val = 2 * prec * rec / (prec + rec + 1e-9)

        st.divider()
        st.subheader("Live Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Detected Flooded", f"{mask.sum():,}")
        m2.metric("True Positives",   f"{tp:,}")
        m1.metric("Precision", f"{prec:.3f}")
        m2.metric("Recall",    f"{rec:.3f}")
        st.metric("F1 Score", f"{f1_val:.3f}")

    with col_map:
        st.subheader(f"Flood Extent Map  (Δ < {threshold} dB)")

        m_sar = folium.Map(location=[-42.88, 147.33], zoom_start=12, tiles="CartoDB positron")

        # Probability heatmap background
        heat_pts = [[r.lat, r.lon, r.flood_prob] for r in df.itertuples()]
        HeatMap(heat_pts, radius=12, blur=14,
                gradient={"0.2": "blue", "0.5": "cyan", "0.75": "lime", "1.0": "red"}
                ).add_to(m_sar)

        # Detected flood markers (capped at 250 for performance)
        det_df = df[mask].sample(min(250, mask.sum()), random_state=0)
        for r in det_df.itertuples():
            folium.CircleMarker(
                location=[r.lat, r.lon], radius=5,
                color="#0D47A1", fill=True, fill_color="#42A5F5", fill_opacity=0.85,
                popup=(
                    f"SAR Δ: {r.sar_change_db:.1f} dB<br>"
                    f"Elevation: {r.elevation_m:.0f} m<br>"
                    f"True label: {'Flooded' if r.flooded else 'Not flooded'}"
                ),
            ).add_to(m_sar)

        st_folium(m_sar, height=460, use_container_width=True)
        st.caption("Blue markers = SAR-detected flood | Heatmap = true flood probability")

    st.divider()
    st.subheader("SAR Backscatter Analysis")

    ca, cb, cc = st.columns(3)

    with ca:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df["sar_pre_db"],   name="Pre-flood",   opacity=0.7, marker_color="#43A047"))
        fig_hist.add_trace(go.Histogram(x=df["sar_flood_db"], name="Flood period", opacity=0.7, marker_color="#1E88E5"))
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold {threshold} dB")
        fig_hist.update_layout(barmode="overlay", title="Backscatter Distributions",
                               xaxis_title="dB", height=280, margin=dict(t=35))
        st.plotly_chart(fig_hist, use_container_width=True)

    with cb:
        fig_box = px.box(
            df, x="flooded", y="sar_change_db",
            color="flooded",
            labels={"flooded": "Flooded", "sar_change_db": "SAR Δ (dB)"},
            title="Backscatter Change by Class",
            color_discrete_map={0: "#43A047", 1: "#E53935"},
        )
        fig_box.update_layout(height=280, margin=dict(t=35), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with cc:
        cm_arr = np.array([[tn, fp], [fn, tp]])
        fig_cm = px.imshow(
            cm_arr, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Flooded", "Flooded"], y=["Not Flooded", "Flooded"],
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        fig_cm.update_layout(height=280, margin=dict(t=35))
        st.plotly_chart(fig_cm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SUSCEPTIBILITY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

with t3:
    st.header("🤖 Flood Susceptibility Model")
    st.markdown("Train an ML model on terrain, hydrological, and SAR features to predict flood susceptibility.")

    col_cfg, col_out = st.columns([1, 2])

    with col_cfg:
        st.subheader("Model Configuration")
        algo = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Logistic Regression"])

        sel_feats = st.multiselect(
            "Predictor Features",
            options=ALL_FEATURES,
            default=ALL_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
        )

        test_pct = st.slider("Test Split (%)", 10, 40, 20)

        st.subheader("Hyperparameters")
        if algo in ("Random Forest", "Gradient Boosting"):
            n_est    = st.slider("n_estimators", 20, 400, 100, 20)
            max_dep  = st.select_slider("max_depth", [2, 3, 4, 5, 7, 10, "None"], value=5)
            max_dep  = None if max_dep == "None" else int(max_dep)
        else:
            n_est, max_dep = 100, 5

        run_btn = st.button("Train Model", type="primary", use_container_width=True)

    with col_out:
        if not sel_feats:
            st.warning("Select at least one feature on the left.")
            st.stop()

        X   = df[sel_feats].values
        y   = df["flooded"].values
        sc  = StandardScaler()
        Xs  = sc.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xs, y, test_size=test_pct / 100, random_state=42, stratify=y
        )

        if algo == "Random Forest":
            mdl = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, random_state=42, n_jobs=-1)
        elif algo == "Gradient Boosting":
            mdl = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_dep or 3, random_state=42)
        else:
            mdl = LogisticRegression(max_iter=600, random_state=42)

        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        y_prob = mdl.predict_proba(X_te)[:, 1]

        acc_s  = accuracy_score(y_te, y_pred)
        f1_s   = f1_score(y_te, y_pred)
        prec_s = precision_score(y_te, y_pred)
        rec_s  = recall_score(y_te, y_pred)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc_s  = auc(fpr, tpr)
        cv_f1  = cross_val_score(mdl, Xs, y, cv=5, scoring="f1")

        st.subheader("Performance Metrics")
        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        pm1.metric("Accuracy",  f"{acc_s:.3f}")
        pm2.metric("Precision", f"{prec_s:.3f}")
        pm3.metric("Recall",    f"{rec_s:.3f}")
        pm4.metric("F1",        f"{f1_s:.3f}")
        pm5.metric("ROC-AUC",   f"{auc_s:.3f}")

        st.caption(f"5-fold CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"AUC = {auc_s:.3f}",
                                          line=dict(color="#1E88E5", width=2.5)))
            fig_roc.add_shape(type="line", line=dict(dash="dash", color="gray"),
                               x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(title="ROC Curve",
                                   xaxis_title="False Positive Rate",
                                   yaxis_title="True Positive Rate",
                                   height=290, margin=dict(t=35))
            st.plotly_chart(fig_roc, use_container_width=True)

        with r1c2:
            if hasattr(mdl, "feature_importances_"):
                fi_df = pd.DataFrame({
                    "Feature": [FEATURE_LABELS[f] for f in sel_feats],
                    "Importance": mdl.feature_importances_,
                }).sort_values("Importance")
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                title="Feature Importance",
                                color="Importance", color_continuous_scale="Blues")
                fig_fi.update_layout(height=290, margin=dict(t=35), showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                coef_df = pd.DataFrame({
                    "Feature": [FEATURE_LABELS[f] for f in sel_feats],
                    "Coefficient": mdl.coef_[0],
                }).sort_values("Coefficient")
                fig_cf = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                                title="Logistic Coefficients",
                                color="Coefficient", color_continuous_scale="RdBu")
                fig_cf.update_layout(height=290, margin=dict(t=35))
                st.plotly_chart(fig_cf, use_container_width=True)

        # Confusion matrix
        cm2 = confusion_matrix(y_te, y_pred)
        fig_cm2 = px.imshow(cm2, text_auto=True,
                             labels=dict(x="Predicted", y="Actual"),
                             x=["Not Flooded", "Flooded"],
                             y=["Not Flooded", "Flooded"],
                             color_continuous_scale="Blues",
                             title="Test Set Confusion Matrix")
        fig_cm2.update_layout(height=270, margin=dict(t=35))
        st.plotly_chart(fig_cm2, use_container_width=True)

    # Susceptibility map
    st.divider()
    st.subheader("Flood Susceptibility Map")
    susc = mdl.predict_proba(Xs)[:, 1]
    df_s = df.assign(susceptibility=susc)

    m_s = folium.Map(location=[-42.88, 147.33], zoom_start=12, tiles="CartoDB positron")
    heat_s = [[r.lat, r.lon, r.susceptibility] for r in df_s.itertuples()]
    HeatMap(heat_s, radius=15, blur=12,
            gradient={"0.2": "green", "0.45": "yellow", "0.65": "orange", "1.0": "red"}
            ).add_to(m_s)
    st_folium(m_s, height=430, use_container_width=True)
    st.caption("Green = Low susceptibility → Red = High susceptibility")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

with t4:
    st.header("📈 Continuous Flood Monitoring Simulation")
    st.markdown(
        "Simulated Sentinel-1 derived flood risk index, SAR backscatter, and rainfall over time. "
        "The May 2018 Hobart flood event is highlighted."
    )

    col_mon, col_cfg2 = st.columns([3, 1])

    with col_cfg2:
        st.subheader("Controls")
        alert_thr = st.slider("Alert Threshold", 0.20, 0.90, 0.55, 0.05)
        show_win  = st.select_slider(
            "Window", ["3 mo", "6 mo", "1 yr", "2 yr"], value="1 yr"
        )
        show_sar  = st.checkbox("Show SAR Backscatter", value=True)
        show_rain = st.checkbox("Show Rainfall", value=True)

        window_days = {"3 mo": 90, "6 mo": 180, "1 yr": 365, "2 yr": 730}[show_win]
        ts_w = ts.head(window_days)

        alert_days = int((ts_w["risk"] > alert_thr).sum())
        may18 = ts_w[ts_w["date"].dt.strftime("%Y-%m") == "2018-05"]
        peak_risk = may18["risk"].max() if len(may18) else float("nan")

        st.divider()
        st.metric("Alert Days", alert_days, f">{alert_thr:.2f} in window")
        if not np.isnan(peak_risk):
            st.metric("May 2018 Peak Risk", f"{peak_risk:.2f}")
            if peak_risk > alert_thr:
                st.error("FLOOD ALERT triggered")
            else:
                st.success("Below alert threshold")

    with col_mon:
        # Risk score
        fig_risk_ts = go.Figure()
        fig_risk_ts.add_trace(go.Scatter(
            x=ts_w["date"], y=ts_w["risk"],
            mode="lines", name="Flood Risk Score",
            fill="tozeroy", line=dict(color="#1E88E5", width=1.5),
            fillcolor="rgba(30,136,229,0.18)",
        ))
        fig_risk_ts.add_hline(y=alert_thr, line_dash="dash", line_color="#EF5350",
                               annotation_text=f"Alert: {alert_thr}")
        if "2018-05" in ts_w["date"].dt.strftime("%Y-%m").values:
            fig_risk_ts.add_vrect(
                x0="2018-05-01", x1="2018-05-31",
                fillcolor="red", opacity=0.12,
                annotation_text="May 2018 Flood", annotation_position="top left",
            )
        fig_risk_ts.update_layout(
            title="Flood Risk Index (Sentinel-1 derived, simulated)",
            yaxis_title="Risk Score (0–1)", height=240,
            margin=dict(t=40, b=10), xaxis_title="",
        )
        st.plotly_chart(fig_risk_ts, use_container_width=True)

        if show_sar:
            fig_sar_ts = go.Figure()
            fig_sar_ts.add_trace(go.Scatter(
                x=ts_w["date"], y=ts_w["sar_db"],
                mode="lines", name="SAR Backscatter",
                line=dict(color="#AB47BC", width=1.3),
            ))
            if "2018-05" in ts_w["date"].dt.strftime("%Y-%m").values:
                fig_sar_ts.add_vrect(x0="2018-05-01", x1="2018-05-31",
                                      fillcolor="red", opacity=0.12)
            fig_sar_ts.update_layout(
                title="SAR Mean Backscatter", yaxis_title="dB", height=200,
                margin=dict(t=35, b=10), xaxis_title="",
            )
            st.plotly_chart(fig_sar_ts, use_container_width=True)

        if show_rain:
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=ts_w["date"], y=ts_w["rain_mm"],
                name="Rainfall", marker_color="#26A69A", opacity=0.75,
            ))
            if "2018-05" in ts_w["date"].dt.strftime("%Y-%m").values:
                fig_rain.add_vrect(x0="2018-05-01", x1="2018-05-31",
                                    fillcolor="red", opacity=0.12)
            fig_rain.update_layout(
                title="Simulated Daily Rainfall", yaxis_title="mm", height=200,
                margin=dict(t=35, b=10),
            )
            st.plotly_chart(fig_rain, use_container_width=True)

    st.divider()
    st.subheader("Monthly Risk Summary")
    ts_m = (ts.assign(month=ts["date"].dt.to_period("M"))
              .groupby("month")
              .agg(mean_risk=("risk", "mean"), max_risk=("risk", "max"),
                   total_rain=("rain_mm", "sum"))
              .reset_index()
              .assign(month_str=lambda d: d["month"].astype(str))
              .tail(24))

    fig_mon = px.bar(
        ts_m, x="month_str", y="max_risk",
        color="max_risk", color_continuous_scale="RdYlGn_r",
        title="Monthly Peak Flood Risk Score",
        labels={"month_str": "Month", "max_risk": "Peak Risk"},
    )
    fig_mon.update_layout(height=300, margin=dict(t=40))
    st.plotly_chart(fig_mon, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with t5:
    st.header("📊 Data Explorer")

    col_d1, col_d2 = st.columns([1, 2])

    with col_d1:
        st.subheader("Dataset Summary")
        st.dataframe(
            df[ALL_FEATURES + ["flooded"]].describe().T[["mean", "std", "min", "max"]].round(2),
            use_container_width=True,
        )

        st.subheader("Upload Your Own Data")
        upf = st.file_uploader("Upload CSV", type=["csv"])
        if upf:
            udf = pd.read_csv(upf)
            st.success(f"Loaded: {len(udf)} rows × {len(udf.columns)} columns")
            st.dataframe(udf.head(8), use_container_width=True)
        else:
            st.info("Upload a CSV to explore it here alongside the simulated data.")

    with col_d2:
        st.subheader("Feature Distribution by Flood Status")
        feat_sel = st.selectbox(
            "Feature", ALL_FEATURES, format_func=lambda x: FEATURE_LABELS[x]
        )
        fig_fd = go.Figure()
        fig_fd.add_trace(go.Histogram(
            x=df.loc[df["flooded"] == 0, feat_sel],
            name="Not Flooded", opacity=0.72, marker_color="#43A047", nbinsx=35,
        ))
        fig_fd.add_trace(go.Histogram(
            x=df.loc[df["flooded"] == 1, feat_sel],
            name="Flooded", opacity=0.72, marker_color="#E53935", nbinsx=35,
        ))
        fig_fd.update_layout(
            barmode="overlay", height=290, margin=dict(t=30),
            title=f"{FEATURE_LABELS[feat_sel]} by Flood Status",
        )
        st.plotly_chart(fig_fd, use_container_width=True)

        st.subheader("Correlation Matrix")
        corr = df[ALL_FEATURES + ["flooded"]].corr()
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            text_auto=".2f", title="Feature Correlation",
        )
        fig_corr.update_layout(height=420, margin=dict(t=40))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    st.subheader("Scatter Matrix — Pairwise Relationships")
    scatter_sel = st.multiselect(
        "Features",
        options=ALL_FEATURES,
        default=["elevation_m", "dist_river_km", "twi", "slope_deg"],
        format_func=lambda x: FEATURE_LABELS[x],
    )
    if scatter_sel:
        fig_sm = px.scatter_matrix(
            df.sample(min(350, len(df)), random_state=0),
            dimensions=scatter_sel,
            color="flooded",
            color_discrete_map={0: "#43A047", 1: "#E53935"},
            opacity=0.45,
            labels=FEATURE_LABELS,
            title="Pairwise Feature Relationships",
        )
        fig_sm.update_layout(height=500)
        st.plotly_chart(fig_sm, use_container_width=True)
    else:
        st.info("Select features above to build the scatter matrix.")
