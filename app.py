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

    n_pts = st.slider(
        "Data Points", 300, 1200, 600, 100,
        help=(
            "Controls how many spatial sample points are generated across the Hobart study area. "
            "Higher values give denser spatial coverage and smoother maps, "
            "but increase rendering time. 600 is a good balance for exploration."
        ),
    )
    r_seed = st.number_input(
        "Random Seed", 0, 9999, 42,
        help=(
            "Changing this generates a completely different random spatial arrangement "
            "of data points. Use this to test whether patterns and model results are "
            "stable across different realisations, or to explore sensitivity."
        ),
    )

    with st.expander("ℹ️ About the simulation data"):
        st.markdown("""
        All data displayed is **synthetically generated** but constrained to
        Hobart's real geography:

        - **Mt Wellington** (~1270 m) in the NW drives the elevation and slope gradient
        - **Derwent River** corridor is the primary flood pathway
        - **SAR backscatter** values follow realistic land-cover-dependent baselines
          (urban ≈ −8 dB, forest ≈ −14 dB, open water ≈ −20 dB)
        - **Flood labels** are derived from a logistic model using elevation,
          distance to river, TWI, and flow accumulation — mimicking real flood physics

        Once real Sentinel-1 imagery from Google Earth Engine is connected,
        the same controls will drive actual satellite processing pipelines.
        """)

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
    st.markdown(
        '<p class="sub-title">Sentinel-1 SAR-based flood mapping & risk intelligence '
        '· May 2018 Hobart case study</p>',
        unsafe_allow_html=True,
    )

    st.info(
        "**What is this platform?**  "
        "This system is a research and prototyping environment for mapping the May 2018 Hobart flood "
        "using Sentinel-1 Synthetic Aperture Radar (SAR) satellite imagery. It also demonstrates how "
        "that retrospective mapping can evolve into a continuously updated flood-risk intelligence "
        "product for insurers, emergency managers, and local councils. "
        "All data shown is simulated from Hobart's real terrain until live GEE outputs are connected."
    )
    st.divider()

    # KPIs
    st.markdown("##### Simulation Snapshot")
    st.caption(
        "These metrics summarise the current synthetic dataset. "
        "Adjust **Data Points** or **Random Seed** in the sidebar and they update instantly."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    flood_pct   = df["flooded"].mean() * 100
    mean_risk   = df["flood_prob"].mean()
    median_chng = df.loc[df["flooded"] == 1, "sar_change_db"].median()
    high_risk   = (df["flood_prob"] > 0.6).sum()

    c1.metric(
        "Total Points", f"{len(df):,}",
        help="Total simulated spatial sample points across the Hobart study area.",
    )
    c2.metric(
        "Flooded Points", f"{df['flooded'].sum():,}", f"{flood_pct:.1f}%",
        help="Points classified as flooded in the simulation. In the real workflow this comes from SAR change detection.",
    )
    c3.metric(
        "Mean Flood Risk", f"{mean_risk:.3f}",
        help="Average modelled flood probability (0–1) across all points. Higher = more area at risk on average.",
    )
    c4.metric(
        "High-Risk Points", f"{high_risk:,}", "> 0.6 probability",
        help="Points where the true flood probability exceeds 0.6 — the zone where flood occurrence is more likely than not.",
    )
    c5.metric(
        "Median SAR Change", f"{median_chng:.1f} dB", "flooded areas",
        help=(
            "Median SAR backscatter drop (dB) observed in flooded pixels. "
            "Open floodwater reflects the radar signal away from the satellite, causing a drop typically between −3 and −8 dB."
        ),
    )

    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Research Phase Progress")
        st.caption(
            "Six sequential phases take this project from a single retrospective flood map "
            "to a commercial near-real-time risk platform. Percentages reflect current completion."
        )
        phases = {
            "Phase 1 · Retrospective Flood Mapping": 70,
            "Phase 2 · Validation & Accuracy":       25,
            "Phase 3 · Susceptibility Modelling":    40,
            "Phase 4 · Automation Pipeline":          10,
            "Phase 5 · Dashboard Prototype":          30,
            "Phase 6 · Commercial Pilot":              5,
        }
        phase_desc = {
            "Phase 1 · Retrospective Flood Mapping":
                "Map the May 2018 Hobart flood using Sentinel-1 SAR imagery in Google Earth Engine.",
            "Phase 2 · Validation & Accuracy":
                "Assess map accuracy against available Tasmanian flood reference datasets.",
            "Phase 3 · Susceptibility Modelling":
                "Train ML models on terrain, hydrology, and SAR features to predict future flood likelihood.",
            "Phase 4 · Automation Pipeline":
                "Schedule automatic SAR ingestion and processing so the system self-updates.",
            "Phase 5 · Dashboard Prototype":
                "Build an interactive product interface for non-technical end users.",
            "Phase 6 · Commercial Pilot":
                "Demo to insurers (e.g. RACT) and emergency agencies; refine based on feedback.",
        }
        for phase, pct in phases.items():
            st.markdown(f"**{phase}**")
            st.caption(phase_desc[phase])
            st.progress(pct / 100, text=f"{pct}%")

    with col_r:
        st.subheader("Flood Susceptibility Distribution")
        st.caption(
            "How flood probability is distributed across the study area. "
            "A bimodal shape (peaks near 0 and 1) means the terrain clearly separates "
            "safe high-ground from vulnerable low-lying areas — a good signal for modelling."
        )
        fig_risk = px.histogram(
            df, x="flood_prob", nbins=40,
            color_discrete_sequence=["#1E88E5"],
            labels={"flood_prob": "Flood Probability"},
            title="Flood Susceptibility Distribution",
        )
        fig_risk.update_layout(showlegend=False, height=340, margin=dict(t=40))
        st.plotly_chart(fig_risk, use_container_width=True)

        with st.expander("How to read this chart"):
            st.markdown("""
            - **X-axis** — modelled flood probability from 0 (no risk) to 1 (certain flood)
            - **Y-axis** — number of spatial points at that probability level
            - A **peak near 0** = large area of safe high ground (Mt Wellington slopes)
            - A **peak near 1** = significant low-lying flood-prone zone (Derwent River flats)
            - A flat or unimodal distribution would suggest weak terrain control over flood risk
            """)

    # Overview map
    st.subheader("Study Area — Hobart, Tasmania")
    st.caption(
        "Click any point for its flood status, elevation, and risk score. "
        "Red = flooded in simulation, Green = not flooded. Only 150 points are shown for map performance."
    )
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
    st.caption("🔴 Flooded  🟢 Not flooded — 150 sampled points | Zoom and pan freely")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SAR FLOOD MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

with t2:
    st.header("🛰️ Sentinel-1 SAR Flood Detection")

    st.info(
        "**What this tab does:** Simulates the core SAR flood detection algorithm. "
        "When a surface floods, open water reflects the radar signal *away* from the satellite, "
        "causing a measurable drop in backscatter (measured in decibels, dB). "
        "This tab lets you control the detection threshold and masking rules, "
        "and shows in real-time how those choices affect flood extent, precision, and recall. "
        "In the real workflow these parameters are applied to Sentinel-1 GRD imagery inside Google Earth Engine."
    )

    col_ctrl, col_map = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Detection Parameters")

        with st.expander("ℹ️ How SAR flood detection works"):
            st.markdown("""
            **Sentinel-1** emits microwave pulses and measures how much energy bounces back
            (backscatter, in dB). The key steps are:

            1. **Pre-flood image** — acquired days/weeks before the event; represents normal surface conditions
            2. **Flood-period image** — acquired during or just after the flood peak
            3. **Change calculation** — flood image minus pre-flood image (dB)
            4. **Thresholding** — pixels where the change falls below a chosen threshold are classified as flooded

            **Why it works:** Calm floodwater acts like a mirror — the radar signal bounces *away*
            from the satellite (specular reflection), so flooded pixels appear much darker than normal.
            Typical drops range from −3 dB (shallow or vegetation-mixed flood) to −8 dB (open deep water).

            **Limitations:** Urban areas, dense vegetation, and steep terrain can confuse the signal.
            That's why masking filters are critical.
            """)

        threshold = st.slider(
            "Backscatter Change Threshold (dB)",
            min_value=-10.0, max_value=-0.5, value=-3.0, step=0.25,
            help=(
                "The key decision boundary. Pixels with a SAR change *below* this value are classified as flooded. "
                "A stricter (more negative) threshold reduces false positives but misses shallow floods. "
                "A lenient (less negative) threshold catches more flood but introduces more noise. "
                "Typical published values range from −2 dB to −4 dB."
            ),
        )

        polarisation = st.selectbox(
            "Polarisation", ["VV", "VH", "VV+VH Combined"],
            help=(
                "Sentinel-1 transmits and receives in two polarisations:\n\n"
                "**VV** (vertical-vertical): Better for detecting open water on flat surfaces. "
                "More sensitive to surface roughness changes caused by flooding.\n\n"
                "**VH** (vertical-horizontal): Better in vegetated areas — "
                "flooded vegetation produces a distinct cross-polarisation signal. "
                "Useful for detecting floods under crops or low shrubs.\n\n"
                "**VV+VH Combined**: Uses both channels together. Generally the most robust "
                "approach for mixed urban/vegetated/open areas like Hobart."
            ),
        )

        orbit_dir = st.selectbox(
            "Orbit Direction", ["Ascending", "Descending"],
            help=(
                "Sentinel-1 passes over Tasmania on both ascending (south-to-north, evening) "
                "and descending (north-to-south, morning) orbits. "
                "It is important to compare images from the **same orbit direction** to avoid "
                "false change signals caused by differences in viewing geometry. "
                "Hobart's mountainous terrain (Mt Wellington) means radar shadows and layover "
                "patterns differ between orbits."
            ),
        )

        st.divider()
        st.subheader("Masking Filters")
        st.caption(
            "Masking removes pixels that would cause systematic errors in flood detection. "
            "Always apply these before interpreting results."
        )

        rm_perm = st.checkbox(
            "Remove Permanent Water", value=True,
            help=(
                "Excludes rivers, reservoirs, and coastal water that appear dark in SAR year-round. "
                "Without this mask, the Derwent River and Hobart's harbour would always be classified as flooded. "
                "In GEE this is typically done using the JRC Global Surface Water dataset."
            ),
        )
        rm_slope = st.checkbox(
            "Mask Steep Slopes", value=True,
            help=(
                "Steep terrain causes two SAR artefacts that mimic floodwater:\n\n"
                "**Radar shadow** — the back side of a hill is not illuminated by the radar pulse, "
                "creating a dark patch that looks like water.\n\n"
                "**Layover** — steep slopes facing the satellite appear compressed and bright, "
                "confusing change detection.\n\n"
                "Masking slopes above the threshold removes these artefacts at the cost of "
                "missing any genuine floods on sloped terrain."
            ),
        )
        slope_thresh = st.slider(
            "Slope Threshold (°)", 5, 40, 15, disabled=not rm_slope,
            help=(
                "Pixels steeper than this angle are excluded from flood classification. "
                "15° is a common default. Reduce it to be more conservative in mountainous areas; "
                "increase it if you want to capture flood signals on gentle hills."
            ),
        )

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
        st.subheader("Live Detection Metrics")
        st.caption(
            "These update instantly as you move the sliders above. "
            "They measure how well the SAR threshold matches the simulated ground truth."
        )

        m1, m2 = st.columns(2)
        m1.metric(
            "Detected Flooded", f"{mask.sum():,}",
            help="Total pixels classified as flooded by the current threshold and masking settings.",
        )
        m2.metric(
            "True Positives", f"{tp:,}",
            help="Pixels correctly identified as flooded (detected AND actually flooded in the simulation).",
        )
        m1.metric(
            "Precision", f"{prec:.3f}",
            help=(
                "Of all pixels the algorithm flagged as flooded, what fraction were actually flooded? "
                "Low precision = many false alarms (non-flooded areas mis-classified as flooded). "
                "Critical for insurance use cases where false alarms trigger unnecessary payouts."
            ),
        )
        m2.metric(
            "Recall", f"{rec:.3f}",
            help=(
                "Of all pixels that were actually flooded, what fraction did the algorithm detect? "
                "Low recall = algorithm misses real floods. "
                "Critical for emergency response where missing a flooded area could be dangerous."
            ),
        )
        st.metric(
            "F1 Score", f"{f1_val:.3f}",
            help=(
                "Harmonic mean of Precision and Recall. Balances both concerns into a single number. "
                "1.0 = perfect detection. 0.0 = complete failure. "
                "A score above 0.7 is generally considered acceptable for SAR flood mapping."
            ),
        )

        with st.expander("ℹ️ Precision vs Recall trade-off"):
            st.markdown("""
            Moving the threshold slider demonstrates the **precision-recall trade-off**:

            - **Threshold → less negative (e.g. −1 dB):** More pixels classified as flooded.
              Recall increases (fewer missed floods) but precision drops (more false alarms).

            - **Threshold → more negative (e.g. −7 dB):** Fewer pixels classified.
              Precision increases (only confident detections) but recall drops (shallow floods missed).

            The **F1 score** is the recommended single metric when you want to balance both.
            For emergency response, favour high recall. For insurance pricing, favour high precision.
            """)

    with col_map:
        st.subheader(f"Flood Extent Map  (Δ < {threshold} dB)")
        st.caption(
            "**Heatmap** shows the underlying true flood probability (blue = low → red = high). "
            "**Blue circle markers** are pixels that the current SAR threshold has flagged as flooded. "
            "Click any marker to see its exact backscatter change, elevation, and true label."
        )

        m_sar = folium.Map(location=[-42.88, 147.33], zoom_start=12, tiles="CartoDB positron")

        heat_pts = [[r.lat, r.lon, r.flood_prob] for r in df.itertuples()]
        HeatMap(heat_pts, radius=12, blur=14,
                gradient={"0.2": "blue", "0.5": "cyan", "0.75": "lime", "1.0": "red"}
                ).add_to(m_sar)

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

    st.divider()
    st.subheader("SAR Backscatter Analysis")
    st.caption(
        "These three charts help you understand *why* a particular threshold is appropriate "
        "and how much separation exists between flooded and non-flooded SAR signals."
    )

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
        st.caption(
            "Overlap between pre-flood (green) and flood-period (blue) distributions shows how much "
            "ambiguity exists. A clear separation means the threshold is easy to set. "
            "Heavy overlap means misclassifications are unavoidable with a simple threshold."
        )

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
        st.caption(
            "Box plot of SAR change (dB) split by true flood label. "
            "Flooded pixels (red) should cluster well below zero. "
            "Outliers in the green box (not flooded but showing large drops) are potential false positives."
        )

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
        st.caption(
            "**Top-left (TN):** Correctly identified as not flooded. "
            "**Top-right (FP):** False alarms — dry areas called flooded. "
            "**Bottom-left (FN):** Missed floods. "
            "**Bottom-right (TP):** Correctly detected floods. "
            "Aim to minimise FP + FN simultaneously."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SUSCEPTIBILITY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

with t3:
    st.header("🤖 Flood Susceptibility Model")

    st.info(
        "**What this tab does:** Trains a machine learning classifier to predict *which areas are "
        "inherently susceptible to flooding* based on terrain, hydrology, rainfall, and SAR features — "
        "not just whether they flooded in May 2018. This is the step that transforms the project from "
        "a one-off historical map into a forward-looking risk intelligence tool. "
        "You can choose the algorithm, select which physical variables to include, tune hyperparameters, "
        "and immediately see the impact on model performance and the resulting susceptibility map."
    )

    col_cfg, col_out = st.columns([1, 2])

    with col_cfg:
        st.subheader("Model Configuration")

        algo = st.selectbox(
            "Algorithm",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"],
            help=(
                "**Random Forest:** Builds many decision trees on random data subsets and averages their votes. "
                "Robust to outliers, handles non-linear relationships, and provides reliable feature importance. "
                "Good default choice.\n\n"
                "**Gradient Boosting:** Builds trees sequentially, each correcting errors of the previous. "
                "Often achieves higher accuracy than Random Forest but is more sensitive to overfitting "
                "and hyperparameter choices.\n\n"
                "**Logistic Regression:** A linear model — fast, interpretable, and works well when "
                "features have roughly linear relationships with flood probability. "
                "Use this as a baseline to check whether non-linear models are actually adding value."
            ),
        )

        with st.expander("ℹ️ Which algorithm should I use?"):
            st.markdown("""
            | Algorithm | Best when | Watch out for |
            |---|---|---|
            | Random Forest | Good general-purpose baseline; noisy data | Slow on large datasets |
            | Gradient Boosting | You need maximum accuracy | Overfitting with few data points |
            | Logistic Regression | Interpretability matters; linear relationships | Misses complex patterns |

            **For this project:** Start with Random Forest for exploration.
            Use Gradient Boosting when you want to squeeze out more performance.
            Use Logistic Regression to verify that non-linear models are justified.
            """)

        sel_feats = st.multiselect(
            "Predictor Features",
            options=ALL_FEATURES,
            default=ALL_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
            help=(
                "The physical variables used to predict flood susceptibility. "
                "Remove features one at a time and observe the change in F1 / AUC "
                "to understand how much each contributes. "
                "Features with near-zero importance in the bar chart can be safely removed."
            ),
        )

        with st.expander("ℹ️ What does each feature represent?"):
            st.markdown("""
            | Feature | What it captures | Why it matters for floods |
            |---|---|---|
            | **Elevation** | Height above sea level (m) | Low elevation = gravitational accumulation of water |
            | **Slope** | Steepness of terrain (°) | Gentle slopes drain slowly → longer inundation |
            | **Distance to River** | Proximity to Derwent River (km) | River overflow reaches closest areas first |
            | **Annual Rainfall** | Long-term precipitation (mm/yr) | Drives overall water availability |
            | **TWI** | Topographic Wetness Index | Combines slope and contributing area; high TWI = natural water collection point |
            | **Flow Accumulation** | How much upslope area drains through a point | High values = natural drainage convergence zones |
            | **SAR Pre-flood** | Normal-conditions backscatter (dB) | Land cover proxy; urban vs forest vs water baseline |
            """)

        test_pct = st.slider(
            "Test Split (%)", 10, 40, 20,
            help=(
                "Percentage of data held out for evaluation (never seen during training). "
                "A 20% split is standard. Reducing it gives the model more training data "
                "but makes the accuracy estimate noisier. "
                "The 5-fold cross-validation score is more reliable than this single split."
            ),
        )

        st.subheader("Hyperparameters")
        st.caption("These control model complexity. Hover each control for guidance.")

        if algo in ("Random Forest", "Gradient Boosting"):
            n_est = st.slider(
                "n_estimators", 20, 400, 100, 20,
                help=(
                    "Number of decision trees to build. More trees = more stable predictions "
                    "and lower variance, but slower training. "
                    "Performance usually plateaus around 100–200 trees. "
                    "If training feels slow, reduce this first."
                ),
            )
            max_dep = st.select_slider(
                "max_depth", [2, 3, 4, 5, 7, 10, "None"], value=5,
                help=(
                    "Maximum depth of each decision tree. Shallow trees (2–3) underfit — "
                    "they're too simple to capture complex flood patterns. "
                    "Deep trees (10+) risk overfitting — they memorise training data instead of generalising. "
                    "Start at 5 and adjust based on whether train vs test F1 diverges."
                ),
            )
            max_dep = None if max_dep == "None" else int(max_dep)
        else:
            n_est, max_dep = 100, 5

        run_btn = st.button("Train Model", type="primary", use_container_width=True)
        st.caption("Model retrains automatically when any setting changes.")

    with col_out:
        if not sel_feats:
            st.warning("Select at least one feature on the left to train the model.")
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
        st.caption(
            "Evaluated on the held-out test set. These numbers tell you how well the model "
            "generalises to unseen locations — the key question for real-world deployment."
        )

        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        pm1.metric("Accuracy",  f"{acc_s:.3f}", help="Overall fraction of correct predictions (flooded + not flooded).")
        pm2.metric("Precision", f"{prec_s:.3f}", help="Of predicted flooded areas, what fraction was truly flooded?")
        pm3.metric("Recall",    f"{rec_s:.3f}", help="Of all truly flooded areas, what fraction did the model find?")
        pm4.metric("F1",        f"{f1_s:.3f}", help="Harmonic mean of Precision and Recall. Primary model comparison metric.")
        pm5.metric("ROC-AUC",   f"{auc_s:.3f}", help="Area Under ROC Curve. 0.5 = random guessing, 1.0 = perfect. Above 0.85 is strong.")

        st.caption(
            f"**5-fold Cross-Validation F1:** {cv_f1.mean():.3f} ± {cv_f1.std():.3f}  "
            "— more reliable than the single test split. High std means results vary across folds (potential overfitting)."
        )

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
            st.caption(
                "The ROC curve shows the precision-recall trade-off at every possible decision threshold. "
                "A curve hugging the top-left corner is ideal. "
                "The dashed diagonal = random guessing. "
                "Use this to choose an operating threshold suited to your use case."
            )

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
                st.caption(
                    "Importance = how much each feature reduces prediction error across all trees. "
                    "Features near zero can be removed without significant accuracy loss. "
                    "High-importance features are the key physical drivers of flood susceptibility in Hobart."
                )
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
                st.caption(
                    "Positive coefficients → higher feature value = higher flood probability. "
                    "Negative coefficients → higher feature value = lower flood probability. "
                    "Magnitude shows strength of the relationship."
                )

        cm2 = confusion_matrix(y_te, y_pred)
        fig_cm2 = px.imshow(cm2, text_auto=True,
                             labels=dict(x="Predicted", y="Actual"),
                             x=["Not Flooded", "Flooded"],
                             y=["Not Flooded", "Flooded"],
                             color_continuous_scale="Blues",
                             title="Test Set Confusion Matrix")
        fig_cm2.update_layout(height=270, margin=dict(t=35))
        st.plotly_chart(fig_cm2, use_container_width=True)
        st.caption(
            "The confusion matrix on the held-out test set. "
            "False Negatives (bottom-left) = flood-prone areas the model missed — dangerous for emergency planning. "
            "False Positives (top-right) = safe areas flagged as risky — costly for insurance pricing."
        )

    # Susceptibility map
    st.divider()
    st.subheader("Flood Susceptibility Map")
    st.caption(
        "Each point is coloured by the model's predicted flood probability (0–1). "
        "This is the core product output: a spatial risk layer that could be overlaid with "
        "property boundaries, infrastructure, or insurance portfolio data. "
        "Green = low susceptibility (high ground, far from river). "
        "Red = high susceptibility (low-lying, near Derwent, high TWI)."
    )

    susc = mdl.predict_proba(Xs)[:, 1]
    df_s = df.assign(susceptibility=susc)

    m_s = folium.Map(location=[-42.88, 147.33], zoom_start=12, tiles="CartoDB positron")
    heat_s = [[r.lat, r.lon, r.susceptibility] for r in df_s.itertuples()]
    HeatMap(heat_s, radius=15, blur=12,
            gradient={"0.2": "green", "0.45": "yellow", "0.65": "orange", "1.0": "red"}
            ).add_to(m_s)
    st_folium(m_s, height=430, use_container_width=True)

    with st.expander("ℹ️ How to use this map operationally"):
        st.markdown("""
        This susceptibility map answers: *"Which parts of Hobart are structurally at risk of flooding?"*
        — independent of whether a specific flood event is happening right now.

        **For insurers:** Overlay property boundaries to calculate portfolio flood exposure.
        Properties in red zones should carry higher flood risk premiums.

        **For emergency managers:** Use the red/orange zones to pre-position resources
        before forecast rainfall events.

        **For councils:** Identify which planned developments fall in high-susceptibility zones
        and require flood mitigation measures.

        **Next step in the research:** Validate this map against official flood extent data
        to confirm the modelled susceptibility reflects observed flood patterns.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

with t4:
    st.header("📈 Continuous Flood Monitoring Simulation")

    st.info(
        "**What this tab does:** Demonstrates what a continuously updated flood monitoring system "
        "would look like. Sentinel-1 revisits Tasmania roughly every 6–12 days, so SAR-derived "
        "flood indicators can be refreshed on that cadence. This tab simulates two years of "
        "daily flood risk scores, SAR backscatter trends, and rainfall — with the May 2018 "
        "Hobart flood event marked as a real reference point. "
        "Adjust the alert threshold to see how many high-risk days would trigger notifications."
    )

    col_mon, col_cfg2 = st.columns([3, 1])

    with col_cfg2:
        st.subheader("Display Controls")

        alert_thr = st.slider(
            "Alert Threshold", 0.20, 0.90, 0.55, 0.05,
            help=(
                "Risk score above which an automated flood alert would be triggered. "
                "Lower values → more sensitive (more alerts, fewer missed events). "
                "Higher values → fewer alerts (only extreme events trigger). "
                "In an operational system this would be calibrated against historical events "
                "and user tolerance for false alarms."
            ),
        )

        show_win = st.select_slider(
            "Time Window", ["3 mo", "6 mo", "1 yr", "2 yr"], value="1 yr",
            help=(
                "How much of the simulated record to display. "
                "Use '2 yr' to see the full seasonal cycle and the May 2018 flood spike. "
                "Use '3 mo' to zoom into short-term variability."
            ),
        )

        show_sar  = st.checkbox(
            "Show SAR Backscatter", value=True,
            help=(
                "SAR mean backscatter (dB) over the study area. "
                "When flooding occurs, the area-average backscatter drops noticeably. "
                "Correlating this with the risk index confirms that the signal is SAR-driven."
            ),
        )
        show_rain = st.checkbox(
            "Show Rainfall", value=True,
            help=(
                "Simulated daily rainfall (mm). "
                "In the real system this would come from the Bureau of Meteorology or ERA5. "
                "Rainfall precedes the SAR-detected flood by 1–3 days — "
                "this lag is visible in the charts when both are displayed."
            ),
        )

        window_days = {"3 mo": 90, "6 mo": 180, "1 yr": 365, "2 yr": 730}[show_win]
        ts_w = ts.head(window_days)

        alert_days = int((ts_w["risk"] > alert_thr).sum())
        may18 = ts_w[ts_w["date"].dt.strftime("%Y-%m") == "2018-05"]
        peak_risk = may18["risk"].max() if len(may18) else float("nan")

        st.divider()
        st.metric(
            "Alert Days", alert_days, f">{alert_thr:.2f} in window",
            help="Number of days in the selected window where the risk score exceeded the alert threshold.",
        )
        if not np.isnan(peak_risk):
            st.metric(
                "May 2018 Peak Risk", f"{peak_risk:.2f}",
                help="Highest simulated risk score during the May 2018 flood month.",
            )
            if peak_risk > alert_thr:
                st.error("⚠️ FLOOD ALERT triggered for May 2018")
            else:
                st.success("May 2018 below current threshold")

        with st.expander("ℹ️ How the risk score is built"):
            st.markdown("""
            The simulated risk score combines:

            1. **Seasonal baseline** — flood risk in Tasmania peaks in austral autumn/winter (April–August)
               when frontal rainfall is most frequent. The seasonal cycle accounts for ~28% of variance.

            2. **Noise** — day-to-day variability from changing SAR acquisition conditions,
               rainfall variability, and soil moisture fluctuations.

            3. **Event spike** — the May 2018 flood is injected as a sharp multi-day spike
               above the seasonal baseline, mimicking what SAR would detect during an actual event.

            In the real system the score would be a weighted combination of:
            SAR backscatter anomaly + rainfall accumulation + antecedent soil moisture + tide level.
            """)

    with col_mon:
        fig_risk_ts = go.Figure()
        fig_risk_ts.add_trace(go.Scatter(
            x=ts_w["date"], y=ts_w["risk"],
            mode="lines", name="Flood Risk Score",
            fill="tozeroy", line=dict(color="#1E88E5", width=1.5),
            fillcolor="rgba(30,136,229,0.18)",
        ))
        fig_risk_ts.add_hline(y=alert_thr, line_dash="dash", line_color="#EF5350",
                               annotation_text=f"Alert threshold: {alert_thr}")
        if "2018-05" in ts_w["date"].dt.strftime("%Y-%m").values:
            fig_risk_ts.add_vrect(
                x0="2018-05-01", x1="2018-05-31",
                fillcolor="red", opacity=0.12,
                annotation_text="May 2018 Flood", annotation_position="top left",
            )
        fig_risk_ts.update_layout(
            title="Flood Risk Index — Sentinel-1 derived (simulated)",
            yaxis_title="Risk Score (0 = no risk, 1 = extreme)",
            height=240, margin=dict(t=40, b=10), xaxis_title="",
        )
        st.plotly_chart(fig_risk_ts, use_container_width=True)
        st.caption(
            "The shaded red rectangle marks the May 2018 Hobart flood. "
            "The dashed red line is the alert threshold you set on the right. "
            "Days where the blue line crosses the dashed line would trigger an automated notification."
        )

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
                title="Area-Average SAR Backscatter",
                yaxis_title="dB (lower = more water on surface)",
                height=200, margin=dict(t=35, b=10), xaxis_title="",
            )
            st.plotly_chart(fig_sar_ts, use_container_width=True)
            st.caption(
                "Backscatter drops when more of the study area is covered with water. "
                "Notice the dip aligning with the May 2018 flood window. "
                "In the real system each Sentinel-1 overpass (~every 6–12 days) provides one data point."
            )

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
                title="Simulated Daily Rainfall",
                yaxis_title="mm/day",
                height=200, margin=dict(t=35, b=10),
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            st.caption(
                "Heavy rainfall events typically precede peak SAR flood detection by 1–3 days "
                "(time for water to accumulate in low-lying areas). "
                "Combining rainfall forecasts with the SAR risk index enables early warning."
            )

    st.divider()
    st.subheader("Monthly Risk Summary")
    st.caption(
        "Peak monthly flood risk over the full 2-year record. "
        "Red bars = high-risk months, green = low-risk months. "
        "This view helps identify which months historically carry the highest flood burden — "
        "useful for seasonal preparedness planning and insurance premium timing."
    )

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

    st.info(
        "**What this tab does:** Lets you interrogate the underlying simulation dataset — "
        "the same data that drives all other tabs. Use it to understand how each physical "
        "variable relates to flood occurrence, which features are most correlated with each other "
        "(potential multicollinearity in the ML model), and which features best separate "
        "flooded from non-flooded areas. You can also upload your own CSV to inspect it here."
    )

    col_d1, col_d2 = st.columns([1, 2])

    with col_d1:
        st.subheader("Dataset Summary Statistics")
        st.caption(
            "Descriptive statistics for each feature across all simulated points. "
            "Use these to sanity-check the simulation ranges against real Hobart values: "
            "elevation should peak near 1270 m (Mt Wellington), "
            "rainfall should be 600–1500 mm/yr depending on location."
        )
        st.dataframe(
            df[ALL_FEATURES + ["flooded"]].describe().T[["mean", "std", "min", "max"]].round(2),
            use_container_width=True,
        )

        st.subheader("Upload Your Own Data")
        st.caption(
            "Upload a CSV file with field measurements, GEE exports, or any tabular data. "
            "Useful for comparing real observations against simulation statistics, "
            "or for loading actual Sentinel-1 extracted values from GEE."
        )
        upf = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="Accepts any CSV. Columns do not need to match the simulation — you can explore any dataset here.",
        )
        if upf:
            udf = pd.read_csv(upf)
            st.success(f"Loaded: {len(udf)} rows × {len(udf.columns)} columns")
            st.dataframe(udf.head(8), use_container_width=True)
        else:
            st.info(
                "No file uploaded yet. Once you export data from Google Earth Engine "
                "or collect field observations, drop the CSV here to explore it."
            )

    with col_d2:
        st.subheader("Feature Distribution by Flood Status")
        st.caption(
            "Select a feature below to see how its distribution differs between flooded and non-flooded points. "
            "A clear separation between red and green peaks means the feature is a strong flood predictor. "
            "Heavy overlap means the feature adds little discriminative power on its own."
        )
        feat_sel = st.selectbox(
            "Feature to inspect", ALL_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
            help="Switch between features to find which ones best separate flooded from non-flooded areas.",
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
            title=f"{FEATURE_LABELS[feat_sel]} — Flooded vs Not Flooded",
        )
        st.plotly_chart(fig_fd, use_container_width=True)

        st.subheader("Feature Correlation Matrix")
        st.caption(
            "Pairwise Pearson correlations between all features and the flood label (−1 to +1). "
            "**Dark red** = strong positive correlation. **Dark blue** = strong negative correlation. "
            "Features that correlate strongly with 'flooded' are good predictors. "
            "Features that correlate strongly with *each other* may be redundant in the ML model "
            "(multicollinearity) — consider dropping one of them."
        )
        corr = df[ALL_FEATURES + ["flooded"]].corr()
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            text_auto=".2f", title="Feature Correlation",
        )
        fig_corr.update_layout(height=420, margin=dict(t=40))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    st.subheader("Scatter Matrix — Pairwise Relationships")
    st.caption(
        "Select 3–5 features to see all pairwise scatter plots at once. "
        "Each subplot shows two features plotted against each other, coloured by flood status. "
        "Look for clear *boundaries* between red and green clusters — those feature combinations "
        "are the ones your ML model will most easily learn to separate. "
        "Diagonal plots show the distribution of each feature individually."
    )

    scatter_sel = st.multiselect(
        "Features to include in scatter matrix",
        options=ALL_FEATURES,
        default=["elevation_m", "dist_river_km", "twi", "slope_deg"],
        format_func=lambda x: FEATURE_LABELS[x],
        help=(
            "Choose 3–5 features for best readability. "
            "Too many features makes the matrix hard to read. "
            "Start with the top features from the Susceptibility Model's importance chart."
        ),
    )
    if scatter_sel:
        fig_sm = px.scatter_matrix(
            df.sample(min(350, len(df)), random_state=0),
            dimensions=scatter_sel,
            color="flooded",
            color_discrete_map={0: "#43A047", 1: "#E53935"},
            opacity=0.45,
            labels=FEATURE_LABELS,
            title="Pairwise Feature Relationships (green = not flooded, red = flooded)",
        )
        fig_sm.update_layout(height=500)
        st.plotly_chart(fig_sm, use_container_width=True)
    else:
        st.info("Select at least two features above to generate the scatter matrix.")
