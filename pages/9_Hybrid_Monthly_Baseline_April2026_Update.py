import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ee_auth import show_ee_connection_block
from utils.gee_presets import AOI_PRESETS, DATASET_PRESETS

st.set_page_config(page_title="Hybrid Baseline + Recent Weekly Update", page_icon="🧭", layout="wide")
st.title("🧭 Hybrid Monitoring: Historical Monthly Baseline + Recent Weekly Update")
st.caption(
    "Build a compact monitoring table for susceptibility updating: monthly historical baseline first, "
    "then 7-day update windows for the previous month + current month. If no dataset is found in 7 days, "
    "the app tries 8, 9, and 10 days."
)

show_ee_connection_block()

import ee

st.info(
    "Corrected design: historical data is monthly, but the recent/current monitoring period uses 7-day windows. "
    "For the current project timing, the recent update period starts from April 2026 and continues through May 2026. "
    "Use the April/May weekly rows to choose a target window in the Dynamic Flood Susceptibility Trainer."
)


def monthly_periods(start_date: str, end_date: str):
    starts = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="MS")
    rows = []
    for i, start in enumerate(starts):
        end = start + pd.offsets.MonthBegin(1)
        rows.append({
            "period_index": i,
            "period_type": "historical_monthly_baseline",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "nominal_days": int((end - start).days),
        })
    return rows


def weekly_periods(start_date: str, end_date: str, step_days: int = 7):
    starts = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq=f"{step_days}D")
    rows = []
    for i, start in enumerate(starts):
        end = min(start + pd.Timedelta(days=step_days), pd.to_datetime(end_date) + pd.Timedelta(days=1))
        rows.append({
            "period_index": i,
            "period_type": "recent_7day_update",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "nominal_days": int((end - start).days),
        })
    return rows


def build_period_table(hist_start, hist_end, update_start, update_end):
    hist = monthly_periods(hist_start, hist_end)
    updates = weekly_periods(update_start, update_end, 7)
    all_rows = hist + updates
    for i, row in enumerate(all_rows):
        row["global_index"] = i
    return all_rows


def adaptive_collection(base_collection, base_start: str, base_end: str, max_extra_days: int):
    """Return a collection that expands from 7 days up to 10 days if needed.

    The original window is used first. If no image/data exists, the end date is
    expanded by 1, 2, then 3 days. For a 7-day recent window, that means 8, 9,
    and 10 days. Historical monthly windows also use the same safety check, but
    they usually already contain data.
    """
    extensions = ee.List.sequence(0, max_extra_days)

    def candidate(extra):
        extra = ee.Number(extra)
        end = ee.Date(base_end).advance(extra, "day")
        count = base_collection.filterDate(base_start, end).size()
        return ee.Feature(None, {
            "extra_days": extra,
            "count": count,
            "actual_end": end.format("YYYY-MM-dd"),
        })

    candidates = ee.FeatureCollection(extensions.map(candidate))
    valid = candidates.filter(ee.Filter.gt("count", 0)).sort("extra_days")
    has_data = valid.size().gt(0)
    best = ee.Feature(ee.Algorithms.If(has_data, valid.first(), candidates.sort("extra_days", False).first()))
    extra_days = ee.Number(best.get("extra_days"))
    actual_end = ee.Date(base_end).advance(extra_days, "day")
    collection = base_collection.filterDate(base_start, actual_end)
    return {
        "collection": collection,
        "count": collection.size(),
        "extra_days": extra_days,
        "actual_end": actual_end.format("YYYY-MM-dd"),
        "found": has_data,
    }


def null_dict(keys):
    return ee.Dictionary({key: None for key in keys})


def build_hybrid_features(region, cfg):
    period_rows = build_period_table(
        cfg["hist_start"],
        cfg["hist_end"],
        cfg["update_start"],
        cfg["update_end"],
    )

    features = []

    for row in period_rows:
        start_s = row["start_date"]
        end_s = row["end_date"]
        props = dict(row)

        if cfg["use_s1"]:
            s1_base = (
                ee.ImageCollection(cfg["s1_id"])
                .filterBounds(region)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            )
            if cfg["orbit"] != "Either":
                s1_base = s1_base.filter(ee.Filter.eq("orbitProperties_pass", cfg["orbit"]))

            s1_info = adaptive_collection(s1_base, start_s, end_s, cfg["max_extra_days"])
            s1_col = s1_info["collection"]

            def s1_metrics():
                vv = s1_col.select("VV").median().clip(region)
                water_like = vv.gte(cfg["vv_min"]).And(vv.lte(cfg["vv_max"])).rename("sar_water_like")
                water_area = water_like.unmask(0).multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=region, scale=cfg["scale"], maxPixels=1e13
                ).get("sar_water_like")
                vv_mean = vv.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=cfg["scale"], maxPixels=1e13
                ).get("VV")
                return ee.Dictionary({
                    "sar_water_like_area_km2": ee.Number(water_area).divide(1e6),
                    "vv_mean_db": vv_mean,
                })

            s1_dict = ee.Dictionary(
                ee.Algorithms.If(
                    s1_info["found"],
                    s1_metrics(),
                    null_dict(["sar_water_like_area_km2", "vv_mean_db"]),
                )
            )
            props.update({
                "s1_image_count": s1_info["count"],
                "s1_extra_days_used": s1_info["extra_days"],
                "s1_actual_end": s1_info["actual_end"],
                "sar_water_like_area_km2": s1_dict.get("sar_water_like_area_km2"),
                "vv_mean_db": s1_dict.get("vv_mean_db"),
            })

        if cfg["use_chirps"]:
            rain_base = ee.ImageCollection(cfg["chirps_id"]).filterBounds(region)
            rain_info = adaptive_collection(rain_base, start_s, end_s, cfg["max_extra_days"])
            rain_col = rain_info["collection"]

            def rain_metrics():
                rain = rain_col.sum().rename("rain_total_mm").unmask(0)
                rain_mean = rain.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=max(cfg["scale"], 5000), maxPixels=1e13
                ).get("rain_total_mm")
                return ee.Dictionary({"rain_total_mean_mm": rain_mean})

            rain_dict = ee.Dictionary(
                ee.Algorithms.If(
                    rain_info["found"],
                    rain_metrics(),
                    null_dict(["rain_total_mean_mm"]),
                )
            )
            props.update({
                "chirps_image_count": rain_info["count"],
                "chirps_extra_days_used": rain_info["extra_days"],
                "chirps_actual_end": rain_info["actual_end"],
                "rain_total_mean_mm": rain_dict.get("rain_total_mean_mm"),
            })

        if cfg["use_smap"]:
            smap_base = ee.ImageCollection(cfg["smap_id"]).filterBounds(region).select("sm_surface")
            smap_info = adaptive_collection(smap_base, start_s, end_s, cfg["max_extra_days"])
            smap_col = smap_info["collection"]

            def smap_metrics():
                smap_img = smap_col.mean().rename("soil_moisture")
                smap_mean = smap_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=max(cfg["scale"], 9000), maxPixels=1e13
                ).get("soil_moisture")
                return ee.Dictionary({"soil_moisture_mean": smap_mean})

            smap_dict = ee.Dictionary(
                ee.Algorithms.If(
                    smap_info["found"],
                    smap_metrics(),
                    null_dict(["soil_moisture_mean"]),
                )
            )
            props.update({
                "smap_image_count": smap_info["count"],
                "smap_extra_days_used": smap_info["extra_days"],
                "smap_actual_end": smap_info["actual_end"],
                "soil_moisture_mean": smap_dict.get("soil_moisture_mean"),
            })

        if cfg["use_s2"]:
            s2_base = (
                ee.ImageCollection(cfg["s2_id"])
                .filterBounds(region)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["s2_cloud_pct"]))
            )
            s2_info = adaptive_collection(s2_base, start_s, end_s, cfg["max_extra_days"])
            s2_col = s2_info["collection"]

            def s2_metrics():
                s2_img = s2_col.median().divide(10000).clip(region)
                ndwi = s2_img.normalizedDifference(["B3", "B8"]).rename("NDWI")
                ndwi_mean = ndwi.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=max(cfg["scale"], 60), maxPixels=1e13
                ).get("NDWI")
                return ee.Dictionary({"ndwi_mean": ndwi_mean})

            s2_dict = ee.Dictionary(
                ee.Algorithms.If(
                    s2_info["found"],
                    s2_metrics(),
                    null_dict(["ndwi_mean"]),
                )
            )
            props.update({
                "s2_image_count": s2_info["count"],
                "s2_extra_days_used": s2_info["extra_days"],
                "s2_actual_end": s2_info["actual_end"],
                "ndwi_mean": s2_dict.get("ndwi_mean"),
            })

        features.append(ee.Feature(None, props))

    return ee.FeatureCollection(features)


with st.sidebar:
    st.header("1. Hybrid time design")
    st.caption("Historical context is monthly. The recent update period is previous month + current month using 7-day windows.")
    hist_start = st.text_input("Historical monthly start", "2018-01-01")
    hist_end = st.text_input("Historical monthly end", "2026-03-01")
    update_start = st.text_input("Recent weekly update start", "2026-04-01")
    update_end = st.text_input("Recent weekly update end", "2026-05-31")
    weekly_step = 7
    st.caption("Recent update interval is fixed at 7 days. If no data is found, the app tries 8, 9, and 10 days.")
    max_extra_days = st.select_slider("Auto-extend if no data", options=[0, 1, 2, 3], value=3)

    st.divider()
    st.header("2. Area of Interest")
    aoi_name = st.selectbox("AOI preset", list(AOI_PRESETS.keys()))
    bbox = AOI_PRESETS[aoi_name]["bbox"]
    st.caption(AOI_PRESETS[aoi_name]["description"])
    xmin = st.number_input("Min longitude", value=float(bbox[0]), step=0.01, format="%.4f")
    ymin = st.number_input("Min latitude", value=float(bbox[1]), step=0.01, format="%.4f")
    xmax = st.number_input("Max longitude", value=float(bbox[2]), step=0.01, format="%.4f")
    ymax = st.number_input("Max latitude", value=float(bbox[3]), step=0.01, format="%.4f")
    scale = st.select_slider("Analysis scale", [30, 50, 100, 250, 500, 1000], value=250)

    st.divider()
    st.header("3. Data streams")
    use_s1 = st.checkbox("Sentinel-1 SAR water-like area", value=True)
    use_chirps = st.checkbox("CHIRPS rainfall", value=True)
    use_smap = st.checkbox("SMAP soil moisture", value=True)
    use_s2 = st.checkbox("Sentinel-2 NDWI", value=True)

    st.divider()
    st.header("4. Dataset IDs and thresholds")
    s1_id = st.text_input("Sentinel-1 dataset", DATASET_PRESETS["Sentinel-1 SAR GRD"]["id"])
    chirps_id = st.text_input("CHIRPS dataset", DATASET_PRESETS["CHIRPS Daily Rainfall"]["id"])
    smap_id = st.text_input("SMAP dataset", DATASET_PRESETS["SMAP Soil Moisture"]["id"])
    s2_id = st.text_input("Sentinel-2 dataset", DATASET_PRESETS["Sentinel-2 Surface Reflectance Harmonized"]["id"])
    orbit = st.selectbox("Sentinel-1 orbit", ["Either", "ASCENDING", "DESCENDING"])
    vv_min = st.slider("VV water-like min dB", -30.0, -10.0, -20.0, 0.5)
    vv_max = st.slider("VV water-like max dB", -25.0, -5.0, -15.0, 0.5)
    s2_cloud_pct = st.slider("Sentinel-2 cloud max %", 5, 80, 40, 5)

cfg = {
    "hist_start": hist_start,
    "hist_end": hist_end,
    "update_start": update_start,
    "update_end": update_end,
    "weekly_step": int(weekly_step),
    "max_extra_days": int(max_extra_days),
    "scale": int(scale),
    "use_s1": use_s1,
    "use_chirps": use_chirps,
    "use_smap": use_smap,
    "use_s2": use_s2,
    "s1_id": s1_id,
    "chirps_id": chirps_id,
    "smap_id": smap_id,
    "s2_id": s2_id,
    "orbit": orbit,
    "vv_min": vv_min,
    "vv_max": vv_max,
    "s2_cloud_pct": s2_cloud_pct,
}

period_rows = build_period_table(hist_start, hist_end, update_start, update_end)
region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

c1, c2, c3 = st.columns(3)
c1.metric("Historical monthly rows", len([r for r in period_rows if r["period_type"] == "historical_monthly_baseline"]))
c2.metric("Recent 7-day rows", len([r for r in period_rows if r["period_type"] == "recent_7day_update"]))
c3.metric("Total periods", len(period_rows))

run_preview = st.button("Build hybrid monitoring preview", type="primary")
export_drive = st.button("Export full hybrid table to Google Drive")

if run_preview:
    try:
        with st.spinner("Building hybrid monthly/weekly Earth Engine table..."):
            fc = build_hybrid_features(region, cfg)
            rows = fc.getInfo().get("features", [])
            df = pd.DataFrame([f.get("properties", {}) for f in rows])
            st.session_state["hybrid_monitoring_df"] = df
            st.session_state["hybrid_monitoring_fc"] = fc
        st.success("Hybrid monitoring table created.")
    except Exception as exc:
        st.error("Could not build hybrid monitoring table.")
        st.exception(exc)

if export_drive:
    try:
        fc = build_hybrid_features(region, cfg)
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description="hobart_hybrid_monthly_baseline_recent_weekly_update",
            folder="GEE_Hobart_Flood_Project",
            fileNamePrefix="hobart_hybrid_monthly_baseline_recent_weekly_update",
            fileFormat="CSV",
        )
        task.start()
        st.success("Hybrid table export started.")
        st.json(task.status())
    except Exception as exc:
        st.error("Could not start export.")
        st.exception(exc)

if "hybrid_monitoring_df" in st.session_state:
    df = st.session_state["hybrid_monitoring_df"]
    st.subheader("Hybrid monitoring preview")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download preview CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="hybrid_monthly_baseline_recent_weekly_update.csv",
        mime="text/csv",
    )

    if "start_date" in df.columns:
        df_plot = df.copy()
        df_plot["start_date"] = pd.to_datetime(df_plot["start_date"])
        numeric_cols = [c for c in df_plot.columns if c not in ["start_date", "end_date"] and pd.api.types.is_numeric_dtype(df_plot[c])]
        if numeric_cols:
            chart_col = st.selectbox("Chart variable", numeric_cols)
            fig = px.line(df_plot, x="start_date", y=chart_col, color="period_type", title=f"Hybrid monitoring: {chart_col}")
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("How this supports updated susceptibility mapping")
st.markdown(
    """
1. Use the monthly rows from 2018 to March 2026 as the historical baseline/context.
2. Use the recent 7-day rows from April 2026 through May 2026 as the current update signal.
3. If a 7-day period has no data, the app checks 8, 9, and 10 days and records the extension in `_extra_days_used` columns.
4. Pick the recent week with high rainfall, high SAR water-like area, or high soil moisture.
5. Open **Dynamic Flood Susceptibility Trainer**.
6. Use that recent week as the target SAR label window and train a new updated susceptibility map.
"""
)
