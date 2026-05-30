import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ee_auth import show_ee_connection_block
from utils.gee_presets import AOI_PRESETS, DATASET_PRESETS, PAGE_HELP

st.set_page_config(page_title="Weekly GEE Time Series", page_icon="📈", layout="wide")
st.title("📈 Weekly Google Earth Engine Time Series")
st.caption(
    "Fetch continuous weekly indicators from Earth Engine. Default setup covers 2018 to 2026 at 7-day intervals for flood monitoring and long-term model updates."
)

show_ee_connection_block()

import ee

st.info(
    "This page creates a weekly table, not a weekly raster stack. It is designed for monitoring indicators such as SAR flood-like area, rainfall, soil moisture, and image availability. "
    "For hundreds of weeks, table export to Google Drive is safer than direct browser download."
)


def make_week_starts(start_date: str, end_date: str, step_days: int):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return pd.date_range(start=start, end=end, freq=f"{step_days}D")


def build_weekly_features(region, cfg):
    week_starts = make_week_starts(cfg["start_date"], cfg["end_date"], cfg["step_days"])
    features = []

    for idx, start in enumerate(week_starts):
        end = start + pd.Timedelta(days=cfg["step_days"])
        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")

        props = {
            "week_index": idx,
            "start_date": start_s,
            "end_date": end_s,
        }

        if cfg["use_s1"]:
            s1 = (
                ee.ImageCollection(cfg["s1_id"])
                .filterBounds(region)
                .filterDate(start_s, end_s)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            )
            if cfg["orbit"] != "Either":
                s1 = s1.filter(ee.Filter.eq("orbitProperties_pass", cfg["orbit"]))

            s1_count = s1.size()
            vv = s1.select("VV").median().clip(region)
            water_like = vv.gte(cfg["vv_min"]).And(vv.lte(cfg["vv_max"])).rename("sar_water_like")
            area = water_like.selfMask().multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=cfg["scale"],
                maxPixels=1e13,
            ).get("sar_water_like")
            vv_mean = vv.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=cfg["scale"],
                maxPixels=1e13,
            ).get("VV")
            props.update({
                "s1_image_count": s1_count,
                "sar_water_like_area_km2": ee.Number(area).divide(1e6),
                "vv_mean_db": vv_mean,
            })

        if cfg["use_chirps"]:
            rain = (
                ee.ImageCollection(cfg["chirps_id"])
                .filterBounds(region)
                .filterDate(start_s, end_s)
                .sum()
                .rename("weekly_rain_mm")
            )
            rain_mean = rain.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=max(cfg["scale"], 5000),
                maxPixels=1e13,
            ).get("weekly_rain_mm")
            props["weekly_rain_mean_mm"] = rain_mean

        if cfg["use_smap"]:
            smap = (
                ee.ImageCollection(cfg["smap_id"])
                .filterBounds(region)
                .filterDate(start_s, end_s)
                .select("sm_surface")
            )
            smap_count = smap.size()
            smap_mean_img = smap.mean().rename("soil_moisture")
            smap_mean = smap_mean_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=max(cfg["scale"], 9000),
                maxPixels=1e13,
            ).get("soil_moisture")
            props.update({
                "smap_image_count": smap_count,
                "soil_moisture_mean": smap_mean,
            })

        if cfg["use_s2"]:
            s2 = (
                ee.ImageCollection(cfg["s2_id"])
                .filterBounds(region)
                .filterDate(start_s, end_s)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["s2_cloud_pct"]))
            )
            s2_count = s2.size()
            # Use raw scaled SR bands only for weekly NDWI summary. Cloud masking is intentionally simple for speed.
            s2_median = s2.median().divide(10000).clip(region)
            ndwi = s2_median.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndwi_mean = ndwi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=max(cfg["scale"], 60),
                maxPixels=1e13,
            ).get("NDWI")
            props.update({
                "s2_image_count": s2_count,
                "ndwi_mean": ndwi_mean,
            })

        features.append(ee.Feature(None, props))

    return ee.FeatureCollection(features)


with st.sidebar:
    st.header("1. Time range")
    st.caption("Default range is continuous weekly monitoring from 2018 to 2026.")
    start_date = st.text_input("Start date", "2018-01-01")
    end_date = st.text_input("End date", "2026-12-31")
    step_days = st.select_slider("Step length", options=[6, 7, 10, 14, 30], value=7)
    st.caption("Use 7 days for weekly monitoring. Sentinel-1 revisit can be around 6-12 days depending on orbit/location, so some weeks may have no image.")

    st.divider()
    st.header("2. Area of Interest")
    aoi_name = st.selectbox("AOI preset", list(AOI_PRESETS.keys()))
    bbox = AOI_PRESETS[aoi_name]["bbox"]
    st.caption(AOI_PRESETS[aoi_name]["description"])
    xmin = st.number_input("Min longitude", value=float(bbox[0]), step=0.01, format="%.4f")
    ymin = st.number_input("Min latitude", value=float(bbox[1]), step=0.01, format="%.4f")
    xmax = st.number_input("Max longitude", value=float(bbox[2]), step=0.01, format="%.4f")
    ymax = st.number_input("Max latitude", value=float(bbox[3]), step=0.01, format="%.4f")
    scale = st.select_slider("Analysis scale", options=[30, 50, 100, 250, 500, 1000], value=250)

    st.divider()
    st.header("3. Weekly indicators")
    st.caption("Select the data streams you want to fetch each week.")
    use_s1 = st.checkbox("Sentinel-1 SAR water-like area", value=True)
    use_chirps = st.checkbox("CHIRPS weekly rainfall", value=True)
    use_smap = st.checkbox("SMAP soil moisture", value=True)
    use_s2 = st.checkbox("Sentinel-2 NDWI mean", value=False)

    st.divider()
    st.header("4. Dataset IDs and thresholds")
    st.caption("Preset IDs are editable. Paste your own GEE ID if you want to replace a source.")
    s1_id = st.text_input("Sentinel-1 dataset", DATASET_PRESETS["Sentinel-1 SAR GRD"]["id"])
    s2_id = st.text_input("Sentinel-2 dataset", DATASET_PRESETS["Sentinel-2 Surface Reflectance Harmonized"]["id"])
    chirps_id = st.text_input("CHIRPS dataset", DATASET_PRESETS["CHIRPS Daily Rainfall"]["id"])
    smap_id = st.text_input("SMAP dataset", DATASET_PRESETS["SMAP Soil Moisture"]["id"])
    orbit = st.selectbox("Sentinel-1 orbit", ["Either", "ASCENDING", "DESCENDING"])
    vv_min = st.slider("VV water-like min dB", -30.0, -10.0, -20.0, 0.5)
    vv_max = st.slider("VV water-like max dB", -25.0, -5.0, -15.0, 0.5)
    s2_cloud_pct = st.slider("Sentinel-2 cloud max %", 5, 80, 40, 5)

cfg = {
    "start_date": start_date,
    "end_date": end_date,
    "step_days": int(step_days),
    "scale": int(scale),
    "use_s1": use_s1,
    "use_chirps": use_chirps,
    "use_smap": use_smap,
    "use_s2": use_s2,
    "s1_id": s1_id,
    "s2_id": s2_id,
    "chirps_id": chirps_id,
    "smap_id": smap_id,
    "orbit": orbit,
    "vv_min": vv_min,
    "vv_max": vv_max,
    "s2_cloud_pct": s2_cloud_pct,
}

region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
week_count = len(make_week_starts(start_date, end_date, step_days))

c1, c2, c3 = st.columns(3)
c1.metric("Start", start_date)
c2.metric("End", end_date)
c3.metric("Weekly windows", week_count)

if week_count > 520:
    st.warning("This is a large time series. Use Google Drive export for the full table. Direct preview is limited.")

run_preview = st.button("Build weekly time series preview", type="primary")
export_drive = st.button("Export full weekly time series to Google Drive")

if run_preview:
    try:
        with st.spinner("Building weekly Earth Engine table. This can take time for long ranges..."):
            fc = build_weekly_features(region, cfg)
            preview_limit = min(week_count, 260)
            rows = fc.limit(preview_limit).getInfo().get("features", [])
            data = [f.get("properties", {}) for f in rows]
            df = pd.DataFrame(data)
            st.session_state["weekly_timeseries_df"] = df
            st.session_state["weekly_timeseries_fc"] = fc
        st.success(f"Preview loaded: {len(df)} rows. Export to Drive for the complete table if needed.")
    except Exception as exc:
        st.error("Could not build weekly time series preview.")
        st.exception(exc)

if export_drive:
    try:
        fc = build_weekly_features(region, cfg)
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description="hobart_weekly_gee_timeseries_2018_2026",
            folder="GEE_Hobart_Flood_Project",
            fileNamePrefix="hobart_weekly_gee_timeseries_2018_2026",
            fileFormat="CSV",
        )
        task.start()
        st.success("Weekly time-series export task started.")
        st.json(task.status())
    except Exception as exc:
        st.error("Could not start weekly time-series export.")
        st.exception(exc)

if "weekly_timeseries_df" in st.session_state:
    df = st.session_state["weekly_timeseries_df"]
    st.subheader("Weekly time-series preview")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download preview CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="weekly_gee_timeseries_preview.csv",
        mime="text/csv",
    )

    numeric_cols = [col for col in df.columns if col not in ["start_date", "end_date"] and pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        y_col = st.selectbox("Chart variable", numeric_cols)
        plot_df = df.copy()
        plot_df["start_date"] = pd.to_datetime(plot_df["start_date"])
        fig = px.line(plot_df, x="start_date", y=y_col, title=f"Weekly {y_col}")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("What each weekly variable means")
st.markdown(
    """
- `s1_image_count`: how many Sentinel-1 images were available in that weekly window.
- `sar_water_like_area_km2`: area of pixels matching the VV water-like threshold. This is not fully validated flood extent by itself.
- `vv_mean_db`: average Sentinel-1 VV backscatter over the AOI.
- `weekly_rain_mean_mm`: mean CHIRPS rainfall total for the week.
- `soil_moisture_mean`: mean SMAP surface soil moisture for the week.
- `s2_image_count`: number of Sentinel-2 images under the cloud threshold.
- `ndwi_mean`: average NDWI for the week, useful for wetness/water context when cloud allows.
"""
)
