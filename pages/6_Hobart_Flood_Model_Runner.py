import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import plotly.express as px

from utils.ee_auth import show_ee_connection_block
from utils.ee_visualization import add_ee_layer, hobart_map, get_default_vis
from utils.flood_pipeline import (
    FloodConfig,
    study_area,
    get_s1_vv,
    get_s2_existing_water,
    build_flood_map,
    build_persistence,
    build_predictors,
    train_rf_susceptibility,
    build_final_overlay,
    area_km2,
)

st.set_page_config(page_title="Hobart Flood Model Runner", page_icon="🌊", layout="wide")
st.title("🌊 Hobart 2018 Flood Model Runner")
st.caption("Run the real Earth Engine flood workflow from Streamlit: SAR flood map, NDWI mask, river distance, soil moisture, RF susceptibility, final overlay, and exports.")

show_ee_connection_block()

import ee

with st.sidebar:
    st.header("AOI and scale")
    xmin = st.number_input("Min lon", value=146.85, step=0.01, format="%.4f")
    ymin = st.number_input("Min lat", value=-43.10, step=0.01, format="%.4f")
    xmax = st.number_input("Max lon", value=147.55, step=0.01, format="%.4f")
    ymax = st.number_input("Max lat", value=-42.65, step=0.01, format="%.4f")
    scale = st.select_slider("Scale", [30, 50, 100, 250, 500, 1000], value=100)

    st.divider()
    st.header("SAR stages")
    s1_stage1_start = st.text_input("Stage 1 start", "2018-05-15")
    s1_stage1_end = st.text_input("Stage 1 end", "2018-05-16")
    s1_stage2_start = st.text_input("Stage 2 start", "2018-05-27")
    s1_stage2_end = st.text_input("Stage 2 end", "2018-05-28")
    orbit = st.selectbox("Orbit", ["Either", "ASCENDING", "DESCENDING"])

    st.divider()
    st.header("Flood thresholds")
    vv_min = st.slider("VV min dB", -30.0, -10.0, -20.0, 0.5)
    vv_max = st.slider("VV max dB", -25.0, -5.0, -15.0, 0.5)
    stuck_change_tolerance = st.slider("Stuck-water tolerance ±dB", 0.5, 6.0, 2.0, 0.5)

    st.divider()
    st.header("NDWI existing-water mask")
    s2_water_start = st.text_input("S2 pre-flood start", "2018-03-01")
    s2_water_end = st.text_input("S2 pre-flood end", "2018-05-09")
    s2_cloud_pct = st.slider("S2 cloud max %", 5, 80, 40, 5)
    ndwi_threshold = st.slider("NDWI threshold", -0.2, 0.5, 0.0, 0.05)

    st.divider()
    st.header("Predictor dates")
    predictor_start = st.text_input("MODIS predictor start", "2018-04-01")
    predictor_end = st.text_input("MODIS predictor end", "2018-05-09")
    month_start = st.text_input("Month start", "2018-05-01")
    month_end = st.text_input("Month end", "2018-06-01")
    event_rain_start = st.text_input("Event rain start", "2018-05-10")
    event_rain_end = st.text_input("Event rain end", "2018-05-13")
    soil_start = st.text_input("Soil moisture start", "2018-05-01")
    soil_end = st.text_input("Soil moisture end", "2018-06-01")

    st.divider()
    st.header("New factors")
    use_river_distance = st.checkbox("Use river distance", value=True)
    river_asset = st.text_input("River asset", "WWF/HydroSHEDS/v1/FreeFlowingRivers")
    river_distance_cap_m = st.slider("River distance cap m", 100, 3000, 1000, 100)
    use_soil_moisture = st.checkbox("Use soil moisture", value=True)

    st.divider()
    st.header("RF and overlay")
    seed = st.number_input("Seed", 0, 9999, 42)
    class_points = st.slider("Samples per class", 100, 2000, 600, 100)
    train_fraction = st.slider("Training fraction", 0.50, 0.90, 0.70, 0.05)
    rf_trees = st.slider("RF trees", 50, 800, 300, 50)
    susceptibility_weight = st.slider("Susceptibility weight", 0.0, 1.0, 0.70, 0.05)
    high_overlay_threshold = st.slider("High overlay threshold", 0.30, 0.90, 0.66, 0.01)

    st.divider()
    st.header("Buildings")
    building_asset = st.text_input("Building asset", "projects/gee-project-493107/assets/Greater_Hobart_Buildings_WGS84")

config = FloodConfig(
    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, scale=scale, seed=int(seed),
    s1_stage1_start=s1_stage1_start, s1_stage1_end=s1_stage1_end,
    s1_stage2_start=s1_stage2_start, s1_stage2_end=s1_stage2_end,
    s2_water_start=s2_water_start, s2_water_end=s2_water_end,
    s2_cloud_pct=s2_cloud_pct, predictor_start=predictor_start, predictor_end=predictor_end,
    month_start=month_start, month_end=month_end,
    event_rain_start=event_rain_start, event_rain_end=event_rain_end,
    soil_start=soil_start, soil_end=soil_end,
    vv_min=vv_min, vv_max=vv_max,
    ndwi_threshold=ndwi_threshold, stuck_change_tolerance=stuck_change_tolerance,
    river_asset=river_asset, river_distance_cap_m=river_distance_cap_m,
    use_river_distance=use_river_distance, use_soil_moisture=use_soil_moisture,
    class_points=class_points, train_fraction=train_fraction,
    rf_trees=rf_trees, susceptibility_weight=susceptibility_weight,
    high_overlay_threshold=high_overlay_threshold,
)

st.warning("Research note: this is an NDWI-masked SAR-derived potential flood extent workflow. It is not field-verified ground truth unless independent validation data is added.")

run = st.button("Run Earth Engine Flood Model", type="primary", use_container_width=True)

if not run:
    st.info("Configure the parameters in the sidebar, then click **Run Earth Engine Flood Model**.")
    st.stop()

with st.status("Building Earth Engine workflow...", expanded=True) as status:
    try:
        region = study_area(config)
        status.write("Loading Sentinel-1 VV images...")
        stage1_vv, stage1_col = get_s1_vv(region, config.s1_stage1_start, config.s1_stage1_end, orbit)
        stage2_vv, stage2_col = get_s1_vv(region, config.s1_stage2_start, config.s1_stage2_end, orbit)

        status.write("Creating Sentinel-2 NDWI existing-water mask...")
        ndwi, existing_water = get_s2_existing_water(region, config)

        status.write("Creating SAR flood map and persistence layer...")
        flood_map = build_flood_map(stage1_vv, existing_water, config)
        vv_change, stuck_water, persistence = build_persistence(stage1_vv, stage2_vv, flood_map, config)

        status.write("Building terrain, spectral, rainfall, river, and soil moisture predictors...")
        predictors, predictor_names, extra_layers = build_predictors(region, config)

        status.write("Training Random Forest susceptibility model in Earth Engine...")
        model_outputs = train_rf_susceptibility(region, predictors, flood_map, predictor_names, config)
        susceptibility = model_outputs["susceptibility"]
        final_overlay, high_overlay = build_final_overlay(susceptibility, stuck_water, config)

        status.write("Calculating summary metrics...")
        stage1_count = stage1_col.size().getInfo()
        stage2_count = stage2_col.size().getInfo()
        flood_area = area_km2(flood_map, "Flood_Map", region, scale).getInfo()
        stuck_area = area_km2(stuck_water, "Stuck_Water", region, scale).getInfo()
        cm_info = model_outputs["confusion_matrix"].getInfo()
        accuracy = model_outputs["confusion_matrix"].accuracy().getInfo()
        kappa = model_outputs["confusion_matrix"].kappa().getInfo()
        importance = model_outputs["importance"].getInfo()

        status.update(label="Earth Engine model run completed", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Earth Engine model run failed", state="error", expanded=True)
        st.exception(exc)
        st.stop()

# Store outputs in session for export buttons during this rerun.
st.session_state["last_ee_outputs"] = {
    "region": region,
    "flood_map": flood_map,
    "persistence": persistence,
    "vv_change": vv_change,
    "stuck_water": stuck_water,
    "susceptibility": susceptibility,
    "final_overlay": final_overlay,
    "high_overlay": high_overlay,
    "predictor_names": predictor_names,
    "importance_fc": model_outputs["importance_fc"],
    "extra_layers": extra_layers,
}

# Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stage 1 S1 images", stage1_count)
m2.metric("Stage 2 S1 images", stage2_count)
m3.metric("Flood area km²", f"{flood_area:.2f}")
m4.metric("Stuck-water km²", f"{stuck_area:.2f}")
m5.metric("RF accuracy", f"{accuracy:.3f}")
st.caption(f"Kappa: {kappa:.3f} · Predictors: {', '.join(predictor_names)}")

# Charts
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Confusion matrix")
    st.write(cm_info)
with c2:
    st.subheader("Variable importance")
    imp_df = pd.DataFrame([{"variable": k, "importance": v} for k, v in importance.items()]).sort_values("importance")
    fig = px.bar(imp_df, x="importance", y="variable", orientation="h", title="Earth Engine RF Variable Importance")
    st.plotly_chart(fig, use_container_width=True)

# Map
st.subheader("Interactive Earth Engine output map")
st.caption("Map interactions are disabled from triggering Streamlit reruns so the model output does not disappear after tiles load.")
with st.spinner("Rendering map tiles..."):
    m = hobart_map(zoom_start=10)
    folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False, tooltip="AOI").add_to(m)
    add_ee_layer(m, existing_water.selfMask(), get_default_vis("existing_water"), "Existing water mask", shown=False)
    add_ee_layer(m, flood_map.selfMask(), get_default_vis("flood_map"), "NDWI-masked SAR flood map", shown=True)
    add_ee_layer(m, vv_change, get_default_vis("vv_change"), "VV change", shown=False)
    add_ee_layer(m, persistence.selfMask(), get_default_vis("persistence"), "Flood persistence", shown=False)
    if "river_distance" in extra_layers:
        add_ee_layer(m, extra_layers["river_distance"], get_default_vis("river_distance"), "River distance capped", shown=False)
        add_ee_layer(m, extra_layers["river_buffer"].selfMask(), get_default_vis("river_buffer"), "River buffer", shown=False)
    if "soil_moisture" in extra_layers:
        add_ee_layer(m, extra_layers["soil_moisture"], get_default_vis("soil_moisture"), "Soil moisture", shown=False)
    add_ee_layer(m, susceptibility, get_default_vis("susceptibility"), "RF susceptibility", shown=True, opacity=0.75)
    add_ee_layer(m, final_overlay, get_default_vis("final_overlay"), "Final overlay", shown=False, opacity=0.75)
    add_ee_layer(m, high_overlay.selfMask(), get_default_vis("high_overlay"), "High overlay zone", shown=False)
    try:
        buildings = ee.FeatureCollection(building_asset).filterBounds(region)
        add_ee_layer(m, buildings.style(color="000000", fillColor="00000000", width=1), {}, "Buildings", shown=False)
    except Exception:
        pass
    folium.LayerControl().add_to(m)
    st_folium(
        m,
        height=720,
        use_container_width=True,
        returned_objects=[],
        key="hobart_flood_output_map_static",
    )

# Downloads and exports
st.subheader("Export current run")
e1, e2, e3, e4 = st.columns(4)
folder = st.text_input("Google Drive export folder", "GEE_Hobart_Flood_Project")

if e1.button("Export flood map"):
    task = ee.batch.Export.image.toDrive(image=flood_map, description="hobart_flood_map_streamlit", folder=folder, fileNamePrefix="hobart_flood_map_streamlit", region=region, scale=scale, maxPixels=1e13)
    task.start()
    st.json(task.status())

if e2.button("Export susceptibility"):
    task = ee.batch.Export.image.toDrive(image=susceptibility, description="hobart_susceptibility_streamlit", folder=folder, fileNamePrefix="hobart_susceptibility_streamlit", region=region, scale=scale, maxPixels=1e13)
    task.start()
    st.json(task.status())

if e3.button("Export final overlay"):
    task = ee.batch.Export.image.toDrive(image=final_overlay, description="hobart_final_overlay_streamlit", folder=folder, fileNamePrefix="hobart_final_overlay_streamlit", region=region, scale=scale, maxPixels=1e13)
    task.start()
    st.json(task.status())

if e4.button("Export importance CSV"):
    task = ee.batch.Export.table.toDrive(collection=model_outputs["importance_fc"], description="hobart_rf_importance_streamlit", folder=folder, fileNamePrefix="hobart_rf_importance_streamlit", fileFormat="CSV")
    task.start()
    st.json(task.status())
