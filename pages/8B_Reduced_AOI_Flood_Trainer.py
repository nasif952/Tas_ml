import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import plotly.express as px
import ee

from utils.ee_auth import show_ee_connection_block
from utils.ee_visualization import add_ee_layer, hobart_map, get_default_vis
from utils.flood_pipeline import (
    FloodConfig,
    get_s1_vv_event_change,
    get_s2_existing_water,
    build_change_flood_map,
    build_persistence,
    build_predictors,
    train_rf_susceptibility,
    build_final_overlay,
    area_km2,
)
from utils.gee_presets import DATASET_PRESETS

DEFAULT_AOI_ASSET = "projects/gee-project-493107/assets/Study_Area"
DEFAULT_BUILDING_ASSET = DATASET_PRESETS["Greater Hobart Buildings asset"]["id"]
SKIP_LGA_CODE = 115

st.set_page_config(page_title="Reduced AOI Flood Trainer", page_icon="🛰️", layout="wide")
st.title("🛰️ Reduced AOI Flood Trainer")
st.caption("Runs the dynamic flood susceptibility workflow using the Study_Area asset while skipping LGA_CODE 115 to reduce Earth Engine memory load.")
show_ee_connection_block()

with st.sidebar:
    st.header("AOI")
    aoi_asset = st.text_input("Study area polygon asset", DEFAULT_AOI_ASSET)
    skip_lga_115 = st.checkbox("Skip LGA_CODE 115", value=True)
    simplify_m = st.slider("Simplify AOI geometry metres", 0, 1000, 100, 50)

    st.header("SAR dates")
    label_start = st.text_input("During-event SAR start", "2018-05-10")
    label_end = st.text_input("During-event SAR end", "2018-05-18")
    recovery_start = st.text_input("Pre-event/reference SAR start", "2018-05-01")
    recovery_end = st.text_input("Pre-event/reference SAR end", "2018-05-09")
    orbit = st.selectbox("Sentinel-1 orbit", ["Either", "ASCENDING", "DESCENDING"])
    max_extra_days = st.select_slider("Auto-extend if no SAR data", [0, 1, 2, 3], value=3)

    st.header("Water mask and thresholds")
    water_start = st.text_input("Pre-event water-mask start", "2018-03-01")
    water_end = st.text_input("Pre-event water-mask end", "2018-05-09")
    ndwi_threshold = st.slider("NDWI threshold", -0.2, 0.5, 0.0, 0.05)
    s2_cloud_pct = st.slider("Sentinel-2 cloud max percent", 5, 80, 40, 5)
    vv_min = st.slider("During VV water-like min dB", -30.0, -10.0, -20.0, 0.5)
    vv_max = st.slider("During VV water-like max dB", -25.0, -5.0, -15.0, 0.5)
    sar_change_threshold_db = st.slider("Flood SAR drop threshold dB", -6.0, 0.0, -2.0, 0.5)
    stuck_change_tolerance = st.slider("Persistence tolerance dB", 0.5, 6.0, 2.0, 0.5)

    st.header("Predictors")
    predictor_start = st.text_input("MODIS predictor start", "2018-04-01")
    predictor_end = st.text_input("MODIS predictor end", "2018-05-18")
    month_start = st.text_input("Monthly rainfall/temp start", "2018-05-01")
    month_end = st.text_input("Monthly rainfall/temp end", "2018-06-01")
    event_rain_start = st.text_input("Event rainfall start", "2018-05-10")
    event_rain_end = st.text_input("Event rainfall end", "2018-05-18")
    soil_start = st.text_input("Soil moisture start", "2018-05-01")
    soil_end = st.text_input("Soil moisture end", "2018-05-18")
    use_river_distance = st.checkbox("Use river distance", True)
    river_asset = st.text_input("River asset", "WWF/HydroSHEDS/v1/FreeFlowingRivers")
    river_distance_cap_m = st.slider("River distance cap m", 100, 3000, 1000, 100)
    use_soil_moisture = st.checkbox("Use soil moisture", True)

    st.header("Random Forest")
    scale = st.select_slider("Scale", [30, 50, 100, 250, 500, 1000], value=250)
    class_points = st.slider("Samples per class", 100, 2000, 300, 100)
    train_fraction = st.slider("Training fraction", 0.50, 0.90, 0.70, 0.05)
    rf_trees = st.slider("RF trees", 50, 800, 200, 50)
    seed = st.number_input("Random seed", 0, 9999, 42)
    susceptibility_weight = st.slider("Final overlay susceptibility weight", 0.0, 1.0, 0.70, 0.05)
    high_overlay_threshold = st.slider("High overlay threshold", 0.30, 0.90, 0.66, 0.01)
    building_asset = st.text_input("Building asset", DEFAULT_BUILDING_ASSET)

config = FloodConfig(
    aoi_asset=aoi_asset,
    scale=int(scale),
    seed=int(seed),
    s1_stage1_start=label_start,
    s1_stage1_end=label_end,
    s1_stage2_start=recovery_start,
    s1_stage2_end=recovery_end,
    s2_water_start=water_start,
    s2_water_end=water_end,
    s2_cloud_pct=int(s2_cloud_pct),
    predictor_start=predictor_start,
    predictor_end=predictor_end,
    month_start=month_start,
    month_end=month_end,
    event_rain_start=event_rain_start,
    event_rain_end=event_rain_end,
    soil_start=soil_start,
    soil_end=soil_end,
    vv_min=float(vv_min),
    vv_max=float(vv_max),
    ndwi_threshold=float(ndwi_threshold),
    stuck_change_tolerance=float(stuck_change_tolerance),
    sar_change_threshold_db=float(sar_change_threshold_db),
    river_asset=river_asset,
    river_distance_cap_m=int(river_distance_cap_m),
    use_river_distance=bool(use_river_distance),
    use_soil_moisture=bool(use_soil_moisture),
    class_points=int(class_points),
    train_fraction=float(train_fraction),
    rf_trees=int(rf_trees),
    susceptibility_weight=float(susceptibility_weight),
    high_overlay_threshold=float(high_overlay_threshold),
    max_extra_days=int(max_extra_days),
)

st.info(f"AOI asset: `{aoi_asset}`")
st.info(f"Skip LGA_CODE 115: `{skip_lga_115}` | scale: `{scale}` m | samples/class: `{class_points}` | RF trees: `{rf_trees}`")

if st.button("Train reduced AOI model", type="primary", use_container_width=True):
    with st.status("Running reduced AOI Earth Engine workflow...", expanded=True) as status:
        try:
            status.write("Preparing AOI...")
            study_area_fc = ee.FeatureCollection(aoi_asset)
            if skip_lga_115:
                study_area_fc = study_area_fc.filter(ee.Filter.neq("LGA_CODE", SKIP_LGA_CODE))
            region = study_area_fc.geometry()
            if int(simplify_m) > 0:
                region = region.simplify(int(simplify_m))

            status.write("Building SAR layers...")
            pre_vv, during_vv, sar_change, pre_col, target_col, pre_info, target_info = get_s1_vv_event_change(
                region, recovery_start, recovery_end, label_start, label_end, orbit, config.max_extra_days
            )
            status.write("Building existing-water mask...")
            ndwi, existing_water, water_info = get_s2_existing_water(region, config, return_info=True)
            status.write("Creating flood label and persistence layers...")
            flood_map = build_change_flood_map(during_vv, sar_change, existing_water, config)
            vv_change, stuck_water, persistence = build_persistence(pre_vv, during_vv, flood_map, config)
            status.write("Building predictors and training RF...")
            predictors, predictor_names, extra_layers = build_predictors(region, config)
            rf_result = train_rf_susceptibility(region, predictors, flood_map, predictor_names, config)
            susceptibility = rf_result["susceptibility"]
            final_overlay, high_overlay = build_final_overlay(susceptibility, stuck_water, config)

            cm = rf_result["confusion_matrix"]
            metrics = {
                "aoi_features": study_area_fc.size().getInfo(),
                "pre_s1_count": pre_col.size().getInfo(),
                "during_s1_count": target_col.size().getInfo(),
                "accuracy": cm.accuracy().getInfo(),
                "kappa": cm.kappa().getInfo(),
                "training_samples": rf_result["training"].size().getInfo(),
                "testing_samples": rf_result["testing"].size().getInfo(),
                "flood_area_km2": area_km2(flood_map, "Flood_Map", region, config.scale).getInfo(),
                "stuck_water_km2": area_km2(stuck_water, "Stuck_Water", region, config.scale).getInfo(),
            }
            status.update(label="Training complete", state="complete")
        except Exception as exc:
            status.update(label="Training failed", state="error")
            st.exception(exc)
            st.stop()

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Kappa", f"{metrics['kappa']:.3f}")
    c3.metric("Flood area km²", f"{metrics['flood_area_km2']:.2f}")
    c4.metric("Stuck water km²", f"{metrics['stuck_water_km2']:.2f}")

    st.write("AOI feature count used:", metrics["aoi_features"])
    st.write("SAR image counts:", {"pre_event": metrics["pre_s1_count"], "during_event": metrics["during_s1_count"]})
    st.write("Training/testing samples:", {"training": metrics["training_samples"], "testing": metrics["testing_samples"]})
    st.write("Predictors used:", predictor_names)

    try:
        importance_features = rf_result["importance_fc"].getInfo()["features"]
        importance_df = pd.DataFrame([f["properties"] for f in importance_features]).sort_values("importance", ascending=False)
        st.subheader("Variable importance")
        st.dataframe(importance_df, use_container_width=True)
        st.plotly_chart(px.bar(importance_df, x="variable", y="importance"), use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render importance table: {exc}")

    st.subheader("Map")
    m = hobart_map(zoom_start=9)
    add_ee_layer(m, study_area_fc.style(color="red", fillColor="00000000", width=2), {}, "Reduced Study Area", True)
    add_ee_layer(m, flood_map.selfMask(), get_default_vis("flood_map"), "SAR-change Flood Map", True)
    add_ee_layer(m, susceptibility, get_default_vis("susceptibility"), "RF Flood Susceptibility", True)
    add_ee_layer(m, final_overlay, get_default_vis("final_overlay"), "Final Overlay", True)
    add_ee_layer(m, high_overlay.selfMask(), get_default_vis("high_overlay"), "High Overlay Zone", False)
    if "river_distance" in extra_layers:
        add_ee_layer(m, extra_layers["river_distance"], get_default_vis("river_distance"), "River Distance", False)
    if "soil_moisture" in extra_layers:
        add_ee_layer(m, extra_layers["soil_moisture"], get_default_vis("soil_moisture"), "Soil Moisture", False)
    folium.LayerControl().add_to(m)
    st_folium(m, height=650, use_container_width=True)
else:
    st.caption("Default reduced settings: skip LGA_CODE 115, simplify 100 m, scale 250 m, 300 samples/class, 200 trees.")
