import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
import folium

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
from utils.gee_presets import AOI_PRESETS, DATASET_PRESETS, VV_THRESHOLD_PRESETS, NDWI_PRESETS, MODEL_PRESETS

st.set_page_config(page_title="Dynamic Flood Susceptibility Trainer", page_icon="🔁", layout="wide")
st.title("🔁 Dynamic Flood Susceptibility Trainer")
st.caption(
    "Train and update a flood susceptibility map for any selected event/week, not only the 2018 Hobart event. "
    "Use weekly monitoring results to choose target windows, then train a new SAR/NDWI-derived model."
)

show_ee_connection_block()

import ee

st.warning(
    "This page trains from a remote-sensing-derived flood label. It is useful for continuous updating and exploratory monitoring, "
    "but it is not independent ground-truth validation unless you add observed flood points or verified flood boundaries."
)

EVENT_LIBRARY = {
    "February 2026 Greater Hobart update": {
        "label_start": "2026-02-02",
        "label_end": "2026-02-16",
        "recovery_start": "2026-01-16",
        "recovery_end": "2026-01-28",
        "water_start": "2025-10-01",
        "water_end": "2025-12-31",
        "predictor_start": "2026-01-01",
        "predictor_end": "2026-02-16",
        "month_start": "2026-01-01",
        "month_end": "2026-02-16",
        "event_rain_start": "2025-01-01",
        "event_rain_end": "2026-02-16",
        "soil_start": "2026-01-01",
        "soil_end": "2026-02-16",
        "aoi_name": "Greater Hobart wider context",
        "ndwi_preset": "Default existing water NDWI > 0.0",
        "ndwi_threshold": 0.0,
        "s2_cloud_pct": 40,
        "vv_preset": "Conservative open water (-20 to -15 dB)",
        "vv_min": -20.0,
        "vv_max": -15.0,
        "orbit": "Either",
        "stuck_change_tolerance": 2.0,
        "use_river_distance": True,
        "river_asset": "WWF/HydroSHEDS/v1/FreeFlowingRivers",
        "river_distance_cap_m": 1000,
        "use_soil_moisture": True,
        "model_name": "Balanced quick test",
        "scale": 100,
        "class_points": 600,
        "train_fraction": 0.70,
        "rf_trees": 300,
        "seed": 42,
        "susceptibility_weight": 0.70,
        "high_overlay_threshold": 0.66,
        "building_asset": "",
        "note": "Your requested February 2026 Greater Hobart setup. Recovery window is before the target window, so persistence is treated as a comparison layer, not post-event recovery.",
    },
    "Hobart May 2018 default": {
        "label_start": "2018-05-15",
        "label_end": "2018-05-16",
        "recovery_start": "2018-05-27",
        "recovery_end": "2018-05-28",
        "water_start": "2018-03-01",
        "water_end": "2018-05-09",
        "predictor_start": "2018-04-01",
        "predictor_end": "2018-05-09",
        "month_start": "2018-05-01",
        "month_end": "2018-06-01",
        "event_rain_start": "2018-05-10",
        "event_rain_end": "2018-05-13",
        "soil_start": "2018-05-01",
        "soil_end": "2018-06-01",
        "aoi_name": "Hobart core / default",
        "ndwi_preset": "Default existing water NDWI > 0.0",
        "ndwi_threshold": 0.0,
        "s2_cloud_pct": 40,
        "vv_preset": "Conservative open water (-20 to -15 dB)",
        "vv_min": -20.0,
        "vv_max": -15.0,
        "orbit": "Either",
        "stuck_change_tolerance": 2.0,
        "use_river_distance": True,
        "river_asset": "WWF/HydroSHEDS/v1/FreeFlowingRivers",
        "river_distance_cap_m": 1000,
        "use_soil_moisture": True,
        "model_name": "Balanced quick test",
        "scale": 100,
        "class_points": 600,
        "train_fraction": 0.70,
        "rf_trees": 300,
        "seed": 42,
        "susceptibility_weight": 0.70,
        "high_overlay_threshold": 0.66,
        "building_asset": DATASET_PRESETS["Greater Hobart Buildings asset"]["id"],
        "note": "Original Hobart prototype event.",
    },
    "Recent/custom weekly update": {
        "label_start": "2025-01-01",
        "label_end": "2025-01-08",
        "recovery_start": "2025-01-15",
        "recovery_end": "2025-01-22",
        "water_start": "2024-10-01",
        "water_end": "2024-12-31",
        "predictor_start": "2024-12-01",
        "predictor_end": "2025-01-01",
        "month_start": "2024-12-01",
        "month_end": "2025-01-08",
        "event_rain_start": "2024-12-25",
        "event_rain_end": "2025-01-08",
        "soil_start": "2024-12-01",
        "soil_end": "2025-01-08",
        "aoi_name": "Greater Hobart wider context",
        "ndwi_preset": "Default existing water NDWI > 0.0",
        "ndwi_threshold": 0.0,
        "s2_cloud_pct": 40,
        "vv_preset": "Conservative open water (-20 to -15 dB)",
        "vv_min": -20.0,
        "vv_max": -15.0,
        "orbit": "Either",
        "stuck_change_tolerance": 2.0,
        "use_river_distance": True,
        "river_asset": "WWF/HydroSHEDS/v1/FreeFlowingRivers",
        "river_distance_cap_m": 1000,
        "use_soil_moisture": True,
        "model_name": "Balanced quick test",
        "scale": 100,
        "class_points": 600,
        "train_fraction": 0.70,
        "rf_trees": 300,
        "seed": 42,
        "susceptibility_weight": 0.70,
        "high_overlay_threshold": 0.66,
        "building_asset": DATASET_PRESETS["Greater Hobart Buildings asset"]["id"],
        "note": "Template for training a new weekly/event model. Edit all dates below.",
    },
    "Manual event dates": {
        "label_start": "2026-01-01",
        "label_end": "2026-01-08",
        "recovery_start": "2026-01-15",
        "recovery_end": "2026-01-22",
        "water_start": "2025-10-01",
        "water_end": "2025-12-31",
        "predictor_start": "2025-12-01",
        "predictor_end": "2026-01-01",
        "month_start": "2025-12-01",
        "month_end": "2026-01-08",
        "event_rain_start": "2025-12-25",
        "event_rain_end": "2026-01-08",
        "soil_start": "2025-12-01",
        "soil_end": "2026-01-08",
        "aoi_name": "Custom bounding box",
        "ndwi_preset": "Default existing water NDWI > 0.0",
        "ndwi_threshold": 0.0,
        "s2_cloud_pct": 40,
        "vv_preset": "Conservative open water (-20 to -15 dB)",
        "vv_min": -20.0,
        "vv_max": -15.0,
        "orbit": "Either",
        "stuck_change_tolerance": 2.0,
        "use_river_distance": True,
        "river_asset": "WWF/HydroSHEDS/v1/FreeFlowingRivers",
        "river_distance_cap_m": 1000,
        "use_soil_moisture": True,
        "model_name": "Balanced quick test",
        "scale": 100,
        "class_points": 600,
        "train_fraction": 0.70,
        "rf_trees": 300,
        "seed": 42,
        "susceptibility_weight": 0.70,
        "high_overlay_threshold": 0.66,
        "building_asset": DATASET_PRESETS["Greater Hobart Buildings asset"]["id"],
        "note": "Fully custom event setup.",
    },
}


def preset_index(options, value):
    return options.index(value) if value in options else 0


def metric_value(value, decimals=3):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "n/a"


with st.sidebar:
    st.header("1. What are you training?")
    event_options = list(EVENT_LIBRARY.keys())
    event_name = st.selectbox("Event/update window preset", event_options, index=0)
    ev = EVENT_LIBRARY[event_name]
    event_key = event_name.lower().replace(" ", "_").replace("/", "_")
    st.caption(ev["note"])
    st.info(
        "Target label window = SAR image period used to detect water-like/flood-like pixels. "
        "Recovery window = later SAR period used to identify persistent/stuck water."
    )

    label_start = st.text_input("Target label SAR start", ev["label_start"], key=f"{event_key}_label_start")
    label_end = st.text_input("Target label SAR end", ev["label_end"], key=f"{event_key}_label_end")
    recovery_start = st.text_input("Recovery SAR start", ev["recovery_start"], key=f"{event_key}_recovery_start")
    recovery_end = st.text_input("Recovery SAR end", ev["recovery_end"], key=f"{event_key}_recovery_end")

    if recovery_start < label_start:
        st.warning("This preset uses a comparison/reference SAR window before the target label window. That is valid for comparison, but it is not a post-event recovery window.")

    st.divider()
    st.header("2. AOI")
    aoi_options = list(AOI_PRESETS.keys())
    aoi_name = st.selectbox("AOI preset", aoi_options, index=preset_index(aoi_options, ev["aoi_name"]), key=f"{event_key}_aoi")
    bbox = AOI_PRESETS[aoi_name]["bbox"]
    st.caption(AOI_PRESETS[aoi_name]["description"])
    xmin = st.number_input("Min longitude", value=float(bbox[0]), step=0.01, format="%.4f", key=f"{event_key}_xmin")
    ymin = st.number_input("Min latitude", value=float(bbox[1]), step=0.01, format="%.4f", key=f"{event_key}_ymin")
    xmax = st.number_input("Max longitude", value=float(bbox[2]), step=0.01, format="%.4f", key=f"{event_key}_xmax")
    ymax = st.number_input("Max latitude", value=float(bbox[3]), step=0.01, format="%.4f", key=f"{event_key}_ymax")

    st.divider()
    st.header("3. Existing-water mask")
    ndwi_options = list(NDWI_PRESETS.keys())
    ndwi_preset_name = st.selectbox("NDWI threshold preset", ndwi_options, index=preset_index(ndwi_options, ev["ndwi_preset"]), key=f"{event_key}_ndwi_preset")
    ndwi_default = float(ev.get("ndwi_threshold", NDWI_PRESETS[ndwi_preset_name]["threshold"]))
    ndwi_threshold = st.slider("NDWI threshold", -0.2, 0.5, ndwi_default, 0.05, key=f"{event_key}_ndwi_threshold")
    water_start = st.text_input("Pre-event water-mask start", ev["water_start"], key=f"{event_key}_water_start")
    water_end = st.text_input("Pre-event water-mask end", ev["water_end"], key=f"{event_key}_water_end")
    s2_cloud_pct = st.slider("Sentinel-2 cloud max %", 5, 80, int(ev["s2_cloud_pct"]), 5, key=f"{event_key}_s2_cloud")

    st.divider()
    st.header("4. SAR threshold")
    vv_options = list(VV_THRESHOLD_PRESETS.keys())
    vv_preset_name = st.selectbox("VV threshold preset", vv_options, index=preset_index(vv_options, ev["vv_preset"]), key=f"{event_key}_vv_preset")
    vv_min = st.slider("VV water-like min dB", -30.0, -10.0, float(ev.get("vv_min", VV_THRESHOLD_PRESETS[vv_preset_name]["vv_min"])), 0.5, key=f"{event_key}_vv_min")
    vv_max = st.slider("VV water-like max dB", -25.0, -5.0, float(ev.get("vv_max", VV_THRESHOLD_PRESETS[vv_preset_name]["vv_max"])), 0.5, key=f"{event_key}_vv_max")
    orbit_options = ["Either", "ASCENDING", "DESCENDING"]
    orbit = st.selectbox("Sentinel-1 orbit", orbit_options, index=preset_index(orbit_options, ev["orbit"]), key=f"{event_key}_orbit")
    stuck_change_tolerance = st.slider("Stuck-water tolerance ±dB", 0.5, 6.0, float(ev["stuck_change_tolerance"]), 0.5, key=f"{event_key}_stuck_tol")

    st.divider()
    st.header("5. Predictors and dates")
    st.caption("These predictors are rebuilt for the selected event/update period.")
    predictor_start = st.text_input("MODIS predictor start", ev["predictor_start"], key=f"{event_key}_predictor_start")
    predictor_end = st.text_input("MODIS predictor end", ev["predictor_end"], key=f"{event_key}_predictor_end")
    month_start = st.text_input("Monthly rainfall/temp start", ev["month_start"], key=f"{event_key}_month_start")
    month_end = st.text_input("Monthly rainfall/temp end", ev["month_end"], key=f"{event_key}_month_end")
    event_rain_start = st.text_input("Event rainfall start", ev["event_rain_start"], key=f"{event_key}_event_rain_start")
    event_rain_end = st.text_input("Event rainfall end", ev["event_rain_end"], key=f"{event_key}_event_rain_end")
    soil_start = st.text_input("Soil moisture start", ev["soil_start"], key=f"{event_key}_soil_start")
    soil_end = st.text_input("Soil moisture end", ev["soil_end"], key=f"{event_key}_soil_end")

    use_river_distance = st.checkbox("Use river distance", value=bool(ev["use_river_distance"]), key=f"{event_key}_use_river")
    river_asset = st.text_input("River asset", ev["river_asset"], key=f"{event_key}_river_asset")
    river_distance_cap_m = st.slider("River distance cap m", 100, 3000, int(ev["river_distance_cap_m"]), 100, key=f"{event_key}_river_cap")
    use_soil_moisture = st.checkbox("Use soil moisture", value=bool(ev["use_soil_moisture"]), key=f"{event_key}_use_soil")

    st.divider()
    st.header("6. Model")
    model_options = list(MODEL_PRESETS.keys())
    model_name = st.selectbox("Model preset", model_options, index=preset_index(model_options, ev["model_name"]), key=f"{event_key}_model")
    model_preset = MODEL_PRESETS[model_name]
    st.caption(model_preset["description"])
    scale = st.select_slider("Scale", [30, 50, 100, 250, 500, 1000], value=int(ev.get("scale", model_preset["scale"])), key=f"{event_key}_scale")
    class_points = st.slider("Samples per class", 100, 2000, int(ev.get("class_points", model_preset["samples"])), 100, key=f"{event_key}_class_points")
    train_fraction = st.slider("Training fraction", 0.50, 0.90, float(ev.get("train_fraction", model_preset["train_fraction"])), 0.05, key=f"{event_key}_train_fraction")
    rf_trees = st.slider("RF trees", 50, 800, int(ev.get("rf_trees", model_preset["trees"])), 50, key=f"{event_key}_rf_trees")
    seed = st.number_input("Random seed", 0, 9999, int(ev["seed"]), key=f"{event_key}_seed")
    susceptibility_weight = st.slider("Final overlay susceptibility weight", 0.0, 1.0, float(ev["susceptibility_weight"]), 0.05, key=f"{event_key}_sus_weight")
    high_overlay_threshold = st.slider("High overlay threshold", 0.30, 0.90, float(ev["high_overlay_threshold"]), 0.01, key=f"{event_key}_high_threshold")

    st.divider()
    default_building_asset = ev["building_asset"] or DATASET_PRESETS["Greater Hobart Buildings asset"]["id"]
    building_asset = st.text_input("Building asset", default_building_asset, key=f"{event_key}_building_asset")

config = FloodConfig(
    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, scale=scale, seed=int(seed),
    s1_stage1_start=label_start, s1_stage1_end=label_end,
    s1_stage2_start=recovery_start, s1_stage2_end=recovery_end,
    s2_water_start=water_start, s2_water_end=water_end,
    s2_cloud_pct=s2_cloud_pct,
    predictor_start=predictor_start, predictor_end=predictor_end,
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

run = st.button("Train / update susceptibility map", type="primary", use_container_width=True)

if run:
    with st.status("Training dynamic susceptibility model...", expanded=True) as status:
        try:
            region = study_area(config)
            status.write("Checking Sentinel-1 availability...")
            stage1_vv, stage1_col = get_s1_vv(region, label_start, label_end, orbit)
            stage2_vv, stage2_col = get_s1_vv(region, recovery_start, recovery_end, orbit)
            stage1_count = stage1_col.size().getInfo()
            stage2_count = stage2_col.size().getInfo()
            if stage1_count == 0:
                st.error("No Sentinel-1 images found in the target label window. Use a wider date range or choose Either orbit.")
                st.stop()

            if stage2_count == 0:
                st.warning("No Sentinel-1 images found in the recovery/comparison window. Stuck-water/persistence will be set to zero so training can continue.")
                stage2_vv = stage1_vv
                vv_change = ee.Image.constant(0).clip(region).rename("Stage2_minus_Stage1").toFloat()
                stuck_water = ee.Image.constant(0).clip(region).rename("Stuck_Water").toInt()
                persistence = ee.Image.constant(0).clip(region).rename("Flood_Persistence").toInt()
                persistence_note = "No recovery/comparison S1 images; persistence was set to zero."
            else:
                persistence_note = "Persistence layer built from selected recovery/comparison SAR window."

            status.write("Building pre-event NDWI existing-water mask...")
            ndwi, existing_water = get_s2_existing_water(region, config)
            status.write("Building SAR-derived event flood label...")
            flood_map = build_flood_map(stage1_vv, existing_water, config)
            if stage2_count > 0:
                vv_change, stuck_water, persistence = build_persistence(stage1_vv, stage2_vv, flood_map, config)

            status.write("Building event-specific predictors with missing-data safeguards...")
            predictors, predictor_names, extra_layers = build_predictors(region, config)
            availability = extra_layers.get("availability")
            availability_info = availability.getInfo() if availability is not None else {}

            status.write("Training Random Forest and generating susceptibility probability raster...")
            model_outputs = train_rf_susceptibility(region, predictors, flood_map, predictor_names, config)
            susceptibility = model_outputs["susceptibility"]
            final_overlay, high_overlay = build_final_overlay(susceptibility, stuck_water, config)
            status.write("Calculating metrics...")
            metrics = {
                "stage1_count": stage1_count,
                "stage2_count": stage2_count,
                "flood_area": area_km2(flood_map, "Flood_Map", region, scale).getInfo(),
                "stuck_area": area_km2(stuck_water, "Stuck_Water", region, scale).getInfo(),
                "accuracy": model_outputs["confusion_matrix"].accuracy().getInfo(),
                "kappa": model_outputs["confusion_matrix"].kappa().getInfo(),
                "cm_info": model_outputs["confusion_matrix"].getInfo(),
                "importance": model_outputs["importance"].getInfo(),
                "predictor_names": predictor_names,
                "availability": availability_info,
                "persistence_note": persistence_note,
            }
            outputs = {
                "region": region,
                "existing_water": existing_water,
                "flood_map": flood_map,
                "vv_change": vv_change,
                "persistence": persistence,
                "stuck_water": stuck_water,
                "susceptibility": susceptibility,
                "final_overlay": final_overlay,
                "high_overlay": high_overlay,
                "extra_layers": extra_layers,
                "importance_fc": model_outputs["importance_fc"],
                "bbox": [xmin, ymin, xmax, ymax],
                "scale": scale,
                "building_asset": building_asset,
                "event_name": event_name,
                "label_start": label_start,
                "label_end": label_end,
            }
            st.session_state["dynamic_flood_outputs"] = outputs
            st.session_state["dynamic_flood_metrics"] = metrics
            status.update(label="Dynamic susceptibility model completed", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Dynamic susceptibility model failed", state="error", expanded=True)
            st.exception(exc)
            st.stop()

if "dynamic_flood_outputs" not in st.session_state:
    st.info("Select any historical/recent update window, then train the model. Use the Weekly Time Series page first to identify weeks with high SAR water-like area/rainfall.")
    st.stop()

outputs = st.session_state["dynamic_flood_outputs"]
metrics = st.session_state["dynamic_flood_metrics"]

st.success(f"Showing dynamic susceptibility model for: {outputs['event_name']} ({outputs['label_start']} to {outputs['label_end']})")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Target S1 images", metrics["stage1_count"])
m2.metric("Recovery S1 images", metrics["stage2_count"])
m3.metric("Flood-like area km²", metric_value(metrics["flood_area"], 2))
m4.metric("Stuck-water km²", metric_value(metrics["stuck_area"], 2))
m5.metric("RF accuracy", metric_value(metrics["accuracy"], 3))
st.caption(f"Kappa: {metric_value(metrics['kappa'], 3)} · Predictors: {', '.join(metrics['predictor_names'])}")
st.info(metrics.get("persistence_note", ""))

availability = metrics.get("availability", {})
if availability:
    with st.expander("Input data availability used by predictor builder", expanded=False):
        st.json(availability)
        missing_items = [k for k, v in availability.items() if v == 0]
        if missing_items:
            st.warning(
                "Some predictor collections had zero images/features and were filled with -9999 fallback values: "
                + ", ".join(missing_items)
            )

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Confusion matrix")
    st.write(metrics["cm_info"])
with c2:
    st.subheader("Variable importance")
    imp_df = pd.DataFrame([{"variable": k, "importance": v} for k, v in metrics["importance"].items()]).sort_values("importance")
    st.plotly_chart(px.bar(imp_df, x="importance", y="variable", orientation="h"), use_container_width=True)

st.subheader("Updated susceptibility map")
region = outputs["region"]
xmin, ymin, xmax, ymax = outputs["bbox"]
with st.spinner("Rendering updated Earth Engine layers..."):
    m = hobart_map(zoom_start=10)
    folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False, tooltip="AOI").add_to(m)
    add_ee_layer(m, outputs["flood_map"].selfMask(), get_default_vis("flood_map"), "Event SAR/NDWI flood-like label", shown=True)
    add_ee_layer(m, outputs["susceptibility"], get_default_vis("susceptibility"), "Updated RF susceptibility", shown=True, opacity=0.75)
    add_ee_layer(m, outputs["final_overlay"], get_default_vis("final_overlay"), "Updated final overlay", shown=False, opacity=0.75)
    add_ee_layer(m, outputs["high_overlay"].selfMask(), get_default_vis("high_overlay"), "High overlay zone", shown=False)
    add_ee_layer(m, outputs["vv_change"], get_default_vis("vv_change"), "VV change", shown=False)
    add_ee_layer(m, outputs["persistence"].selfMask(), get_default_vis("persistence"), "Persistence", shown=False)
    if "river_distance" in outputs["extra_layers"]:
        add_ee_layer(m, outputs["extra_layers"]["river_distance"], get_default_vis("river_distance"), "River distance", shown=False)
    if "soil_moisture" in outputs["extra_layers"]:
        add_ee_layer(m, outputs["extra_layers"]["soil_moisture"], get_default_vis("soil_moisture"), "Soil moisture", shown=False)
    try:
        buildings = ee.FeatureCollection(outputs["building_asset"]).filterBounds(region)
        add_ee_layer(m, buildings.style(color="000000", fillColor="00000000", width=1), {}, "Buildings", shown=False)
    except Exception:
        pass
    folium.LayerControl().add_to(m)
    st_folium(m, height=720, use_container_width=True, returned_objects=[], key="dynamic_flood_map_static")

st.subheader("Export updated model output")
folder = st.text_input("Google Drive folder", "GEE_Hobart_Flood_Project")
e1, e2, e3 = st.columns(3)
if e1.button("Export updated susceptibility"):
    task = ee.batch.Export.image.toDrive(image=outputs["susceptibility"], description="updated_dynamic_flood_susceptibility", folder=folder, fileNamePrefix="updated_dynamic_flood_susceptibility", region=region, scale=outputs["scale"], maxPixels=1e13)
    task.start()
    st.json(task.status())
if e2.button("Export updated final overlay"):
    task = ee.batch.Export.image.toDrive(image=outputs["final_overlay"], description="updated_dynamic_final_overlay", folder=folder, fileNamePrefix="updated_dynamic_final_overlay", region=region, scale=outputs["scale"], maxPixels=1e13)
    task.start()
    st.json(task.status())
if e3.button("Export variable importance CSV"):
    task = ee.batch.Export.table.toDrive(collection=outputs["importance_fc"], description="updated_dynamic_rf_importance", folder=folder, fileNamePrefix="updated_dynamic_rf_importance", fileFormat="CSV")
    task.start()
    st.json(task.status())
