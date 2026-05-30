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
    get_s1_vv_event_change,
    get_s2_existing_water,
    build_change_flood_map,
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
    "Train and update a flood susceptibility map using pre-event vs during-event Sentinel-1 SAR change detection. "
    "Flood labels are now based on a dB drop from the closest pre-event SAR image to the mean during-event SAR image."
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
        "sar_change_threshold_db": -2.0,
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
        "max_extra_days": 3,
        "building_asset": "",
        "note": "Uses 2026-01-16 to 2026-01-28 as pre-event/reference SAR and 2026-02-02 to 2026-02-16 as during-event SAR.",
    },
    "Hobart May 2018 default": {
        "label_start": "2018-05-10",
        "label_end": "2018-05-18",
        "recovery_start": "2018-05-01",
        "recovery_end": "2018-05-09",
        "water_start": "2018-03-01",
        "water_end": "2018-05-09",
        "predictor_start": "2018-04-01",
        "predictor_end": "2018-05-18",
        "month_start": "2018-05-01",
        "month_end": "2018-06-01",
        "event_rain_start": "2018-05-10",
        "event_rain_end": "2018-05-18",
        "soil_start": "2018-05-01",
        "soil_end": "2018-05-18",
        "aoi_name": "Hobart core / default",
        "ndwi_preset": "Default existing water NDWI > 0.0",
        "ndwi_threshold": 0.0,
        "s2_cloud_pct": 40,
        "vv_preset": "Conservative open water (-20 to -15 dB)",
        "vv_min": -20.0,
        "vv_max": -15.0,
        "sar_change_threshold_db": -2.0,
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
        "max_extra_days": 3,
        "building_asset": DATASET_PRESETS["Greater Hobart Buildings asset"]["id"],
        "note": "Hobart 2018 setup using pre-event SAR before the flood and during-event SAR during the flood window.",
    },
    "Recent/custom weekly update": {
        "label_start": "2025-01-01",
        "label_end": "2025-01-08",
        "recovery_start": "2024-12-20",
        "recovery_end": "2024-12-31",
        "water_start": "2024-10-01",
        "water_end": "2024-12-31",
        "predictor_start": "2024-12-01",
        "predictor_end": "2025-01-08",
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
        "sar_change_threshold_db": -2.0,
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
        "max_extra_days": 3,
        "building_asset": DATASET_PRESETS["Greater Hobart Buildings asset"]["id"],
        "note": "Template for training a new weekly/event SAR-change model. Edit all dates below.",
    },
    "Manual event dates": {
        "label_start": "2026-01-01",
        "label_end": "2026-01-08",
        "recovery_start": "2025-12-20",
        "recovery_end": "2025-12-31",
        "water_start": "2025-10-01",
        "water_end": "2025-12-31",
        "predictor_start": "2025-12-01",
        "predictor_end": "2026-01-08",
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
        "sar_change_threshold_db": -2.0,
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
        "max_extra_days": 3,
        "building_asset": DATASET_PRESETS["Greater Hobart Buildings asset"]["id"],
        "note": "Fully custom pre-event vs during-event SAR-change setup.",
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
        "Current logic: find the closest pre-event SAR image, build a mean during-event SAR image, "
        "then detect flood where during SAR drops by the selected dB threshold and existing water is masked."
    )

    label_start = st.text_input("During-event SAR start", ev["label_start"], key=f"{event_key}_label_start")
    label_end = st.text_input("During-event SAR end", ev["label_end"], key=f"{event_key}_label_end")
    recovery_start = st.text_input("Pre-event/reference SAR search start", ev["recovery_start"], key=f"{event_key}_recovery_start")
    recovery_end = st.text_input("Pre-event/reference SAR search end", ev["recovery_end"], key=f"{event_key}_recovery_end")
    max_extra_days = st.select_slider(
        "Auto-extend if no data",
        options=[0, 1, 2, 3],
        value=int(ev.get("max_extra_days", 3)),
        key=f"{event_key}_max_extra_days",
    )
    st.caption("0 = exact date range only. 3 = try nominal range, then +1, +2, +3 days.")

    if recovery_end >= label_start:
        st.warning("For true pre/during SAR change detection, the pre-event/reference SAR end should be before the during-event SAR start.")

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
    st.caption("Existing water is masked using Sentinel-2 NDWI OR MODIS NDWI before flood area is counted.")

    st.divider()
    st.header("4. SAR change detection")
    vv_options = list(VV_THRESHOLD_PRESETS.keys())
    vv_preset_name = st.selectbox("During-event VV water-like preset", vv_options, index=preset_index(vv_options, ev["vv_preset"]), key=f"{event_key}_vv_preset")
    vv_min = st.slider("During VV water-like min dB", -30.0, -10.0, float(ev.get("vv_min", VV_THRESHOLD_PRESETS[vv_preset_name]["vv_min"])), 0.5, key=f"{event_key}_vv_min")
    vv_max = st.slider("During VV water-like max dB", -25.0, -5.0, float(ev.get("vv_max", VV_THRESHOLD_PRESETS[vv_preset_name]["vv_max"])), 0.5, key=f"{event_key}_vv_max")
    sar_change_threshold_db = st.slider("Flood SAR drop threshold dB", -6.0, 0.0, float(ev.get("sar_change_threshold_db", -2.0)), 0.5, key=f"{event_key}_sar_drop")
    st.caption("Flood condition: during_mean_VV - pre_closest_VV must be less than or equal to this value. Example: -2 dB means at least a 2 dB drop.")
    orbit_options = ["Either", "ASCENDING", "DESCENDING"]
    orbit = st.selectbox("Sentinel-1 orbit", orbit_options, index=preset_index(orbit_options, ev["orbit"]), key=f"{event_key}_orbit")
    stuck_change_tolerance = st.slider("Reference similarity tolerance ±dB", 0.5, 6.0, float(ev["stuck_change_tolerance"]), 0.5, key=f"{event_key}_stuck_tol")

    st.divider()
    st.header("5. Predictors and dates")
    st.caption("These predictors are rebuilt for the selected event/update period. Each collection also uses adaptive +0 to +3 day extension if needed.")
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
    sar_change_threshold_db=sar_change_threshold_db,
    river_asset=river_asset, river_distance_cap_m=river_distance_cap_m,
    use_river_distance=use_river_distance, use_soil_moisture=use_soil_moisture,
    class_points=class_points, train_fraction=train_fraction,
    rf_trees=rf_trees, susceptibility_weight=susceptibility_weight,
    high_overlay_threshold=high_overlay_threshold,
    max_extra_days=int(max_extra_days),
)

run = st.button("Train / update susceptibility map", type="primary", use_container_width=True)

if run:
    with st.status("Training SAR-change dynamic susceptibility model...", expanded=True) as status:
        try:
            region = study_area(config)
            adaptive_summary = {}

            status.write("Building pre-event closest SAR and during-event mean SAR...")
            pre_vv, during_vv, sar_change, pre_col, target_col, pre_info, target_info = get_s1_vv_event_change(
                region=region,
                pre_start=recovery_start,
                pre_end=recovery_end,
                target_start=label_start,
                target_end=label_end,
                orbit=orbit,
                max_extra_days=config.max_extra_days,
            )
            pre_count = pre_col.size().getInfo()
            target_count = target_col.size().getInfo()
            pre_meta = ee.Dictionary({
                "pre_s1_count": pre_info["count"],
                "pre_s1_extra_days_used": pre_info["extra_days"],
                "pre_s1_actual_end": pre_info["actual_end"],
                "pre_s1_found": pre_info["found"],
            }).getInfo()
            target_meta = ee.Dictionary({
                "during_s1_count": target_info["count"],
                "during_s1_extra_days_used": target_info["extra_days"],
                "during_s1_actual_end": target_info["actual_end"],
                "during_s1_found": target_info["found"],
            }).getInfo()
            adaptive_summary.update(pre_meta)
            adaptive_summary.update(target_meta)

            if pre_count == 0:
                st.error("No pre-event Sentinel-1 image found, even after adaptive extension. Widen the pre-event/reference SAR search window or choose Either orbit.")
                st.stop()
            if target_count == 0:
                st.error("No during-event Sentinel-1 image found, even after adaptive extension. Widen the during-event SAR window or choose Either orbit.")
                st.stop()

            status.write("Building Sentinel-2/MODIS NDWI existing-water mask...")
            ndwi, existing_water, s2_water_info = get_s2_existing_water(region, config, return_info=True)
            adaptive_summary.update(s2_water_info.getInfo())

            status.write("Building SAR-change flood label: during water-like AND dB drop AND not existing water...")
            flood_map = build_change_flood_map(during_vv, sar_change, existing_water, config)
            vv_change = sar_change.rename("During_minus_Pre")
            stuck_water = flood_map.And(sar_change.abs().lte(config.stuck_change_tolerance)).rename("Stuck_Water")
            persistence = flood_map.add(stuck_water).rename("Flood_Persistence")
            persistence_note = (
                "Flood label built from SAR change: during-event mean VV minus closest pre-event VV <= "
                f"{config.sar_change_threshold_db:.1f} dB, with Sentinel-2/MODIS existing-water mask applied."
            )

            status.write("Building event-specific predictors with adaptive date extension and missing-data safeguards...")
            predictors, predictor_names, extra_layers = build_predictors(region, config)
            availability = extra_layers.get("availability")
            availability_info = availability.getInfo() if availability is not None else {}
            availability_info.update(adaptive_summary)
            availability_info["sar_change_threshold_db"] = config.sar_change_threshold_db

            status.write("Training Random Forest and generating susceptibility probability raster...")
            model_outputs = train_rf_susceptibility(region, predictors, flood_map, predictor_names, config)
            susceptibility = model_outputs["susceptibility"]
            final_overlay, high_overlay = build_final_overlay(susceptibility, stuck_water, config)
            status.write("Calculating metrics...")
            metrics = {
                "stage1_count": target_count,
                "stage2_count": pre_count,
                "flood_area": area_km2(flood_map, "Flood_Map", region, scale).getInfo(),
                "stuck_area": area_km2(stuck_water, "Stuck_Water", region, scale).getInfo(),
                "accuracy": model_outputs["confusion_matrix"].accuracy().getInfo(),
                "kappa": model_outputs["confusion_matrix"].kappa().getInfo(),
                "cm_info": model_outputs["confusion_matrix"].getInfo(),
                "importance": model_outputs["importance"].getInfo(),
                "predictor_names": predictor_names,
                "availability": availability_info,
                "persistence_note": persistence_note,
                "max_extra_days": int(max_extra_days),
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
            status.update(label="SAR-change dynamic susceptibility model completed", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="SAR-change dynamic susceptibility model failed", state="error", expanded=True)
            st.exception(exc)
            st.stop()

if "dynamic_flood_outputs" not in st.session_state:
    st.info("Select any event/update window, then train the model. Use a real flood window as the during-event SAR period and a dry/normal pre-event SAR window before it.")
    st.stop()

outputs = st.session_state["dynamic_flood_outputs"]
metrics = st.session_state["dynamic_flood_metrics"]

st.success(f"Showing SAR-change dynamic susceptibility model for: {outputs['event_name']} ({outputs['label_start']} to {outputs['label_end']})")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("During S1 images", metrics["stage1_count"])
m2.metric("Pre-event S1 images", metrics["stage2_count"])
m3.metric("SAR-change flood km²", metric_value(metrics["flood_area"], 2))
m4.metric("Stable/change-filtered km²", metric_value(metrics["stuck_area"], 2))
m5.metric("RF accuracy", metric_value(metrics["accuracy"], 3))
st.caption(f"Kappa: {metric_value(metrics['kappa'], 3)} · Predictors: {', '.join(metrics['predictor_names'])}")
st.info(metrics.get("persistence_note", ""))

availability = metrics.get("availability", {})
if availability:
    with st.expander("Input data availability, adaptive dates, and SAR-change threshold", expanded=True):
        st.json(availability)
        not_found = [k for k, v in availability.items() if k.endswith("_found") and v is False]
        extra_used = {k: v for k, v in availability.items() if k.endswith("_extra_days_used") and v not in [0, "0"]}
        if extra_used:
            st.info("Some datasets needed adaptive date extension. The exact extra days and actual end dates are shown above.")
        if not_found:
            st.warning("Some datasets still had no data after adaptive extension and used fallback values: " + ", ".join(not_found))

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Confusion matrix")
    st.write(metrics["cm_info"])
with c2:
    st.subheader("Variable importance")
    imp_df = pd.DataFrame([{"variable": k, "importance": v} for k, v in metrics["importance"].items()]).sort_values("importance")
    st.plotly_chart(px.bar(imp_df, x="importance", y="variable", orientation="h"), use_container_width=True)

st.subheader("Updated SAR-change susceptibility map")
region = outputs["region"]
xmin, ymin, xmax, ymax = outputs["bbox"]
with st.spinner("Rendering updated Earth Engine layers..."):
    m = hobart_map(zoom_start=10)
    folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]], color="black", weight=2, fill=False, tooltip="AOI").add_to(m)
    add_ee_layer(m, outputs["flood_map"].selfMask(), get_default_vis("flood_map"), "SAR-change flood label", shown=True)
    add_ee_layer(m, outputs["susceptibility"], get_default_vis("susceptibility"), "Updated RF susceptibility", shown=True, opacity=0.75)
    add_ee_layer(m, outputs["final_overlay"], get_default_vis("final_overlay"), "Updated final overlay", shown=False, opacity=0.75)
    add_ee_layer(m, outputs["high_overlay"].selfMask(), get_default_vis("high_overlay"), "High overlay zone", shown=False)
    add_ee_layer(m, outputs["vv_change"], get_default_vis("vv_change"), "During minus pre SAR change", shown=True)
    add_ee_layer(m, outputs["persistence"].selfMask(), get_default_vis("persistence"), "Persistence/change consistency", shown=False)
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
