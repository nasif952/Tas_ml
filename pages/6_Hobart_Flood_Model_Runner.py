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
from utils.gee_presets import (
    AOI_PRESETS,
    DATASET_PRESETS,
    FLOOD_EVENT_PRESETS,
    MODEL_PRESETS,
    NDWI_PRESETS,
    PAGE_HELP,
    S1_ORBIT_OPTIONS,
    S1_POLARIZATION_OPTIONS,
    VV_THRESHOLD_PRESETS,
)

st.set_page_config(page_title="Hobart Flood Model Runner", page_icon="🌊", layout="wide")
st.title("🌊 Hobart 2018 Flood Model Runner")
st.caption(
    "Run the real Earth Engine flood workflow from Streamlit: SAR flood map, NDWI mask, "
    "river distance, soil moisture, RF susceptibility, final overlay, and exports."
)

show_ee_connection_block()

import ee

st.warning(
    "Research note: this is an NDWI-masked SAR-derived potential flood extent workflow. "
    "It is not field-verified ground truth unless independent validation data is added."
)

# -----------------------------------------------------------------------------
# Sidebar: guided presets + custom inputs.
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Guided configuration")
    st.caption("Choose presets to auto-guide the settings, then edit any field manually.")

    aoi_name = st.selectbox("AOI preset", list(AOI_PRESETS.keys()))
    aoi_preset = AOI_PRESETS[aoi_name]
    st.caption(aoi_preset["description"])
    axmin, aymin, axmax, aymax = aoi_preset["bbox"]

    event_name = st.selectbox("Flood event/date preset", list(FLOOD_EVENT_PRESETS.keys()))
    event_preset = FLOOD_EVENT_PRESETS[event_name]
    st.caption(event_preset["description"])

    model_name = st.selectbox("Model speed/detail preset", list(MODEL_PRESETS.keys()))
    model_preset = MODEL_PRESETS[model_name]
    st.caption(model_preset["description"])

    vv_preset_name = st.selectbox("SAR VV threshold preset", list(VV_THRESHOLD_PRESETS.keys()))
    vv_preset = VV_THRESHOLD_PRESETS[vv_preset_name]
    st.caption(vv_preset["description"])

    ndwi_preset_name = st.selectbox("NDWI water-mask preset", list(NDWI_PRESETS.keys()))
    ndwi_preset = NDWI_PRESETS[ndwi_preset_name]
    st.caption(ndwi_preset["description"])

    st.divider()

    with st.expander("1) AOI and scale", expanded=True):
        st.caption(PAGE_HELP["aoi"])
        xmin = st.number_input("Min longitude", value=float(axmin), step=0.01, format="%.4f")
        ymin = st.number_input("Min latitude", value=float(aymin), step=0.01, format="%.4f")
        xmax = st.number_input("Max longitude", value=float(axmax), step=0.01, format="%.4f")
        ymax = st.number_input("Max latitude", value=float(aymax), step=0.01, format="%.4f")
        scale = st.select_slider(
            "Processing/export scale",
            [30, 50, 100, 250, 500, 1000],
            value=int(model_preset["scale"]),
            help="Smaller scale = more detail but slower. 100 m is a good web-app balance."
        )

    with st.expander("2) Sentinel-1 SAR flood detection", expanded=True):
        st.caption(PAGE_HELP["s1"])
        s1_stage1_start = st.text_input("Stage 1 / flood-period start", event_preset["stage1_start"])
        s1_stage1_end = st.text_input("Stage 1 / flood-period end", event_preset["stage1_end"])
        s1_stage2_start = st.text_input("Stage 2 / recovery-period start", event_preset["stage2_start"])
        s1_stage2_end = st.text_input("Stage 2 / recovery-period end", event_preset["stage2_end"])
        orbit = st.selectbox("Orbit pass", list(S1_ORBIT_OPTIONS.keys()))
        st.caption(S1_ORBIT_OPTIONS[orbit])
        polarization = st.selectbox("Polarization", list(S1_POLARIZATION_OPTIONS.keys()), index=0)
        st.caption(S1_POLARIZATION_OPTIONS[polarization])
        if polarization != "VV":
            st.info("The current production pipeline still uses VV. VH/VV+VH are shown as future configuration options.")
        vv_min = st.slider("VV water-like minimum dB", -30.0, -10.0, float(vv_preset["vv_min"]), 0.5)
        vv_max = st.slider("VV water-like maximum dB", -25.0, -5.0, float(vv_preset["vv_max"]), 0.5)
        stuck_change_tolerance = st.slider(
            "Stuck-water change tolerance ±dB",
            0.5, 6.0, 2.0, 0.5,
            help="Lower value = only pixels that remain very similar between flood and recovery stages are treated as persistent/stuck water."
        )

    with st.expander("3) Sentinel-2 NDWI existing-water mask", expanded=True):
        st.caption(PAGE_HELP["s2"])
        s2_dataset = st.text_input(
            "Sentinel-2 dataset ID",
            DATASET_PRESETS["Sentinel-2 Surface Reflectance Harmonized"]["id"],
            help="Current pipeline uses this preset internally, but this field documents the source and can guide future custom versions."
        )
        s2_water_start = st.text_input("Pre-flood Sentinel-2 start", event_preset["s2_start"])
        s2_water_end = st.text_input("Pre-flood Sentinel-2 end", event_preset["s2_end"])
        s2_cloud_pct = st.slider("Maximum Sentinel-2 cloud percentage", 5, 80, 40, 5)
        ndwi_threshold = st.slider("Existing-water NDWI threshold", -0.2, 0.5, float(ndwi_preset["threshold"]), 0.05)

    with st.expander("4) Predictor datasets and dates", expanded=True):
        st.caption(PAGE_HELP["predictors"])
        st.markdown("**Dataset IDs used by the pipeline**")
        st.caption("These are pre-filled but still editable so you can document or later swap sources.")
        srtm_id = st.text_input("DEM dataset", DATASET_PRESETS["SRTM DEM 30m"]["id"])
        chirps_id = st.text_input("Rainfall dataset", DATASET_PRESETS["CHIRPS Daily Rainfall"]["id"])
        modis_reflectance_id = st.text_input("NDVI/NDBI MODIS dataset", DATASET_PRESETS["MODIS Surface Reflectance 8-day"]["id"])
        modis_lst_id = st.text_input("Temperature MODIS dataset", DATASET_PRESETS["MODIS Land Surface Temperature"]["id"])
        smap_id = st.text_input("Soil moisture dataset", DATASET_PRESETS["SMAP Soil Moisture"]["id"])
        st.caption("Note: current backend functions use the standard dataset IDs above. Custom dataset execution can be added later per source.")

        predictor_start = st.text_input("MODIS predictor start", "2018-04-01")
        predictor_end = st.text_input("MODIS predictor end", "2018-05-09")
        month_start = st.text_input("Monthly rainfall/temp start", "2018-05-01")
        month_end = st.text_input("Monthly rainfall/temp end", "2018-06-01")
        event_rain_start = st.text_input("Event rainfall start", event_preset["event_rain_start"])
        event_rain_end = st.text_input("Event rainfall end", event_preset["event_rain_end"])
        soil_start = st.text_input("Soil moisture start", "2018-05-01")
        soil_end = st.text_input("Soil moisture end", "2018-06-01")

    with st.expander("5) River, soil moisture, and exposure factors", expanded=True):
        use_river_distance = st.checkbox("Use river distance predictor", value=True)
        river_asset = st.text_input("River FeatureCollection asset", DATASET_PRESETS["HydroSHEDS Free Flowing Rivers"]["id"])
        st.caption("Use HydroSHEDS for testing, but a local Tasmania hydrography / Geofabric river asset is better for final research.")
        river_distance_cap_m = st.slider("River distance cap m", 100, 3000, 1000, 100)
        use_soil_moisture = st.checkbox("Use soil moisture predictor", value=True)
        building_asset = st.text_input("Building FeatureCollection asset", DATASET_PRESETS["Greater Hobart Buildings asset"]["id"])

    with st.expander("6) Random Forest and final overlay", expanded=True):
        st.caption(PAGE_HELP["rf"])
        seed = st.number_input("Random seed", 0, 9999, 42)
        class_points = st.slider("Samples per class", 100, 2000, int(model_preset["samples"]), 100)
        train_fraction = st.slider("Training fraction", 0.50, 0.90, float(model_preset["train_fraction"]), 0.05)
        rf_trees = st.slider("RF trees", 50, 800, int(model_preset["trees"]), 50)
        susceptibility_weight = st.slider("Final overlay susceptibility weight", 0.0, 1.0, 0.70, 0.05)
        high_overlay_threshold = st.slider("High overlay threshold", 0.30, 0.90, 0.66, 0.01)

    clear_last = st.button("Clear last model output", use_container_width=True)
    if clear_last:
        st.session_state.pop("last_hobart_model", None)
        st.session_state.pop("last_hobart_metrics", None)
        st.session_state.pop("last_hobart_config", None)

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

run = st.button("Run Earth Engine Flood Model", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Execute model only when requested, otherwise reuse last successful output.
# -----------------------------------------------------------------------------
if run:
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
            metrics = {
                "stage1_count": stage1_col.size().getInfo(),
                "stage2_count": stage2_col.size().getInfo(),
                "flood_area": area_km2(flood_map, "Flood_Map", region, scale).getInfo(),
                "stuck_area": area_km2(stuck_water, "Stuck_Water", region, scale).getInfo(),
                "cm_info": model_outputs["confusion_matrix"].getInfo(),
                "accuracy": model_outputs["confusion_matrix"].accuracy().getInfo(),
                "kappa": model_outputs["confusion_matrix"].kappa().getInfo(),
                "importance": model_outputs["importance"].getInfo(),
                "predictor_names": predictor_names,
            }

            outputs = {
                "region": region,
                "existing_water": existing_water,
                "flood_map": flood_map,
                "persistence": persistence,
                "vv_change": vv_change,
                "stuck_water": stuck_water,
                "susceptibility": susceptibility,
                "final_overlay": final_overlay,
                "high_overlay": high_overlay,
                "importance_fc": model_outputs["importance_fc"],
                "extra_layers": extra_layers,
                "building_asset": building_asset,
                "bbox": [xmin, ymin, xmax, ymax],
                "scale": scale,
            }

            st.session_state["last_hobart_model"] = outputs
            st.session_state["last_hobart_metrics"] = metrics
            st.session_state["last_hobart_config"] = config
            status.update(label="Earth Engine model run completed", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Earth Engine model run failed", state="error", expanded=True)
            st.exception(exc)
            st.stop()

if "last_hobart_model" not in st.session_state:
    st.info("Configure the parameters in the sidebar, then click **Run Earth Engine Flood Model**. After one successful run, the output will stay visible during map reloads/reruns.")
    st.stop()

outputs = st.session_state["last_hobart_model"]
metrics = st.session_state["last_hobart_metrics"]

region = outputs["region"]
existing_water = outputs["existing_water"]
flood_map = outputs["flood_map"]
persistence = outputs["persistence"]
vv_change = outputs["vv_change"]
stuck_water = outputs["stuck_water"]
susceptibility = outputs["susceptibility"]
final_overlay = outputs["final_overlay"]
high_overlay = outputs["high_overlay"]
extra_layers = outputs["extra_layers"]
importance_fc = outputs["importance_fc"]
building_asset = outputs["building_asset"]
scale = outputs["scale"]
xmin, ymin, xmax, ymax = outputs["bbox"]

# -----------------------------------------------------------------------------
# Results display.
# -----------------------------------------------------------------------------
st.success("Showing last successful Earth Engine model output. Change settings and click Run again to update it.")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stage 1 S1 images", metrics["stage1_count"])
m2.metric("Stage 2 S1 images", metrics["stage2_count"])
m3.metric("Flood area km²", f"{metrics['flood_area']:.2f}")
m4.metric("Stuck-water km²", f"{metrics['stuck_area']:.2f}")
m5.metric("RF accuracy", f"{metrics['accuracy']:.3f}")
st.caption(f"Kappa: {metrics['kappa']:.3f} · Predictors: {', '.join(metrics['predictor_names'])}")

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Confusion matrix")
    st.caption("This compares the RF model prediction against the SAR/NDWI-derived label split. It is not independent field validation.")
    st.write(metrics["cm_info"])
with c2:
    st.subheader("Variable importance")
    st.caption("Higher values mean the RF used that variable more often/usefully to reproduce the SAR-derived flood label.")
    imp_df = pd.DataFrame([{"variable": k, "importance": v} for k, v in metrics["importance"].items()]).sort_values("importance")
    fig = px.bar(imp_df, x="importance", y="variable", orientation="h", title="Earth Engine RF Variable Importance")
    st.plotly_chart(fig, use_container_width=True)

# Map
st.subheader("Interactive Earth Engine output map")
st.caption("Map interactions are disabled from triggering Streamlit reruns. Use the layer control to switch between flood, susceptibility, river, soil moisture, and building layers.")
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
st.caption(PAGE_HELP["export"])
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
    task = ee.batch.Export.table.toDrive(collection=importance_fc, description="hobart_rf_importance_streamlit", folder=folder, fileNamePrefix="hobart_rf_importance_streamlit", fileFormat="CSV")
    task.start()
    st.json(task.status())
