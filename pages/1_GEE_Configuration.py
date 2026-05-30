import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(
    page_title="GEE Configuration · Tasmania Flood Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.3rem; }
.main-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0.2rem; }
.sub-title { color: #90CAF9; font-size: 1rem; margin-top: 0; }
.small-note { color: #B0BEC5; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Synthetic Hobart-like data for live Streamlit model tuning.
# This does not replace the real GEE pipeline. It lets the deployed app behave
# like a live lab where thresholds/features/hyperparameters can be tested.
# -----------------------------------------------------------------------------

@st.cache_data
def make_live_dataset(n: int, seed: int, river_cap_km: float, soil_moisture_strength: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-43.10, -42.65, n)
    lon = rng.uniform(146.85, 147.55, n)

    # Approximate Derwent/urban river corridor geometry for simulation only.
    river_lon_at_lat = 147.335 + (lat + 42.88) * 0.45
    dist_river_km = np.abs(lon - river_lon_at_lat) * 82.0
    river_distance_1km = np.minimum(dist_river_km, river_cap_km)

    # Elevation and terrain simulation.
    mt_lat, mt_lon = -42.896, 147.234
    dist_mt = np.sqrt((lat - mt_lat) ** 2 + (lon - mt_lon) ** 2)
    elevation_m = 1200 * np.exp(-dist_mt ** 2 / 0.006) + rng.normal(0, 22, n)
    elevation_m = np.clip(elevation_m, 0, 1270)
    slope_deg = np.clip(rng.gamma(2.0, 3.0, n) + elevation_m * 0.012, 0, 45)
    twi = np.clip(12.5 - slope_deg * 0.13 - elevation_m * 0.003 + rng.normal(0, 1.1, n), 1, 18)

    # Spectral/environmental predictors.
    ndvi = np.clip(rng.normal(0.45, 0.16, n) - elevation_m / 5000, -0.2, 0.9)
    ndbi = np.clip(rng.normal(0.12, 0.14, n) + np.exp(-dist_river_km / 2.5) * 0.18, -0.4, 0.65)
    precipitation = np.clip(75 + rng.normal(0, 15, n) + elevation_m * 0.015, 15, 180)
    event_rainfall = np.clip(30 + rng.gamma(2.0, 8.0, n) + np.exp(-dist_river_km / 3) * 18, 0, 120)
    temperature = np.clip(11.5 - elevation_m * 0.004 + rng.normal(0, 1.5, n), 2, 20)
    soil_moisture = np.clip(
        0.20 + event_rainfall / 300 + np.exp(-dist_river_km / 4) * 0.08 + rng.normal(0, 0.035, n),
        0.05,
        0.65,
    )

    # SAR and existing water simulation.
    existing_water = (dist_river_km < 0.12).astype(int)
    sar_pre_db = -9.5 - ndbi * 3.5 - existing_water * 8 + rng.normal(0, 1.4, n)

    logit = (
        -0.0075 * elevation_m
        -0.12 * slope_deg
        -0.55 * river_distance_1km
        + 0.30 * twi
        + 1.70 * ndbi
        + 0.025 * event_rainfall
        + soil_moisture_strength * soil_moisture
        + 1.8
    )
    true_prob = 1 / (1 + np.exp(-logit))
    flooded = (rng.random(n) < true_prob).astype(int)
    sar_change_db = flooded * rng.uniform(-8, -3, n) + rng.normal(0, 1.1, n)
    sar_flood_db = sar_pre_db + sar_change_db

    return pd.DataFrame({
        "lat": lat,
        "lon": lon,
        "elevation": elevation_m,
        "slope": slope_deg,
        "TWI_simple": twi,
        "NDVI": ndvi,
        "NDBI": ndbi,
        "precipitation": precipitation,
        "event_rainfall": event_rainfall,
        "temperature": temperature,
        "river_distance_1km": river_distance_1km,
        "soil_moisture": soil_moisture,
        "existing_water": existing_water,
        "sar_pre_db": sar_pre_db,
        "sar_change_db": sar_change_db,
        "sar_flood_db": sar_flood_db,
        "true_prob": true_prob,
        "flooded": flooded,
    })


FEATURE_LABELS = {
    "elevation": "Elevation",
    "slope": "Slope",
    "TWI_simple": "TWI simple",
    "NDVI": "NDVI",
    "NDBI": "NDBI",
    "precipitation": "May precipitation",
    "event_rainfall": "Event rainfall",
    "temperature": "Temperature",
    "river_distance_1km": "River distance capped at 1 km",
    "soil_moisture": "Soil moisture",
}


def build_gee_code(cfg: dict) -> str:
    river_asset = cfg["river_asset"].strip() or "WWF/HydroSHEDS/v1/FreeFlowingRivers"
    building_asset = cfg["building_asset"].strip()
    river_block = f"var rivers = ee.FeatureCollection('{river_asset}').filterBounds(studyArea);"
    building_block = f"var buildings = ee.FeatureCollection('{building_asset}').filterBounds(studyArea);" if building_asset else "// var buildings = ee.FeatureCollection('YOUR_BUILDINGS_ASSET').filterBounds(studyArea);"

    predictor_names = [
        "elevation", "slope", "TWI_simple", "NDVI", "NDBI",
        "precipitation", "event_rainfall", "temperature"
    ]
    if cfg["use_river"]:
        predictor_names.append("river_distance_1km")
    if cfg["use_soil"]:
        predictor_names.append("soil_moisture")

    predictor_names_js = ",\n  ".join([f"'{p}'" for p in predictor_names])
    optional_bands = ""
    if cfg["use_river"]:
        optional_bands += "\n  .addBands(riverDistance1km)"
    if cfg["use_soil"]:
        optional_bands += "\n  .addBands(soilMoisture)"

    optional_layers = ""
    if cfg["use_river"]:
        optional_layers += """
Map.addLayer(riverDistance1km, {min: 0, max: riverDistanceCapM, palette: ['blue', 'cyan', 'yellow', 'red']}, 'River Distance capped');
Map.addLayer(riverBuffer.selfMask(), {palette: ['purple']}, 'River Buffer');
"""
    if cfg["use_soil"]:
        optional_layers += """
Map.addLayer(soilMoisture, {min: 0, max: 0.5, palette: ['brown', 'yellow', 'green', 'blue']}, 'Soil Moisture');
"""

    return f"""// =====================================================
// GENERATED GEE CODE FROM STREAMLIT CONFIGURATION PAGE
// Hobart 2018 Flood: NDWI-masked SAR flood map + RF susceptibility
// Includes configurable river distance and soil moisture predictors
// =====================================================

// -----------------------------
// 0. Configuration
// -----------------------------
var studyArea = ee.Geometry.Rectangle([{cfg['xmin']}, {cfg['ymin']}, {cfg['xmax']}, {cfg['ymax']}]);
var scale = {cfg['scale']};
var seed = {cfg['seed']};

var s1Stage1Start = '{cfg['s1_stage1_start']}';
var s1Stage1End   = '{cfg['s1_stage1_end']}';
var s1Stage2Start = '{cfg['s1_stage2_start']}';
var s1Stage2End   = '{cfg['s1_stage2_end']}';

var s2WaterStart = '{cfg['s2_water_start']}';
var s2WaterEnd   = '{cfg['s2_water_end']}';
var predictorStart = '{cfg['predictor_start']}';
var predictorEnd   = '{cfg['predictor_end']}';
var eventRainStart = '{cfg['event_rain_start']}';
var eventRainEnd   = '{cfg['event_rain_end']}';

var vvMin = {cfg['vv_min']};
var vvMax = {cfg['vv_max']};
var ndwiThreshold = {cfg['ndwi_threshold']};
var stuckChangeTolerance = {cfg['stuck_change_tolerance']};
var riverDistanceCapM = {cfg['river_cap_m']};
var rfTrees = {cfg['rf_trees']};
var classPoints = {cfg['class_points']};
var finalSusceptibilityWeight = {cfg['sus_weight']};
var finalStuckWaterWeight = {cfg['stuck_weight']};
var highOverlayThreshold = {cfg['high_overlay_threshold']};

Map.centerObject(studyArea, 10);

// -----------------------------
// 1. Helper functions
// -----------------------------
function getS1VV(start, end) {{
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(studyArea)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select('VV');
}}

function maskS2Clouds(img) {{
  var qa = img.select('QA60');
  var cloud = 1 << 10;
  var cirrus = 1 << 11;
  var mask = qa.bitwiseAnd(cloud).eq(0).and(qa.bitwiseAnd(cirrus).eq(0));
  return img.updateMask(mask).divide(10000);
}}

function areaKm2(img, band, name) {{
  var area = img.selfMask().multiply(ee.Image.pixelArea()).reduceRegion({{
    reducer: ee.Reducer.sum(),
    geometry: studyArea,
    scale: scale,
    maxPixels: 1e13
  }});
  print(name, ee.Number(area.get(band)).divide(1e6));
}}

// -----------------------------
// 2. Sentinel-1 SAR stages
// -----------------------------
var stage1VV = getS1VV(s1Stage1Start, s1Stage1End).median().clip(studyArea).rename('VV');
var stage2VV = getS1VV(s1Stage2Start, s1Stage2End).median().clip(studyArea).rename('Stage2_VV');
print('Stage 1 VV image count:', getS1VV(s1Stage1Start, s1Stage1End).size());
print('Stage 2 VV image count:', getS1VV(s1Stage2Start, s1Stage2End).size());

// -----------------------------
// 3. Sentinel-2 NDWI existing-water mask
// -----------------------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyArea)
  .filterDate(s2WaterStart, s2WaterEnd)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', {cfg['s2_cloud_pct']}))
  .map(maskS2Clouds)
  .median()
  .clip(studyArea);

var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var existingWater = ndwi.gt(ndwiThreshold).rename('Existing_Water');

// -----------------------------
// 4. MODIS NDVI / NDBI predictors
// -----------------------------
var modis = ee.ImageCollection('MODIS/061/MOD09A1')
  .filterBounds(studyArea)
  .filterDate(predictorStart, predictorEnd)
  .median()
  .clip(studyArea)
  .multiply(0.0001);

var ndvi = modis.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('NDVI');
var ndbi = modis.normalizedDifference(['sur_refl_b06', 'sur_refl_b02']).rename('NDBI');

// -----------------------------
// 5. SAR flood map label
// -----------------------------
var floodMap = stage1VV
  .lte(vvMax)
  .and(stage1VV.gte(vvMin))
  .and(existingWater.not())
  .rename('Flood_Map');

var vvChange = stage2VV.subtract(stage1VV).rename('Stage2_minus_Stage1');
var stuckWater = floodMap
  .and(stage2VV.lte(vvMax))
  .and(stage2VV.gte(vvMin))
  .and(vvChange.abs().lte(stuckChangeTolerance))
  .rename('Stuck_Water');
var persistence = floodMap.add(stuckWater).rename('Flood_Persistence');

// -----------------------------
// 6. Predictor layers
// -----------------------------
var dem = ee.Image('USGS/SRTMGL1_003').clip(studyArea).rename('elevation');
var slope = ee.Terrain.slope(dem).rename('slope');
var twi = ee.Image(1).divide(slope.multiply(Math.PI).divide(180).tan().add(0.001)).log().rename('TWI_simple');

var precipitation = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(studyArea)
  .filterDate('{cfg['month_rain_start']}', '{cfg['month_rain_end']}')
  .sum()
  .clip(studyArea)
  .rename('precipitation');

var eventRainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(studyArea)
  .filterDate(eventRainStart, eventRainEnd)
  .sum()
  .clip(studyArea)
  .rename('event_rainfall');

var temperature = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterBounds(studyArea)
  .filterDate('{cfg['month_rain_start']}', '{cfg['month_rain_end']}')
  .select('LST_Day_1km')
  .mean()
  .multiply(0.02)
  .subtract(273.15)
  .clip(studyArea)
  .rename('temperature');

// River distance layer
{river_block}
var riverRaster = ee.Image().byte().paint({{featureCollection: rivers, color: 1}}).clip(studyArea);
var riverDistance = riverRaster.fastDistanceTransform(1024).sqrt().multiply(ee.Image.pixelArea().sqrt()).clip(studyArea).rename('river_distance_m');
var riverDistance1km = riverDistance.min(riverDistanceCapM).rename('river_distance_1km');
var riverBuffer = riverDistance.lte(riverDistanceCapM).rename('river_buffer');

// Soil moisture layer
var soilMoisture = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
  .filterBounds(studyArea)
  .filterDate('{cfg['soil_start']}', '{cfg['soil_end']}')
  .select('sm_surface')
  .mean()
  .clip(studyArea)
  .rename('soil_moisture');

// -----------------------------
// 7. Predictor stack
// -----------------------------
var predictorNames = [
  {predictor_names_js}
];

var predictors = dem
  .addBands(slope)
  .addBands(twi)
  .addBands(ndvi)
  .addBands(ndbi)
  .addBands(precipitation)
  .addBands(eventRainfall)
  .addBands(temperature){optional_bands}
  .unmask(-9999);

var label = floodMap.unmask(0).toInt().rename('label');
var stack = predictors.addBands(label);

// -----------------------------
// 8. Train/test RF model
// -----------------------------
var samples = stack.stratifiedSample({{
  numPoints: 0,
  classBand: 'label',
  classValues: [0, 1],
  classPoints: [classPoints, classPoints],
  region: studyArea,
  scale: scale,
  seed: seed,
  geometries: true,
  dropNulls: false
}});

var randomSamples = samples.randomColumn('random', seed);
var trainingSamples = randomSamples.filter(ee.Filter.lt('random', {cfg['train_frac']}));
var testingSamples = randomSamples.filter(ee.Filter.gte('random', {cfg['train_frac']}));

var rfClassifier = ee.Classifier.smileRandomForest({{numberOfTrees: rfTrees, seed: seed}}).train({{
  features: trainingSamples,
  classProperty: 'label',
  inputProperties: predictorNames
}});

var tested = testingSamples.classify(rfClassifier);
var cm = tested.errorMatrix('label', 'classification');
print('Confusion Matrix:', cm);
print('Overall Accuracy:', cm.accuracy());
print('Kappa:', cm.kappa());
print('Producer Accuracy:', cm.producersAccuracy());
print('Consumer Accuracy:', cm.consumersAccuracy());

var importance = ee.Dictionary(rfClassifier.explain().get('importance'));
var importanceTable = ee.FeatureCollection(importance.keys().map(function(key) {{
  return ee.Feature(null, {{variable: key, importance: importance.get(key)}});
}}));
print('Variable Importance Table:', importanceTable);

var importanceChart = ui.Chart.feature.byFeature({{features: importanceTable, xProperty: 'variable', yProperties: ['importance']}})
  .setChartType('ColumnChart')
  .setOptions({{title: 'Random Forest Variable Importance', hAxis: {{title: 'Variables'}}, vAxis: {{title: 'Importance'}}, legend: {{position: 'none'}}}});
print(importanceChart);

var rfProbability = ee.Classifier.smileRandomForest({{numberOfTrees: rfTrees, seed: seed}})
  .setOutputMode('PROBABILITY')
  .train({{features: trainingSamples, classProperty: 'label', inputProperties: predictorNames}});

var susceptibility = predictors.select(predictorNames).classify(rfProbability).rename('Flood_Susceptibility');
var finalOverlay = susceptibility.multiply(finalSusceptibilityWeight).add(stuckWater.multiply(finalStuckWaterWeight)).rename('Final_Overlay');

// -----------------------------
// 9. Display layers
// -----------------------------
Map.addLayer(existingWater.selfMask(), {{palette: ['darkblue']}}, 'Existing Water Mask from S2 NDWI');
Map.addLayer(floodMap.selfMask(), {{palette: ['cyan']}}, 'NDWI-Masked SAR Flood Map');
Map.addLayer(vvChange, {{min: -5, max: 5, palette: ['blue', 'white', 'red']}}, 'VV Change');
Map.addLayer(persistence.selfMask(), {{min: 1, max: 2, palette: ['lightblue', 'red']}}, 'Flood Persistence');
{optional_layers}
Map.addLayer(susceptibility, {{min: 0, max: 1, palette: ['green', 'yellow', 'red']}}, 'RF Flood Susceptibility');
Map.addLayer(finalOverlay, {{min: 0, max: 1, palette: ['green', 'yellow', 'orange', 'red']}}, 'Final Overlay');

// -----------------------------
// 10. Area and exports
// -----------------------------
areaKm2(floodMap, 'Flood_Map', 'Flood area km2:');
areaKm2(stuckWater, 'Stuck_Water', 'Stuck-water area km2:');
areaKm2(riverBuffer, 'river_buffer', 'River influence zone area km2:');

Export.image.toDrive({{image: floodMap, description: 'Hobart_NDWI_Masked_SAR_Flood_Map', folder: 'GEE_Hobart_Flood_Project', fileNamePrefix: 'hobart_ndwi_masked_sar_flood_map', region: studyArea, scale: scale, maxPixels: 1e13}});
Export.image.toDrive({{image: susceptibility, description: 'Hobart_RF_Flood_Susceptibility_Configured', folder: 'GEE_Hobart_Flood_Project', fileNamePrefix: 'hobart_rf_flood_susceptibility_configured', region: studyArea, scale: scale, maxPixels: 1e13}});
Export.image.toDrive({{image: finalOverlay, description: 'Hobart_Final_Overlay_Configured', folder: 'GEE_Hobart_Flood_Project', fileNamePrefix: 'hobart_final_overlay_configured', region: studyArea, scale: scale, maxPixels: 1e13}});
Export.table.toDrive({{collection: importanceTable, description: 'Hobart_RF_Variable_Importance_Configured', folder: 'GEE_Hobart_Flood_Project', fileNamePrefix: 'hobart_rf_variable_importance_configured', fileFormat: 'CSV'}});

// -----------------------------
// 11. Optional building exposure overlay
// -----------------------------
{building_block}
var highFinalOverlay = finalOverlay.gte(highOverlayThreshold).unmask(0).toInt().rename('high_final_overlay');
Map.addLayer(highFinalOverlay.selfMask(), {{palette: ['purple']}}, 'High Final Overlay Zone');
"""


st.markdown('<p class="main-title">🛰️ GEE Configuration & Live Model Tuning</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Configure the Hobart 2018 flood GEE JavaScript pipeline, test model settings in the Streamlit app, and export a runnable Code Editor script.</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("GEE Run Settings")
    st.caption("These controls generate the JavaScript code and also drive the live Streamlit model simulation.")

    with st.expander("AOI and scale", expanded=True):
        xmin = st.number_input("Min longitude", value=146.85, step=0.01, format="%.4f")
        ymin = st.number_input("Min latitude", value=-43.10, step=0.01, format="%.4f")
        xmax = st.number_input("Max longitude", value=147.55, step=0.01, format="%.4f")
        ymax = st.number_input("Max latitude", value=-42.65, step=0.01, format="%.4f")
        scale = st.select_slider("Export/model scale", options=[30, 50, 100, 250, 500, 1000], value=100)

    with st.expander("Dates", expanded=True):
        s1_stage1_start = st.text_input("Stage 1 SAR start", "2018-05-15")
        s1_stage1_end = st.text_input("Stage 1 SAR end", "2018-05-16")
        s1_stage2_start = st.text_input("Stage 2 SAR start", "2018-05-27")
        s1_stage2_end = st.text_input("Stage 2 SAR end", "2018-05-28")
        s2_water_start = st.text_input("Pre-flood Sentinel-2 start", "2018-03-01")
        s2_water_end = st.text_input("Pre-flood Sentinel-2 end", "2018-05-09")
        predictor_start = st.text_input("MODIS predictor start", "2018-04-01")
        predictor_end = st.text_input("MODIS predictor end", "2018-05-09")
        month_rain_start = st.text_input("Monthly rainfall/temp start", "2018-05-01")
        month_rain_end = st.text_input("Monthly rainfall/temp end", "2018-06-01")
        event_rain_start = st.text_input("Event rainfall start", "2018-05-10")
        event_rain_end = st.text_input("Event rainfall end", "2018-05-13")
        soil_start = st.text_input("Soil moisture start", "2018-05-01")
        soil_end = st.text_input("Soil moisture end", "2018-06-01")

    with st.expander("Flood detection thresholds", expanded=True):
        vv_min = st.slider("VV water-like min dB", -30.0, -10.0, -20.0, 0.5)
        vv_max = st.slider("VV water-like max dB", -25.0, -5.0, -15.0, 0.5)
        ndwi_threshold = st.slider("NDWI existing-water threshold", -0.2, 0.5, 0.0, 0.05)
        s2_cloud_pct = st.slider("Sentinel-2 cloud percentage max", 5, 80, 40, 5)
        stuck_change_tolerance = st.slider("Stuck-water change tolerance ±dB", 0.5, 6.0, 2.0, 0.5)

    with st.expander("New factors", expanded=True):
        use_river = st.checkbox("Use river distance predictor", value=True)
        river_cap_m = st.slider("River distance cap", 100, 3000, 1000, 100)
        river_asset = st.text_input("River FeatureCollection asset", "WWF/HydroSHEDS/v1/FreeFlowingRivers")
        use_soil = st.checkbox("Use soil moisture predictor", value=True)

    with st.expander("RF training and overlay", expanded=True):
        seed = st.number_input("Random seed", 0, 9999, 42)
        class_points = st.slider("Samples per class", 100, 2000, 600, 100)
        rf_trees = st.slider("Random Forest trees", 50, 800, 300, 50)
        train_frac = st.slider("Training fraction", 0.50, 0.90, 0.70, 0.05)
        sus_weight = st.slider("Final overlay susceptibility weight", 0.0, 1.0, 0.70, 0.05)
        stuck_weight = round(1.0 - sus_weight, 2)
        high_overlay_threshold = st.slider("High overlay threshold", 0.30, 0.90, 0.66, 0.01)

    with st.expander("Buildings", expanded=False):
        building_asset = st.text_input("Building asset", "projects/sturdy-apricot-405823/assets/Greater_Hobart_Buildings_WGS84")

cfg = dict(
    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, scale=scale, seed=int(seed),
    s1_stage1_start=s1_stage1_start, s1_stage1_end=s1_stage1_end,
    s1_stage2_start=s1_stage2_start, s1_stage2_end=s1_stage2_end,
    s2_water_start=s2_water_start, s2_water_end=s2_water_end,
    predictor_start=predictor_start, predictor_end=predictor_end,
    month_rain_start=month_rain_start, month_rain_end=month_rain_end,
    event_rain_start=event_rain_start, event_rain_end=event_rain_end,
    soil_start=soil_start, soil_end=soil_end,
    vv_min=vv_min, vv_max=vv_max, ndwi_threshold=ndwi_threshold, s2_cloud_pct=s2_cloud_pct,
    stuck_change_tolerance=stuck_change_tolerance,
    use_river=use_river, river_cap_m=river_cap_m, river_asset=river_asset,
    use_soil=use_soil,
    class_points=class_points, rf_trees=rf_trees, train_frac=train_frac,
    sus_weight=sus_weight, stuck_weight=stuck_weight, high_overlay_threshold=high_overlay_threshold,
    building_asset=building_asset,
)

# -----------------------------------------------------------------------------
# Live app simulation and model training.
# -----------------------------------------------------------------------------

n_points = min(max(class_points * 2, 300), 4000)
df = make_live_dataset(n_points, int(seed), river_cap_m / 1000.0, soil_moisture_strength=2.8 if use_soil else 0.0)

# Build a simulated SAR/NDWI-masked flood label matching the GEE idea.
sar_water_like = (df["sar_flood_db"] >= vv_min) & (df["sar_flood_db"] <= vv_max)
ndwi_existing_water_mask = df["existing_water"].astype(bool)
sim_flood_map = sar_water_like & ~ndwi_existing_water_mask
sim_stuck_water = sim_flood_map & (np.abs(df["sar_change_db"]) <= stuck_change_tolerance)
df["sim_flood_map"] = sim_flood_map.astype(int)
df["sim_stuck_water"] = sim_stuck_water.astype(int)

available_features = ["elevation", "slope", "TWI_simple", "NDVI", "NDBI", "precipitation", "event_rainfall", "temperature"]
if use_river:
    available_features.append("river_distance_1km")
if use_soil:
    available_features.append("soil_moisture")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Configuration Summary",
    "🤖 Live Training/Test",
    "🗺️ Live Map Preview",
    "💻 Generated GEE JavaScript",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AOI scale", f"{scale} m")
    c2.metric("RF trees", f"{rf_trees}")
    c3.metric("River cap", f"{river_cap_m} m")
    c4.metric("Overlay weights", f"{sus_weight:.2f}/{stuck_weight:.2f}")

    st.subheader("Current model factors")
    factor_df = pd.DataFrame({
        "Factor": [FEATURE_LABELS[f] for f in available_features],
        "Band name": available_features,
        "Used in RF": ["Yes"] * len(available_features),
    })
    st.dataframe(factor_df, use_container_width=True, hide_index=True)

    st.info(
        "This page gives real-time Streamlit training/testing on a Hobart-like simulated dataset and generates a runnable GEE JavaScript script. "
        "For true Earth Engine execution inside Streamlit, the deployed app needs Earth Engine Python API authentication using Streamlit secrets."
    )

with tab2:
    st.subheader("Live model training/testing")
    col_l, col_r = st.columns([1, 2])

    with col_l:
        algo = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        selected_features = st.multiselect(
            "Predictor features",
            available_features,
            default=available_features,
            format_func=lambda x: FEATURE_LABELS[x],
        )
        target = st.selectbox(
            "Training label",
            ["sim_flood_map", "flooded"],
            format_func=lambda x: "SAR/NDWI-derived flood label" if x == "sim_flood_map" else "Simulated true flood label",
        )
        test_size = 1.0 - train_frac

    with col_r:
        if not selected_features:
            st.warning("Select at least one feature.")
        else:
            X = df[selected_features].values
            y = df[target].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                Xs, y, test_size=test_size, random_state=int(seed), stratify=y if len(np.unique(y)) > 1 else None
            )

            if algo == "Random Forest":
                model = RandomForestClassifier(n_estimators=rf_trees, random_state=int(seed), n_jobs=-1)
            elif algo == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=min(rf_trees, 300), random_state=int(seed))
            else:
                model = LogisticRegression(max_iter=700, random_state=int(seed))

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            auc_score = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1", f"{f1:.3f}")
            m5.metric("ROC-AUC", "n/a" if np.isnan(auc_score) else f"{auc_score:.3f}")

            cm = confusion_matrix(y_test, pred)
            cma, cmb = st.columns(2)
            with cma:
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
            with cmb:
                if hasattr(model, "feature_importances_"):
                    fi = pd.DataFrame({
                        "Feature": [FEATURE_LABELS[f] for f in selected_features],
                        "Importance": model.feature_importances_,
                    }).sort_values("Importance")
                    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Feature Importance")
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    coef = pd.DataFrame({
                        "Feature": [FEATURE_LABELS[f] for f in selected_features],
                        "Coefficient": model.coef_[0],
                    }).sort_values("Coefficient")
                    fig_cf = px.bar(coef, x="Coefficient", y="Feature", orientation="h", title="Model Coefficients")
                    st.plotly_chart(fig_cf, use_container_width=True)

with tab3:
    st.subheader("Live map preview")
    st.caption("This map is generated from app-side simulated points. The real flood raster is generated in GEE using the script in the next tab.")
    map_choice = st.radio("Map layer", ["SAR/NDWI flood label", "Stuck water", "True flood probability"], horizontal=True)
    if map_choice == "SAR/NDWI flood label":
        value_col = "sim_flood_map"
    elif map_choice == "Stuck water":
        value_col = "sim_stuck_water"
    else:
        value_col = "true_prob"

    m = folium.Map(location=[-42.88, 147.33], zoom_start=11, tiles="CartoDB positron")
    heat = [[r.lat, r.lon, float(getattr(r, value_col))] for r in df.itertuples()]
    HeatMap(heat, radius=13, blur=14, gradient={"0.2": "blue", "0.5": "cyan", "0.75": "yellow", "1.0": "red"}).add_to(m)
    sample = df.sample(min(250, len(df)), random_state=int(seed))
    for r in sample.itertuples():
        is_flood = bool(getattr(r, value_col) > 0.5)
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=3,
            color="#E53935" if is_flood else "#43A047",
            fill=True,
            fill_opacity=0.6,
            popup=(
                f"Layer value: {getattr(r, value_col):.3f}<br>"
                f"NDBI: {r.NDBI:.3f}<br>"
                f"River dist: {r.river_distance_1km:.2f} km<br>"
                f"Soil moisture: {r.soil_moisture:.3f}"
            ),
        ).add_to(m)
    st_folium(m, height=560, use_container_width=True)

with tab4:
    st.subheader("Generated runnable GEE JavaScript")
    gee_code = build_gee_code(cfg)
    st.download_button(
        "Download GEE JavaScript",
        data=gee_code,
        file_name="hobart_flood_configured_gee.js",
        mime="text/javascript",
        use_container_width=True,
    )
    st.code(gee_code, language="javascript")
