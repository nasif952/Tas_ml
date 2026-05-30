# Streamlit + Google Earth Engine UI Visualization Research Handoff

**Repository:** `nasif952/Tas_ml`  
**Branch:** `claude/research-docs-review-P3Vbg`  
**Purpose:** Handoff document for another AI/developer to implement a full Google Earth Engine powered Streamlit flood intelligence interface.  
**Source:** Uploaded deep research report: `deep-research-report.md`.

---

## 1. Executive Direction

The Streamlit app should evolve from a prototype/simulation dashboard into a production-style **Google Earth Engine flood intelligence interface**.

The target design is a **multipage Streamlit application** with four major surfaces:

1. **Browse / Data Catalog Page**
   - Browse user Earth Engine project assets.
   - Browse curated public datasets.
   - Inspect metadata.
   - Detect whether an asset is an `Image`, `ImageCollection`, `FeatureCollection`, table, or derived product.

2. **Map / Visualization Page**
   - Interactively render Earth Engine layers.
   - Preview rasters, vectors, thumbnails, and map tiles.
   - Configure AOI, date ranges, bands, scale, palette, thresholds, and reducers.

3. **Download / Export Page**
   - Allow small direct downloads using Earth Engine download URLs.
   - Allow large batch exports to Google Drive, Cloud Storage, or Earth Engine assets.
   - Track task status inside the Streamlit UI.

4. **Admin / Settings Page**
   - Test Earth Engine connection.
   - Show current project ID and service account email.
   - Test read/list/map/export permissions.
   - Manage non-secret app settings.
   - Keep secret credentials outside GitHub.

The app should not treat all Earth Engine requests the same way. It must separate **interactive operations** from **batch operations**.

---

## 2. Core Architecture

Recommended architecture:

```text
User
  ↓
Streamlit app
  ↓ reads secrets
st.secrets / environment variables
  ↓
Google service account credentials
  ↓
ee.Initialize(project=...)
  ↓
Google Earth Engine API
  ↓
Images / ImageCollections / FeatureCollections / tables / map tiles / exports
```

Practical implementation layers:

```text
app.py
pages/
  1_GEE_Configuration.py
  2_Earth_Engine_Status.py
  3_GEE_Data_Browser.py
  4_GEE_Map_Viewer.py
  5_GEE_Download_Export.py
  6_Flood_Model_Training.py
utils/
  ee_auth.py
  ee_assets.py
  ee_visualization.py
  ee_exports.py
  flood_pipeline.py
  ui_components.py
research/
  15_streamlit_gee_ui_visualization_research.md
```

---

## 3. Earth Engine Data Types the UI Should Support

The app should be designed around these Earth Engine object/product types.

| Earth Engine type/product | Access method | UI use |
|---|---|---|
| `ee.Image` | `ee.Image(asset_id)` | Single raster layer, DEM, flood map, soil moisture image, NDVI image |
| `ee.ImageCollection` | `ee.ImageCollection(asset_id)` | Sentinel-1, Sentinel-2, CHIRPS, MODIS, SMAP, time filtering, compositing |
| `ee.FeatureCollection` | `ee.FeatureCollection(asset_id)` | Buildings, rivers, flood points, admin boundaries, roads |
| Table asset | `ee.FeatureCollection(asset_id)` | CSV/shapefile style uploaded assets |
| Project assets | `ee.data.listAssets`, `ee.data.getAsset` | Asset browser and metadata inspector |
| Map tiles / Map IDs | `getMapId()` or geemap wrapper | Interactive map preview |
| Thumbnail | `getThumbURL()` | Fast static preview cards |
| Direct download URL | `getDownloadURL()` | Small raster/vector downloads |
| Batch export | `ee.batch.Export.*` | Large outputs to Drive, Cloud Storage, or EE assets |
| FeatureView | FeatureView assets | Large vector visualization |

---

## 4. UI Data Fetching Capabilities

The Streamlit UI should be able to fetch and preview data by user settings.

### 4.1 Asset Browser

UI controls:

- Project ID text/select field.
- Asset root path.
- Refresh button.
- Search/filter box.
- Asset type filter.
- Table of assets.
- Metadata viewer.

Backend pattern:

```python
import ee
import pandas as pd
import streamlit as st

@st.cache_data(ttl=300)
def list_assets(parent: str, limit: int = 500) -> pd.DataFrame:
    rows = []
    token = None

    while len(rows) < limit:
        params = {"pageSize": min(100, limit - len(rows))}
        if token:
            params["pageToken"] = token

        resp = ee.data.listAssets(parent, params)
        for asset in resp.get("assets", []):
            rows.append({
                "name": asset.get("name"),
                "type": asset.get("type"),
                "updateTime": asset.get("updateTime"),
                "sizeBytes": asset.get("sizeBytes"),
            })

        token = resp.get("nextPageToken")
        if not token:
            break

    return pd.DataFrame(rows)
```

### 4.2 Public Dataset Presets

The app should include curated dataset presets rather than trying to build a full Earth Engine catalog search.

Recommended flood-related presets:

| Dataset | Earth Engine ID | Type | Use |
|---|---|---|---|
| Sentinel-1 SAR GRD | `COPERNICUS/S1_GRD` | ImageCollection | Flood detection |
| Sentinel-2 SR Harmonized | `COPERNICUS/S2_SR_HARMONIZED` | ImageCollection | NDWI, permanent water masking |
| SRTM DEM | `USGS/SRTMGL1_003` | Image | Elevation, slope, terrain factors |
| CHIRPS Daily Rainfall | `UCSB-CHG/CHIRPS/DAILY` | ImageCollection | Rainfall/event rainfall |
| MODIS Surface Reflectance | `MODIS/061/MOD09A1` | ImageCollection | NDVI/NDBI predictors |
| MODIS Land Surface Temperature | `MODIS/061/MOD11A2` | ImageCollection | Temperature predictor |
| SMAP Soil Moisture | `NASA/SMAP/SPL4SMGP/007` | ImageCollection | Soil moisture predictor |
| HydroSHEDS rivers | candidate river dataset/asset | FeatureCollection | River distance |
| User building asset | `projects/.../assets/Greater_Hobart_Buildings_WGS84` | FeatureCollection | Exposure analysis |

---

## 5. Visualization Strategy

### 5.1 Interactive Maps

Best first approach:

```python
import geemap.foliumap as geemap

m = geemap.Map()
m.addLayer(ee_image, vis_params, "Layer name")
m.centerObject(geometry_or_feature, 10)
m.to_streamlit(height=650)
```

This allows Earth Engine layers to be viewed inside Streamlit with minimal manual tile handling.

### 5.2 Static Thumbnails

Use thumbnails when:

- A quick preview is enough.
- The layer is heavy.
- The user only needs a static report image.
- Interactive tile rendering is slow.

Pattern:

```python
thumb_url = image.getThumbURL({
    "min": 0,
    "max": 1,
    "palette": ["green", "yellow", "red"],
    "dimensions": 768,
    "region": region,
})
st.image(thumb_url)
```

### 5.3 Vector Visualization

For small to medium vectors:

```python
styled = fc.style(color="yellow", fillColor="00000000", width=2)
m.addLayer(styled, {}, "Vector layer")
```

For large vectors:

- Prefer FeatureView if available.
- Avoid calling `getInfo()` on large FeatureCollections.
- Use `limit()` or `listFeatures()` for previews.

---

## 6. Full Configuration Options Needed in the UI

### 6.1 AOI / Study Area

Controls:

- Bounding box inputs: min lon, min lat, max lon, max lat.
- Preset: Hobart rectangle.
- Upload GeoJSON AOI.
- Future: map-drawing polygon.

Backend:

```python
region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
```

### 6.2 Sentinel-1 SAR Flood Detection

Controls:

- Stage 1/pre-flood date range.
- Stage 2/recovery date range.
- Polarisation: VV, VH, VV + VH.
- Orbit: ascending, descending, either.
- VV minimum threshold.
- VV maximum threshold.
- Change threshold.
- Speckle filter on/off.
- Slope masking on/off.
- Slope threshold.

Flood map logic currently used:

```javascript
var floodMap = stage1VV
  .lte(vvMax)
  .and(stage1VV.gte(vvMin))
  .and(existingWater.not())
  .rename('Flood_Map');
```

### 6.3 Sentinel-2 NDWI Mask

Controls:

- Pre-flood Sentinel-2 date range.
- Cloud percentage threshold.
- NDWI threshold.
- Existing water mask on/off.

Logic:

```javascript
var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var existingWater = ndwi.gt(ndwiThreshold).rename('Existing_Water');
```

### 6.4 Predictor Layer Settings

Controls:

- Use elevation.
- Use slope.
- Use TWI.
- Use NDVI.
- Use NDBI.
- Use precipitation.
- Use event rainfall.
- Use temperature.
- Use river distance.
- Use soil moisture.
- River distance cap: default 1000 m.
- Soil moisture date range.

The app should allow toggling each factor on/off and then retraining the model.

### 6.5 Random Forest Model Tuning

Controls:

- Number of trees.
- Samples per class.
- Train/test split.
- Random seed.
- Output mode: classification/probability.
- Class balance mode.
- Feature list.

Metrics to show:

- Confusion matrix.
- Accuracy.
- Kappa.
- Producer accuracy.
- Consumer accuracy.
- Precision.
- Recall.
- F1.
- Feature importance chart.

### 6.6 Final Overlay Tuning

Controls:

- RF susceptibility weight.
- Stuck-water weight.
- High overlay threshold.
- Building exposure threshold.

Current idea:

```javascript
var finalOverlay = susceptibility
  .multiply(0.7)
  .add(stuckWater.multiply(0.3))
  .rename('Final_Overlay');
```

In the UI this should become:

```python
final_overlay = susceptibility_weight * susceptibility + stuck_weight * stuck_water
```

---

## 7. Download and Export Design

### 7.1 Small Downloads

Use direct URLs when output is small.

Raster:

```python
url = image.getDownloadURL({
    "scale": scale,
    "region": region_geojson,
    "format": "GEO_TIFF",
    "filePerBand": False,
})
st.markdown(f"[Download GeoTIFF]({url})")
```

Vector/table:

```python
url = fc.getDownloadURL(filetype="CSV", filename="selected_features")
st.markdown(f"[Download CSV]({url})")
```

### 7.2 Large Exports

Use batch exports for full maps, large raster clips, large building exposure tables, or publication-ready layers.

Image to Drive:

```python
task = ee.batch.Export.image.toDrive(
    image=image,
    description="hobart_final_overlay",
    folder="GEE_Hobart_Flood_Project",
    fileNamePrefix="hobart_final_overlay",
    region=region,
    scale=scale,
    maxPixels=1e13,
)
task.start()
```

Track status:

```python
status = task.status()
st.json(status)
```

Recommended UI:

- Start export button.
- Export destination selector: Drive / Cloud Storage / Earth Engine Asset.
- Export scale.
- File name prefix.
- Task status panel.
- Warning that large exports are asynchronous.

---

## 8. Performance Rules

Important performance rules for Claude/developer:

1. Do not call `getInfo()` on large Earth Engine objects.
2. Use `getInfo()` only for small metadata, counts, and small dictionaries.
3. Use `getMapId()`/geemap for interactive maps.
4. Use `getThumbURL()` for static previews.
5. Use `getDownloadURL()` only for small outputs.
6. Use batch exports for large outputs.
7. Cache Earth Engine initialization with `st.cache_resource`.
8. Cache asset metadata and lists with `st.cache_data(ttl=300)`.
9. Avoid heavy Earth Engine processing inside Streamlit tabs because tabs render even when not selected.
10. Prefer multipage Streamlit structure for heavy GEE pages.

---

## 9. Security Rules

The app already uses a service-account secret stored in Streamlit Cloud.

Rules:

- Never commit JSON keys to GitHub.
- Never print the private key in the app.
- Mask service-account details except project ID and service account email.
- Rotate exposed keys.
- Use minimum required Earth Engine role.
- For browsing only, viewer role may work.
- For map IDs, thumbnails, and exports, writer-level Earth Engine permission may be required.
- Admin actions should be restricted.

Recommended secret structure currently used in this project:

```toml
[earthengine]
project = "gee-project-493107"

type = "service_account"
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----
"""
client_email = "tas-ml-earthengine@gee-project-493107.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

Existing Streamlit initialization pattern:

```python
import ee
import streamlit as st
from google.oauth2 import service_account

@st.cache_resource(show_spinner=False)
def init_earth_engine():
    ee_secrets = st.secrets["earthengine"]

    service_account_info = {
        "type": ee_secrets["type"],
        "project_id": ee_secrets["project"],
        "private_key_id": ee_secrets["private_key_id"],
        "private_key": ee_secrets["private_key"],
        "client_email": ee_secrets["client_email"],
        "client_id": ee_secrets["client_id"],
        "auth_uri": ee_secrets["auth_uri"],
        "token_uri": ee_secrets["token_uri"],
        "auth_provider_x509_cert_url": ee_secrets["auth_provider_x509_cert_url"],
        "client_x509_cert_url": ee_secrets["client_x509_cert_url"],
    }

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )

    ee.Initialize(credentials, project=ee_secrets["project"])
    return True
```

---

## 10. Recommended New Pages for the Streamlit App

### Page 1: Earth Engine Status

Already added as:

```text
pages/2_Earth_Engine_Status.py
```

Should test:

- Secret exists.
- Service account email loads.
- `ee.Initialize()` works.
- A simple SRTM reducer works.
- Optional: asset root listing works.

### Page 2: GEE Configuration

Already added as:

```text
pages/1_GEE_Configuration.py
```

Currently generates GEE JavaScript and runs app-side simulated training. Next upgrade should make it actually execute the Earth Engine pipeline using Python API.

### Page 3: GEE Data Browser

New page to implement:

```text
pages/3_GEE_Data_Browser.py
```

Must support:

- Browse `projects/gee-project-493107/assets`.
- Display asset type.
- Display metadata.
- Preview bands for images.
- Preview row samples for tables.
- Preview collection count for ImageCollections.

### Page 4: GEE Map Viewer

New page to implement:

```text
pages/4_GEE_Map_Viewer.py
```

Must support:

- Select dataset/asset.
- Choose visualization parameters.
- Add layer to map.
- Toggle flood map, susceptibility, final overlay, buildings, rivers.
- Generate thumbnail fallback.

### Page 5: Download / Export

New page to implement:

```text
pages/5_GEE_Download_Export.py
```

Must support:

- Direct small raster download.
- Direct small vector/table download.
- Large export to Drive.
- Large export to Cloud Storage later.
- Task status.

### Page 6: Real Flood Model Runner

New page to implement:

```text
pages/6_Hobart_Flood_Model_Runner.py
```

Must convert the current GEE JavaScript pipeline into Python Earth Engine:

- Build stage 1 VV image.
- Build stage 2 VV image.
- Build Sentinel-2 NDWI water mask.
- Build flood map label.
- Build persistence/stuck water.
- Build predictors.
- Train RF inside Earth Engine.
- Output susceptibility raster.
- Output final overlay raster.
- Visualize all layers.
- Export selected outputs.

---

## 11. Hobart Flood Intelligence Specific UI Plan

### Top-level modes

```text
Mode 1: Explore Data
Mode 2: Configure Flood Detection
Mode 3: Train Susceptibility Model
Mode 4: Visualize Outputs
Mode 5: Download / Export
Mode 6: Building Exposure Analysis
```

### Key outputs to visualize

| Output | Type | Visualization |
|---|---|---|
| Existing water mask | raster binary | dark blue mask |
| NDWI-masked SAR flood map | raster binary | cyan mask |
| VV change | raster continuous | blue-white-red diverging palette |
| Flood persistence | raster class | light blue/red |
| River distance 1 km | raster continuous | blue-cyan-yellow-red |
| River buffer | raster binary/vector | purple |
| Soil moisture | raster continuous | brown-yellow-green-blue |
| RF susceptibility | raster probability | green-yellow-red |
| Final overlay | raster probability | green-yellow-orange-red |
| High overlay zone | raster binary | purple |
| Buildings | FeatureCollection | black outlines |
| Selected building centroids | FeatureCollection/table | yellow points |

---

## 12. Immediate Implementation Checklist for Claude

1. Keep existing app structure but add utility modules.
2. Move repeated EE auth code into `utils/ee_auth.py`.
3. Add `geemap` to `requirements.txt`.
4. Create `pages/3_GEE_Data_Browser.py`.
5. Create `pages/4_GEE_Map_Viewer.py`.
6. Create `pages/5_GEE_Download_Export.py`.
7. Convert current GEE JS flood pipeline into Python EE functions.
8. Make each pipeline parameter controlled by Streamlit sidebar widgets.
9. Render Earth Engine outputs with `geemap.foliumap.Map(...).to_streamlit()`.
10. Add direct small downloads using `getDownloadURL()`.
11. Add large exports using `ee.batch.Export.image.toDrive()` and `ee.batch.Export.table.toDrive()`.
12. Add task status polling.
13. Add a research warning: SAR/NDWI flood map is a remote-sensing-derived potential flood extent, not field-verified truth.
14. Add a security warning if secrets are missing or invalid.
15. Keep private keys out of all logs and UI.

---

## 13. Recommended First Claude Task

Ask Claude to do this first:

> Convert the existing GEE JavaScript flood workflow into Python Earth Engine functions inside `utils/flood_pipeline.py`, then create a new Streamlit page `pages/6_Hobart_Flood_Model_Runner.py` that exposes all major parameters through UI controls and visualizes the resulting Earth Engine layers using geemap.

The most important functions to create:

```python
def get_s1_vv(study_area, start, end, orbit=None):
    pass


def get_s2_existing_water(study_area, start, end, cloud_pct, ndwi_threshold):
    pass


def build_flood_map(stage1_vv, existing_water, vv_min, vv_max):
    pass


def build_persistence(stage1_vv, stage2_vv, flood_map, vv_min, vv_max, change_tolerance):
    pass


def build_predictors(study_area, config):
    pass


def train_rf_susceptibility(predictors, label, predictor_names, config):
    pass


def build_final_overlay(susceptibility, stuck_water, susceptibility_weight):
    pass
```

---

## 14. Final Notes

This project is now technically ready to move from a static prototype to a real Earth Engine-backed Streamlit system.

The main design principle:

> Use Streamlit for configuration, controls, visualization, and workflow orchestration; use Earth Engine for geospatial computation, raster/vector processing, map tiles, and exports.

The next serious engineering step is not more research; it is implementation of a real Earth Engine execution page that replaces the simulated data workflow with live Earth Engine objects.
