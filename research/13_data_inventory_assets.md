# Data Inventory and Asset Notes

This document records the current datasets and assets used or discussed in the Hobart 2018 Flood project.

## 1. Study area asset

Current GEE study area:

```text
projects/sturdy-apricot-405823/assets/Study_Area
```

Purpose:

- Main analysis boundary.
- Used for filtering imagery.
- Used for clipping images.
- Used as export region.
- Used for sampling Random Forest training/testing points.

Recommended GEE setup:

```javascript
var studyAreaFC = ee.FeatureCollection(
  'projects/sturdy-apricot-405823/assets/Study_Area'
);
var studyArea = studyAreaFC.geometry();
```

## 2. Council boundaries

Recent uploaded council boundary ZIPs:

- Sorell
- Huon Valley
- Brighton

Processing notes:

- Use municipality polygon shapefiles, not boundary segment line shapefiles.
- Convert to WGS84 / EPSG:4326 before GEE upload.
- Use Merge to keep boundaries separate.
- Use Dissolve only if a single combined AOI is required.

Preferred output for council-level work:

```text
hobart_councils_merged_wgs84_GEE.zip
```

This keeps Sorell, Huon Valley, and Brighton as separate polygon features inside one shapefile.

## 3. Building footprint asset

Current building asset:

```text
projects/sturdy-apricot-405823/assets/Greater_Hobart_Buildings_WGS84
```

Purpose:

- Building exposure screening.
- Centroid sampling against high final-overlay zones.

Important limitation:

- Centroid inside high-risk zone does not always mean the whole building polygon is flooded.
- Future improvement should compare full building polygon overlap with flood/final-overlay zones.

## 4. Sentinel-1 SAR

Dataset:

```text
COPERNICUS/S1_GRD
```

Current use:

- VV polarisation.
- Stage 1 event-period flood-like water classification.
- Stage 2 persistence / stuck-water analysis.

Current dates:

- Stage 1: 2018-05-15 to 2018-05-16
- Stage 2: 2018-05-27 to 2018-05-28

Important future task:

- Confirm whether these image dates are the best possible Sentinel-1 dates relative to the actual flood peak.

## 5. Sentinel-2 NDWI

Dataset:

```text
COPERNICUS/S2_SR_HARMONIZED
```

Current use:

- Pre-flood NDWI.
- Existing water mask.

Current date range:

```text
2018-03-01 to 2018-05-09
```

Purpose:

- Remove existing/permanent water from Sentinel-1 flood-like classification.

## 6. DEM and terrain

Dataset:

```text
USGS/SRTMGL1_003
```

Derived variables:

- elevation
- slope
- TWI_simple

Limitation:

- The current TWI is simplified and only based on slope. A stronger future version should use flow accumulation and drainage structure.

## 7. Rainfall

Dataset:

```text
UCSB-CHG/CHIRPS/DAILY
```

Current variables:

- precipitation: May 2018 total rainfall.
- event_rainfall: 2018-05-10 to 2018-05-13 rainfall.

## 8. Temperature

Dataset:

```text
MODIS/061/MOD11A2
```

Current variable:

- temperature from LST_Day_1km.

## 9. NDVI and NDBI

Dataset:

```text
MODIS/061/MOD09A1
```

Current variables:

- NDVI
- NDBI

NDBI is currently important in the RF output. Interpretation should be careful because it may reflect both true urban flood exposure and SAR behaviour in built-up areas.

## 10. River distance

Current river dataset in script:

```text
WWF/HydroSHEDS/v1/FreeFlowingRivers
```

Current derived variables:

- river_distance_1km
- river_buffer_1km

Important note:

If the HydroSHEDS FreeFlowingRivers dataset fails in GEE or does not represent Hobart's river network well, replace it with a local river/hydrography asset such as a Tasmanian hydrography or Geofabric river network layer.

## 11. Soil moisture

Dataset:

```text
NASA/SMAP/SPL4SMGP/007
```

Current variable:

- soil_moisture from `sm_surface`

Date range:

```text
2018-05-01 to 2018-06-01
```

Limitation:

- SMAP is coarse resolution, so it should be treated as broad antecedent moisture context, not property-level soil wetness.

## 12. TSFM / water-level resources

Earlier project work discussed TSFM calibration and water-level datasets.

Current interpretation:

- These may be useful for contextual validation.
- They may not be direct flood extent or flood depth observations.
- The project must confirm the meaning of each dataset before using it for accuracy assessment.

## 13. Best-practice asset management

Recommended future GEE asset naming:

```text
Study_Area
Greater_Hobart_Buildings_WGS84
hobart_councils_merged_wgs84
hobart_councils_dissolved_wgs84
hobart_rivers_wgs84
hobart_validation_points_wgs84
```

Recommended CRS for uploaded vector layers:

```text
EPSG:4326 / WGS 84
```
