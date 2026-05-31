# Recent Progress Update

This document records the latest project progress after the initial handoff documentation was created.

## 1. Study area update

The project has moved from a simple rectangular AOI to an uploaded Google Earth Engine polygon asset:

```javascript
var studyAreaFC = ee.FeatureCollection(
  'projects/sturdy-apricot-405823/assets/Study_Area'
);
var studyArea = studyAreaFC.geometry();
```

This is important because the model now uses the actual study-area boundary rather than a coarse rectangle.

## 2. Council boundary work

Council boundary files were prepared for GEE use.

Recent boundary workflow:

- Sorell boundary ZIP uploaded.
- Huon Valley boundary ZIP uploaded.
- Brighton boundary ZIP uploaded.
- Municipality polygon layers were selected instead of boundary-segment line layers.
- Layers were merged while keeping council boundaries separate.
- Output was converted to WGS84 / EPSG:4326.
- A dissolved one-feature version was also created, but the preferred version for separate councils is the merged non-dissolved file.

Important distinction:

- Merge = one layer with separate council polygon features.
- Dissolve = one combined polygon feature with internal boundaries removed.

For the current project, the separate-boundary merged file is preferred when council-level comparison is needed.

## 3. Current GEE flood and susceptibility model

The current GEE workflow now includes:

- Sentinel-1 VV flood-like pixel detection.
- Sentinel-2 NDWI pre-flood water masking.
- Improved stuck-water / persistence layer.
- Random Forest susceptibility modelling.
- River distance capped at 1 km.
- Soil moisture from SMAP.
- Building centroid selection inside high final-overlay zones.

The current code file is:

```text
gee/hobart_rf_flood_susceptibility_river_soilmoisture.js
```

## 4. New predictor variables added

Two important flood susceptibility predictors were added:

### River distance within 1 km

The model now includes `river_distance_1km`.

Interpretation:

- 0 m = on or very near a river.
- 1000 m = 1 km or farther from a river.

A continuous distance variable is preferred over a simple yes/no river buffer because Random Forest can learn gradients of risk near drainage corridors.

### Soil moisture

The model now includes `soil_moisture` from SMAP for May 2018.

Interpretation:

- Higher soil moisture may represent wetter antecedent ground conditions.
- Wet ground may contribute to runoff and flood susceptibility.

## 5. NDWI-masked SAR flood map interpretation

The current flood map should be described as:

**NDWI-masked SAR-derived flood extent map**

or:

**Potential flood inundation after removing pre-existing surface water.**

This is because Sentinel-1 SAR identifies water-like backscatter, while Sentinel-2 NDWI removes water that was already present before the flood.

It should not be described as a fully field-verified flood boundary.

## 6. NDBI variable importance interpretation

NDBI became highly important in the Random Forest variable importance output. This likely means that built-up and impervious urban surfaces are strongly associated with the SAR-derived flood susceptibility pattern.

This is reasonable for Hobart because flood impacts may occur around roads, buildings, drainage corridors, and urban surfaces. However, the interpretation must be careful because the model is trained using the SAR-derived flood label, not independent field-observed flood points.

Recommended wording:

> NDBI showed high importance, indicating that built-up and impervious surfaces were strongly associated with the SAR-derived flood susceptibility pattern. This may reflect true urban flood exposure, but it may also reflect SAR classification behaviour in complex urban areas.

## 7. Building exposure screening

The workflow now includes a step to identify buildings whose centroids fall inside high final-overlay zones.

This is useful for future insurance and exposure analysis, but it should be described as a screening method rather than a confirmed building-level flood impact assessment.

## 8. Current research status

The project has moved from early concept to an active prototype with:

- Defined study-area asset.
- GEE code structure.
- Flood map generation.
- Susceptibility modelling.
- New hydrological and moisture predictors.
- Council boundary processing.
- Building exposure screening.

The next major research priority is validation using reliable independent flood reference data.
