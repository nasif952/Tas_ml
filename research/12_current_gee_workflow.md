# Current Google Earth Engine Workflow

This document describes the current operational GEE workflow for the Hobart 2018 Flood project.

## 1. Study area

The current workflow uses the uploaded GEE table asset:

```javascript
var studyAreaFC = ee.FeatureCollection(
  'projects/sturdy-apricot-405823/assets/Study_Area'
);
var studyArea = studyAreaFC.geometry();
```

This replaces the earlier rectangular AOI:

```javascript
var studyArea = ee.Geometry.Rectangle([146.85, -43.10, 147.55, -42.65]);
```

Using the polygon asset is better because it avoids processing unnecessary land outside the intended study area.

## 2. Flood label creation

The flood map is created from Sentinel-1 VV backscatter using a water-like threshold:

```javascript
-20 dB <= VV <= -15 dB
```

Then pre-existing water is removed using Sentinel-2 NDWI from before the flood.

Conceptually:

```text
Flood label = SAR water-like pixels minus pre-flood NDWI water
```

This creates a potential flood extent layer, not a ground-truth flood layer.

## 3. Existing water mask

Sentinel-2 NDWI is calculated using:

```text
NDWI = (Green - NIR) / (Green + NIR)
```

In the script:

```javascript
var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var existingWater = ndwi.gt(0).rename('Existing_Water');
```

This is used to remove permanent or pre-existing water from the SAR flood classification.

## 4. Persistence / stuck-water layer

The workflow uses two Sentinel-1 dates:

- Stage 1: 2018-05-15 to 2018-05-16
- Stage 2: 2018-05-27 to 2018-05-28

The persistence layer identifies areas that were flood-like in Stage 1 and still water-like in Stage 2 with limited VV change.

Concept:

```text
Stuck water = Stage 1 flood pixel + Stage 2 still water-like + low VV change
```

## 5. Random Forest predictors

The current predictor list is:

```javascript
var predictorNames = [
  'elevation',
  'slope',
  'TWI_simple',
  'NDVI',
  'NDBI',
  'precipitation',
  'event_rainfall',
  'temperature',
  'river_distance_1km',
  'soil_moisture'
];
```

## 6. Predictor meaning

| Predictor | Meaning |
|---|---|
| elevation | Height above sea level from DEM |
| slope | Terrain slope |
| TWI_simple | Simplified wetness index from slope |
| NDVI | Vegetation condition |
| NDBI | Built-up / impervious surface indicator |
| precipitation | Monthly rainfall accumulation |
| event_rainfall | Rainfall during event window |
| temperature | Mean land surface temperature |
| river_distance_1km | Distance to nearest river, capped at 1 km |
| soil_moisture | Surface soil moisture from SMAP |

## 7. Model structure

The script uses Random Forest with:

```javascript
numberOfTrees: 300
seed: 42
```

Samples are created using stratified sampling:

```javascript
classPoints: [600, 600]
```

The train/test split is:

- 70% training
- 30% testing

## 8. Model outputs

The GEE workflow currently outputs:

- NDWI-masked SAR flood map.
- Flood persistence map.
- River distance map.
- River buffer map.
- Soil moisture map.
- Random Forest flood susceptibility map.
- Final overlay map.
- Variable importance CSV.
- Selected building centroids inside high final-overlay zone.

## 9. Final overlay

The final overlay currently uses:

```text
Final overlay = 70% RF susceptibility + 30% stuck-water layer
```

This is useful for visual risk screening, but the weighting is currently heuristic and should be justified or tested later.

## 10. Building exposure screening

Buildings are loaded from:

```javascript
projects/sturdy-apricot-405823/assets/Greater_Hobart_Buildings_WGS84
```

The script converts building polygons to centroids and samples the high final-overlay layer. Buildings are selected if the centroid falls inside the high-risk/final-overlay zone.

This gives an approximate exposure screening result. It does not prove that each selected building was flooded.
