# Next Research Steps

This document updates the project roadmap based on recent progress.

## 1. Immediate technical fixes

### 1.1 Replace rectangle AOI everywhere

The script should consistently use:

```javascript
var studyAreaFC = ee.FeatureCollection(
  'projects/sturdy-apricot-405823/assets/Study_Area'
);
var studyArea = studyAreaFC.geometry();
```

Check that all filtering, clipping, sampling, and exports use `studyArea`.

### 1.2 Confirm river dataset

The current script uses:

```text
WWF/HydroSHEDS/v1/FreeFlowingRivers
```

If this fails or does not represent Hobart streams properly, replace it with a local hydrography layer.

Recommended future river asset:

```text
hobart_rivers_wgs84
```

### 1.3 Check SMAP soil moisture availability

Confirm that the SMAP dataset returns images for May 2018 over the study area.

The script already prints image count. If count is zero, change dataset or date range.

## 2. Improve flood label

The current SAR flood label uses a fixed VV threshold:

```text
-20 dB <= VV <= -15 dB
```

Next improvements:

- Test several thresholds.
- Compare VV and VH.
- Test ratio or difference between pre-flood and flood images.
- Add terrain shadow/slope masks.
- Remove small isolated noisy pixels.
- Compare against visual basemaps.

## 3. Improve Random Forest design

Current model uses the SAR-derived flood map as label. This is acceptable for susceptibility prototyping but not final validation.

Next steps:

- Train model with independent flood points if available.
- Separate training and validation spatially.
- Avoid random split only if spatial autocorrelation is a concern.
- Test feature importance stability across multiple random seeds.
- Compare Random Forest with XGBoost or Gradient Boosted Trees.

## 4. Add stronger hydrological predictors

Recommended next predictors:

1. HAND: Height Above Nearest Drainage.
2. Flow accumulation.
3. Distance to coast.
4. Distance to stormwater/drainage network.
5. Land cover class.
6. Impervious surface percentage.
7. Building density.
8. Road density.

HAND is especially important because it often explains flood susceptibility better than simple elevation.

## 5. Improve building exposure analysis

Current method:

```text
building centroid inside high final-overlay zone
```

Next method:

```text
building polygon intersects high final-overlay zone
```

Better method:

```text
percentage of each building polygon covered by high final-overlay zone
```

Useful outputs:

- Selected building count.
- Building area exposed.
- Council-wise exposed building count.
- Exposure summary by final-overlay threshold.

## 6. Council-level reporting

Because separate council boundaries are now available, produce council-level flood/risk summaries.

Possible outputs:

- Flood area by council.
- High final-overlay area by council.
- Number of exposed building centroids by council.
- Mean susceptibility by council.
- Top council by exposure.

## 7. Validation priority

The strongest next research task is validation.

Search for:

- official Hobart 2018 flood extent
- flood depth layers
- flood marks
- council reports
- emergency management reports
- insurance or damage datasets
- aerial imagery after the event
- geolocated flood photographs

The project should not make strong accuracy claims until independent validation data is prepared.

## 8. Documentation priority

Update documentation after each major result:

- new dataset added
- GEE code changed
- model accuracy changed
- variable importance changed
- validation data found
- exported map produced
- conference abstract/paper drafted

## 9. Near-term deliverables

Recommended deliverables for the next milestone:

1. Final GEE script using `Study_Area` polygon.
2. Exported flood map GeoTIFF.
3. Exported susceptibility map GeoTIFF.
4. Variable importance CSV.
5. Council-level exposure table.
6. Building exposure CSV.
7. Validation plan with available reference sources.
8. Short results summary for conference use.

## 10. Long-term goal

The long-term goal remains to build a regularly updated flood-risk intelligence system for Tasmania. The Hobart 2018 event is the first case study and proof-of-concept.
