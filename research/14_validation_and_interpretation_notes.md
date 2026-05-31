# Validation and Interpretation Notes

This document records the key interpretation rules for the current Hobart 2018 flood workflow.

## 1. Current flood map is not ground truth

The current flood map is generated from Sentinel-1 VV thresholding and Sentinel-2 NDWI masking.

It should be described as:

```text
NDWI-masked SAR-derived flood extent map
```

or:

```text
Potential flood inundation map after removing pre-existing water
```

Avoid calling it a fully verified flood boundary unless independent validation data confirms it.

## 2. Why NDWI masking is acceptable

NDWI masking is acceptable because SAR water-like thresholding can detect both floodwater and permanent/pre-existing water.

The NDWI mask helps remove water that existed before the flood, such as:

- rivers
- lakes
- coastal water
- reservoirs
- permanent water bodies

Conceptually:

```text
SAR water-like pixels - pre-flood NDWI water = potential new inundation
```

## 3. NDWI masking limitations

NDWI masking may introduce uncertainty because:

- Sentinel-2 is optical and affected by cloud.
- NDWI may classify wet surfaces as water.
- High river levels before the event may be removed even if hydrologically relevant.
- Sentinel-1 and Sentinel-2 have different sensing behaviour and spatial resolution.
- NDWI threshold choice affects the amount of water removed.

## 4. Random Forest accuracy limitation

The Random Forest model is trained using the SAR-derived flood map as the label.

This means the model is learning to reproduce or explain the SAR-derived label. It is not yet independently validated against observed flood truth.

Therefore, current accuracy metrics describe agreement with the SAR-derived label, not necessarily real-world flood accuracy.

## 5. NDBI importance interpretation

NDBI showing high importance likely means that built-up and impervious surfaces are strongly associated with the SAR-derived flood label.

Possible explanations:

1. Urban areas were genuinely affected by flooding.
2. Flood-prone drainage corridors overlap with built-up areas.
3. SAR behaviour in urban areas affects the flood-like classification.
4. The model is learning from a SAR-derived label, not independent ground truth.

Recommended wording:

> NDBI showed high importance, suggesting that built-up and impervious surfaces were strongly associated with the SAR-derived flood susceptibility pattern. However, this should be interpreted cautiously because the model was trained using a SAR-derived flood label rather than independent field-observed flood data.

Avoid saying:

```text
NDBI caused the flood.
```

Better wording:

```text
NDBI had the strongest association with the SAR-derived flood susceptibility output.
```

## 6. River distance interpretation

River distance is a hydrologically meaningful predictor because flood susceptibility often increases near rivers, streams, drainage corridors, and low-lying waterways.

Current variable:

```text
river_distance_1km
```

Meaning:

- 0 = on or very near a river.
- 1000 = 1 km or more from a river.

A capped distance variable is useful because it focuses the model on the local influence zone around rivers.

## 7. Soil moisture interpretation

Soil moisture can represent antecedent wetness. Higher antecedent moisture can reduce infiltration capacity and increase runoff.

However, SMAP soil moisture is coarse resolution. It should be interpreted as a broad environmental condition, not fine-scale local soil wetness.

## 8. Building exposure interpretation

The building analysis currently selects buildings whose centroid falls inside the high final-overlay zone.

This should be described as:

```text
building exposure screening
```

not:

```text
confirmed flooded buildings
```

Future improvement:

- Use polygon overlap rather than centroid sampling.
- Calculate percentage of each building footprint intersecting flood/high-risk zones.
- Combine with floor height, address data, or insurance exposure information if available.

## 9. Best validation hierarchy

Preferred validation sources, from strongest to weakest:

1. Field-observed flood extent polygons.
2. Official flood-depth or inundation model outputs verified by authorities.
3. Field-surveyed flood points.
4. High-resolution aerial imagery during or immediately after flood.
5. News/report geolocated flood observations.
6. Water-level or river-stage data as contextual evidence.
7. Manually interpreted points from uncertain imagery.

## 10. Validation goal

The next major research goal is to obtain or create an independent validation dataset so the project can report more defensible accuracy metrics.

Until then, all accuracy outputs should be treated as internal model assessment rather than final flood-map validation.
