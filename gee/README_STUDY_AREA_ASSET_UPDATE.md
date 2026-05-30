# Study Area Asset Update

Use this Earth Engine table asset for the study area:

```text
projects/gee-project-493107/assets/Study_Area
```

Replace the old rectangle AOI block:

```javascript
var studyArea = ee.Geometry.Rectangle([146.85, -43.10, 147.55, -42.65]);
var scale = 100;
Map.centerObject(studyArea, 10);
```

with this polygon asset AOI block:

```javascript
// -----------------------------
// 1. AOI - Study Area Polygon Asset
// -----------------------------

var studyAreaFC = ee.FeatureCollection(
  'projects/gee-project-493107/assets/Study_Area'
);

var studyArea = studyAreaFC.geometry();
var scale = 100;

Map.centerObject(studyAreaFC, 10);

Map.addLayer(studyAreaFC.style({
  color: 'red',
  fillColor: '00000000',
  width: 2
}), {}, 'Study Area Boundary');
```

For the Streamlit app, the study area asset input/default should also be:

```text
projects/gee-project-493107/assets/Study_Area
```

This should fix the previous permission/path mismatch caused by using:

```text
projects/sturdy-apricot-405823/assets/Study_Area
```
