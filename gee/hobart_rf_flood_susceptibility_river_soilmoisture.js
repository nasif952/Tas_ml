// =====================================================
// UPDATED FINAL GEE CODE
// S2 NDWI-Masked Flood + Improved Stuck Water
// + RF Susceptibility + River Distance + Soil Moisture
// + Final Overlay + Accuracy + Variable Importance Chart
// Hobart 2018 Flood
// =====================================================


// -----------------------------
// 1. AOI - Study Area Polygon Asset
// -----------------------------

// Your uploaded GEE study area polygon
var studyAreaFC = ee.FeatureCollection(
  'projects/sturdy-apricot-405823/assets/Study_Area'
);

// Convert FeatureCollection to geometry for clipping, filtering, sampling, and export
var studyArea = studyAreaFC.geometry();

var scale = 100;

Map.centerObject(studyAreaFC, 10);

Map.addLayer(studyAreaFC.style({
  color: 'red',
  fillColor: '00000000',
  width: 2
}), {}, 'Study Area Boundary');


// -----------------------------
// 2. Helper functions
// -----------------------------

function getS1VV(start, end) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(studyArea)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select('VV');
}

function maskS2Clouds(img) {
  var qa = img.select('QA60');
  var cloud = 1 << 10;
  var cirrus = 1 << 11;

  var mask = qa.bitwiseAnd(cloud).eq(0)
    .and(qa.bitwiseAnd(cirrus).eq(0));

  return img.updateMask(mask).divide(10000);
}

function areaKm2(img, band, name) {
  var area = img.selfMask()
    .multiply(ee.Image.pixelArea())
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: studyArea,
      scale: scale,
      maxPixels: 1e13
    });

  print(name, ee.Number(area.get(band)).divide(1e6));
}


// -----------------------------
// 3. Sentinel-1 VV images
// -----------------------------

var stage1VV = getS1VV('2018-05-15', '2018-05-16')
  .median()
  .clip(studyArea)
  .rename('VV');

var stage2VV = getS1VV('2018-05-27', '2018-05-28')
  .median()
  .clip(studyArea)
  .rename('Stage2_VV');

print('Stage 1 VV image count:', getS1VV('2018-05-15', '2018-05-16').size());
print('Stage 2 VV image count:', getS1VV('2018-05-27', '2018-05-28').size());


// -----------------------------
// 4. Sentinel-2 NDWI existing-water mask
// -----------------------------

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyArea)
  .filterDate('2018-03-01', '2018-05-09')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2Clouds)
  .median()
  .clip(studyArea);

// NDWI = (Green - NIR) / (Green + NIR)
var ndwi = s2.normalizedDifference(['B3', 'B8'])
  .rename('NDWI');

// Existing/pre-flood water
var existingWater = ndwi.gt(0)
  .rename('Existing_Water');


// -----------------------------
// 5. MODIS NDVI / NDBI predictors
// -----------------------------

var modis = ee.ImageCollection('MODIS/061/MOD09A1')
  .filterBounds(studyArea)
  .filterDate('2018-04-01', '2018-05-09')
  .median()
  .clip(studyArea)
  .multiply(0.0001);

var ndvi = modis.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
  .rename('NDVI');

var ndbi = modis.normalizedDifference(['sur_refl_b06', 'sur_refl_b02'])
  .rename('NDBI');


// -----------------------------
// 6. NDWI-masked flood map
// -----------------------------

// Flood = SAR water-like pixels minus Sentinel-2 NDWI existing water
// SAR water-like rule: -20 dB <= VV <= -15 dB

var floodMap = stage1VV
  .lte(-15)
  .and(stage1VV.gte(-20))
  .and(existingWater.not())
  .rename('Flood_Map');


// -----------------------------
// 7. Improved stuck-water / persistence map
// -----------------------------

// VV change between Stage 2 and Stage 1
var vvChange = stage2VV
  .subtract(stage1VV)
  .rename('Stage2_minus_Stage1');

// Improved stuck water:
// 1. Pixel was flood in Stage 1
// 2. Pixel is still water-like in Stage 2
// 3. VV did not change much between stages: abs(change) <= 2 dB

var stuckWater = floodMap
  .and(stage2VV.lte(-15))
  .and(stage2VV.gte(-20))
  .and(vvChange.abs().lte(2))
  .rename('Stuck_Water');

// Persistence:
// 1 = flooded in Stage 1 but recovered by Stage 2
// 2 = still water-like / slow recovery

var persistence = floodMap
  .add(stuckWater)
  .rename('Flood_Persistence');


// -----------------------------
// 8. RF predictor layers
// -----------------------------

var dem = ee.Image('USGS/SRTMGL1_003')
  .clip(studyArea)
  .rename('elevation');

var slope = ee.Terrain.slope(dem)
  .rename('slope');

var twi = ee.Image(1)
  .divide(slope.multiply(Math.PI).divide(180).tan().add(0.001))
  .log()
  .rename('TWI_simple');

var precipitation = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(studyArea)
  .filterDate('2018-05-01', '2018-06-01')
  .sum()
  .clip(studyArea)
  .rename('precipitation');

var eventRainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(studyArea)
  .filterDate('2018-05-10', '2018-05-13')
  .sum()
  .clip(studyArea)
  .rename('event_rainfall');

var temperature = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterBounds(studyArea)
  .filterDate('2018-05-01', '2018-06-01')
  .select('LST_Day_1km')
  .mean()
  .multiply(0.02)
  .subtract(273.15)
  .clip(studyArea)
  .rename('temperature');


// -----------------------------
// 8B. River distance and soil moisture predictors
// -----------------------------

// River distance factor:
// Uses HydroSHEDS Free Flowing Rivers.
// Distance is capped at 1000 m so the RF focuses on the 1 km river influence zone.
// Value meaning:
// 0 = river/very near river
// 1000 = 1 km or more from river

var rivers = ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
  .filterBounds(studyArea);

var riverRaster = ee.Image()
  .byte()
  .paint({
    featureCollection: rivers,
    color: 1
  })
  .clip(studyArea);

var riverDistance = riverRaster
  .fastDistanceTransform(1024)
  .sqrt()
  .multiply(ee.Image.pixelArea().sqrt())
  .clip(studyArea)
  .rename('river_distance_m');

var riverDistance1km = riverDistance
  .min(1000)
  .rename('river_distance_1km');

// Optional binary 1 km river buffer layer for display/interpretation.
// This is not used in the RF model because continuous distance is better.
var riverBuffer1km = riverDistance
  .lte(1000)
  .rename('river_buffer_1km');

// Soil moisture factor:
// SMAP surface soil moisture for May 2018.
// Higher soil moisture can indicate wetter antecedent ground condition.

var soilMoisture = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
  .filterBounds(studyArea)
  .filterDate('2018-05-01', '2018-06-01')
  .select('sm_surface')
  .mean()
  .clip(studyArea)
  .rename('soil_moisture');

print('River features in study area:', rivers.size());
print('Soil moisture image count:', ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
  .filterBounds(studyArea)
  .filterDate('2018-05-01', '2018-06-01')
  .select('sm_surface')
  .size());


// -----------------------------
// 9. Predictor stack
// -----------------------------

// Do not use VV or NDWI as RF predictors.
// VV and NDWI already created the flood label.
// New factors added:
// 1. river_distance_1km
// 2. soil_moisture

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

var predictors = dem
  .addBands(slope)
  .addBands(twi)
  .addBands(ndvi)
  .addBands(ndbi)
  .addBands(precipitation)
  .addBands(eventRainfall)
  .addBands(temperature)
  .addBands(riverDistance1km)
  .addBands(soilMoisture)
  .unmask(-9999);

var label = floodMap
  .unmask(0)
  .toInt()
  .rename('label');

var stack = predictors.addBands(label);

print('Predictor names:', predictorNames);


// -----------------------------
// 10. Samples and train/test split
// -----------------------------

var samples = stack.stratifiedSample({
  numPoints: 0,
  classBand: 'label',
  classValues: [0, 1],
  classPoints: [2000, 2000],
  region: studyArea,
  scale: scale,
  seed: 42,
  geometries: true,
  dropNulls: false
});

var randomSamples = samples.randomColumn('random', 42);

var trainingSamples = randomSamples.filter(ee.Filter.lt('random', 0.7));
var testingSamples = randomSamples.filter(ee.Filter.gte('random', 0.7));

print('Total samples:', samples.size());
print('Training samples:', trainingSamples.size());
print('Testing samples:', testingSamples.size());
print('Class count:', samples.aggregate_histogram('label'));


// -----------------------------
// 11. RF classifier accuracy
// -----------------------------

var rfClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 300,
  seed: 42
}).train({
  features: trainingSamples,
  classProperty: 'label',
  inputProperties: predictorNames
});

var tested = testingSamples.classify(rfClassifier);
var cm = tested.errorMatrix('label', 'classification');

print('Confusion Matrix:', cm);
print('Overall Accuracy:', cm.accuracy());
print('Kappa:', cm.kappa());
print('Producer Accuracy:', cm.producersAccuracy());
print('Consumer Accuracy:', cm.consumersAccuracy());


// -----------------------------
// 12. Variable importance chart
// -----------------------------

var importance = ee.Dictionary(rfClassifier.explain().get('importance'));

var importanceTable = ee.FeatureCollection(
  importance.keys().map(function(key) {
    return ee.Feature(null, {
      variable: key,
      importance: importance.get(key)
    });
  })
);

print('Variable Importance Table:', importanceTable);

var importanceChart = ui.Chart.feature.byFeature({
  features: importanceTable,
  xProperty: 'variable',
  yProperties: ['importance']
})
.setChartType('ColumnChart')
.setOptions({
  title: 'Random Forest Variable Importance with River Distance and Soil Moisture',
  hAxis: {title: 'Variables'},
  vAxis: {title: 'Importance'},
  legend: {position: 'none'}
});

print(importanceChart);


// -----------------------------
// 13. RF probability model for susceptibility
// -----------------------------

var rfProbability = ee.Classifier.smileRandomForest({
  numberOfTrees: 300,
  seed: 42
})
.setOutputMode('PROBABILITY')
.train({
  features: trainingSamples,
  classProperty: 'label',
  inputProperties: predictorNames
});

var susceptibility = predictors
  .select(predictorNames)
  .classify(rfProbability)
  .rename('Flood_Susceptibility');


// -----------------------------
// 14. Final overlay
// -----------------------------

// Final overlay = 70% RF susceptibility + 30% improved stuck-water layer

var finalOverlay = susceptibility
  .multiply(0.7)
  .add(stuckWater.multiply(0.3))
  .rename('Final_Overlay');


// -----------------------------
// 15. Display required layers
// -----------------------------

Map.addLayer(existingWater.selfMask(), {
  palette: ['darkblue']
}, 'Existing Water Mask from S2 NDWI');

Map.addLayer(floodMap.selfMask(), {
  palette: ['cyan']
}, '1. NDWI-Masked Flood Map');

Map.addLayer(vvChange, {
  min: -5,
  max: 5,
  palette: ['blue', 'white', 'red']
}, 'VV Change: Stage 2 - Stage 1');

Map.addLayer(persistence.selfMask(), {
  min: 1,
  max: 2,
  palette: ['lightblue', 'red']
}, '2. Flood Persistence Map');

Map.addLayer(riverDistance1km, {
  min: 0,
  max: 1000,
  palette: ['blue', 'cyan', 'yellow', 'red']
}, 'River Distance capped at 1 km');

Map.addLayer(riverBuffer1km.selfMask(), {
  palette: ['purple']
}, 'River Buffer 1 km');

Map.addLayer(soilMoisture, {
  min: 0,
  max: 0.5,
  palette: ['brown', 'yellow', 'green', 'blue']
}, 'Soil Moisture May 2018');

Map.addLayer(susceptibility, {
  min: 0,
  max: 1,
  palette: ['green', 'yellow', 'red']
}, '3. Random Forest Flood Susceptibility Map');

Map.addLayer(finalOverlay, {
  min: 0,
  max: 1,
  palette: ['green', 'yellow', 'orange', 'red']
}, '4. Final Overlay Map');


// -----------------------------
// 16. Area calculation
// -----------------------------

areaKm2(floodMap, 'Flood_Map', 'Flood area km²:');
areaKm2(stuckWater, 'Stuck_Water', 'Improved stuck-water area km²:');
areaKm2(riverBuffer1km, 'river_buffer_1km', 'Area within 1 km of rivers km²:');


// -----------------------------
// 17. Exports
// -----------------------------

Export.image.toDrive({
  image: floodMap,
  description: 'Hobart_S2_NDWI_Masked_Flood_Map',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_s2_ndwi_masked_flood_map',
  region: studyArea,
  scale: scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: persistence,
  description: 'Hobart_Improved_Flood_Persistence_Map',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_improved_flood_persistence_map',
  region: studyArea,
  scale: scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: riverDistance1km,
  description: 'Hobart_River_Distance_1km',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_river_distance_1km',
  region: studyArea,
  scale: scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: riverBuffer1km,
  description: 'Hobart_River_Buffer_1km',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_river_buffer_1km',
  region: studyArea,
  scale: scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: soilMoisture,
  description: 'Hobart_Soil_Moisture_May_2018',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_soil_moisture_may_2018',
  region: studyArea,
  scale: scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: susceptibility,
  description: 'Hobart_RF_Flood_Susceptibility_Map_With_River_SoilMoisture',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_rf_flood_susceptibility_map_with_river_soilmoisture',
  region: studyArea,
  scale: scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: finalOverlay,
  description: 'Hobart_Final_Overlay_Map_With_River_SoilMoisture',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_final_overlay_map_with_river_soilmoisture',
  region: studyArea,
  scale: scale,
  maxPixels: 1e13
});

Export.table.toDrive({
  collection: importanceTable,
  description: 'Hobart_RF_Variable_Importance_With_River_SoilMoisture',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'hobart_rf_variable_importance_with_river_soilmoisture',
  fileFormat: 'CSV'
});


// =====================================================
// SELECT BUILDINGS INSIDE HIGH FINAL OVERLAY ZONE
// Use this in the SAME script where finalOverlay exists
// =====================================================

// 1. Load buildings
var buildings = ee.FeatureCollection(
  'projects/sturdy-apricot-405823/assets/Greater_Hobart_Buildings_WGS84'
).filterBounds(studyArea);

print('Total buildings in study area:', buildings.size());


// 2. Check final overlay value range
var overlayStats = finalOverlay.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea,
  scale: scale,
  maxPixels: 1e13
});

print('Final overlay min/max:', overlayStats);


// 3. Create high final overlay zone
// Try 0.66 first. If no buildings appear, test 0.5.
var highFinalOverlay = finalOverlay
  .gte(0.66)
  .unmask(0)
  .toInt()
  .rename('high_final_overlay');


// 4. Convert building polygons to centroid points
var buildingCentroids = buildings.map(function(feature) {
  return feature.setGeometry(feature.geometry().centroid(1));
});


// 5. Extract final overlay zone value at building centroids
var sampledBuildings = highFinalOverlay.sampleRegions({
  collection: buildingCentroids,
  properties: buildings.first().propertyNames(),
  scale: scale,
  geometries: true
});


// 6. Select buildings where centroid falls inside high final overlay zone
var selectedCentroids = sampledBuildings.filter(
  ee.Filter.eq('high_final_overlay', 1)
);

print('Buildings in high final overlay zone:', selectedCentroids.size());


// 7. Display building layers
Map.addLayer(highFinalOverlay.selfMask(), {
  palette: ['purple']
}, 'High Final Overlay Zone');

Map.addLayer(buildings.style({
  color: 'black',
  fillColor: '00000000',
  width: 1
}), {}, 'All Buildings');

Map.addLayer(selectedCentroids.style({
  color: 'yellow',
  pointSize: 3,
  width: 1
}), {}, 'Selected Building Centroids in Final Overlay Zone');


// 8. Export selected building centroids
Export.table.toDrive({
  collection: selectedCentroids,
  description: 'Selected_Buildings_High_Final_Overlay_Centroids_With_River_SoilMoisture',
  folder: 'GEE_Hobart_Flood_Project',
  fileNamePrefix: 'selected_buildings_high_final_overlay_centroids_with_river_soilmoisture',
  fileFormat: 'CSV'
});
