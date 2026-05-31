// =====================================================
// Hobart 2018 Flood Mapping - Traceable GEE Script
// Purpose: show print() and Map.addLayer() output after every major step
// =====================================================

// -----------------------------
// USER SETTINGS
// -----------------------------
var START_DATE = '2018-05-01';
var END_DATE = '2018-05-31';
var BEFORE_START = '2018-04-01';
var BEFORE_END = '2018-04-30';
var AFTER_START = '2018-05-10';
var AFTER_END = '2018-05-20';

var POLARIZATION = 'VV';
var ORBIT_PASS = 'DESCENDING';
var WATER_THRESHOLD_DB = -17;
var CHANGE_THRESHOLD_DB = -1.25;

// Optional validation/calibration asset.
// If this asset does not exist or is not accessible, comment this section out.
var WATER_LEVEL_ASSET = 'projects/sturdy-apricot-405823/assets/TSFM_2018_05_WaterLevel_WGS84_GEE';

// -----------------------------
// STEP 1: Define Hobart ROI
// -----------------------------
var tasmania = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Australia'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Tasmania'));

var hobartPoint = ee.Geometry.Point([147.3272, -42.8821]);
var roi = hobartPoint.buffer(35000).bounds();

print('STEP 1 - Tasmania boundary:', tasmania);
print('STEP 1 - Hobart ROI:', roi);
Map.centerObject(roi, 10);
Map.addLayer(tasmania, {color: 'gray'}, 'STEP 1 - Tasmania boundary');
Map.addLayer(roi, {color: 'red'}, 'STEP 1 - Hobart ROI');

// -----------------------------
// STEP 2: Load Sentinel-1 collection
// -----------------------------
var s1Base = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', POLARIZATION))
  .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS))
  .select(POLARIZATION);

print('STEP 2 - Sentinel-1 base collection:', s1Base);
print('STEP 2 - Total Sentinel-1 image count in ROI:', s1Base.size());

// -----------------------------
// STEP 3: Filter before-event images
// -----------------------------
var beforeCollection = s1Base.filterDate(BEFORE_START, BEFORE_END);
print('STEP 3 - Before-event collection:', beforeCollection);
print('STEP 3 - Before-event image count:', beforeCollection.size());
print('STEP 3 - Before-event image dates:', beforeCollection.aggregate_array('system:time_start'));

var before = beforeCollection.median().clip(roi);
print('STEP 3 - Before-event median image:', before);
Map.addLayer(before, {min: -25, max: 0}, 'STEP 3 - Before Sentinel-1 VV median');

// -----------------------------
// STEP 4: Filter flood/after-event images
// -----------------------------
var afterCollection = s1Base.filterDate(AFTER_START, AFTER_END);
print('STEP 4 - After-event collection:', afterCollection);
print('STEP 4 - After-event image count:', afterCollection.size());
print('STEP 4 - After-event image dates:', afterCollection.aggregate_array('system:time_start'));

var after = afterCollection.median().clip(roi);
print('STEP 4 - After-event median image:', after);
Map.addLayer(after, {min: -25, max: 0}, 'STEP 4 - After/Flood Sentinel-1 VV median');

// -----------------------------
// STEP 5: Speckle smoothing
// -----------------------------
var beforeSmooth = before.focal_mean({radius: 30, units: 'meters'});
var afterSmooth = after.focal_mean({radius: 30, units: 'meters'});

print('STEP 5 - Smoothed before image:', beforeSmooth);
print('STEP 5 - Smoothed after image:', afterSmooth);
Map.addLayer(beforeSmooth, {min: -25, max: 0}, 'STEP 5 - Smoothed before');
Map.addLayer(afterSmooth, {min: -25, max: 0}, 'STEP 5 - Smoothed after');

// -----------------------------
// STEP 6: Static water/flood threshold mask
// -----------------------------
var afterWaterMask = afterSmooth.lt(WATER_THRESHOLD_DB).rename('after_water_mask');
print('STEP 6 - After water/flood mask using threshold:', afterWaterMask);
Map.addLayer(afterWaterMask.selfMask(), {palette: ['0000ff']}, 'STEP 6 - Threshold water/flood mask');

// -----------------------------
// STEP 7: Change detection flood mask
// More negative backscatter after flood can indicate new water.
// -----------------------------
var change = afterSmooth.subtract(beforeSmooth).rename('after_minus_before_db');
print('STEP 7 - SAR change image after minus before:', change);
Map.addLayer(change, {min: -5, max: 5}, 'STEP 7 - Change image after-before dB');

var changeFloodMask = change.lt(CHANGE_THRESHOLD_DB).and(afterWaterMask).rename('change_flood_mask');
print('STEP 7 - Change-based flood mask:', changeFloodMask);
Map.addLayer(changeFloodMask.selfMask(), {palette: ['00ffff']}, 'STEP 7 - Change-based flood mask');

// -----------------------------
// STEP 8: Remove very small noisy patches
// -----------------------------
var connectedPixels = changeFloodMask.connectedPixelCount(100, true);
var cleanedFloodMask = changeFloodMask.updateMask(connectedPixels.gte(8)).rename('cleaned_flood_mask');

print('STEP 8 - Connected pixel count:', connectedPixels);
print('STEP 8 - Cleaned flood mask:', cleanedFloodMask);
Map.addLayer(connectedPixels, {min: 0, max: 100}, 'STEP 8 - Connected pixel count');
Map.addLayer(cleanedFloodMask.selfMask(), {palette: ['ff0000']}, 'STEP 8 - Cleaned flood mask');

// -----------------------------
// STEP 9: Calculate flood area
// -----------------------------
var floodAreaImage = cleanedFloodMask.multiply(ee.Image.pixelArea()).rename('flood_area_m2');
var floodAreaStats = floodAreaImage.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: roi,
  scale: 10,
  maxPixels: 1e13
});

var floodAreaKm2 = ee.Number(floodAreaStats.get('flood_area_m2')).divide(1e6);
print('STEP 9 - Flood area stats m2:', floodAreaStats);
print('STEP 9 - Estimated flood area km2:', floodAreaKm2);

// -----------------------------
// STEP 10: Load optional TSFM water-level validation asset
// -----------------------------
var waterLevel = ee.Image(WATER_LEVEL_ASSET).clip(roi);
print('STEP 10 - TSFM water-level asset:', waterLevel);
Map.addLayer(waterLevel, {min: 0, max: 5, palette: ['white', 'yellow', 'orange', 'red']}, 'STEP 10 - TSFM water level');

// -----------------------------
// STEP 11: Compare SAR flood mask with TSFM water-level areas
// Any positive water level is treated as validation flood/water presence.
// -----------------------------
var validationMask = waterLevel.gt(0).rename('validation_water_level_gt_0');
print('STEP 11 - Validation mask water level greater than 0:', validationMask);
Map.addLayer(validationMask.selfMask(), {palette: ['00ff00']}, 'STEP 11 - Validation mask');

var sarAndValidation = cleanedFloodMask.and(validationMask).rename('sar_and_validation');
var sarOnly = cleanedFloodMask.and(validationMask.not()).rename('sar_only');
var validationOnly = validationMask.and(cleanedFloodMask.not()).rename('validation_only');

print('STEP 11 - SAR and validation overlap:', sarAndValidation);
print('STEP 11 - SAR only:', sarOnly);
print('STEP 11 - Validation only:', validationOnly);

Map.addLayer(sarAndValidation.selfMask(), {palette: ['00ff00']}, 'STEP 11 - Match: SAR + validation');
Map.addLayer(sarOnly.selfMask(), {palette: ['ff0000']}, 'STEP 11 - SAR only');
Map.addLayer(validationOnly.selfMask(), {palette: ['ffff00']}, 'STEP 11 - Validation only');

// -----------------------------
// STEP 12: Accuracy-style pixel counts
// -----------------------------
var comparison = ee.Image.cat([
  cleanedFloodMask.unmask(0).rename('sar'),
  validationMask.unmask(0).rename('validation')
]);

var counts = comparison.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: roi,
  scale: 10,
  maxPixels: 1e13
});

print('STEP 12 - Comparison bands:', comparison);
print('STEP 12 - Pixel count histograms:', counts);

// -----------------------------
// STEP 13: Export cleaned flood mask
// -----------------------------
Export.image.toDrive({
  image: cleanedFloodMask.uint8(),
  description: 'Hobart_2018_SAR_Flood_Mask_Traceable',
  folder: 'GEE_Hobart_Flood',
  fileNamePrefix: 'hobart_2018_sar_flood_mask_traceable',
  region: roi,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

print('STEP 13 - Export task created: Hobart_2018_SAR_Flood_Mask_Traceable');
print('DONE - Check Console for each step output and Layers panel for every display layer.');
