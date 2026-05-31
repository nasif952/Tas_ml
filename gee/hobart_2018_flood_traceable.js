// =====================================================
// Hobart 2018 Flood Mapping - Traceable + Map-Safe GEE Script
// Fix: prevents browser/GEE app reload by limiting visible map layers,
// reducing heavy console prints, and making debug layers optional.
// =====================================================

// -----------------------------
// USER SETTINGS
// -----------------------------
var BEFORE_START = '2018-04-01';
var BEFORE_END = '2018-04-30';
var AFTER_START = '2018-05-10';
var AFTER_END = '2018-05-20';

var POLARIZATION = 'VV';
var ORBIT_PASS = 'DESCENDING';
var WATER_THRESHOLD_DB = -17;
var CHANGE_THRESHOLD_DB = -1.25;

// IMPORTANT:
// Keep this false while tracing. Turn true only if you want all intermediate layers.
// Too many active layers can make the GEE map reload/crash.
var SHOW_DEBUG_LAYERS = false;

// Optional validation/calibration asset.
var WATER_LEVEL_ASSET = 'projects/sturdy-apricot-405823/assets/TSFM_2018_05_WaterLevel_WGS84_GEE';

// -----------------------------
// HELPER FUNCTIONS
// -----------------------------
function addLayerSafe(image, vis, name, shown) {
  Map.addLayer(image, vis, name, shown);
}

function printStep(name, object) {
  print(name, object);
}

// -----------------------------
// STEP 1: Define Hobart ROI
// -----------------------------
var tasmania = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Australia'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Tasmania'));

var hobartPoint = ee.Geometry.Point([147.3272, -42.8821]);
var roi = hobartPoint.buffer(35000).bounds();

printStep('STEP 1 - ROI ready', roi);
Map.centerObject(roi, 10);
addLayerSafe(roi, {color: 'red'}, 'STEP 1 - Hobart ROI', true);

// -----------------------------
// STEP 2: Load Sentinel-1 collection
// -----------------------------
var s1Base = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', POLARIZATION))
  .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS))
  .select(POLARIZATION);

printStep('STEP 2 - Total Sentinel-1 image count in ROI', s1Base.size());

// -----------------------------
// STEP 3: Before-event images
// -----------------------------
var beforeCollection = s1Base.filterDate(BEFORE_START, BEFORE_END);
printStep('STEP 3 - Before-event image count', beforeCollection.size());
printStep('STEP 3 - First before image', beforeCollection.first());

var before = beforeCollection.median().clip(roi);
printStep('STEP 3 - Before-event median image ready', before.bandNames());
addLayerSafe(before, {min: -25, max: 0}, 'STEP 3 - Before Sentinel-1 VV median', SHOW_DEBUG_LAYERS);

// -----------------------------
// STEP 4: After/flood-event images
// -----------------------------
var afterCollection = s1Base.filterDate(AFTER_START, AFTER_END);
printStep('STEP 4 - After-event image count', afterCollection.size());
printStep('STEP 4 - First after image', afterCollection.first());

var after = afterCollection.median().clip(roi);
printStep('STEP 4 - After-event median image ready', after.bandNames());
addLayerSafe(after, {min: -25, max: 0}, 'STEP 4 - After/Flood Sentinel-1 VV median', true);

// -----------------------------
// STEP 5: Speckle smoothing
// -----------------------------
var beforeSmooth = before.focal_mean({radius: 30, units: 'meters'});
var afterSmooth = after.focal_mean({radius: 30, units: 'meters'});

printStep('STEP 5 - Smoothed images ready', ee.List(['beforeSmooth', 'afterSmooth']));
addLayerSafe(beforeSmooth, {min: -25, max: 0}, 'STEP 5 - Smoothed before', SHOW_DEBUG_LAYERS);
addLayerSafe(afterSmooth, {min: -25, max: 0}, 'STEP 5 - Smoothed after', SHOW_DEBUG_LAYERS);

// -----------------------------
// STEP 6: Static threshold water/flood mask
// -----------------------------
var afterWaterMask = afterSmooth.lt(WATER_THRESHOLD_DB).rename('after_water_mask');
printStep('STEP 6 - Threshold water/flood mask ready', afterWaterMask.bandNames());
addLayerSafe(afterWaterMask.selfMask(), {palette: ['0000ff']}, 'STEP 6 - Threshold water/flood mask', SHOW_DEBUG_LAYERS);

// -----------------------------
// STEP 7: Change detection flood mask
// -----------------------------
var change = afterSmooth.subtract(beforeSmooth).rename('after_minus_before_db');
printStep('STEP 7 - Change image ready', change.bandNames());
addLayerSafe(change, {min: -5, max: 5}, 'STEP 7 - Change image after-before dB', SHOW_DEBUG_LAYERS);

var changeFloodMask = change.lt(CHANGE_THRESHOLD_DB).and(afterWaterMask).rename('change_flood_mask');
printStep('STEP 7 - Change-based flood mask ready', changeFloodMask.bandNames());
addLayerSafe(changeFloodMask.selfMask(), {palette: ['00ffff']}, 'STEP 7 - Change-based flood mask', SHOW_DEBUG_LAYERS);

// -----------------------------
// STEP 8: Clean noisy patches
// -----------------------------
var connectedPixels = changeFloodMask.connectedPixelCount(100, true);
var cleanedFloodMask = changeFloodMask.updateMask(connectedPixels.gte(8)).rename('cleaned_flood_mask');

printStep('STEP 8 - Cleaned flood mask ready', cleanedFloodMask.bandNames());
// Do NOT show connectedPixels by default. It is a heavy layer and can crash/reload the map.
addLayerSafe(connectedPixels, {min: 0, max: 100}, 'STEP 8 - Connected pixel count DEBUG', false);
addLayerSafe(cleanedFloodMask.selfMask(), {palette: ['ff0000']}, 'STEP 8 - Cleaned flood mask FINAL', true);

// -----------------------------
// STEP 9: Flood area calculation
// -----------------------------
var floodAreaImage = cleanedFloodMask.multiply(ee.Image.pixelArea()).rename('flood_area_m2');
var floodAreaStats = floodAreaImage.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: roi,
  scale: 30,
  maxPixels: 1e13,
  tileScale: 4
});

var floodAreaKm2 = ee.Number(floodAreaStats.get('flood_area_m2')).divide(1e6);
printStep('STEP 9 - Estimated flood area km2', floodAreaKm2);

// -----------------------------
// STEP 10: Load TSFM water-level validation asset
// -----------------------------
var waterLevel = ee.Image(WATER_LEVEL_ASSET).clip(roi);
printStep('STEP 10 - TSFM water-level asset ready', waterLevel.bandNames());
addLayerSafe(waterLevel, {min: 0, max: 5, palette: ['white', 'yellow', 'orange', 'red']}, 'STEP 10 - TSFM water level', false);

// -----------------------------
// STEP 11: Compare SAR flood mask with TSFM water-level areas
// -----------------------------
var validationMask = waterLevel.gt(0).rename('validation_water_level_gt_0');
printStep('STEP 11 - Validation mask ready', validationMask.bandNames());
addLayerSafe(validationMask.selfMask(), {palette: ['00ff00']}, 'STEP 11 - Validation mask', false);

var sarAndValidation = cleanedFloodMask.and(validationMask).rename('sar_and_validation');
var sarOnly = cleanedFloodMask.and(validationMask.not()).rename('sar_only');
var validationOnly = validationMask.and(cleanedFloodMask.not()).rename('validation_only');

printStep('STEP 11 - Match/SAR-only/Validation-only layers ready', ee.List([
  'sarAndValidation',
  'sarOnly',
  'validationOnly'
]));

addLayerSafe(sarAndValidation.selfMask(), {palette: ['00ff00']}, 'STEP 11 - Match: SAR + validation', true);
addLayerSafe(sarOnly.selfMask(), {palette: ['ff0000']}, 'STEP 11 - SAR only', false);
addLayerSafe(validationOnly.selfMask(), {palette: ['ffff00']}, 'STEP 11 - Validation only', false);

// -----------------------------
// STEP 12: Accuracy-style pixel counts
// -----------------------------
// Use coarser scale + tileScale so it does not overload the browser/session.
var comparison = ee.Image.cat([
  cleanedFloodMask.unmask(0).rename('sar'),
  validationMask.unmask(0).rename('validation')
]);

var counts = comparison.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: roi,
  scale: 30,
  maxPixels: 1e13,
  tileScale: 4
});

printStep('STEP 12 - Pixel count histograms at 30m scale', counts);

// -----------------------------
// STEP 13: Export cleaned flood mask
// -----------------------------
Export.image.toDrive({
  image: cleanedFloodMask.uint8(),
  description: 'Hobart_2018_SAR_Flood_Mask_Traceable_MapSafe',
  folder: 'GEE_Hobart_Flood',
  fileNamePrefix: 'hobart_2018_sar_flood_mask_traceable_map_safe',
  region: roi,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

printStep('STEP 13 - Export task created', 'Hobart_2018_SAR_Flood_Mask_Traceable_MapSafe');
print('DONE - Map-safe version loaded. Open hidden layers one by one if needed.');
