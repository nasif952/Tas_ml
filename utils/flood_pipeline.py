from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FloodConfig:
    xmin: float = 146.85
    ymin: float = -43.10
    xmax: float = 147.55
    ymax: float = -42.65
    scale: int = 100
    seed: int = 42
    s1_stage1_start: str = "2018-05-15"
    s1_stage1_end: str = "2018-05-16"
    s1_stage2_start: str = "2018-05-27"
    s1_stage2_end: str = "2018-05-28"
    s2_water_start: str = "2018-03-01"
    s2_water_end: str = "2018-05-09"
    s2_cloud_pct: int = 40
    predictor_start: str = "2018-04-01"
    predictor_end: str = "2018-05-09"
    month_start: str = "2018-05-01"
    month_end: str = "2018-06-01"
    event_rain_start: str = "2018-05-10"
    event_rain_end: str = "2018-05-13"
    soil_start: str = "2018-05-01"
    soil_end: str = "2018-06-01"
    vv_min: float = -20.0
    vv_max: float = -15.0
    ndwi_threshold: float = 0.0
    stuck_change_tolerance: float = 2.0
    river_asset: str = "WWF/HydroSHEDS/v1/FreeFlowingRivers"
    river_distance_cap_m: int = 1000
    use_river_distance: bool = True
    use_soil_moisture: bool = True
    class_points: int = 600
    train_fraction: float = 0.70
    rf_trees: int = 300
    susceptibility_weight: float = 0.70
    high_overlay_threshold: float = 0.66


def study_area(config: FloodConfig):
    import ee
    return ee.Geometry.Rectangle([config.xmin, config.ymin, config.xmax, config.ymax])


def get_s1_vv(region, start: str, end: str, orbit: Optional[str] = None):
    import ee
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    if orbit and orbit.lower() != "either":
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit.upper()))
    return col.median().clip(region).rename("VV"), col


def mask_s2_clouds(img):
    import ee
    qa = img.select("QA60")
    cloud = 1 << 10
    cirrus = 1 << 11
    mask = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
    return img.updateMask(mask).divide(10000)


def get_s2_existing_water(region, config: FloodConfig):
    import ee
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(config.s2_water_start, config.s2_water_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.s2_cloud_pct))
        .map(mask_s2_clouds)
        .median()
        .clip(region)
    )
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    existing_water = ndwi.gt(config.ndwi_threshold).rename("Existing_Water")
    return ndwi, existing_water


def build_flood_map(stage1_vv, existing_water, config: FloodConfig):
    return (
        stage1_vv.lte(config.vv_max)
        .And(stage1_vv.gte(config.vv_min))
        .And(existing_water.Not())
        .rename("Flood_Map")
    )


def build_persistence(stage1_vv, stage2_vv, flood_map, config: FloodConfig):
    vv_change = stage2_vv.subtract(stage1_vv).rename("Stage2_minus_Stage1")
    stuck_water = (
        flood_map
        .And(stage2_vv.lte(config.vv_max))
        .And(stage2_vv.gte(config.vv_min))
        .And(vv_change.abs().lte(config.stuck_change_tolerance))
        .rename("Stuck_Water")
    )
    persistence = flood_map.add(stuck_water).rename("Flood_Persistence")
    return vv_change, stuck_water, persistence


def build_predictors(region, config: FloodConfig):
    import ee
    dem = ee.Image("USGS/SRTMGL1_003").clip(region).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    twi = ee.Image(1).divide(slope.multiply(3.141592653589793).divide(180).tan().add(0.001)).log().rename("TWI_simple")

    modis = (
        ee.ImageCollection("MODIS/061/MOD09A1")
        .filterBounds(region)
        .filterDate(config.predictor_start, config.predictor_end)
        .median()
        .clip(region)
        .multiply(0.0001)
    )
    ndvi = modis.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI")
    ndbi = modis.normalizedDifference(["sur_refl_b06", "sur_refl_b02"]).rename("NDBI")

    precipitation = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(region)
        .filterDate(config.month_start, config.month_end)
        .sum()
        .clip(region)
        .rename("precipitation")
    )
    event_rainfall = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(region)
        .filterDate(config.event_rain_start, config.event_rain_end)
        .sum()
        .clip(region)
        .rename("event_rainfall")
    )
    temperature = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(region)
        .filterDate(config.month_start, config.month_end)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .clip(region)
        .rename("temperature")
    )

    predictor_names: List[str] = [
        "elevation", "slope", "TWI_simple", "NDVI", "NDBI",
        "precipitation", "event_rainfall", "temperature"
    ]

    predictors = dem.addBands(slope).addBands(twi).addBands(ndvi).addBands(ndbi).addBands(precipitation).addBands(event_rainfall).addBands(temperature)

    extra_layers: Dict[str, object] = {}

    if config.use_river_distance:
        rivers = ee.FeatureCollection(config.river_asset).filterBounds(region)
        river_raster = ee.Image().byte().paint(featureCollection=rivers, color=1).clip(region)
        river_distance = river_raster.fastDistanceTransform(1024).sqrt().multiply(ee.Image.pixelArea().sqrt()).clip(region).rename("river_distance_m")
        river_distance_1km = river_distance.min(config.river_distance_cap_m).rename("river_distance_1km")
        river_buffer = river_distance.lte(config.river_distance_cap_m).rename("river_buffer")
        predictors = predictors.addBands(river_distance_1km)
        predictor_names.append("river_distance_1km")
        extra_layers["rivers"] = rivers
        extra_layers["river_distance"] = river_distance_1km
        extra_layers["river_buffer"] = river_buffer

    if config.use_soil_moisture:
        soil_moisture = (
            ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
            .filterBounds(region)
            .filterDate(config.soil_start, config.soil_end)
            .select("sm_surface")
            .mean()
            .clip(region)
            .rename("soil_moisture")
        )
        predictors = predictors.addBands(soil_moisture)
        predictor_names.append("soil_moisture")
        extra_layers["soil_moisture"] = soil_moisture

    return predictors.unmask(-9999), predictor_names, extra_layers


def train_rf_susceptibility(region, predictors, label, predictor_names: List[str], config: FloodConfig):
    import ee
    stack = predictors.addBands(label.unmask(0).toInt().rename("label"))
    samples = stack.stratifiedSample(
        numPoints=0,
        classBand="label",
        classValues=[0, 1],
        classPoints=[config.class_points, config.class_points],
        region=region,
        scale=config.scale,
        seed=config.seed,
        geometries=True,
        dropNulls=False,
    )
    random_samples = samples.randomColumn("random", config.seed)
    training = random_samples.filter(ee.Filter.lt("random", config.train_fraction))
    testing = random_samples.filter(ee.Filter.gte("random", config.train_fraction))

    rf = ee.Classifier.smileRandomForest(numberOfTrees=config.rf_trees, seed=config.seed).train(
        features=training,
        classProperty="label",
        inputProperties=predictor_names,
    )
    tested = testing.classify(rf)
    cm = tested.errorMatrix("label", "classification")

    importance = ee.Dictionary(rf.explain().get("importance"))
    importance_fc = ee.FeatureCollection(
        importance.keys().map(lambda key: ee.Feature(None, {"variable": key, "importance": importance.get(key)}))
    )

    rf_probability = ee.Classifier.smileRandomForest(numberOfTrees=config.rf_trees, seed=config.seed).setOutputMode("PROBABILITY").train(
        features=training,
        classProperty="label",
        inputProperties=predictor_names,
    )
    susceptibility = predictors.select(predictor_names).classify(rf_probability).rename("Flood_Susceptibility")

    return {
        "samples": samples,
        "training": training,
        "testing": testing,
        "classifier": rf,
        "confusion_matrix": cm,
        "importance": importance,
        "importance_fc": importance_fc,
        "susceptibility": susceptibility,
    }


def build_final_overlay(susceptibility, stuck_water, config: FloodConfig):
    stuck_weight = 1 - config.susceptibility_weight
    final_overlay = susceptibility.multiply(config.susceptibility_weight).add(stuck_water.multiply(stuck_weight)).rename("Final_Overlay")
    high_overlay = final_overlay.gte(config.high_overlay_threshold).unmask(0).toInt().rename("high_final_overlay")
    return final_overlay, high_overlay


def area_km2(image, band: str, region, scale: int):
    import ee
    area = image.selfMask().multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=scale, maxPixels=1e13
    )
    return ee.Number(area.get(band)).divide(1e6)
