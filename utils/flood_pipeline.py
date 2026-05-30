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
    sar_change_threshold_db: float = -2.0
    river_asset: str = "WWF/HydroSHEDS/v1/FreeFlowingRivers"
    river_distance_cap_m: int = 1000
    use_river_distance: bool = True
    use_soil_moisture: bool = True
    class_points: int = 600
    train_fraction: float = 0.70
    rf_trees: int = 300
    susceptibility_weight: float = 0.70
    high_overlay_threshold: float = 0.66
    max_extra_days: int = 3


def study_area(config: FloodConfig):
    import ee
    return ee.Geometry.Rectangle([config.xmin, config.ymin, config.xmax, config.ymax])


def adaptive_collection(base_collection, base_start: str, base_end: str, max_extra_days: int = 3):
    """Try the requested date window, then extend the end date if empty."""
    import ee

    max_extra_days = max(0, int(max_extra_days))
    extensions = ee.List.sequence(0, max_extra_days)

    def candidate(extra):
        extra = ee.Number(extra)
        actual_end = ee.Date(base_end).advance(extra, "day")
        count = base_collection.filterDate(base_start, actual_end).size()
        return ee.Feature(None, {
            "extra_days": extra,
            "count": count,
            "actual_end": actual_end.format("YYYY-MM-dd"),
        })

    candidates = ee.FeatureCollection(extensions.map(candidate))
    valid = candidates.filter(ee.Filter.gt("count", 0)).sort("extra_days")
    found = valid.size().gt(0)
    best = ee.Feature(
        ee.Algorithms.If(
            found,
            valid.first(),
            candidates.sort("extra_days", False).first(),
        )
    )
    extra_days = ee.Number(best.get("extra_days"))
    actual_end_date = ee.Date(base_end).advance(extra_days, "day")
    collection = base_collection.filterDate(base_start, actual_end_date)

    return {
        "collection": collection,
        "count": collection.size(),
        "extra_days": extra_days,
        "actual_end": actual_end_date.format("YYYY-MM-dd"),
        "found": found,
    }


def _s1_vv_base_collection(region, orbit: Optional[str] = None):
    import ee
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    if orbit and orbit.lower() != "either":
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit.upper()))
    return col


def get_s1_vv(region, start: str, end: str, orbit: Optional[str] = None):
    col = _s1_vv_base_collection(region, orbit).filterDate(start, end)
    return col.median().clip(region).rename("VV"), col


def get_s1_vv_adaptive(region, start: str, end: str, orbit: Optional[str] = None, max_extra_days: int = 3):
    base_col = _s1_vv_base_collection(region, orbit)
    info = adaptive_collection(base_col, start, end, max_extra_days)
    col = info["collection"]
    return col.median().clip(region).rename("VV"), col, info


def get_s1_vv_event_change(
    region,
    pre_start: str,
    pre_end: str,
    target_start: str,
    target_end: str,
    orbit: Optional[str] = None,
    max_extra_days: int = 3,
):
    """Build SAR layers for flood change detection.

    Pre-event: closest available Sentinel-1 VV image inside/adapted pre window.
    During-event: mean Sentinel-1 VV composite inside/adapted target window.
    Change: during_mean_vv - pre_closest_vv. Flooding usually causes a negative
    dB drop over land because smooth water has lower SAR backscatter.
    """
    import ee

    base_col = _s1_vv_base_collection(region, orbit)

    pre_info = adaptive_collection(base_col, pre_start, pre_end, max_extra_days)
    pre_col = pre_info["collection"]
    pre_vv = (
        ee.Image(pre_col.sort("system:time_start", False).first())
        .clip(region)
        .rename("Pre_Closest_VV")
    )

    target_info = adaptive_collection(base_col, target_start, target_end, max_extra_days)
    target_col = target_info["collection"]
    during_vv = target_col.mean().clip(region).rename("During_Mean_VV")

    sar_change = during_vv.subtract(pre_vv).rename("SAR_Change_DuringMinusPre")

    return pre_vv, during_vv, sar_change, pre_col, target_col, pre_info, target_info


def mask_s2_clouds(img):
    import ee
    qa = img.select("QA60")
    cloud = 1 << 10
    cirrus = 1 << 11
    mask = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
    return img.updateMask(mask).divide(10000)


def get_s2_existing_water(region, config: FloodConfig, return_info: bool = False):
    import ee

    s2_base = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.s2_cloud_pct))
        .map(mask_s2_clouds)
        .select(["B3", "B8"])
    )
    s2_info = adaptive_collection(s2_base, config.s2_water_start, config.s2_water_end, config.max_extra_days)
    s2_col = s2_info["collection"]
    fallback_s2 = ee.Image.constant([0, 0]).rename(["B3", "B8"]).clip(region).toFloat()
    safe_s2_col = ee.ImageCollection(
        ee.Algorithms.If(s2_info["found"], s2_col, ee.ImageCollection([fallback_s2]))
    )
    s2 = safe_s2_col.median().clip(region)
    s2_ndwi = s2.normalizedDifference(["B3", "B8"]).rename("S2_NDWI").unmask(-9999)
    s2_valid = ee.Image.constant(ee.Algorithms.If(s2_info["found"], 1, 0)).toByte()
    s2_existing_water = s2_ndwi.gt(config.ndwi_threshold).And(s2_valid).rename("S2_Existing_Water")

    modis_water_base = (
        ee.ImageCollection("MODIS/061/MOD09A1")
        .filterBounds(region)
        .select(["sur_refl_b04", "sur_refl_b02"])
    )
    modis_info = adaptive_collection(modis_water_base, config.s2_water_start, config.s2_water_end, config.max_extra_days)
    modis_col = modis_info["collection"]
    fallback_modis = (
        ee.Image.constant([0, 0])
        .rename(["sur_refl_b04", "sur_refl_b02"])
        .clip(region)
        .toFloat()
    )
    safe_modis_col = ee.ImageCollection(
        ee.Algorithms.If(modis_info["found"], modis_col, ee.ImageCollection([fallback_modis]))
    )
    modis = safe_modis_col.median().clip(region).multiply(0.0001)
    modis_ndwi = modis.normalizedDifference(["sur_refl_b04", "sur_refl_b02"]).rename("MODIS_NDWI").unmask(-9999)
    modis_valid = ee.Image.constant(ee.Algorithms.If(modis_info["found"], 1, 0)).toByte()
    modis_existing_water = modis_ndwi.gt(config.ndwi_threshold).And(modis_valid).rename("MODIS_Existing_Water")

    existing_water = s2_existing_water.Or(modis_existing_water).rename("Existing_Water")
    combined_ndwi = ee.ImageCollection([s2_ndwi.rename("NDWI"), modis_ndwi.rename("NDWI")]).max().rename("NDWI")

    if return_info:
        info_dict = ee.Dictionary({
            "s2_water_mask_count": s2_info["count"],
            "s2_water_mask_extra_days_used": s2_info["extra_days"],
            "s2_water_mask_actual_end": s2_info["actual_end"],
            "s2_water_mask_found": s2_info["found"],
            "modis_ndwi_water_mask_count": modis_info["count"],
            "modis_ndwi_water_mask_extra_days_used": modis_info["extra_days"],
            "modis_ndwi_water_mask_actual_end": modis_info["actual_end"],
            "modis_ndwi_water_mask_found": modis_info["found"],
            "combined_existing_water_mask_found": ee.Algorithms.If(s2_info["found"], True, modis_info["found"]),
        })
        return combined_ndwi, existing_water, info_dict

    return combined_ndwi, existing_water


def build_flood_map(stage1_vv, existing_water, config: FloodConfig):
    return (
        stage1_vv.lte(config.vv_max)
        .And(stage1_vv.gte(config.vv_min))
        .And(existing_water.Not())
        .rename("Flood_Map")
    )


def build_change_flood_map(during_vv, sar_change, existing_water, config: FloodConfig):
    """Flood label from during-event water-like SAR and pre-to-during dB drop."""
    return (
        during_vv.lte(config.vv_max)
        .And(during_vv.gte(config.vv_min))
        .And(sar_change.lte(config.sar_change_threshold_db))
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


def _constant_band(region, name: str, value: float = -9999):
    import ee
    return ee.Image.constant(value).clip(region).rename(name).toFloat()


def _safe_single_band_collection(raw_col, fallback_img):
    import ee
    return ee.ImageCollection(
        ee.Algorithms.If(raw_col.size().gt(0), raw_col, ee.ImageCollection([fallback_img]))
    )


def build_predictors(region, config: FloodConfig):
    import ee
    dem = ee.Image("USGS/SRTMGL1_003").clip(region).rename("elevation").toFloat()
    slope = ee.Terrain.slope(dem).rename("slope").toFloat()
    twi = (
        ee.Image(1)
        .divide(slope.multiply(3.141592653589793).divide(180).tan().add(0.001))
        .log()
        .rename("TWI_simple")
        .toFloat()
    )

    extra_layers: Dict[str, object] = {}
    availability = {}

    modis_base = (
        ee.ImageCollection("MODIS/061/MOD09A1")
        .filterBounds(region)
        .select(["sur_refl_b01", "sur_refl_b02", "sur_refl_b06"])
    )
    modis_info = adaptive_collection(modis_base, config.predictor_start, config.predictor_end, config.max_extra_days)
    modis_raw = modis_info["collection"]
    availability.update({
        "modis_surface_reflectance_count": modis_info["count"],
        "modis_surface_reflectance_extra_days_used": modis_info["extra_days"],
        "modis_surface_reflectance_actual_end": modis_info["actual_end"],
        "modis_surface_reflectance_found": modis_info["found"],
    })
    modis_fallback = (
        ee.Image.constant([0, 0, 0])
        .rename(["sur_refl_b01", "sur_refl_b02", "sur_refl_b06"])
        .clip(region)
        .toFloat()
    )
    modis = (
        ee.ImageCollection(
            ee.Algorithms.If(modis_info["found"], modis_raw, ee.ImageCollection([modis_fallback]))
        )
        .median()
        .clip(region)
        .multiply(0.0001)
    )
    ndvi = modis.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI").unmask(-9999)
    ndbi = modis.normalizedDifference(["sur_refl_b06", "sur_refl_b02"]).rename("NDBI").unmask(-9999)

    precipitation_base = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(region)
        .select("precipitation")
    )
    precipitation_info = adaptive_collection(precipitation_base, config.month_start, config.month_end, config.max_extra_days)
    precipitation_raw = precipitation_info["collection"]
    availability.update({
        "monthly_chirps_count": precipitation_info["count"],
        "monthly_chirps_extra_days_used": precipitation_info["extra_days"],
        "monthly_chirps_actual_end": precipitation_info["actual_end"],
        "monthly_chirps_found": precipitation_info["found"],
    })
    precipitation = (
        _safe_single_band_collection(
            precipitation_raw,
            _constant_band(region, "precipitation", -9999),
        )
        .sum()
        .clip(region)
        .rename("precipitation")
        .toFloat()
    )

    event_rainfall_base = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(region)
        .select("precipitation")
    )
    event_rainfall_info = adaptive_collection(event_rainfall_base, config.event_rain_start, config.event_rain_end, config.max_extra_days)
    event_rainfall_raw = event_rainfall_info["collection"]
    availability.update({
        "event_chirps_count": event_rainfall_info["count"],
        "event_chirps_extra_days_used": event_rainfall_info["extra_days"],
        "event_chirps_actual_end": event_rainfall_info["actual_end"],
        "event_chirps_found": event_rainfall_info["found"],
    })
    event_rainfall = (
        _safe_single_band_collection(
            event_rainfall_raw,
            _constant_band(region, "event_rainfall", -9999),
        )
        .sum()
        .clip(region)
        .rename("event_rainfall")
        .toFloat()
    )

    temperature_base = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(region)
        .select("LST_Day_1km")
    )
    temperature_info = adaptive_collection(temperature_base, config.month_start, config.month_end, config.max_extra_days)
    temperature_raw = temperature_info["collection"]
    availability.update({
        "modis_lst_count": temperature_info["count"],
        "modis_lst_extra_days_used": temperature_info["extra_days"],
        "modis_lst_actual_end": temperature_info["actual_end"],
        "modis_lst_found": temperature_info["found"],
    })
    temperature = (
        _safe_single_band_collection(
            temperature_raw,
            _constant_band(region, "LST_Day_1km", -9999),
        )
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .clip(region)
        .rename("temperature")
        .toFloat()
    )

    predictor_names: List[str] = [
        "elevation", "slope", "TWI_simple", "NDVI", "NDBI",
        "precipitation", "event_rainfall", "temperature"
    ]

    predictors = (
        dem.addBands(slope)
        .addBands(twi)
        .addBands(ndvi)
        .addBands(ndbi)
        .addBands(precipitation)
        .addBands(event_rainfall)
        .addBands(temperature)
    )

    if config.use_river_distance:
        rivers = ee.FeatureCollection(config.river_asset).filterBounds(region)
        availability["river_feature_count"] = rivers.size()
        river_raster = ee.Image().byte().paint(featureCollection=rivers, color=1).unmask(0).clip(region)
        river_distance = (
            river_raster
            .fastDistanceTransform(1024)
            .sqrt()
            .multiply(ee.Image.pixelArea().sqrt())
            .clip(region)
            .rename("river_distance_m")
            .toFloat()
        )
        river_distance_1km = river_distance.min(config.river_distance_cap_m).rename("river_distance_1km").toFloat()
        river_buffer = river_distance.lte(config.river_distance_cap_m).rename("river_buffer")
        predictors = predictors.addBands(river_distance_1km)
        predictor_names.append("river_distance_1km")
        extra_layers["rivers"] = rivers
        extra_layers["river_distance"] = river_distance_1km
        extra_layers["river_buffer"] = river_buffer

    if config.use_soil_moisture:
        smap_base = (
            ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
            .filterBounds(region)
            .select("sm_surface")
        )
        smap_info = adaptive_collection(smap_base, config.soil_start, config.soil_end, config.max_extra_days)
        smap_raw = smap_info["collection"]
        availability.update({
            "smap_soil_moisture_count": smap_info["count"],
            "smap_soil_moisture_extra_days_used": smap_info["extra_days"],
            "smap_soil_moisture_actual_end": smap_info["actual_end"],
            "smap_soil_moisture_found": smap_info["found"],
        })
        soil_moisture = (
            _safe_single_band_collection(
                smap_raw,
                _constant_band(region, "sm_surface", -9999),
            )
            .mean()
            .clip(region)
            .rename("soil_moisture")
            .toFloat()
        )
        predictors = predictors.addBands(soil_moisture)
        predictor_names.append("soil_moisture")
        extra_layers["soil_moisture"] = soil_moisture

    predictors = predictors.select(predictor_names).unmask(-9999).toFloat()
    extra_layers["availability"] = ee.Dictionary(availability)
    return predictors, predictor_names, extra_layers


def train_rf_susceptibility(region, predictors, label, predictor_names: List[str], config: FloodConfig):
    import ee

    clean_predictors = predictors.select(predictor_names).unmask(-9999).toFloat()
    stack = clean_predictors.addBands(label.unmask(0).toInt().rename("label"))

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

    predictor_list = ee.List(predictor_names)

    def fill_missing_properties(feature):
        feature = ee.Feature(feature)
        original_names = feature.propertyNames()

        def set_one_property(name, accumulator):
            name = ee.String(name)
            accumulator = ee.Feature(accumulator)
            value = ee.Algorithms.If(original_names.contains(name), accumulator.get(name), -9999)
            return accumulator.set(name, value)

        return ee.Feature(predictor_list.iterate(set_one_property, feature))

    samples = samples.map(fill_missing_properties)

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
    susceptibility = clean_predictors.classify(rf_probability).rename("Flood_Susceptibility")

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
    value = ee.Number(ee.Algorithms.If(area.get(band), area.get(band), 0))
    return value.divide(1e6)
