"""Curated Earth Engine presets for the Streamlit UI.

These are not every dataset in Earth Engine. They are practical presets for
flood mapping, flood susceptibility, hydrology, remote sensing, and Hobart
2018 research. Every page still exposes custom text inputs so users can paste
any valid Earth Engine asset or dataset ID.
"""

AOI_PRESETS = {
    "Hobart core / default": {
        "bbox": [146.85, -43.10, 147.55, -42.65],
        "description": "Default study rectangle used in the Hobart 2018 flood prototype.",
    },
    "Greater Hobart wider context": {
        "bbox": [146.60, -43.25, 147.75, -42.45],
        "description": "Wider context for checking river corridor and outer urban areas. Slower to run.",
    },
    "Derwent corridor focus": {
        "bbox": [146.95, -43.05, 147.45, -42.70],
        "description": "Smaller river-corridor focus. Useful for faster testing and exposure checks.",
    },
    "Custom bounding box": {
        "bbox": [146.85, -43.10, 147.55, -42.65],
        "description": "Use the coordinate fields below to define your own rectangle.",
    },
}

DATASET_PRESETS = {
    "Sentinel-1 SAR GRD": {
        "id": "COPERNICUS/S1_GRD",
        "type": "ImageCollection",
        "best_for": "Flood detection from radar backscatter, cloud-independent mapping.",
        "notes": "Use VV/VH, orbit filters, and before/during event dates. Main flood dataset in this app.",
    },
    "Sentinel-2 Surface Reflectance Harmonized": {
        "id": "COPERNICUS/S2_SR_HARMONIZED",
        "type": "ImageCollection",
        "best_for": "NDWI, water masking, vegetation/built-up spectral context when cloud-free.",
        "notes": "Optical data is cloud-sensitive. Use pre-flood period for permanent/existing water masking.",
    },
    "SRTM DEM 30m": {
        "id": "USGS/SRTMGL1_003",
        "type": "Image",
        "best_for": "Elevation, slope, simple terrain wetness variables.",
        "notes": "Useful for susceptibility; not a hydrodynamic model by itself.",
    },
    "CHIRPS Daily Rainfall": {
        "id": "UCSB-CHG/CHIRPS/DAILY",
        "type": "ImageCollection",
        "best_for": "Rainfall and event rainfall totals.",
        "notes": "Good for broad rainfall signal. Local gauges may be better for city-scale calibration.",
    },
    "MODIS Surface Reflectance 8-day": {
        "id": "MODIS/061/MOD09A1",
        "type": "ImageCollection",
        "best_for": "NDVI and NDBI style predictors at coarse resolution.",
        "notes": "Coarse compared with Sentinel-2, but stable for quick modelling.",
    },
    "MODIS Land Surface Temperature": {
        "id": "MODIS/061/MOD11A2",
        "type": "ImageCollection",
        "best_for": "Temperature predictor for environmental context.",
        "notes": "Usually less important than terrain/rainfall for flood, but useful as contextual variable.",
    },
    "SMAP Soil Moisture": {
        "id": "NASA/SMAP/SPL4SMGP/007",
        "type": "ImageCollection",
        "best_for": "Surface soil moisture / antecedent wetness signal.",
        "notes": "Coarse resolution; interpret carefully at urban scale.",
    },
    "HydroSHEDS Free Flowing Rivers": {
        "id": "WWF/HydroSHEDS/v1/FreeFlowingRivers",
        "type": "FeatureCollection",
        "best_for": "River distance factor when a local river asset is not available.",
        "notes": "May not be the best Tasmania hydrography source. Prefer a local river/Geofabric asset if you have one.",
    },
    "Greater Hobart Buildings asset": {
        "id": "projects/gee-project-493107/assets/Greater_Hobart_Buildings_WGS84",
        "type": "FeatureCollection",
        "best_for": "Building exposure overlay and selected building exports.",
        "notes": "Custom project asset. Service account needs access.",
    },
}

FLOOD_EVENT_PRESETS = {
    "Hobart May 2018 default": {
        "stage1_start": "2018-05-15",
        "stage1_end": "2018-05-16",
        "stage2_start": "2018-05-27",
        "stage2_end": "2018-05-28",
        "s2_start": "2018-03-01",
        "s2_end": "2018-05-09",
        "event_rain_start": "2018-05-10",
        "event_rain_end": "2018-05-13",
        "description": "Current working event setup from the prototype. Uses Stage 1 flood-like SAR and Stage 2 persistence/recovery comparison.",
    },
    "Hobart May 2018 wider SAR windows": {
        "stage1_start": "2018-05-10",
        "stage1_end": "2018-05-18",
        "stage2_start": "2018-05-24",
        "stage2_end": "2018-06-03",
        "s2_start": "2018-02-01",
        "s2_end": "2018-05-09",
        "event_rain_start": "2018-05-09",
        "event_rain_end": "2018-05-14",
        "description": "Uses wider date windows, useful if exact same-day Sentinel-1 images are missing or noisy.",
    },
    "Custom event dates": {
        "stage1_start": "2018-05-15",
        "stage1_end": "2018-05-16",
        "stage2_start": "2018-05-27",
        "stage2_end": "2018-05-28",
        "s2_start": "2018-03-01",
        "s2_end": "2018-05-09",
        "event_rain_start": "2018-05-10",
        "event_rain_end": "2018-05-13",
        "description": "Use manual date inputs below.",
    },
}

S1_ORBIT_OPTIONS = {
    "Either": "Use all matching Sentinel-1 passes. More images, but mixed geometry can add noise.",
    "ASCENDING": "Use ascending passes only. More consistent geometry if images exist.",
    "DESCENDING": "Use descending passes only. More consistent geometry if images exist.",
}

S1_POLARIZATION_OPTIONS = {
    "VV": "Default. Often strong for open-water flood detection.",
    "VH": "Can help with vegetation/roughness but is not implemented in the current RF pipeline yet.",
    "VV + VH": "Future option. The UI documents it, but current production pipeline runs VV.",
}

VV_THRESHOLD_PRESETS = {
    "Conservative open water (-20 to -15 dB)": {
        "vv_min": -20.0,
        "vv_max": -15.0,
        "description": "Current default. Targets water-like SAR backscatter while avoiding very dark noise extremes.",
    },
    "Broader water-like range (-23 to -14 dB)": {
        "vv_min": -23.0,
        "vv_max": -14.0,
        "description": "Captures more possible water, but may increase false positives.",
    },
    "Strict darker water (-25 to -18 dB)": {
        "vv_min": -25.0,
        "vv_max": -18.0,
        "description": "More conservative for very dark smooth water, but may miss urban/shallow flooding.",
    },
    "Custom threshold": {
        "vv_min": -20.0,
        "vv_max": -15.0,
        "description": "Set VV min/max manually.",
    },
}

NDWI_PRESETS = {
    "Default existing water NDWI > 0.0": {
        "threshold": 0.0,
        "description": "Common simple water threshold for masking pre-existing surface water.",
    },
    "Stricter water NDWI > 0.2": {
        "threshold": 0.2,
        "description": "Masks only stronger optical water signal; may leave some permanent water unmasked.",
    },
    "Sensitive water NDWI > -0.05": {
        "threshold": -0.05,
        "description": "Masks more wet/dark surfaces; may remove valid flood candidates if too aggressive.",
    },
    "Custom NDWI threshold": {
        "threshold": 0.0,
        "description": "Set manually.",
    },
}

MODEL_PRESETS = {
    "Balanced quick test": {
        "scale": 100,
        "samples": 600,
        "trees": 300,
        "train_fraction": 0.70,
        "description": "Good default for Streamlit testing. Reasonable speed and stable results.",
    },
    "Fast rough preview": {
        "scale": 250,
        "samples": 300,
        "trees": 100,
        "train_fraction": 0.70,
        "description": "Faster but less detailed. Use when checking if settings work.",
    },
    "More detailed slower run": {
        "scale": 50,
        "samples": 1000,
        "trees": 500,
        "train_fraction": 0.75,
        "description": "More detailed and slower. Better for final exploratory output.",
    },
    "Custom model settings": {
        "scale": 100,
        "samples": 600,
        "trees": 300,
        "train_fraction": 0.70,
        "description": "Set model parameters manually.",
    },
}

MAP_VIS_PRESETS = {
    "DEM / elevation": {"bands": "", "min": 0.0, "max": 1200.0, "palette": "006633,E5FFCC,662A00,D8D8D8"},
    "Probability 0-1 green-yellow-red": {"bands": "", "min": 0.0, "max": 1.0, "palette": "green,yellow,red"},
    "Binary mask cyan": {"bands": "", "min": 0.0, "max": 1.0, "palette": "cyan"},
    "VV SAR dB": {"bands": "VV", "min": -25.0, "max": 0.0, "palette": "black,white"},
    "VV change diverging": {"bands": "", "min": -5.0, "max": 5.0, "palette": "blue,white,red"},
    "Custom visualization": {"bands": "", "min": 0.0, "max": 3000.0, "palette": ""},
}

PAGE_HELP = {
    "aoi": "The Area of Interest controls where Earth Engine clips, samples, maps, and exports data. Smaller areas run faster.",
    "s1": "Sentinel-1 is radar. It is useful for flood mapping because it can see through cloud and works at night.",
    "s2": "Sentinel-2 is optical. Here it is used before the flood to identify existing water so the SAR flood map focuses on new potential inundation.",
    "predictors": "Predictors are environmental factors used by the Random Forest susceptibility model. They explain where SAR-derived flood-like pixels are likely to occur.",
    "rf": "Random Forest learns the relationship between the SAR-derived flood label and predictor layers. Accuracy here evaluates the label/model split, not independent field truth.",
    "export": "Direct downloads are only for small outputs. Use batch exports for full maps or large building/table results.",
}
