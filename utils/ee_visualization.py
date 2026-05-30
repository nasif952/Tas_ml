import folium


def add_ee_layer(m: folium.Map, ee_object, vis_params: dict, name: str, shown: bool = True, opacity: float = 1.0):
    """Add an Earth Engine Image or styled FeatureCollection image to a Folium map."""
    map_id_dict = ee_object.getMapId(vis_params or {})
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=shown,
        opacity=opacity,
    ).add_to(m)
    return m


def hobart_map(zoom_start: int = 10) -> folium.Map:
    return folium.Map(
        location=[-42.88, 147.33],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )


def get_default_vis(layer_name: str) -> dict:
    presets = {
        "existing_water": {"palette": ["darkblue"]},
        "flood_map": {"palette": ["cyan"]},
        "vv_change": {"min": -5, "max": 5, "palette": ["blue", "white", "red"]},
        "persistence": {"min": 1, "max": 2, "palette": ["lightblue", "red"]},
        "river_distance": {"min": 0, "max": 1000, "palette": ["blue", "cyan", "yellow", "red"]},
        "river_buffer": {"palette": ["purple"]},
        "soil_moisture": {"min": 0, "max": 0.5, "palette": ["brown", "yellow", "green", "blue"]},
        "susceptibility": {"min": 0, "max": 1, "palette": ["green", "yellow", "red"]},
        "final_overlay": {"min": 0, "max": 1, "palette": ["green", "yellow", "orange", "red"]},
        "high_overlay": {"palette": ["purple"]},
        "dem": {"min": 0, "max": 1200, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8"]},
    }
    return presets.get(layer_name, {})
