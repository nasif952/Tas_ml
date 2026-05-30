import streamlit as st
from streamlit_folium import st_folium
import folium

from utils.ee_auth import show_ee_connection_block
from utils.ee_visualization import add_ee_layer, hobart_map
from utils.gee_presets import AOI_PRESETS, DATASET_PRESETS, MAP_VIS_PRESETS

st.set_page_config(page_title="GEE Map Viewer", page_icon="🗺️", layout="wide")
st.title("🗺️ Google Earth Engine Map Viewer")
st.caption("Visualize Earth Engine images, image collections, tables, thumbnails, and styled vector layers from the UI. Choose a preset or paste your own dataset/asset ID.")

show_ee_connection_block()

import ee

with st.sidebar:
    st.header("1. Layer source")
    st.caption("Use a known flood/geospatial dataset preset or manually paste your own Earth Engine ID.")
    preset_names = list(DATASET_PRESETS.keys())
    dataset_preset = st.selectbox("Dataset preset", preset_names)
    preset_info = DATASET_PRESETS[dataset_preset]
    st.caption("Best for: " + preset_info["best_for"])
    st.caption("Notes: " + preset_info["notes"])

    source_type = st.selectbox(
        "Object type",
        ["Image", "ImageCollection", "FeatureCollection / Table"],
        index=0 if preset_info["type"] == "Image" else 1 if preset_info["type"] == "ImageCollection" else 2,
    )
    asset_id = st.text_input("Earth Engine asset/dataset ID", preset_info["id"])

    st.divider()
    st.header("2. AOI")
    st.caption("The AOI clips the displayed layer. Smaller AOIs load faster.")
    aoi_name = st.selectbox("AOI preset", list(AOI_PRESETS.keys()))
    bbox = AOI_PRESETS[aoi_name]["bbox"]
    st.caption(AOI_PRESETS[aoi_name]["description"])
    xmin = st.number_input("Min lon", value=float(bbox[0]), step=0.01, format="%.4f")
    ymin = st.number_input("Min lat", value=float(bbox[1]), step=0.01, format="%.4f")
    xmax = st.number_input("Max lon", value=float(bbox[2]), step=0.01, format="%.4f")
    ymax = st.number_input("Max lat", value=float(bbox[3]), step=0.01, format="%.4f")
    zoom = st.slider("Map zoom", 6, 15, 10)

    st.divider()
    st.header("3. ImageCollection filters")
    st.caption("Only used for ImageCollection layers. The app filters by date and creates one composite image.")
    start = st.text_input("Start date", "2018-05-01")
    end = st.text_input("End date", "2018-06-01")
    composite = st.selectbox("Composite", ["median", "mean", "first", "mosaic"])

    st.divider()
    st.header("4. Visualization")
    st.caption("These settings only affect display colour and contrast, not the actual data.")
    vis_preset_name = st.selectbox("Visualization preset", list(MAP_VIS_PRESETS.keys()))
    vis_preset = MAP_VIS_PRESETS[vis_preset_name]
    bands_text = st.text_input("Bands comma-separated", vis_preset["bands"])
    vis_min = st.number_input("Min", value=float(vis_preset["min"]))
    vis_max = st.number_input("Max", value=float(vis_preset["max"]))
    palette = st.text_input("Palette comma-separated", vis_preset["palette"])
    opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.05)
    vector_color = st.color_picker("Vector color", "#FFFF00")
    vector_width = st.slider("Vector width", 1, 6, 2)
    show_thumb = st.checkbox("Also create thumbnail preview", value=True)

region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

col_map, col_info = st.columns([2, 1])

with col_info:
    st.subheader("Layer setup")
    st.write("Type:", source_type)
    st.code(asset_id)
    st.info("This page uses Earth Engine map tiles and thumbnails for visualization. Use the Download/Export page for files.")

try:
    if source_type == "Image":
        ee_obj = ee.Image(asset_id).clip(region)
        layer_name = asset_id.split("/")[-1]
    elif source_type == "ImageCollection":
        col = ee.ImageCollection(asset_id).filterBounds(region).filterDate(start, end)
        count = col.size().getInfo()
        with col_info:
            st.metric("Filtered image count", count)
        if composite == "median":
            ee_obj = col.median().clip(region)
        elif composite == "mean":
            ee_obj = col.mean().clip(region)
        elif composite == "first":
            ee_obj = ee.Image(col.first()).clip(region)
        else:
            ee_obj = col.mosaic().clip(region)
        layer_name = f"{asset_id.split('/')[-1]} {composite}"
    else:
        fc = ee.FeatureCollection(asset_id).filterBounds(region)
        styled = fc.style(color=vector_color.replace("#", ""), fillColor="00000000", width=vector_width)
        ee_obj = styled
        layer_name = asset_id.split("/")[-1]

    vis = {}
    if source_type != "FeatureCollection / Table":
        if bands_text.strip():
            vis["bands"] = [b.strip() for b in bands_text.split(",") if b.strip()]
        vis["min"] = vis_min
        vis["max"] = vis_max
        if palette.strip():
            vis["palette"] = [p.strip() for p in palette.split(",") if p.strip()]

    with col_map:
        st.subheader("Interactive map")
        m = hobart_map(zoom_start=zoom)
        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            color="black",
            weight=2,
            fill=False,
            tooltip="AOI",
        ).add_to(m)
        add_ee_layer(m, ee_obj, vis, layer_name, opacity=opacity)
        folium.LayerControl().add_to(m)
        st_folium(m, height=650, use_container_width=True, returned_objects=[], key="gee_map_viewer_static")

    if show_thumb and source_type != "FeatureCollection / Table":
        with col_info:
            st.subheader("Static thumbnail")
            thumb_params = dict(vis)
            thumb_params.update({"region": region, "dimensions": 512})
            try:
                st.image(ee_obj.getThumbURL(thumb_params), caption="Earth Engine thumbnail preview")
            except Exception as exc:
                st.info("Thumbnail not available for these settings.")
                st.caption(str(exc))

except Exception as exc:
    st.error("Could not visualize this Earth Engine object. Check asset ID, permissions, bands, dates, and visualization parameters.")
    st.exception(exc)
