import pandas as pd
import streamlit as st

from utils.download_utils import (
    dataframe_downloads,
    dict_downloads,
    list_repository_files,
    repository_text_file_download,
)
from utils.ee_auth import show_ee_connection_block
from utils.gee_presets import AOI_PRESETS, DATASET_PRESETS

st.set_page_config(page_title="Local Download Center", page_icon="💾", layout="wide")
st.title("💾 Local Download Center")
st.caption(
    "Download generated tables, metadata, research markdown files, and small Earth Engine outputs directly to your computer. "
    "Large rasters should still use Earth Engine export to Google Drive."
)

show_ee_connection_block()

import ee

st.info(
    "Use local downloads for small/medium CSV, JSON, TXT, Markdown, and small vector tables. "
    "Use Google Drive export for large GeoTIFFs, large FeatureCollections, or full-resolution model outputs."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Generated session tables",
    "Research/docs files",
    "Small GEE vector/table download",
    "Small GEE raster download",
])

with tab1:
    st.subheader("Generated session tables")
    st.caption("If you created tables in Hybrid Monitoring or Weekly Time Series during this app session, download them here.")

    session_tables = {
        "Hybrid monitoring table": "hybrid_monitoring_df",
        "Weekly time-series preview": "weekly_timeseries_df",
    }

    available = [name for name, key in session_tables.items() if key in st.session_state]
    if not available:
        st.warning("No generated session table is available yet. Run a monitoring/time-series page first.")
    else:
        selected = st.selectbox("Available table", available)
        key = session_tables[selected]
        df = st.session_state[key]
        st.dataframe(df, use_container_width=True, hide_index=True)
        dataframe_downloads(df, selected.lower().replace(" ", "_"), "Download table as")

with tab2:
    st.subheader("Research/docs files from repository")
    st.caption("Download Markdown/TXT/CSV/JSON files that exist in the deployed repository, especially files inside the research folder.")
    files = list_repository_files("research", suffixes=(".md", ".txt", ".csv", ".json"))
    if not files:
        st.warning("No downloadable files found in the research folder of the deployed app.")
    else:
        selected_file = st.selectbox("Repository file", files)
        repository_text_file_download(selected_file, label=f"Download {selected_file.split('/')[-1]}")

with tab3:
    st.subheader("Small Earth Engine vector/table download")
    st.caption("Use this for small FeatureCollections/tables. For large tables, use Earth Engine Drive export.")

    preset = st.selectbox("Vector/table preset", ["Greater Hobart Buildings asset", "HydroSHEDS Free Flowing Rivers", "Custom"], key="vec_preset")
    if preset == "Custom":
        fc_id = st.text_input("FeatureCollection/Table ID", "projects/gee-project-493107/assets/Greater_Hobart_Buildings_WGS84")
    else:
        fc_id = st.text_input("FeatureCollection/Table ID", DATASET_PRESETS[preset]["id"])

    aoi_name = st.selectbox("AOI preset", list(AOI_PRESETS.keys()), key="vec_aoi")
    bbox = AOI_PRESETS[aoi_name]["bbox"]
    xmin = st.number_input("Min longitude", value=float(bbox[0]), step=0.01, format="%.4f", key="vec_xmin")
    ymin = st.number_input("Min latitude", value=float(bbox[1]), step=0.01, format="%.4f", key="vec_ymin")
    xmax = st.number_input("Max longitude", value=float(bbox[2]), step=0.01, format="%.4f", key="vec_xmax")
    ymax = st.number_input("Max latitude", value=float(bbox[3]), step=0.01, format="%.4f", key="vec_ymax")
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    limit = st.slider("Preview/local table row limit", 10, 5000, 500, 10)
    filetype = st.selectbox("Earth Engine direct download format", ["CSV", "GeoJSON", "KML", "KMZ"])

    c1, c2 = st.columns(2)
    if c1.button("Preview and local-download vector table", type="primary"):
        try:
            fc = ee.FeatureCollection(fc_id).filterBounds(region).limit(limit)
            features = fc.getInfo().get("features", [])
            rows = []
            for f in features:
                props = f.get("properties", {})
                geom = f.get("geometry") or {}
                props["geometry_type"] = geom.get("type")
                rows.append(props)
            df = pd.DataFrame(rows)
            st.session_state["local_vector_preview_df"] = df
            st.success(f"Loaded {len(df)} rows for local download.")
        except Exception as exc:
            st.error("Could not preview vector/table locally. Try a smaller AOI or lower row limit.")
            st.exception(exc)

    if c2.button("Create direct GEE download URL"):
        try:
            fc = ee.FeatureCollection(fc_id).filterBounds(region)
            url = fc.getDownloadURL(filetype=filetype, filename="gee_vector_download")
            st.success("Direct Earth Engine download URL created.")
            st.markdown(f"[Download {filetype} from Earth Engine]({url})")
        except Exception as exc:
            st.error("Could not create direct Earth Engine table download URL. Use Drive export for large tables.")
            st.exception(exc)

    if "local_vector_preview_df" in st.session_state:
        st.dataframe(st.session_state["local_vector_preview_df"], use_container_width=True, hide_index=True)
        dataframe_downloads(st.session_state["local_vector_preview_df"], "gee_vector_preview", "Download preview as")

with tab4:
    st.subheader("Small Earth Engine raster download")
    st.caption("Use this for small clipped raster downloads. Large/full-resolution rasters should use Drive export.")

    raster_preset = st.selectbox("Raster preset", ["SRTM DEM 30m", "Custom Image ID"], key="raster_preset")
    if raster_preset == "Custom Image ID":
        image_id = st.text_input("Image ID", "USGS/SRTMGL1_003")
    else:
        image_id = st.text_input("Image ID", DATASET_PRESETS[raster_preset]["id"])

    aoi_name_r = st.selectbox("AOI preset", list(AOI_PRESETS.keys()), key="ras_aoi")
    bbox_r = AOI_PRESETS[aoi_name_r]["bbox"]
    rxmin = st.number_input("Min longitude", value=float(bbox_r[0]), step=0.01, format="%.4f", key="ras_xmin")
    rymin = st.number_input("Min latitude", value=float(bbox_r[1]), step=0.01, format="%.4f", key="ras_ymin")
    rxmax = st.number_input("Max longitude", value=float(bbox_r[2]), step=0.01, format="%.4f", key="ras_xmax")
    rymax = st.number_input("Max latitude", value=float(bbox_r[3]), step=0.01, format="%.4f", key="ras_ymax")
    rregion = ee.Geometry.Rectangle([rxmin, rymin, rxmax, rymax])

    scale = st.select_slider("Raster scale", [30, 50, 100, 250, 500, 1000], value=100)
    raster_format = st.selectbox("Raster format", ["GEO_TIFF", "ZIPPED_GEO_TIFF", "NPY"])

    if st.button("Create small raster download URL", type="primary"):
        try:
            img = ee.Image(image_id).clip(rregion)
            url = img.getDownloadURL({
                "scale": scale,
                "region": rregion,
                "format": raster_format,
                "filePerBand": False,
            })
            st.success("Raster download URL created.")
            st.markdown(f"[Download raster from Earth Engine]({url})")
        except Exception as exc:
            st.error("Could not create raster download URL. Use a smaller AOI/coarser scale or Drive export.")
            st.exception(exc)
