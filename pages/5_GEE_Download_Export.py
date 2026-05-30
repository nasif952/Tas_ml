import time
import streamlit as st

from utils.ee_auth import show_ee_connection_block

st.set_page_config(page_title="GEE Download & Export", page_icon="⬇️", layout="wide")
st.title("⬇️ Google Earth Engine Download & Export Center")
st.caption("Create small direct download links or start large Earth Engine batch exports from the UI.")

show_ee_connection_block()

import ee

with st.sidebar:
    st.header("AOI")
    xmin = st.number_input("Min lon", value=146.85, step=0.01, format="%.4f")
    ymin = st.number_input("Min lat", value=-43.10, step=0.01, format="%.4f")
    xmax = st.number_input("Max lon", value=147.55, step=0.01, format="%.4f")
    ymax = st.number_input("Max lat", value=-42.65, step=0.01, format="%.4f")
    scale = st.select_slider("Scale", options=[30, 50, 100, 250, 500, 1000], value=100)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

st.info("Small outputs can use direct download URLs. Large rasters/tables should use batch exports to Google Drive or Earth Engine assets.")

t1, t2, t3 = st.tabs(["Small raster download", "Small vector/table download", "Large batch export"])

with t1:
    st.subheader("Small raster download URL")
    image_id = st.text_input("Image ID", "USGS/SRTMGL1_003")
    file_format = st.selectbox("Format", ["GEO_TIFF", "ZIPPED_GEO_TIFF", "NPY"])
    if st.button("Create raster download URL", type="primary"):
        try:
            image = ee.Image(image_id).clip(region)
            url = image.getDownloadURL({
                "scale": scale,
                "region": region,
                "format": file_format,
                "filePerBand": False,
            })
            st.success("Download URL created.")
            st.markdown(f"[Download raster file]({url})")
            st.caption("If this fails, the requested area is probably too large for direct download. Use batch export instead.")
        except Exception as exc:
            st.error("Could not create raster download URL.")
            st.exception(exc)

with t2:
    st.subheader("Small vector/table download URL")
    fc_id = st.text_input("FeatureCollection/Table ID", "projects/gee-project-493107/assets/Greater_Hobart_Buildings_WGS84")
    vector_format = st.selectbox("Vector format", ["CSV", "GeoJSON", "KML", "KMZ"])
    if st.button("Create vector/table download URL", type="primary"):
        try:
            fc = ee.FeatureCollection(fc_id).filterBounds(region)
            url = fc.getDownloadURL(filetype=vector_format, filename="gee_vector_download")
            st.success("Download URL created.")
            st.markdown(f"[Download vector/table file]({url})")
        except Exception as exc:
            st.error("Could not create vector/table download URL.")
            st.exception(exc)

with t3:
    st.subheader("Large batch export")
    export_type = st.radio("Export type", ["Image to Drive", "Table to Drive"], horizontal=True)
    description = st.text_input("Task description / file prefix", "hobart_gee_export")
    folder = st.text_input("Google Drive folder", "GEE_Hobart_Flood_Project")

    if export_type == "Image to Drive":
        export_image_id = st.text_input("Image ID to export", "USGS/SRTMGL1_003", key="export_img")
    else:
        export_fc_id = st.text_input("FeatureCollection/Table ID to export", "projects/gee-project-493107/assets/Greater_Hobart_Buildings_WGS84", key="export_fc")
        table_format = st.selectbox("Table export format", ["CSV", "GeoJSON", "KML", "KMZ", "SHP"])

    if st.button("Start batch export", type="primary"):
        try:
            if export_type == "Image to Drive":
                img = ee.Image(export_image_id).clip(region)
                task = ee.batch.Export.image.toDrive(
                    image=img,
                    description=description,
                    folder=folder,
                    fileNamePrefix=description,
                    region=region,
                    scale=scale,
                    maxPixels=1e13,
                )
            else:
                fc = ee.FeatureCollection(export_fc_id).filterBounds(region)
                task = ee.batch.Export.table.toDrive(
                    collection=fc,
                    description=description,
                    folder=folder,
                    fileNamePrefix=description,
                    fileFormat=table_format,
                )
            task.start()
            st.success("Export task started.")
            st.json(task.status())
            st.caption("Open Google Earth Engine Tasks or check Google Drive after completion.")
        except Exception as exc:
            st.error("Could not start export task.")
            st.exception(exc)

    st.divider()
    st.subheader("Recent task list")
    if st.button("Refresh Earth Engine tasks"):
        try:
            tasks = ee.batch.Task.list()
            rows = []
            for task in tasks[:20]:
                status = task.status()
                rows.append({
                    "id": status.get("id"),
                    "description": status.get("description"),
                    "state": status.get("state"),
                    "creation_timestamp_ms": status.get("creation_timestamp_ms"),
                })
            st.dataframe(rows, use_container_width=True)
        except Exception as exc:
            st.error("Could not fetch task list.")
            st.exception(exc)
