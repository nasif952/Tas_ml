import pandas as pd
import streamlit as st

from utils.ee_auth import init_earth_engine, show_ee_connection_block

st.set_page_config(page_title="GEE Data Browser", page_icon="🗂️", layout="wide")
st.title("🗂️ Google Earth Engine Data Browser")
st.caption("Browse project assets, inspect metadata, preview image bands, and prepare assets for visualization or export.")

show_ee_connection_block()

import ee


@st.cache_data(ttl=300, show_spinner=False)
def list_assets(parent: str, limit: int = 500) -> pd.DataFrame:
    """List Earth Engine assets under a parent.

    Newer earthengine-api versions expect ee.data.listAssets(params_dict),
    not ee.data.listAssets(parent, params).
    """
    rows = []
    token = None
    while len(rows) < limit:
        params = {
            "parent": parent,
            "pageSize": min(100, limit - len(rows)),
        }
        if token:
            params["pageToken"] = token
        resp = ee.data.listAssets(params)
        for asset in resp.get("assets", []):
            rows.append({
                "name": asset.get("name"),
                "type": asset.get("type"),
                "updateTime": asset.get("updateTime"),
                "sizeBytes": asset.get("sizeBytes"),
            })
        token = resp.get("nextPageToken")
        if not token:
            break
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def get_asset_metadata(asset_name: str) -> dict:
    return ee.data.getAsset(asset_name)


@st.cache_data(ttl=300, show_spinner=False)
def image_band_names(asset_name: str):
    return ee.Image(asset_name).bandNames().getInfo()


@st.cache_data(ttl=300, show_spinner=False)
def image_collection_summary(asset_name: str, start: str = None, end: str = None, limit: int = 5):
    col = ee.ImageCollection(asset_name)
    if start and end:
        col = col.filterDate(start, end)
    count = col.size().getInfo()
    first = col.first()
    bands = first.bandNames().getInfo() if count else []
    return {"count": count, "first_bands": bands[:50]}


@st.cache_data(ttl=300, show_spinner=False)
def table_preview(asset_name: str, limit: int = 10):
    fc = ee.FeatureCollection(asset_name)
    features = fc.limit(limit).getInfo().get("features", [])
    rows = []
    for f in features:
        props = f.get("properties", {})
        geom_type = f.get("geometry", {}).get("type") if f.get("geometry") else None
        props["geometry_type"] = geom_type
        rows.append(props)
    return pd.DataFrame(rows)


ctx = init_earth_engine()
default_root = f"projects/{ctx['project']}/assets"

with st.sidebar:
    st.header("Browse settings")
    root = st.text_input("Asset root", default_root)
    limit = st.slider("Max assets", 50, 1000, 500, 50)
    refresh = st.button("Refresh asset list", use_container_width=True)
    st.divider()
    st.markdown("**Curated public dataset IDs**")
    presets = {
        "Sentinel-1 SAR GRD": "COPERNICUS/S1_GRD",
        "Sentinel-2 SR Harmonized": "COPERNICUS/S2_SR_HARMONIZED",
        "SRTM DEM": "USGS/SRTMGL1_003",
        "CHIRPS Daily Rainfall": "UCSB-CHG/CHIRPS/DAILY",
        "MODIS Surface Reflectance": "MODIS/061/MOD09A1",
        "MODIS LST": "MODIS/061/MOD11A2",
        "SMAP Soil Moisture": "NASA/SMAP/SPL4SMGP/007",
    }
    preset_name = st.selectbox("Preset", list(presets.keys()))
    preset_id = st.text_input("Selected public dataset", presets[preset_name])

if refresh:
    st.cache_data.clear()

left, right = st.columns([1.35, 1])
selected_asset = None

with left:
    st.subheader("Project assets")
    try:
        assets = list_assets(root, limit)
        if assets.empty:
            st.warning("No assets found in this root, or the service account cannot list it.")
        else:
            type_filter = st.multiselect("Filter by type", sorted(assets["type"].dropna().unique()), default=sorted(assets["type"].dropna().unique()))
            visible = assets[assets["type"].isin(type_filter)] if type_filter else assets
            st.dataframe(visible, use_container_width=True, hide_index=True)
            selected_asset = st.selectbox("Select project asset", visible["name"].tolist())
    except Exception as exc:
        st.error("Could not list project assets.")
        st.exception(exc)
        selected_asset = None

with right:
    st.subheader("Metadata inspector")
    inspect_mode = st.radio("Inspect", ["Project asset", "Public dataset ID"], horizontal=True)
    asset_to_inspect = selected_asset if inspect_mode == "Project asset" else preset_id
    if asset_to_inspect:
        try:
            meta = get_asset_metadata(asset_to_inspect)
            st.json(meta)
            asset_type = meta.get("type")
            st.divider()
            if asset_type == "IMAGE":
                st.markdown("**Image bands**")
                st.write(image_band_names(asset_to_inspect))
            elif asset_type == "IMAGE_COLLECTION":
                st.markdown("**ImageCollection summary**")
                c1, c2 = st.columns(2)
                start = c1.text_input("Start date", "2018-05-01")
                end = c2.text_input("End date", "2018-06-01")
                if st.button("Summarize collection"):
                    st.json(image_collection_summary(asset_to_inspect, start, end))
            elif asset_type == "TABLE":
                st.markdown("**Table preview")
                st.dataframe(table_preview(asset_to_inspect), use_container_width=True)
        except Exception as exc:
            st.error("Could not inspect this asset/dataset. Check the ID and permissions.")
            st.exception(exc)

st.info("Use this page to find valid asset IDs, then paste them into the Map Viewer, Download/Export, or Hobart Flood Model Runner pages.")
