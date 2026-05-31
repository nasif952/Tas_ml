"""
Lazy one-layer-at-a-time Streamlit + geemap viewer.

Use this helper in the flood susceptibility Streamlit page to stop the app
from refreshing/crashing when the map section renders.

Why this exists:
- Rendering many Earth Engine layers at once can overload the browser iframe.
- This helper only creates the map after the user clicks a button.
- It renders only the selected layer.
- Changing layer clears the previous map by rerunning with a new selection.

Example usage inside your Streamlit page:

    from utils.lazy_geemap_viewer import render_lazy_gee_map

    layer_options = {
        "RF Susceptibility": {
            "ee_object": rf_classified,
            "vis": rf_vis,
            "name": "RF Susceptibility",
        },
        "Flood Mask": {
            "ee_object": flood_mask.selfMask(),
            "vis": flood_vis,
            "name": "Flood Mask",
        },
        "SAR Before": {
            "ee_object": pre_sar,
            "vis": sar_vis,
            "name": "SAR Before",
        },
        "SAR During": {
            "ee_object": during_sar,
            "vis": sar_vis,
            "name": "SAR During",
        },
        "Building Exposure": {
            "ee_object": exposed_buildings,
            "vis": {},
            "name": "Building Exposure",
        },
    }

    render_lazy_gee_map(
        layer_options=layer_options,
        center=(-42.88, 147.33),
        zoom=10,
        height=650,
        key_prefix="b_reduced_aoi_flood_trainer",
    )
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import streamlit as st

try:
    import geemap.foliumap as geemap
except Exception:  # pragma: no cover
    import geemap  # type: ignore


LayerConfig = Mapping[str, Any]
LayerOptions = Mapping[str, LayerConfig]


def render_lazy_gee_map(
    *,
    layer_options: LayerOptions,
    center: Tuple[float, float] = (-42.88, 147.33),
    zoom: int = 10,
    height: int = 650,
    key_prefix: str = "lazy_gee_map",
) -> None:
    """Render one Earth Engine map layer only after a button click.

    Parameters
    ----------
    layer_options:
        Dictionary where each key is the dropdown label and each value contains:
        - ee_object: Earth Engine Image, FeatureCollection, Geometry, etc.
        - vis: visualization dictionary
        - name: optional layer name
    center:
        Map center as (latitude, longitude).
    zoom:
        Initial map zoom.
    height:
        Streamlit map iframe height.
    key_prefix:
        Unique key prefix for Streamlit widgets.
    """

    st.subheader("Map")
    st.caption(
        "Map is lazy-loaded to prevent browser refresh. Select one layer, then click Load selected map."
    )

    if not layer_options:
        st.info("No map layers are available yet. Run the Earth Engine workflow first.")
        return

    layer_names = ["None"] + list(layer_options.keys())

    selected_layer = st.selectbox(
        "Choose one map layer to display",
        layer_names,
        index=0,
        key=f"{key_prefix}_selected_layer",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        load_clicked = st.button(
            "Load selected map",
            key=f"{key_prefix}_load_button",
            disabled=(selected_layer == "None"),
        )
    with col2:
        clear_clicked = st.button(
            "Clear map",
            key=f"{key_prefix}_clear_button",
        )

    if clear_clicked:
        st.session_state.pop(f"{key_prefix}_active_layer", None)
        st.info("Map cleared. Select a layer and click Load selected map.")
        return

    if load_clicked and selected_layer != "None":
        st.session_state[f"{key_prefix}_active_layer"] = selected_layer

    active_layer = st.session_state.get(f"{key_prefix}_active_layer")

    if not active_layer:
        st.info("Select one map layer and click Load selected map.")
        return

    layer = layer_options[active_layer]
    ee_object = layer.get("ee_object")
    vis = dict(layer.get("vis", {}))
    layer_name = layer.get("name", active_layer)

    if ee_object is None:
        st.warning(f"Layer '{active_layer}' is not available.")
        return

    st.success(f"Showing one layer only: {active_layer}")

    # Create the map ONLY after click/state activation.
    # This avoids mounting a heavy iframe automatically after workflow results.
    m = geemap.Map(center=center, zoom=zoom)
    m.add_basemap("HYBRID")
    m.addLayer(ee_object, vis, layer_name)

    # Render exactly one selected layer.
    m.to_streamlit(height=height)


def build_layer_options(**layers: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Small convenience wrapper for readable layer dict construction."""
    return dict(layers)
