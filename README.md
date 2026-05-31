# Tas_ml

Research repository for the Hobart 2018 Flood and Tasmania flood-risk monitoring project.

## Current project direction

This project builds a remote-sensing and machine-learning workflow for flood mapping and flood susceptibility modelling in Hobart / southern Tasmania.

Current workflow:

- Sentinel-1 SAR VV imagery for water-like flood detection.
- Sentinel-2 NDWI to mask pre-existing surface water.
- DEM, slope, simple TWI, NDVI, NDBI, rainfall, event rainfall, temperature, river distance, and soil moisture as Random Forest predictors.
- GEE `Study_Area` polygon asset instead of a rectangular AOI.
- Council boundary layers converted to WGS84 for GEE upload.
- Building centroid screening inside high final-overlay zones.

## Main research idea

The project starts from the May 2018 Hobart flood and is intended to grow into a regularly updated flood-risk intelligence workflow for Tasmania.

## Key caution

The current flood label is an NDWI-masked SAR-derived flood map, not field-verified ground truth. Accuracy and variable importance must be interpreted as model behaviour relative to the SAR-derived label.

## Repository structure

- `gee/` — Google Earth Engine JavaScript code.
- `research/` — research handoff, methodology notes, validation notes, data inventory, and roadmap.

## Important research files

- `research/10_full_research_handoff.md`
- `research/11_recent_progress_update.md`
- `research/12_current_gee_workflow.md`
- `research/13_data_inventory_assets.md`
- `research/14_validation_and_interpretation_notes.md`
- `research/15_next_research_steps.md`
