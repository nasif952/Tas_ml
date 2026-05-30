# Methods

## Data Sources
- Sentinel-1 SAR
- Tasmania flood calibration datasets
- Tasmania water-level datasets
- DEM and terrain variables (future)
- Rainfall and hydrological variables (future)

## Flood Mapping Workflow
1. Select pre-flood Sentinel-1 image.
2. Select flood-period Sentinel-1 image.
3. Apply preprocessing.
4. Calculate backscatter change.
5. Extract flood extent.
6. Remove permanent water where appropriate.
7. Export results.

## Validation Workflow
1. Prepare reference data.
2. Convert reference observations into comparable format.
3. Compare flood extent with observations.
4. Calculate accuracy metrics.

## Future ML Workflow
Flood susceptibility modelling using:
- SAR indicators
- DEM
- Slope
- Rainfall
- Distance to river
- Land cover
- Additional environmental predictors
