# Full Research Handoff: Hobart 2018 Flood Mapping and Continuous Flood-Risk Intelligence System

**Repository:** `nasif952/Tas_ml`  
**Project theme:** Hobart 2018 Flood, Sentinel-1 SAR flood mapping, validation, flood susceptibility modelling, and future operational flood-risk intelligence for Tasmania.  
**Prepared for:** Future research continuation, conference development, thesis/project writing, supervisor discussion, and product development planning.

---

## 1. Executive Summary

This project aims to develop a flood-mapping and flood-risk intelligence workflow for Tasmania, beginning with the May 2018 Hobart flood event as the first major case study. The initial technical focus is to use Sentinel-1 Synthetic Aperture Radar (SAR) imagery in Google Earth Engine (GEE) to identify flood-affected areas during the May 2018 flood. The broader goal is to convert this retrospective flood-mapping workflow into a continuously updated flood-risk monitoring system.

The long-term vision is not only academic. The project is intended to become a practical decision-support system for organisations such as insurance companies, emergency management agencies, local councils, infrastructure managers, and disaster-risk analysts. A possible commercial direction is to provide continuously updated flood-risk information to companies such as RACT or other insurance providers, allowing them to understand flood exposure, support claims analysis, improve risk pricing, and issue early risk alerts.

The project is currently in the early research and prototyping stage. The main workflow has been conceptually defined, and early data exploration has been conducted. Sentinel-1 SAR is considered suitable because it can detect surface water under cloud conditions and has regular revisit capability. Google Earth Engine is the preferred processing environment because it provides direct access to satellite imagery, scalable cloud processing, and export capability.

However, the main research challenge is validation. The available Tasmanian flood calibration and water-level datasets appear to provide water-level or river-related information rather than clean, field-verified flood extent or flood-depth observations. Some uploaded assets showed mainly river features rather than actual inundation surfaces. This means that accuracy assessment must be carefully designed, and the project must clearly distinguish between flood extent, flood depth, water level, river level, permanent water, and modelled calibration data.

---

## 2. Project Motivation

Floods are high-impact hazards that affect people, buildings, transport networks, insurance portfolios, and emergency services. Hobart and other parts of Tasmania are exposed to flash flooding, river flooding, stormwater overflow, coastal interactions, and terrain-driven runoff. The May 2018 Hobart flood provides a useful case study because it was a significant event and falls within the Sentinel-1 data era.

Traditional flood mapping often depends on field observations, hydrological modelling, aerial imagery, or post-event reports. These sources are valuable but can be delayed, incomplete, expensive, or spatially limited. Satellite remote sensing, especially SAR, offers a way to observe flood extent over large areas even during cloudy or rainy conditions.

The motivation of this project is to move from one-time flood mapping toward continuous flood-risk intelligence. Instead of only asking where flooding occurred after an event, the project aims to eventually answer questions such as:

- Which areas are currently at higher flood risk?
- Which properties or assets are exposed?
- How has flood susceptibility changed over time?
- Can satellite and environmental data provide early warning indicators?
- Can insurers use this information to improve risk assessment?

---

## 3. Core Research Problem

The central research problem is:

> How accurately can Sentinel-1 SAR imagery detect the May 2018 Hobart flood extent, and how can this workflow be extended into a continuous flood-risk monitoring system for Tasmania?

This problem has two connected parts.

First, there is a remote sensing problem: detecting flood extent from SAR backscatter changes. Floodwater typically produces low backscatter in open areas because smooth water surfaces reflect the radar signal away from the satellite. However, SAR flood mapping is complicated by urban areas, vegetation, slopes, layover, shadow, permanent water bodies, and surface roughness.

Second, there is an applied GeoAI problem: building a system that can repeatedly update flood-risk information over time. This requires satellite processing, validation, historical event analysis, environmental predictors, machine learning, uncertainty handling, and potentially integration with insurance or asset-exposure datasets.

---

## 4. Main Research Question

**Main question:**

How can Sentinel-1 SAR data be used to map the May 2018 Hobart flood, validate the result with available Tasmanian flood and water-level resources, and form the basis of a continuous flood-risk monitoring system?

---

## 5. Sub-Questions

1. Which Sentinel-1 pre-flood and flood-period images best represent the May 2018 Hobart flood event?
2. What SAR-based flood detection method is most appropriate for the study area?
3. How should permanent water, rivers, slope effects, and urban artefacts be handled?
4. What reference data is available for validation?
5. Are Tasmanian flood calibration/water-level datasets suitable for accuracy assessment?
6. Can water-level data be converted into useful validation points or constraints?
7. What are the limitations of using SAR in Hobart's terrain and urban environment?
8. What environmental variables should be included in a future flood susceptibility model?
9. How can the workflow be automated for periodic updates?
10. How can the outputs be turned into a practical product for insurance or disaster-risk management?

---

## 6. Intended Outputs

The project should eventually produce the following outputs:

1. A Sentinel-1 SAR flood extent map for the May 2018 Hobart flood.
2. A cleaned and documented GEE script for flood detection.
3. A validation strategy and accuracy assessment report.
4. A data inventory describing all available resources.
5. A limitations report explaining what the data can and cannot prove.
6. A flood susceptibility model for future prediction.
7. A repeatable processing pipeline.
8. A dashboard or prototype interface.
9. A conference abstract or paper.
10. A business-facing product concept for insurance companies.

---

## 7. Study Area

The current study area is Hobart, Tasmania, with the May 2018 flood as the target event. The area is complex for flood mapping because Hobart contains urban surfaces, rivers, steep terrain, coastal areas, and drainage networks. These conditions make flood detection more difficult than in flat rural floodplains.

The study area may later expand to Greater Hobart and eventually Tasmania-wide coverage. A Tasmania-wide product would require consistent preprocessing, terrain correction, hydrological context, and event-specific calibration.

---

## 8. Event Focus: May 2018 Hobart Flood

The May 2018 Hobart flood is the first retrospective case study. The event is suitable because it occurred after the launch of Sentinel-1, meaning SAR imagery is available. The main challenge is selecting suitable pre-event and event-period imagery that captures flood conditions as closely as possible.

The flood detection workflow should use images from before the flood and during or immediately after the flood. The exact dates should be confirmed carefully in GEE using Sentinel-1 image availability, orbit direction, polarisation, pass timing, and spatial coverage.

Important date-selection rules:

- Use a stable pre-flood reference period.
- Avoid comparing images with very different acquisition geometry.
- Prefer same orbit direction where possible.
- Use VV and/or VH polarisation depending on performance.
- Consider multi-date compositing to reduce speckle.

---

## 9. Why Sentinel-1 SAR?

Sentinel-1 SAR is central to the project because:

- It can observe the surface during cloud cover.
- It works during day and night.
- It has regular revisit capability.
- It is available in Google Earth Engine.
- It is widely used in flood mapping studies.
- It can support continuous monitoring.

For flood mapping, SAR can detect water because open water often appears darker than surrounding land. This is because calm water surfaces cause specular reflection, sending the radar signal away from the sensor. However, the interpretation is not always straightforward. Flooded vegetation and urban flooding may not appear dark. In cities, double-bounce effects can sometimes increase backscatter rather than decrease it.

Therefore, SAR flood maps should be treated as modelled flood estimates rather than absolute truth.

---

## 10. Why Google Earth Engine?

Google Earth Engine is useful because it provides:

- Direct Sentinel-1 access.
- Scalable processing.
- Built-in filtering by date, orbit, polarisation, and location.
- Integration with DEM and land-cover datasets.
- Export to Drive or Earth Engine assets.
- Repeatable scripting.

The current project uses GEE as the main processing platform for flood mapping. Python may later be used for machine learning, validation, reporting, and dashboard development.

---

## 11. Current Data Resources

The following resources have been discussed or considered:

### 11.1 Sentinel-1 SAR
Primary dataset for flood detection. The workflow should filter Sentinel-1 GRD imagery by:

- Region of interest
- Date range
- Instrument mode
- Polarisation
- Orbit direction
- Resolution

### 11.2 TSFM Calibration / Water-Level Data
Several Tasmanian flood-related datasets were explored. Some appear to represent water-level or calibration data rather than direct flood-depth or flood-extent surfaces. These may still be valuable, but they need careful interpretation.

### 11.3 Water-Level Assets
A GEE asset path was discussed for a water-level dataset. There were issues where the asset was not found, not accessible, or not recognised as an image. This suggests the asset type must be confirmed before use.

Possible asset types:

- Image
- ImageCollection
- FeatureCollection
- Table
- Vector layer
- Raster with limited bands

### 11.4 Georeferenced Flood Image
A flood image was discussed for georeferencing and point extraction. This can support approximate validation, but it should not be treated as equivalent to official field-surveyed flood points unless its source and accuracy are known.

### 11.5 Future Datasets
Future modelling should include:

- Digital elevation model
- Slope
- Flow accumulation
- Distance to river
- Drainage density
- Rainfall
- Soil moisture
- Land cover
- Impervious surface
- Historical flood records
- Building footprints
- Road and infrastructure layers
- Insurance exposure or claim data if available

---

## 12. Key Data Issue: Flood Level vs Flood Extent

A major issue identified in the project is the difference between flood level and flood extent.

Flood extent means the spatial area covered by water during a flood. Flood level means water height or stage at certain locations or along a hydrological model. A water-level dataset may not directly show all flooded areas.

This distinction is critical for validation. If the Sentinel-1 output is a flood extent map, then the validation data should ideally be flood extent polygons, observed flood points, or high-confidence inundation footprints. Water-level lines or river gauge values may support contextual validation but may not be enough for pixel-level accuracy assessment.

Therefore, any accuracy assessment must explicitly state what type of reference data is being used.

---

## 13. Current Understanding of What Is Working

The following parts of the project are currently strong:

1. Sentinel-1 is a suitable data source for flood mapping.
2. GEE is a suitable processing platform.
3. The May 2018 Hobart event is within the Sentinel-1 observation period.
4. A before/during SAR comparison workflow is technically feasible.
5. The project has a clear long-term purpose beyond a single map.
6. The insurance and risk-intelligence product direction is realistic as a future application.

---

## 14. Current Understanding of What Is Not Working Yet

The following parts remain unresolved:

1. The validation dataset is not yet confirmed.
2. Some uploaded datasets appear to show rivers rather than flood extents.
3. Some GEE assets produced errors or were not recognised as images.
4. Flood-depth information may not be available.
5. Extracted flood points from images may be approximate and uncertain.
6. Accuracy metrics may be weak if reference data quality is poor.
7. Urban SAR flood detection may produce false positives and false negatives.

---

## 15. Technical Methodology: Flood Mapping Workflow

The initial SAR flood mapping workflow should follow these stages.

### Step 1: Define Region of Interest
Use Hobart or Greater Hobart boundary as the region of interest. The ROI should be simplified only if necessary for processing speed.

### Step 2: Select Pre-Flood SAR Images
Choose Sentinel-1 images before the May 2018 flood. The pre-flood period should represent normal conditions. A multi-date median or mosaic may reduce speckle and noise.

### Step 3: Select Flood-Period SAR Images
Choose Sentinel-1 images during or immediately after the flood. The closer the acquisition is to the flood peak, the better.

### Step 4: Filter by Orbit and Polarisation
Where possible, use consistent orbit direction and polarisation. Mixing different geometries can create false change signals.

### Step 5: Preprocess SAR
GEE Sentinel-1 GRD data is already partially processed, but additional steps may include:

- Speckle filtering
- Terrain masking
- Slope masking
- Conversion to linear or dB scale depending on method
- Clipping to ROI

### Step 6: Detect Change
Common approaches:

- Difference: flood image minus pre-flood image
- Ratio: flood image divided by pre-flood image
- Thresholding of flood-period backscatter
- Combined threshold and change approach

### Step 7: Remove Permanent Water
Use a permanent water dataset to avoid classifying rivers, lakes, and coastal water as new floodwater unless the objective includes all water presence.

### Step 8: Remove Terrain Artefacts
Mask steep slopes and areas affected by radar shadow or layover. This is important in Hobart because terrain can create SAR artefacts.

### Step 9: Clean Classification
Apply connected-pixel filtering or minimum mapping unit to remove isolated noisy pixels.

### Step 10: Export Flood Map
Export outputs as:

- GeoTIFF
- GEE asset
- Vector polygon
- CSV summary
- Map visualisation

---

## 16. Validation Strategy

Validation is one of the most important and difficult parts of the project. The project should not claim high accuracy unless the reference data supports it.

Potential validation approaches:

### 16.1 Point-Based Validation
Use known flooded and non-flooded points. These points can come from field reports, official observations, manually interpreted imagery, or trusted flood datasets.

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### 16.2 Polygon-Based Validation
Compare SAR flood extent with official flood extent polygons.

Metrics:

- Intersection over Union
- Overlap percentage
- Commission error
- Omission error
- Area difference

### 16.3 Water-Level Assisted Validation
If only water-level data is available, it can be used indirectly. For example:

- Confirm whether detected flooding occurs near expected water-level zones.
- Compare flood detection near river corridors.
- Use water-level points as supporting evidence, not complete ground truth.

### 16.4 Visual Validation
Use high-resolution basemaps, news images, reports, or georeferenced flood images. This is useful but should be clearly described as qualitative or semi-quantitative validation.

---

## 17. Accuracy Assessment Warning

The project must avoid overstating accuracy. If validation points are manually extracted from an uncertain image, the result should be described as approximate. If water-level data is used, the report should say that it is not a direct flood extent reference.

A strong statement would be:

> The SAR-derived flood map was evaluated against the best available reference information; however, the absence of field-verified flood extent points limits the certainty of the accuracy assessment.

This protects the research from making unsupported claims.

---

## 18. Main Limitations

### 18.1 Reference Data Limitation
The biggest limitation is the lack of confirmed flood extent or flood-depth validation points.

### 18.2 SAR Urban Limitation
Urban floodwater may be hidden by buildings, roads, radar geometry, and double-bounce effects.

### 18.3 Terrain Limitation
Hobart's terrain can cause radar shadow, layover, and slope-related backscatter changes.

### 18.4 Timing Limitation
Sentinel-1 may not capture the exact flood peak. If the satellite passed before or after maximum inundation, flood extent may be underestimated.

### 18.5 Permanent Water Confusion
Rivers and coastal water can be confused with floodwater if not masked properly.

### 18.6 Threshold Sensitivity
Flood classification can change significantly depending on the threshold selected.

---

## 19. Future Flood Susceptibility Modelling

After flood extent mapping, the next stage is flood susceptibility modelling. This would shift the project from event mapping to predictive risk modelling.

Potential predictor variables:

- Elevation
- Slope
- Aspect
- Curvature
- Topographic wetness index
- Distance to river
- Drainage density
- Flow accumulation
- Rainfall
- Soil moisture
- Land cover
- Impervious surface
- Historical flood frequency
- SAR-derived wetness indicators

Possible models:

- Random Forest
- XGBoost
- Gradient Boosting
- Support Vector Machine
- Logistic Regression
- Neural Networks
- Ensemble models

Model outputs:

- Flood susceptibility classes
- Probability map
- Risk score per grid cell
- Risk score per property or asset

---

## 20. Continuous Monitoring Concept

The long-term project aims to update flood risk continuously or periodically. Sentinel-1 has regular revisit capability, so SAR-derived indicators can be refreshed as new imagery becomes available.

A future update cycle could look like this:

1. New Sentinel-1 image arrives.
2. System checks rainfall and hydrological conditions.
3. SAR preprocessing runs automatically.
4. Flood-water probability is calculated.
5. Flood-risk map is updated.
6. Alerts are generated for high-risk zones.
7. Dashboard updates for users.

This is the key transition from academic flood mapping to operational flood intelligence.

---

## 21. Proposed System Architecture

A future system could include:

### Data Layer
- Sentinel-1 SAR
- Rainfall data
- DEM
- River network
- Land cover
- Building footprints
- Historical flood records

### Processing Layer
- Google Earth Engine scripts
- Python processing notebooks
- ML model training
- Validation scripts

### Storage Layer
- GeoTIFF outputs
- Vector flood polygons
- PostgreSQL/PostGIS database
- Cloud storage

### Intelligence Layer
- Flood detection model
- Susceptibility model
- Risk scoring model
- Alert logic

### Application Layer
- Web dashboard
- API
- Insurance report export
- Map viewer
- Automated notifications

---

## 22. Insurance Use Case

Insurance companies may benefit from flood intelligence in several ways:

1. Exposure analysis: identifying which insured properties are in flood-prone areas.
2. Claims validation: comparing claims locations with satellite-derived flood extent.
3. Risk pricing: improving premium calculation based on dynamic flood risk.
4. Portfolio management: understanding regional concentration of flood exposure.
5. Early warning: notifying customers or internal teams before or during flood conditions.
6. Post-event assessment: rapidly estimating affected areas after an event.

The product should not initially promise exact property-level flood depth unless reliable depth modelling is added. A more realistic early product is flood exposure and flood likelihood mapping.

---

## 23. Research-to-Product Roadmap

### Phase 1: Retrospective Flood Mapping
- Map Hobart 2018 flood using Sentinel-1.
- Produce initial flood extent map.
- Document method and limitations.

### Phase 2: Validation and Accuracy Assessment
- Identify best available reference data.
- Perform point/polygon/visual validation.
- Report accuracy honestly.

### Phase 3: Susceptibility Modelling
- Create predictor dataset.
- Train ML model.
- Produce flood susceptibility map.

### Phase 4: Automation
- Automate Sentinel-1 ingestion.
- Schedule periodic updates.
- Generate automatic outputs.

### Phase 5: Dashboard Prototype
- Build map dashboard.
- Show risk layers and flood extent.
- Add export reports.

### Phase 6: Commercial Pilot
- Prepare demo for insurance or council users.
- Test with sample exposure data.
- Refine product based on user needs.

---

## 24. Conference or Paper Angle

A possible conference title:

**Sentinel-1 SAR-Based Mapping of the 2018 Hobart Flood as a Foundation for Continuous Flood-Risk Intelligence in Tasmania**

Possible abstract direction:

This study investigates the use of Sentinel-1 SAR imagery for mapping the May 2018 Hobart flood and explores how retrospective flood mapping can support future continuous flood-risk monitoring. A Google Earth Engine workflow is developed to compare pre-flood and flood-period SAR backscatter, classify potential inundation, and evaluate results using available Tasmanian flood calibration and water-level resources. The study highlights both the value and limitations of SAR-based flood mapping in complex urban and terrain settings. The findings provide a foundation for integrating SAR-derived flood indicators with terrain, rainfall, and hydrological variables for future flood susceptibility modelling and operational risk intelligence.

---

## 25. What Must Be Done Before Strong Claims

Before the project claims high accuracy or operational readiness, the following must be completed:

1. Confirm exact Sentinel-1 dates used.
2. Confirm asset type and meaning of all TSFM datasets.
3. Identify reliable validation points or polygons.
4. Separate permanent water from floodwater.
5. Test multiple thresholds.
6. Quantify uncertainty.
7. Document false positives and false negatives.
8. Compare results with known event reports.
9. Produce reproducible scripts.
10. Create clear figures and maps.

---

## 26. Recommended Repository Structure

Recommended future structure:

```text
Tas_ml/
├── README.md
├── research/
│   ├── 00_project_handoff.md
│   ├── 01_research_questions.md
│   ├── 02_literature_review.md
│   ├── 03_data_inventory.md
│   ├── 04_methodology.md
│   ├── 05_validation_strategy.md
│   ├── 06_results_log.md
│   ├── 07_limitations.md
│   ├── 08_future_work.md
│   ├── 09_business_case.md
│   └── 10_full_research_handoff.md
├── gee/
│   ├── flood_mapping_v1.js
│   ├── validation_v1.js
│   └── susceptibility_v1.js
├── notebooks/
│   ├── validation_analysis.ipynb
│   └── susceptibility_model.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── outputs/
│   ├── maps/
│   ├── figures/
│   └── tables/
└── docs/
    ├── presentation/
    └── reports/
```

---

## 27. Immediate Action Checklist

- [ ] Add final GEE flood mapping script.
- [ ] Confirm Sentinel-1 image dates.
- [ ] Confirm ROI boundary.
- [ ] Load and inspect TSFM datasets.
- [ ] Determine whether each dataset is raster or vector.
- [ ] Identify flood extent, flood depth, water level, or river-only content.
- [ ] Create first flood extent map.
- [ ] Export map from GEE.
- [ ] Generate candidate validation points.
- [ ] Calculate first confusion matrix if possible.
- [ ] Write limitations clearly.
- [ ] Prepare conference-style results.

---

## 28. Key Wording for Reports

Useful wording:

> This project uses Sentinel-1 SAR imagery to detect flood-related surface water changes during the May 2018 Hobart flood. The study develops a Google Earth Engine workflow for retrospective flood mapping and investigates the feasibility of extending the workflow into an operational flood-risk monitoring system.

> The validation stage is constrained by the availability and suitability of reference data. Some available Tasmanian flood resources appear to represent water-level or calibration information rather than direct field-observed flood extent. Therefore, accuracy assessment must be interpreted cautiously.

> The long-term aim is to integrate SAR-derived flood observations with environmental variables and machine learning to produce dynamic flood susceptibility and risk information for Tasmania.

---

## 29. Final Handoff Summary

This project should be understood as a staged research-to-product workflow. The first stage is not to build the full product immediately, but to prove that Sentinel-1 SAR can reasonably detect the May 2018 Hobart flood. The second stage is to validate this flood map using the best available reference data, while being transparent about uncertainty. The third stage is to use this mapped flood information to train or support a flood susceptibility model. The fourth stage is to automate the workflow so it can update over time. The fifth stage is to turn the system into a practical flood-risk intelligence product.

The biggest strength of the project is its clear applied direction: it connects remote sensing, machine learning, disaster risk, and insurance use cases. The biggest weakness at the moment is validation data uncertainty. The project should therefore prioritise data inventory, validation strategy, and transparent limitation reporting before making strong claims.

If developed carefully, this project can support academic outputs, supervisor discussions, PhD/research applications, conference presentations, and a future commercial flood-risk monitoring platform for Tasmania.
