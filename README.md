# Tree Species Classification from Sentinel-2 Time Series

Master's thesis project ("Puuliikide tuvastamine Eesti metsades Sentinel-2
satelliidipiltide aegridadest", University of Tartu): classifying the
dominant tree species of Estonian forest stands — using one forestry
sub-region ("Subregion 2") as a case study — from dense Sentinel-2 satellite
time series (Random Forest vs. XGBoost, plain and combined with canopy
height and soil data), plus a secondary stand-composition regression task
(predicting each species' % share, not just the dominant one) and a
pixel-based (10 m) species map built from the winning classifier.

**This is a curated, minimal reproduction package** — the ~76-experiment
exploratory history has been pruned down to only what the current thesis
text actually cites (see "What was removed and why" below). Raw geodata,
trained models, and generated outputs are not included (large, and/or
license-restricted); this README tells you how to regenerate them.

## Reproduction order

Run these in order. Each step's script(s) and output are listed; runtimes
are order-of-magnitude estimates from this project's own history, not
formal benchmarks.

### 1. Environment

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (see below)
```

Python 3.14 was used for this pruned repo's own verification pass; the
original project ran on Python 3.12 for most of its history. Package
versions in `requirements.txt` are pinned to what was verified working —
see the comment at the top of that file.

`configs/paths.yaml` defines every data/output directory as a path
relative to the repo root, read by `src/paths.py` (`PATHS.raw`,
`PATHS.processed`, `PATHS.intermediate`, `PATHS.reference`,
`PATHS.models_trained`, `PATHS.outputs_tables`, etc.). No script hardcodes
an absolute path — populate the directories `configs/paths.yaml` names and
everything resolves.

**Copernicus Data Space Ecosystem (CDSE) credentials**: the Sentinel-2
downloads use CDSE's OAuth2 client-credentials flow. Register an OAuth
client at https://dataspace.copernicus.eu/ (Dashboard → OAuth clients) and
put the client ID/secret in `.env`. Never commit `.env` (it's in
`.gitignore`).

### 2. Reference data (not included — see licensing note)

Place these under `data/raw/` (paths per `configs/paths.yaml`):

| File | Role | Source |
|---|---|---|
| `puuliik_filt_pindala/puuliik_filt_pindala2.shp` | Labeled stand polygons, dominant-species share ≥ 80% already applied | Estonian Forest Register (Metsaregister) export — **not redistributable**, request your own extract |
| `puuliik_pindala_SR2.shp` | Subregion-2 stand polygons, used for the clearcut-overlap check | Same source |
| `Subregions/*.shp` | 8 forestry sub-region boundaries (Lang et al., 2018) | Supplied by thesis supervisor / Estonian Environment Agency |
| `H_20_21_22_23.tif` | Multi-year elevation/canopy-height raster, EPSG:3301 | Estonian Environment Agency (Keskkonnaagentuur) |
| `soil_siffer_10m.tif` | Rasterized national soil-code map, EPSG:3301 | Estonian Land Board (Maa-amet) |
| `metsamask.tif` | National forest mask, used to restrict pixel-map prediction to forest pixels | Estonian Environment Agency |

`data/external/` (see its own `README.md`) additionally needs a national
clearcut-polygon shapefile (`EestiLRjaan2025_kuni_veebr2026/`, Mar
2023–Feb 2026 coverage) used only to filter clearcut-contaminated training
stands out.

None of the above are included in this repo: the Forest Register data is
licensed/access-controlled, and the rasters are large. This is also why
the actual train/val/test **parcel ID lists are not included here** — they
are a derived slice of licensed Forest Register data. Regenerating the
split (`scripts/split.py` + `scripts/rename_classes.py`, step 3 below)
reproduces the same IDs deterministically from the same source shapefile
(`random_state=42` throughout — see "Random seeds" below), so the parcel
identifiers are fully reproducible, just not distributed as a static file.

### 3. Build the canonical train/val/test split

```bash
python scripts/split.py            # -> data/processed/final_split.gpkg
python scripts/rename_classes.py   # -> data/processed/final_split_clean.gpkg
```

`split.py` groups stands into a 5×5 km grid by centroid and applies
`sklearn.model_selection.GroupShuffleSplit` twice (70/30 train/temp, then
50/50 val/test on the remainder) so that spatially adjacent stands can't
end up split across train and test — mitigating (not eliminating; see
thesis §5.5) spatial autocorrelation leakage. `rename_classes.py` adds the
final `target` label column, collapsing everything outside the six main
species (MA, KS, KU, LV, HB, LM) into `MUU`.

**Runtime**: seconds (no downloads yet, just a spatial join and split).

### 4. Filter clearcut-contaminated stands

```bash
python scripts/clearcut_overlap.py
# -> data/reference/clearcut_overlap/stands_overlapping_clearcuts_over20pct.csv
```

Flags stands whose area overlaps >20% with a logged clearcut polygon
(Mar 2023–Feb 2026 coverage) — these are excluded from training everywhere
via `src/preprocessing/clearcut.py::remove_clearcut_ids()`, called at
data-load time in every experiment script (not baked into a separate
pre-filtered split file).

### 5. Download Sentinel-2 time series

```bash
python scripts/download_sentinel2_10day_statistical.py   # edit SELECT_SPLIT at the
                                                            # top: "train", "val", "test"
                                                            # in turn -> 3 runs
python scripts/download_sentinel2_daily_statistical.py    # same pattern, daily resolution
                                                            # (needed for the extended/
                                                            # mixed-stand dataset, step 7)
```

Uses the Sentinel Hub **Statistical API** (10-day aggregated stats, one
request per stand polygon — this is why per-pixel raster tiles aren't
needed at this stage) against the CDSE endpoint
(`https://sh.dataspace.copernicus.eu`), full 2025 calendar year, `"P10D"`
(10-day) aggregation periods for the main dataset. The exact evalscript
(`scripts/download_sentinel2_10day_statistical.py`, `EVALSCRIPT` constant):

```javascript
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "dataMask"],
      units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE",
              "REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN","DN"]
    }],
    output: [
      { id: "bands", bands: 10, sampleType: "FLOAT32" },
      { id: "indices", bands: 6, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}
function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);
  let evi = 2.5 * ((sample.B08 - sample.B04) / (sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0 + 1e-6));
  let cire = (sample.B08 / (sample.B05 + 1e-6)) - 1.0;
  if (good) {
    return { bands: [sample.B02,sample.B03,sample.B04,sample.B05,sample.B06,sample.B07,sample.B08,sample.B8A,sample.B11,sample.B12],
             indices: [ndvi, ndre, ndmi, nbr, evi, cire], dataMask: [1] };
  } else {
    return { bands: [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN], indices: [NaN,NaN,NaN,NaN,NaN,NaN], dataMask: [0] };
  }
}
```

Cloud/shadow rejection is via Scene Classification Layer (`SCL` band):
only pixels classified vegetation/bare-soil/water (`SCL` 4/5/6) count as
"good"; everything else (cloud, cloud shadow, snow, etc.) is masked to
`NaN` and excluded from the per-stand statistical aggregation. There is no
separate `maxCloudCoverage` request parameter for this Statistical-API
path — filtering happens per-pixel inside the evalscript itself.

**Runtime**: this is the slow step — CDSE enforces per-account processing
unit quotas; downloading the full Subregion-2 training set historically
required switching between multiple CDSE accounts and took on the order of
hours to low-single-digit days depending on quota resets, not a fixed
number benchmarked here.

### 6. Add height and soil features

```bash
python experiments/EXP027_subregion2_daily_stats/scripts/01_add_height_soil_features.py
# -> experiments/EXP027_subregion2_daily_stats/tables/subregion2_parcel_height_soil.csv
```

Zonal statistics (mean, median, std, min, max) of the elevation raster and
majority soil code per stand polygon. Consumed by `EXP032`, `EXP038`, and
the controlled ablation study's `AUX_FILE`.

### 7. Pure-stand classification: the two final models

```bash
python experiments/EXP032_deciduous_further_split/exp032.py   # best Random Forest
python experiments/EXP038_xgboost_hierarchical/exp038.py       # best XGBoost — thesis's
                                                                 # primary/final result
```

Both use the identical 4-stage hierarchical architecture (conifer/deciduous
+ MUU → species within each group → grey alder split off from the
remaining deciduous species) and the same 18 hand-built species-signature
features (see "Feature definitions" below) — only the classifier family
differs. Reported test-set result (`EXP038`): Accuracy 0.8695 / Macro F1
0.692 / Kappa 0.8203, outperforming `EXP032`'s Random Forest on every
metric. See `experiments/EXP041_xgboost_tuning_hierarchical/` and
`experiments/EXP042_rf_tuning_hierarchical/` for the hyperparameter grid
searches behind each model's configuration (thesis Lisa 2.2, tables L2.2/L2.3).

**Runtime**: minutes, once step 6's data exists — no downloads at this
stage.

### 8. Controlled ablation study (Table 4.9)

```bash
cd experiments/controlled_ablation_study_2026_08
python scripts/step1_baseline_sentinel2_only.py
python scripts/step2_height_soil_ablation.py
python scripts/step3_feature_representation.py
python scripts/step3b_topN_selection_method_check.py
python scripts/step4_class_balancing.py
python scripts/step5_rf_hyperparameter_search.py
python scripts/step6a_architecture_flat_vs_4stage.py
python scripts/step6b_architecture_2stage_rf.py
python scripts/step6c_architecture_2stage_xgboost.py
python scripts/step7_xgboost_hyperparameter_search.py
python scripts/step8_final_test_evaluation.py       # the ONE test-set touch for this pipeline
python scripts/peelback_ablation_from_exp038.py
python scripts/peelback_final_test_b1_candidate.py  # the ONE test-set touch for this direction
```

See that folder's own `README.md` for what each step isolates and how it
maps to Table 4.9's rows — this replaces an earlier, methodologically
confounded ~80-experiment series a thesis reviewer flagged (multiple
variables changed per comparison; three early experiments accidentally run
on a non-canonical dataset). Every step here changes exactly one variable
and is evaluated on validation only, except the two scripts explicitly
marked as the pipeline's single test-set evaluation.

**Runtime**: minutes per script; the hyperparameter grid searches
(`step5`, `step7`) are the slowest, still well under an hour each on a
single machine.

### 9. Extended (pure + mixed stand) dataset and composition regression

```bash
python experiments/EXP047_composition_daily_statistical/download_daily.py   # downloads the
                                                                              # extended dataset
                                                                              # (pure ≥80% +
                                                                              # mixed 20-79% stands)
python experiments/EXP048_composition_rf/exp048.py     # RF, pure+mixed classification
python experiments/EXP049_composition_xgb/exp049.py    # XGBoost, pure+mixed classification
python experiments/EXP050_composition_regression/exp050.py    # RF, species-share regression
python experiments/EXP076_xgboost_composition_regression/exp076.py  # XGBoost, same task
```

Composition (species-share) targets come from the Forest Register's own
recorded species mix per stand (up to 5 species + % share each),
renormalized to sum to exactly 100%. Both regressors are independent
per-species models (`RandomForestRegressor` natively multi-output;
XGBoost wrapped in `sklearn.multioutput.MultiOutputRegressor`, since it has
no native multi-output support) with post-hoc renormalization of predicted
shares to sum to 100% — a deliberate simplicity/pipeline-consistency choice
over dedicated compositional-data methods (log-ratio transforms,
simplex-constrained models); see thesis ch. 2.4 for that reasoning and
ch. 5.5 for the acknowledged limitation.

### 10. Pixel-based species map

```bash
python scripts/download_pixel_stack_subregion2_pilot.py    # one 25x25km pilot tile
python scripts/predict_pixel_species_map_exp032.py         # RF pilot-tile map
python scripts/predict_pixel_species_map_exp038.py         # XGBoost pilot-tile map
python scripts/build_subregion2_species_map.py             # RF, remaining tiles + mosaic
python scripts/build_subregion2_species_map_xgb.py         # XGBoost, remaining tiles + mosaic
python scripts/validate_pixel_map_against_parcels.py       # thesis Tables 4.11/4.12
```

Note the mosaic step (`build_subregion2_species_map*.py`) depends on the
pilot-tile script's output already existing on disk
(`outputs/predictions/pixel_species_map_exp03{2,8}_pilot/`) — run the pilot
scripts first. Height/soil features at pixel resolution use a genuine
~1 ha spatial moving window (`scipy.ndimage` filters) matching the
smallest stand size in training data; Sentinel-2 spectral features use
each pixel's own raw value directly (no further spatial aggregation, since
the model doesn't use separate min/max/median spectral statistics); the
soil feature degenerates to the pixel's own soil code (no "majority"
neighborhood at pixel resolution) — see thesis ch. 4.11/5.5 for the full
discussion of this scale mismatch.

**Runtime**: the full Subregion 2 map (283,360 ha, per-tile Sentinel-2
downloads across the whole growing season) took multiple hours spread
across several days in this project's own history, due to CDSE per-account
processing-unit quota limits requiring periodic waits/account switches —
not a controlled benchmark.

## Random seeds

`random_state=42` is used consistently — verified by grep across every
kept experiment and `scripts/split.py` (66 occurrences, zero exceptions) —
for the spatial split, all `RandomForestClassifier`/`RandomForestRegressor`
instantiations, all `XGBClassifier`/`XGBRegressor` instantiations, and the
manual oversampling step (`src/models/sampling.py`) where used.

## Feature definitions

- **Sentinel-2 spectral features actually used by the final models**
  (`EXP032`/`EXP038`, `TS_FEATURE_COLS`): 10 reflectance band means
  (B02–B12, excl. B01/B09/B10) + 4 of the 6 vegetation indices the
  evalscript computes (NDVI, NDRE, NDMI, NBR — EVI and CIRE are downloaded
  in step 5 but not used as model inputs), aggregated into 26 nominal
  10-day windows across the 2025 growing season (one further window has no
  valid observation for any stand and is dropped entirely —
  `align_columns()`/`drop_all_nan_train_columns()` in
  `src/preprocessing/timeseries.py` handle this) — 14 × 26 = 364 features.
- **Height/soil auxiliary features**: `height_mean`, `height_median`,
  `height_std`, `height_min`, `height_max` (zonal stats over the elevation
  raster) + `soil_majority` (majority soil code, one-hot encoded — 357
  categories after clearcut filtering).
- **18 hand-built species-signature features**: defined identically (byte-
  near-identical) in `add_species_signature_features()` inside
  `experiments/EXP021_species_signature_features/exp021.py`,
  `EXP032_deciduous_further_split/exp032.py`, and
  `EXP038_xgboost_hierarchical/exp038.py` — domain-motivated combinations
  of specific bands/indices at specific dates per species (e.g. an aspen
  spring+autumn index, a grey-alder July SWIR index, a black-alder height
  index), derived from a statistical analysis of class separability
  (Cohen's d, ANOVA F-statistic, Kruskal-Wallis on height) run on
  train+val+test combined — see thesis ch. 3.7/5.5 for why that
  combined-split analysis is a disclosed methodological limitation, not a
  train/test leakage issue in the strict sense (no model was trained or
  evaluated on test during that analysis).
- **Controlled ablation study's winning Top-50 feature set**: not a fixed
  literal list — re-derived at the top of each `controlled_ablation_study_2026_08`
  script via `RandomForestClassifier.feature_importances_` on the training
  split, deterministically (`random_state=42`), so it doesn't need to be
  distributed as a static file either.

## Model files

Not included — `models/trained/*.joblib` bundles (fitted model + imputer +
feature list, one per experiment) are regenerated by running the scripts
above, not distributed (they're several MB each and trivially
reproducible given `random_state=42` and the same input data).

## What was removed and why

This started as a ~76-experiment exploratory project. Kept: everything the
current thesis text actually cites — `EXP000` (baseline, thesis Table
4.1/4.2, Figure 4.1), `EXP021` (species-signature features, referenced in
Table 4.9's discussion), `EXP027` (the height/soil + statistical-analysis
dependency both final models load), `EXP032`/`EXP038` (final RF/XGBoost
models), `EXP041`/`EXP042` (their hyperparameter searches), `EXP047`
through `EXP050` and `EXP076` (extended-dataset classification and
composition regression), plus the new `controlled_ablation_study_2026_08/`
(Table 4.9). Everything else — ~65 experiment folders covering rejected
feature ideas, an AutoML/FLAML comparison, a pixel-level CNN, dozens of
minimal-feature-set variants, and other exploratory dead ends — was
removed, along with a handful of `scripts/` utilities nothing in the kept
set actually calls (a raster-cleaning one-off, a superseded prediction
prototype, an unused split-rebuild variant). `torch`, `flaml[automl]`,
`scikit-image`, and `seaborn` were dropped from `requirements.txt` for the
same reason — nothing kept imports them.

Also removed: the whole `docs/` folder — `docs/references/` (149 MB of
cited PDFs; citations are in the thesis's own reference list) and the
thesis figures (`docs/diagrams/`, available in the thesis itself) — none of
it is needed to run any code and the ATI
thesis-guide PDF (university-internal grading material, not this project's
own output — cite/link it rather than redistribute it if you need it).

## Known limitations

See thesis ch. 5.5 ("Uuringu piirangud") for the full, honest discussion:
depth of spatial cross-validation (a single 5×5 km grid split, not
repeated across multiple random draws or grid sizes), the compositional-
data modeling choice (independent per-species regressors + renormalization
vs. dedicated simplex-aware methods), the pixel/parcel feature-scale
mismatch in the pixel map (above), and the train+val+test-combined
statistical analysis behind the 18 hand-built signature features (above).
None of these were treated as reasons to withhold results — they're
disclosed, scoped limitations on an otherwise-reproducible pipeline.
