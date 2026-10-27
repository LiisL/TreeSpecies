# EXP027 - Subregion2 daily-data statistics, height/soil features, spectral signatures

Descriptive/EDA work for the thesis, built on the Subregion2 **daily** Sentinel-2
time series (`data/intermediate/Subregion2_{train,val,test}_sentinel2_daily_timeseries.csv`)
rather than the 10-day-window model input. Not a model-training experiment
(no RF here) - kept in `analysis/` rather than `experiments/`, numbered EXP027
to stay in sequence with the experiment log.

## Scripts (run in order)

1. **`scripts/01_add_height_soil_features.py`**
   Computes per-parcel height statistics (mean, median, std, min, max - from
   `data/raw/H_20_21_22_23.tif`) and a soil "siffer" majority code (from
   `data/raw/soil_siffer_10m.tif`) via `rasterstats.zonal_stats`, for every
   Subregion2 parcel in `data/processed/final_split_clean.gpkg` (confirmed to
   match the daily CSVs' `id` column: 1613/1613 Subregion2 test IDs found).
   Writes:
   - `tables/subregion2_parcel_height_soil.csv` - one row per parcel
   - `data/processed/Subregion2_{split}_sentinel2_daily_timeseries_with_height_soil.csv`
     (gitignored, same convention as the rest of `data/processed/`) - each
     original daily CSV with the height/soil columns broadcast onto every row

   **Gotcha hit and fixed**: `rasterstats.zonal_stats` does not return its
   result dict keys in the order the `stats=[...]` list was given - it has its
   own fixed internal order. Renaming columns positionally (`df.columns = [...]`)
   silently scrambled height_mean/median/min/max/count on the first pass
   (caught because `height_mean` came out ~15m lower than `height_median` for
   no physical reason). Fixed by renaming via a `{raw_key: new_name}` dict
   instead - order-independent by construction.

2. **`scripts/02_split_species_stats.py`**
   Parcel counts per split (train/val/test) and per split x main species
   (target class). Writes `tables/subregion2_split_species_counts.csv` and two
   figures (Estonian-labelled, per this repo's convention that figure-facing
   text stays Estonian):
   - `figures/subregion2_parcels_per_split.png`
   - `figures/subregion2_species_by_split.png`

3. **`scripts/03_spectral_signature_curves.py`**
   Spectral signature curves per species per band/index across the 2025
   season, using the **raw daily observation dates** (no 10-day binning - a
   deliberate choice, since this is meant to show the true daily resolution,
   distinct from EXP021's model-input grid). For each (date, species) the
   value plotted is the **median across parcels of the parcel-level spatial
   median** (`<band>_median` columns) - median-of-medians, chosen over mean
   because a raw daily dataset is full of exactly the kind of per-parcel
   outliers (partial cloud/shadow, mixed edge pixels, occasional mislabeled
   stands) a median resists and a mean doesn't. `MUU` (catch-all "other") is
   excluded - not a coherent species signature. Two complementary views:

   - **By band** (panel = band, lines = species) - "how do species differ on
     this one band, through the year?"
     - `figures/spectral_signature_reflectance_bands.png` (B2-B12, 10 panels)
     - `figures/spectral_signature_vegetation_indices.png` (NDVI/NDRE/NDMI/NBR/EVI/CIRE)
   - **By species** (panel = species, lines = bands) - "how does this one
     species move across all its bands together, through the year?" Since
     the bands sit on very different raw scales (reflectance ~0-0.5, EVI up
     to ~1.6, CIRE up to ~6), each line is min-max normalized to 0-1 *within
     that species' own year* before plotting - the normalized axis is for
     comparing shape/timing across bands, not absolute magnitude (use the
     "by band" figures for that). Reflectance bands originally used a
     sequential ramp ordered by wavelength, but 10 shades of one hue were
     too close to tell apart in a 10-line overlay - switched to 10 distinct
     categorical colors (readability over strict physical-ordering
     encoding); vegetation indices already used fixed categorical hues.
     - `figures/spectral_signature_by_species_reflectance.png`
     - `figures/spectral_signature_by_species_indices.png`

   Writes `tables/subregion2_species_band_daily_signature.csv` (shared source
   table for all four figures).

   Curves show a plausible phenology (evergreen `mänd`/pine holding higher
   winter NDVI/NIR than the deciduous species, spring green-up from ~May,
   autumn senescence from ~October) but are visibly noisy at the very start
   of January/February - real, not a bug: only 1-2 valid dates exist that
   early in the year (heavy winter cloud cover), so the per-date sample of
   parcels is tiny and easily swung by a handful of snow-contaminated pixels.
   The by-species view also shows `mänd` (evergreen) keeping its bands much
   closer together year-round than the deciduous species, which fan out
   sharply in summer - a visible signature of the evergreen/deciduous split.

4. **`scripts/04_spectral_separability_analysis.py`**
   Class separability analysis - NOT feature engineering (that constructs new
   derived variables; see `experiments/EXP028_daily_full_features_height_soil/
   signature_feature_engineering.py`). This is univariate feature *selection*/
   statistical class-separability analysis: for every (band/index, date)
   column, how well does it discriminate between the six main species,
   independent of any trained model? Two classic, model-free tools:
   - **ANOVA F-statistic** (`sklearn.feature_selection.f_classif`) per column
     across all 6 species at once - complements (doesn't duplicate) the RF
     feature importance from EXP028 step 2, which is entangled with tree-
     splitting behaviour; this is a clean, independent per-variable statistic.
   - **Pairwise Cohen's d** (standardized mean difference) between every
     species pair, per column - which band/date best tells two *specific*
     species apart, connecting directly to confusion-matrix patterns already
     seen throughout EXP028.

   Combines all three splits (max statistical power, same reasoning as
   script 03). **Reliability guard, found and fixed while building this**:
   for sparse dates, a whole species group can have zero real observations
   for a column, so after median-imputation every parcel in that group gets
   the identical value, std becomes exactly 0, and Cohen's d divides by
   zero -> `inf`. Any column where a relevant species has fewer than 20 real
   (non-imputed) observations is excluded from rankings/figures (kept in the
   saved CSV, flagged via `min_real_n_across_species`) - an artifact of
   imputation is not a real separability finding.

   Outputs:
   - `tables/subregion2_separability_by_date_band.csv` (every column, ranked)
   - `tables/subregion2_pairwise_separability.csv` (best discriminator per species pair)
   - `figures/separability_reflectance_bands.png` / `separability_vegetation_indices.png`
     (F-statistic over the year, same panel-per-band layout as script 03,
     mean + median both plotted - they come out visually near-identical,
     because many small Estonian forest stands cover so few Sentinel-2
     pixels that spatial mean and median collapse to the same value, which
     is also why RF importance found them "tied" back in EXP028 step 2 -
     this analysis explains that finding.)
   - `figures/separability_top30_datetime_band.png`
   - `figures/separability_pairwise_heatmap.png`

   **Findings**: mid-June (14.06, 04.06) on B7/B8A/B6 is the single most
   species-discriminating (band, date) combination - independently
   confirming EXP028 step 2's RF-based finding (June most important month,
   B7/B6/B8A top bands) through a completely different statistical method.
   EVI has a striking peak separability in late June/July (F-statistic over
   1200, higher than NBR/NDRE's ~800 peaks) despite ranking lower in step
   2's *aggregate* importance - its useful signal is concentrated in a
   narrow window that aggregate-across-all-dates ranking dilutes, a candidate
   for future signature-feature engineering. Pairwise, **HB-LV is the
   hardest-to-separate species pair** (max achievable |Cohen's d| = 0.73),
   followed by LM-LV (0.76) and HB-KS (0.81) - matching the confusion-matrix
   hot spots (HB repeatedly confused with KS/LV) seen throughout every
   EXP028 model variant.

5. **`scripts/05_one_vs_rest_species_windows.py`**
   One-vs-rest species signature analysis, using **10-day windows** instead
   of single dates - a direct follow-up to a negative result: training an RF
   on the top-30 single-date (band, date) pairs from script 04 collapsed LM's
   recall to 0.08, because 60% of those top single dates clustered in one
   narrow June event and completely missed LM's October signal (see
   `EXPERIMENTS.md`, `top30_separability_features.py`). This asks a different,
   complementary question per species: not "which single date/pair separates
   species most" but **"what makes THIS species distinct from all five others
   combined, and when"** - one-vs-rest Cohen's d per (species, band, 10-day
   window), same reliability guard as script 04 (>=20 real observations per
   group).

   Outputs:
   - `tables/subregion2_one_vs_rest_by_species_window_band.csv` (full ranked table)
   - `tables/subregion2_one_vs_rest_top10_per_species.csv` (top 10 per species)
   - `figures/one_vs_rest_heatmap_grid.png` (6-panel band x 10-day-window heatmap, shared color scale)
   - `figures/one_vs_rest_best_score_per_species.png` (best achievable |Cohen's d| per species, ranked)

   **Findings**: `mänd` (pine) and `lepp` (alder) both peak at the exact same
   window (10 June, B7) but in opposite directions - conifers read lower on
   red-edge/NIR than broadleaf at peak canopy, confirming the reversal found
   in step 5's signature-feature engineering. `haab` (aspen) shows the
   visually broadest one-vs-rest signature (nearly every band elevated
   Feb-May) despite being the worst-performing, most-confused class in every
   EXP028 model - not a contradiction: one-vs-rest measures distinctness from
   the *average* of the other five species, so HB can look different from
   that average while still overlapping heavily with one specific neighbor
   (KS or LV) pairwise, exactly what the confusion matrices already show.
   `kuusk` (spruce) has the flattest heatmap and the lowest best-case score
   (|d|=0.62, vs. 0.91-1.66 for the rest) - visually confirms it lacks a
   strong unique identity, consistent with its middling-but-never-great
   performance throughout every EXP028 model. `sanglepp` (black alder)'s
   single strongest signature is in **November** (NDRE, |d|=1.02), refining
   step 5's October-focused assumption.

6. **`scripts/06_march_ndvi_threshold_and_rf_rules.py`**
   Domain-expert hypothesis check: "if NDVI is high in March, it's most
   likely a conifer" - find the real threshold, and show what threshold(s)
   the already-trained EXP028 Random Forest actually uses. Answers a
   conceptual question directly: a decision tree IS a sequence of threshold
   rules; RF is 400 of them voting. sklearn has no API to hand a fixed rule
   to `RandomForestClassifier`, but the thresholds it already learned can be
   read straight out of the trained model's tree structure.

   Three parts: (1) per-parcel whole-month March NDVI by species (boxplot);
   (2) a single-feature, depth-1 decision tree (the simplest possible
   "threshold rule") separating conifer (KU, MA) from broadleaf (HB, KS, LM,
   LV) - finds the empirically optimal threshold; (3) every March-NDVI split
   threshold actually used across all 400 trees of the trained "everything"
   model (`models/trained/EXP028_rf_daily_full_features_height_soil.joblib`
   - the only saved model with individual March-NDVI columns in its feature
   set).

   Outputs:
   - `tables/subregion2_march_ndvi_by_species.csv`
   - `figures/march_ndvi_by_species_boxplot.png`
   - `figures/march_ndvi_threshold_stump.png`
   - `figures/march_ndvi_rf_actual_thresholds.png`

   **Findings**: the pattern is real - March NDVI median is 0.72 (MA) and
   0.61 (KU) vs. 0.44-0.52 for the four broadleaf species - but **the
   optimal single threshold is 0.67, not the expert's guessed ~0.8**, and
   even that only reaches 75% accuracy (58.5% conifer recall, 88.6%
   broadleaf recall) because KU's distribution has a long tail overlapping
   broadleaf. The trained RF's own thresholds (2,132 actual splits on
   March-NDVI columns across 400 trees) cluster around an even **lower**
   median of 0.54 - lower than both the expert guess and the simple binary
   stump - because the RF is solving the full 6-class problem at each node
   (often separating one specific species from a specific subset, not just
   "conifer vs. everything"), so its thresholds are more varied and
   generally lower than a clean binary rule would suggest.

7. **`scripts/07_species_forestry_profile.py`**
   Empirical forestry-characteristic profile per species, from this project's
   *own* Subregion2 data - not textbook/dendrology facts (a different,
   not-derivable-from-Sentinel-2-or-height-data kind of claim, e.g. flowering
   timing - see the March-NDVI script's phenology fact-check for why those
   need a real forestry source, not this pipeline). Uses only unambiguous,
   standard Metsaregister fields straight from `final_split_clean.gpkg`:
   height (m, already computed via zonal_stats in script 01), age (`vanus`,
   years), standing timber volume (`tagavara`, m³/ha), parcel area
   (`pindala`, ha), and main-species share (`osakaal`, %). Deliberately does
   **not** decode `arengukl_k` (development class) or `klass` into named
   categories - no verified code lookup for them exists in this repo (same
   reason `soil_majority` stays an uninterpreted numeric code: checked the
   soil raster for an embedded attribute/color table, found none).

   Outputs:
   - `tables/subregion2_species_forestry_profile.csv` (median + IQR per species)
   - `figures/species_forestry_profile_boxplots.png` (5-panel: height/age/volume/area/purity)

   **Findings** (median [IQR], n parcels): `mänd` is oldest (72 [54-90] a),
   tallest (18.3 [12.9-22.3] m) and highest-volume (205 [138-265] m³/ha) -
   consistent with pine stands skewing toward established, commercially
   mature forest in this dataset. `haab` is notably young (median age only
   10 [3-70] a, very wide spread) despite a respectable median height (16.2 m)
   - consistent with aspen's known pioneering/fast-early-growth ecology.
   `lepp` is the shortest and youngest of the deciduous species (12.6 [6.9-
   17.3] m, 18 [7-45] a) - directly relevant to a user question about
   whether "lepp grows to about 15m": the empirical median here is 12.6 m,
   with the *upper* quartile reaching 17.3 m, so ~15 m is a plausible
   mid-to-upper figure but not the typical value, and depends heavily on
   stand age (this dataset's alder stands are mostly young). All species
   maintain high main-species purity (85-100% IQR) - a sanity check that the
   original clean-stand filtering (this thesis §3.2) is working as intended.

8. **`scripts/08_forestry_profile_by_age.py`**
   Script 07's profile, stratified by age group at the **20-year threshold**
   already established elsewhere in this thesis
   (`experiments/EXP021_species_signature_features/age_split_analysis.py`
   found a dramatic classification accuracy jump at exactly this cutoff,
   0.653 for <20y vs. 0.886 for >=20y, independently confirmed by Lang et
   al. (2018)'s own 20-year threshold). A different split (e.g. 10 years)
   would introduce an inconsistency with that existing analysis - not used,
   even though sample sizes would support it too (checked: minimum group
   size at either threshold is well above 70 parcels per species).

   Outputs:
   - `tables/subregion2_species_forestry_profile_by_age.csv`
   - `figures/species_forestry_profile_by_age.png` (height + tagavara, grouped boxplots)

   **Findings**: age dominates the raw (unstratified) numbers from script 07
   - e.g. `haab`'s unstratified median height (16.2 m) blended a very young
   subsample (<20y median 4.1 m) with a much taller mature subsample (>=20y
   median **24.2 m - the tallest mature median of any species**). For the
   user's original alder question: mature (>=20y) `lepp` has median height
   **17.0 m [14.4-19.0]** - much closer to "~15 m" than the unstratified
   12.6 m median was, since this dataset's alder stands skew young overall.
   Controlling for age gives a materially more useful "typical mature
   height" figure than the unstratified profile for every species.

## Numbers (for quick reference)

Subregion2 parcels with daily-timeseries data: **6,906 train / 1,251 val / 1,613 test**
(9,776 total in the geometry file; a small number - about 0.4-0.6% of rows -
have no Sentinel-2 daily download and are excluded). Species breakdown is in
`tables/subregion2_split_species_counts.csv`.
