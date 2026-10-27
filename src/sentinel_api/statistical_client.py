"""Sentinel Hub Statistical API client shared by scripts/download_sentinel2_10day_statistical.py and
scripts/download_sentinel2_daily_statistical.py. Canonicalized to scripts/download_sentinel2_daily_statistical.py's implementation,
which is a strict superset of scripts/download_sentinel2_10day_statistical.py's earlier version:
handles HTTP 500/504 in addition to 502/503, accepts list-valued filters
(scripts/download_sentinel2_10day_statistical.py only ever passed single values, which still work
here via `normalize_filter_to_list`), and wraps stat lookups in `safe_get_stats`.

The EVALSCRIPT string (and therefore the exact band/index formulas, e.g. the
NDRE calculation) is NOT part of this shared module - it stays inline in each
script since scripts/download_pixel_stack.py's and scripts/download_sentinel2_daily_statistical.py's
EVALSCRIPTs genuinely compute NDRE differently (B08 vs B8A in the formula).
That discrepancy is preserved as-is, not merged away - see EXPERIMENTS.md.
"""

import time
import random
import requests

from sentinelhub import SentinelHubStatistical

DEFAULT_BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
]
DEFAULT_INDEX_NAMES = ["NDVI", "NDRE", "NDMI", "NBR", "EVI", "CIRE"]


def normalize_filter_to_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def apply_filters(gdf, subregion, split, subregion_col="subregion", split_col="dataset_split"):
    subregions = normalize_filter_to_list(subregion)
    splits = normalize_filter_to_list(split)

    if subregions is not None:
        gdf = gdf[gdf[subregion_col].isin(subregions)].copy()

    if splits is not None:
        gdf = gdf[gdf[split_col].isin(splits)].copy()

    return gdf


def make_request(geom, config, evalscript, time_range, aggregation_interval, data_collection,
                  maxcc=0.3, resolution=(10, 10)):
    start, end = time_range

    return SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=(start, end),
            aggregation_interval=aggregation_interval,
            resolution=resolution
        ),
        input_data=[
            SentinelHubStatistical.input_data(
                data_collection=data_collection,
                maxcc=maxcc
            )
        ],
        geometry=geom,
        config=config,
        calculations={
            "default": {
                "statistics": {
                    "default": {
                        "percentiles": {
                            "k": [50]
                        }
                    }
                }
            }
        }
    )


def request_with_retry(row_id, geom, config, evalscript, time_range, aggregation_interval,
                        data_collection, refresh_config_fn, max_retries=12, maxcc=0.3):
    for attempt in range(1, max_retries + 1):
        try:
            request = make_request(geom, config, evalscript, time_range, aggregation_interval,
                                    data_collection, maxcc=maxcc)
            data = request.get_data()
            return data, config

        except Exception as e:
            txt = str(e)

            if "unauthorized_client" in txt or "Invalid client or Invalid client credentials" in txt:
                print(f"Credential error for object {row_id}")
                print(txt)
                return None, config

            if "401" in txt or "AccessToken signature expired" in txt or "Unauthorized" in txt:
                print(f"401 token issue for object {row_id}, refreshing config ({attempt}/{max_retries})")
                config = refresh_config_fn()
                wait = min(2 * attempt, 20)
                time.sleep(wait)
                continue

            if "429" in txt or "Too Many Requests" in txt or "RATE_LIMIT_EXCEEDED" in txt:
                wait = min((2 ** attempt) + random.uniform(0, 2), 90)
                print(f"429 rate limit for object {row_id}, waiting {wait:.1f} s ({attempt}/{max_retries})")
                time.sleep(wait)
                continue

            if (
                "500" in txt or
                "502" in txt or
                "503" in txt or
                "504" in txt or
                "Bad Gateway" in txt or
                "Service Unavailable" in txt
            ):
                wait = min((2 ** attempt) + random.uniform(0, 2), 90)
                print(f"Temporary server error for object {row_id}, waiting {wait:.1f} s ({attempt}/{max_retries})")
                time.sleep(wait)
                continue

            print(f"Error for object {row_id}: {e}")
            return None, config

    print(f"Error for object {row_id}: max retry count reached")
    return None, config


def get_stat_mean_and_median(stats_dict):
    mean_val = stats_dict.get("mean")
    median_val = stats_dict.get("percentiles", {}).get("50.0")
    return mean_val, median_val


def safe_get_stats(stats_container, band_name):
    try:
        return stats_container[band_name]["stats"]
    except Exception:
        return {}


def parse_stats(data, row, band_names=None, index_names=None):
    band_names = band_names or DEFAULT_BAND_NAMES
    index_names = index_names or DEFAULT_INDEX_NAMES

    out = []

    if not data or "data" not in data[0]:
        return out

    for interval in data[0]["data"]:
        outputs = interval.get("outputs", {})

        if "bands" not in outputs or "indices" not in outputs:
            continue

        band_stats = outputs["bands"]["bands"]
        idx_stats = outputs["indices"]["bands"]

        row_out = {
            "id": row["id"],
            "date_from": interval["interval"]["from"],
            "date_to": interval["interval"]["to"],
        }

        for i, name in enumerate(band_names):
            mean_val, median_val = get_stat_mean_and_median(safe_get_stats(band_stats, f"B{i}"))
            row_out[f"{name}_mean"] = mean_val
            row_out[f"{name}_median"] = median_val

        for i, name in enumerate(index_names):
            mean_val, median_val = get_stat_mean_and_median(safe_get_stats(idx_stats, f"B{i}"))
            row_out[f"{name}_mean"] = mean_val
            row_out[f"{name}_median"] = median_val

        row_out["target"] = row.get("target")
        row_out["split"] = row.get("dataset_split")
        row_out["subregion"] = row.get("subregion")

        out.append(row_out)

    return out
