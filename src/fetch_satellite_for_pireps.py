# fetch_satellite_for_pireps.py
# Authors: Team Celestial Blue
# Spring 2025
# Purpose: For each PIREP in a clean CSV, fetch GOES-16 satellite images
#          at 15-min intervals, crop around the PIREP location, and save as .npz.
# Usage: python -u fetch_satellite_for_pireps.py <input_csv> <output_dir> [chunk_start] [chunk_end]
#
# Changes from original:
#   - PIREPs are sorted by datetime before processing (maximizes cache reuse)
#   - Optional chunk_start / chunk_end args for parallelism across SLURM array sub-jobs
#   - Resume logic: skips PIREPs whose .npz already exists in output_dir
#   - Per-step timing diagnostics (fetch / project / smooth) to identify bottlenecks

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time as time_module

# Add src/ to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satellite as sat
from consts import GRID_RANGE, MAP_RANGE, BANDS, NUM_FRAMES, FRAME_INTERVAL_MIN, CROP_SIZE


def get_crop_indices(pirep_lat, pirep_lon):
    """
    Convert PIREP lat/lon to grid indices and compute crop bounds.
    Returns (row_start, row_end, col_start, col_end) or None if out of bounds.
    """
    half = CROP_SIZE // 2

    row = int(round(
        (pirep_lat - MAP_RANGE["LAT"]["MIN"]) /
        (MAP_RANGE["LAT"]["MAX"] - MAP_RANGE["LAT"]["MIN"]) *
        (GRID_RANGE["LAT"] - 1)
    ))
    col = int(round(
        (pirep_lon - MAP_RANGE["LON"]["MIN"]) /
        (MAP_RANGE["LON"]["MAX"] - MAP_RANGE["LON"]["MIN"]) *
        (GRID_RANGE["LON"] - 1)
    ))

    row_start = max(0, row - half)
    row_end = row_start + CROP_SIZE
    if row_end > GRID_RANGE["LAT"]:
        row_end = GRID_RANGE["LAT"]
        row_start = row_end - CROP_SIZE

    col_start = max(0, col - half)
    col_end = col_start + CROP_SIZE
    if col_end > GRID_RANGE["LON"]:
        col_end = GRID_RANGE["LON"]
        col_start = col_end - CROP_SIZE

    if row_start < 0 or col_start < 0:
        return None

    return row_start, row_end, col_start, col_end


def round_to_interval(dt, interval_min=FRAME_INTERVAL_MIN):
    """Round a datetime down to the nearest interval for cache keying."""
    minutes = (dt.minute // interval_min) * interval_min
    return dt.replace(minute=minutes, second=0, microsecond=0)


def fetch_and_process_image(timestamp, goes_sat, cache, diagnostics):
    """
    Fetch individual L1b-RadC band files, convert radiance to brightness temp,
    project onto CONUS grid, and smooth.
    Uses an in-memory cache to avoid re-fetching the same timestamp.
    Returns the smoothed full CONUS array (1, 1500, 2500, 6) or None.

    diagnostics: dict accumulating timing stats for fetch/project/smooth steps.
    """
    cache_key = round_to_interval(timestamp)

    if cache_key in cache:
        diagnostics["cache_hits"] = diagnostics.get("cache_hits", 0) + 1
        return cache[cache_key]

    try:
        t0 = time_module.time()
        band_data, first_ds = sat.fetch_l1b_bands(timestamp, BANDS)
        diagnostics["fetch_s"] = diagnostics.get("fetch_s", 0) + (time_module.time() - t0)

        lat, lon = sat.calculate_coordinates(first_ds)

        t0 = time_module.time()
        projected = sat.project(lat, lon, band_data)
        diagnostics["project_s"] = diagnostics.get("project_s", 0) + (time_module.time() - t0)

        t0 = time_module.time()
        smoothed = sat.smooth(projected)
        diagnostics["smooth_s"] = diagnostics.get("smooth_s", 0) + (time_module.time() - t0)

        diagnostics["cache_misses"] = diagnostics.get("cache_misses", 0) + 1
        cache[cache_key] = smoothed
        return smoothed
    except Exception as e:
        print(f"  WARNING: Failed to fetch image at {timestamp}: {e}", flush=True)
        cache[cache_key] = None
        return None


def fetch_frames_for_pirep(pirep_dt, pirep_lat, pirep_lon, goes_sat, cache, diagnostics):
    """
    Fetch satellite frames for a single PIREP, using cached images where available.
    Returns numpy array of shape (NUM_FRAMES, CROP_SIZE, CROP_SIZE, 6) or None.
    """
    crop = get_crop_indices(pirep_lat, pirep_lon)
    if crop is None:
        return None

    row_start, row_end, col_start, col_end = crop
    frames = np.zeros((NUM_FRAMES, CROP_SIZE, CROP_SIZE, len(BANDS)), dtype=np.float32)
    frames_fetched = 0

    for i in range(NUM_FRAMES):
        timestamp = pirep_dt - timedelta(minutes=i * FRAME_INTERVAL_MIN)
        smoothed = fetch_and_process_image(timestamp, goes_sat, cache, diagnostics)

        if smoothed is not None:
            frames[NUM_FRAMES - 1 - i] = smoothed[0, row_start:row_end, col_start:col_end, :]
            frames_fetched += 1

    if frames_fetched == 0:
        return None

    return frames


def main():
    if len(sys.argv) not in (3, 5):
        print(f"Usage: python -u {sys.argv[0]} <input_csv> <output_dir> [chunk_start chunk_end]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    pireps_df = pd.read_csv(input_csv)

    # --- Sort by datetime so temporally-close PIREPs are processed together,
    #     maximising cache reuse within a chunk. The original row index is
    #     preserved in 'orig_idx' so output filenames stay stable.
    pireps_df = pireps_df.sort_values('datetime').reset_index(drop=False).rename(
        columns={'index': 'orig_idx'}
    )

    # --- Optional chunking for parallelism (chunk_start is inclusive, chunk_end exclusive)
    if len(sys.argv) == 5:
        chunk_start = int(sys.argv[3])
        chunk_end = int(sys.argv[4])
        pireps_df = pireps_df.iloc[chunk_start:chunk_end].reset_index(drop=True)
        print(f"Processing chunk [{chunk_start}:{chunk_end}] "
              f"({len(pireps_df)} PIREPs) from {input_csv}", flush=True)
    else:
        print(f"Processing all {len(pireps_df)} PIREPs from {input_csv}", flush=True)

    goes_sat = None  # No longer needed — using direct L1b-RadC S3 access per band
    num_saved = 0
    num_skipped = 0
    num_failed = 0

    # In-memory cache for projected satellite images (keyed by rounded timestamp)
    image_cache = {}
    # Accumulated timing stats across all fetch_and_process_image calls
    diagnostics = {}
    start_time = time_module.time()

    for local_idx, pirep in pireps_df.iterrows():
        orig_idx = int(pirep['orig_idx'])
        output_path = os.path.join(output_dir, f"{orig_idx:07d}.npz")

        # --- Resume logic: skip if already generated
        if os.path.exists(output_path):
            print(f"[{local_idx+1}/{len(pireps_df)}] SKIPPING {output_path} (already exists)",
                  flush=True)
            num_skipped += 1
            continue

        pirep_dt = datetime.fromisoformat(pirep['datetime'])
        pirep_lat = pirep['LAT']
        pirep_lon = pirep['LON']

        pirep_start = time_module.time()
        print(f"[{local_idx+1}/{len(pireps_df)}] PIREP at ({pirep_lat:.2f}, {pirep_lon:.2f}) "
              f"FL{int(pirep['FL'])} {pirep_dt} (cache: {len(image_cache)} images)", flush=True)

        try:
            frames = fetch_frames_for_pirep(
                pirep_dt, pirep_lat, pirep_lon, goes_sat, image_cache, diagnostics
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            num_failed += 1
            continue

        if frames is None:
            print(f"  SKIPPED: no frames fetched", flush=True)
            num_failed += 1
            continue

        np.savez_compressed(
            output_path,
            images=frames,
            turb_label=pirep['turb_label'],
            sample_weight=pirep['sample_weight'],
            lat=pirep_lat,
            lon=pirep_lon,
            fl=pirep['FL'],
            datetime=str(pirep['datetime']),
        )
        num_saved += 1

        pirep_elapsed = time_module.time() - pirep_start
        # Per-PIREP timing breakdown so bottlenecks are visible in the log
        fetch_t   = diagnostics.get("fetch_s", 0)
        project_t = diagnostics.get("project_s", 0)
        smooth_t  = diagnostics.get("smooth_s", 0)
        hits      = diagnostics.get("cache_hits", 0)
        misses    = diagnostics.get("cache_misses", 0)
        print(
            f"  SAVED {output_path} ({pirep_elapsed:.1f}s total | "
            f"fetch={fetch_t:.1f}s project={project_t:.1f}s smooth={smooth_t:.1f}s | "
            f"cache hits={hits} misses={misses})",
            flush=True
        )
        # Reset per-PIREP diagnostics counters (cumulative totals were for this PIREP only)
        diagnostics.clear()

    total_time = time_module.time() - start_time
    print(
        f"\nDone. Saved {num_saved}, skipped {num_skipped}, failed {num_failed} "
        f"/ {len(pireps_df)} PIREPs in chunk. Total time: {total_time/60:.1f} min",
        flush=True
    )
    print(f"Unique satellite images in cache at exit: {len(image_cache)}", flush=True)


if __name__ == "__main__":
    main()