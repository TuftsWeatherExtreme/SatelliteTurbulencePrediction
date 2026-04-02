# fetch_satellite_for_pireps.py
# Authors: Team Celestial Blue
# Spring 2025
# Purpose: For each PIREP in a clean CSV, fetch GOES-16 satellite images
#          at 15-min intervals, crop around the PIREP location, and save as .npz.
# Usage: python -u fetch_satellite_for_pireps.py <input_csv> <output_dir>

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from goes2go import GOES
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


def fetch_and_process_image(timestamp, goes_sat, cache):
    """
    Fetch a single satellite image, project, and smooth it.
    Uses an in-memory cache to avoid re-fetching the same timestamp.
    Returns the smoothed full CONUS array (1, 1500, 2500, 6) or None.
    """
    cache_key = round_to_interval(timestamp)

    if cache_key in cache:
        return cache[cache_key]

    try:
        data = sat.fetch(timestamp, goes_sat)
        lat, lon = sat.calculate_coordinates(data)
        band_data = sat.fetch_bands(data, BANDS)
        projected = sat.project(lat, lon, band_data.values)
        smoothed = sat.smooth(projected)
        cache[cache_key] = smoothed
        return smoothed
    except Exception as e:
        print(f"  WARNING: Failed to fetch image at {timestamp}: {e}", flush=True)
        cache[cache_key] = None
        return None


def fetch_frames_for_pirep(pirep_dt, pirep_lat, pirep_lon, goes_sat, cache):
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
        smoothed = fetch_and_process_image(timestamp, goes_sat, cache)

        if smoothed is not None:
            frames[NUM_FRAMES - 1 - i] = smoothed[0, row_start:row_end, col_start:col_end, :]
            frames_fetched += 1

    if frames_fetched == 0:
        return None

    return frames


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python -u {sys.argv[0]} <input_csv> <output_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    pireps_df = pd.read_csv(input_csv)
    print(f"Processing {len(pireps_df)} PIREPs from {input_csv}", flush=True)

    goes_sat = GOES(satellite=16, product="ABI", domain="C")
    num_saved = 0
    num_failed = 0

    # In-memory cache for projected satellite images (keyed by rounded timestamp)
    # This avoids re-fetching the same image for PIREPs at similar times
    image_cache = {}
    start_time = time_module.time()

    for idx, pirep in pireps_df.iterrows():
        pirep_dt = datetime.fromisoformat(pirep['datetime'])
        pirep_lat = pirep['LAT']
        pirep_lon = pirep['LON']

        pirep_start = time_module.time()
        print(f"[{idx+1}/{len(pireps_df)}] PIREP at ({pirep_lat:.2f}, {pirep_lon:.2f}) "
              f"FL{int(pirep['FL'])} {pirep_dt} (cache: {len(image_cache)} images)", flush=True)

        try:
            frames = fetch_frames_for_pirep(pirep_dt, pirep_lat, pirep_lon, goes_sat, image_cache)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            num_failed += 1
            continue

        if frames is None:
            print(f"  SKIPPED: no frames fetched", flush=True)
            num_failed += 1
            continue

        output_path = os.path.join(output_dir, f"{idx:07d}.npz")
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
        elapsed = time_module.time() - pirep_start
        print(f"  SAVED {output_path} ({elapsed:.1f}s)", flush=True)

    total_time = time_module.time() - start_time
    print(f"\nDone. Saved {num_saved}/{len(pireps_df)} model inputs, "
          f"{num_failed} failed. Total time: {total_time/60:.1f} min", flush=True)
    print(f"Unique satellite images fetched: {len(image_cache)}", flush=True)


if __name__ == "__main__":
    main()
