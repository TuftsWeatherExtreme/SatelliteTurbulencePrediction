# fetch_satellite_for_pireps.py
# Authors: Team Celestial Blue
# Spring 2025
# Purpose: For each PIREP in a clean CSV, fetch 9 consecutive hourly GOES-16
#          satellite images (bands 8,9,10,13,14,15), crop around the PIREP
#          location, and save as .npz model inputs.
# Usage: python fetch_satellite_for_pireps.py <input_csv> <output_dir>

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from goes2go import GOES

# Add src/ to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satellite as sat
from consts import GRID_RANGE, MAP_RANGE
from utils.convert import convert_coord

from consts import BANDS, NUM_FRAMES, FRAME_INTERVAL_MIN, CROP_SIZE

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


def fetch_frames_for_pirep(pirep_dt, pirep_lat, pirep_lon, goes_sat):
    """
    Fetch 9 hourly satellite frames ending at the PIREP time,
    project onto the CONUS grid, smooth, and crop around PIREP location.

    Returns numpy array of shape (9, CROP_SIZE, CROP_SIZE, 6) or None on failure.
    """
    crop = get_crop_indices(pirep_lat, pirep_lon)
    if crop is None:
        return None

    row_start, row_end, col_start, col_end = crop
    frames = np.zeros((NUM_FRAMES, CROP_SIZE, CROP_SIZE, len(BANDS)), dtype=np.float32)

    for i in range(NUM_FRAMES):
        # t=0 is the PIREP time, t=1 is 30 min before, etc.
        timestamp = pirep_dt - timedelta(minutes=i * FRAME_INTERVAL_MIN)
        try:
            data = sat.fetch(timestamp, goes_sat)
            lat, lon = sat.calculate_coordinates(data)
            band_data = sat.fetch_bands(data, BANDS)
            projected = sat.project(lat, lon, band_data.values)
            smoothed = sat.smooth(projected)
            # Crop around PIREP location
            frames[NUM_FRAMES - 1 - i] = smoothed[0, row_start:row_end, col_start:col_end, :]
        except Exception as e:
            print(f"WARNING: Failed to fetch frame at {timestamp}: {e}")
            # Leave as zeros

    return frames


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_csv> <output_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    pireps_df = pd.read_csv(input_csv)
    print(f"Processing {len(pireps_df)} PIREPs from {input_csv}")

    goes_sat = GOES(satellite=16, product="ABI", domain="C")
    num_saved = 0
    num_failed = 0

    for idx, pirep in pireps_df.iterrows():
        pirep_dt = datetime.fromisoformat(pirep['datetime'])
        pirep_lat = pirep['LAT']
        pirep_lon = pirep['LON']

        try:
            frames = fetch_frames_for_pirep(pirep_dt, pirep_lat, pirep_lon, goes_sat)
        except Exception as e:
            print(f"ERROR processing PIREP {idx}: {e}")
            num_failed += 1
            continue

        if frames is None:
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

        if (idx + 1) % max(1, len(pireps_df) // 20) == 0:
            print(f"Progress: {idx + 1}/{len(pireps_df)} ({num_saved} saved, {num_failed} failed)")

    print(f"Done. Saved {num_saved}/{len(pireps_df)} model inputs, {num_failed} failed.")


if __name__ == "__main__":
    main()
