# predict_satellite_conus.py
# Team Celestial Blue
# Spring 2025
# Purpose: Pull current GOES-16 satellite data, tile CONUS into 128x128 patches,
#          run the trained model on each patch, and output a GeoJSON file.
#          Maintains a rolling cache of the last 75 minutes of satellite images
#          so that only new frames need to be fetched each cycle.
#
# Usage: python -u predict_satellite_conus.py --model-type conv3d \
#            --weights model.pth --output predictions.geojson [--cycle-minutes 30]
#
# The script:
#   1. Fetches the last 75 min of GOES-16 L1b-RadC images (15 frames at 5-min intervals)
#   2. Projects and smooths the full CONUS grid once per frame
#   3. Tiles the 1500x2500 CONUS grid into overlapping 128x128 patches
#   4. Feeds each patch through the trained model
#   5. Outputs GeoJSON with turbulence probabilities per patch

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time as time_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satellite as sat
from consts import GRID_RANGE, MAP_RANGE, BANDS, NUM_FRAMES, FRAME_INTERVAL_MIN, CROP_SIZE

# Import models
from train_and_test_model import ConvLSTMModel, Conv3DModel

MODEL_FACTORIES = {
    "convlstm": ConvLSTMModel,
    "conv3d": Conv3DModel,
}

# Tile parameters
TILE_SIZE = CROP_SIZE  # 128 pixels
TILE_STRIDE = 96       # overlap of 32 pixels between tiles


class RollingImageCache:
    """
    Maintains a rolling window of projected/smoothed CONUS satellite images.
    When refreshed, only fetches new timestamps and drops expired ones.
    """
    def __init__(self):
        self.cache = {}  # timestamp -> (1, 1500, 2500, 6) array

    def get_frames(self, end_time, num_frames, interval_min):
        """
        Get num_frames of satellite data ending at end_time.
        Fetches only missing frames, reuses cached ones.
        Returns (num_frames, 1500, 2500, 6) array or None.
        """
        timestamps = []
        for i in range(num_frames):
            ts = end_time - timedelta(minutes=i * interval_min)
            ts = ts.replace(second=0, microsecond=0)
            # Round to nearest interval
            ts = ts.replace(minute=(ts.minute // interval_min) * interval_min)
            timestamps.append(ts)
        timestamps.reverse()  # oldest first

        frames = []
        new_fetches = 0
        cache_hits = 0

        for ts in timestamps:
            if ts in self.cache:
                frames.append(self.cache[ts])
                cache_hits += 1
            else:
                try:
                    band_data, first_ds = sat.fetch_l1b_bands(ts, BANDS)
                    lat, lon = sat.calculate_coordinates(first_ds)
                    projected = sat.project(lat, lon, band_data)
                    smoothed = sat.smooth(projected)
                    self.cache[ts] = smoothed
                    frames.append(smoothed)
                    new_fetches += 1
                except Exception as e:
                    print(f"  WARNING: Failed frame at {ts}: {e}", flush=True)
                    # Use zeros as fallback
                    fallback = np.zeros((1, GRID_RANGE["LAT"], GRID_RANGE["LON"], len(BANDS)),
                                       dtype=np.float32)
                    frames.append(fallback)

        print(f"  Frames: {cache_hits} cached, {new_fetches} new fetches", flush=True)

        # Evict old timestamps (keep only what we need + some buffer)
        cutoff = end_time - timedelta(minutes=(num_frames + 5) * interval_min)
        expired = [k for k in self.cache if k < cutoff]
        for k in expired:
            del self.cache[k]
        if expired:
            print(f"  Evicted {len(expired)} expired frames from cache", flush=True)

        # Stack: each is (1, 1500, 2500, 6) -> (num_frames, 1500, 2500, 6)
        return np.concatenate(frames, axis=0)


def generate_tiles():
    """
    Generate tile coordinates (row_start, col_start) covering the CONUS grid.
    Returns list of (row_start, col_start, center_lat, center_lon).
    """
    tiles = []
    lat_per_pixel = (MAP_RANGE["LAT"]["MAX"] - MAP_RANGE["LAT"]["MIN"]) / GRID_RANGE["LAT"]
    lon_per_pixel = (MAP_RANGE["LON"]["MAX"] - MAP_RANGE["LON"]["MIN"]) / GRID_RANGE["LON"]

    row = 0
    while row + TILE_SIZE <= GRID_RANGE["LAT"]:
        col = 0
        while col + TILE_SIZE <= GRID_RANGE["LON"]:
            center_row = row + TILE_SIZE // 2
            center_col = col + TILE_SIZE // 2
            center_lat = MAP_RANGE["LAT"]["MIN"] + center_row * lat_per_pixel
            center_lon = MAP_RANGE["LON"]["MIN"] + center_col * lon_per_pixel
            tiles.append((row, col, center_lat, center_lon))
            col += TILE_STRIDE
        row += TILE_STRIDE

    return tiles


def predict_tiles(model, conus_frames, tiles, device, batch_size=32):
    """
    Extract each tile from the full CONUS frames and run model inference.

    Args:
        model: trained PyTorch model
        conus_frames: (num_frames, 1500, 2500, 6) array
        tiles: list of (row_start, col_start, center_lat, center_lon)
        device: torch device
        batch_size: inference batch size

    Returns:
        list of dicts with predictions per tile
    """
    results = []
    batch_tensors = []
    batch_tiles = []

    for row, col, clat, clon in tiles:
        # Crop tile: (num_frames, 128, 128, 6) -> transpose to (num_frames, 6, 128, 128)
        tile_data = conus_frames[:, row:row+TILE_SIZE, col:col+TILE_SIZE, :]
        tile_tensor = torch.tensor(
            tile_data.transpose(0, 3, 1, 2), dtype=torch.float32
        )
        tile_tensor = torch.nan_to_num(tile_tensor, nan=0.0)
        batch_tensors.append(tile_tensor)
        batch_tiles.append((clat, clon))

        if len(batch_tensors) == batch_size:
            x = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                output = model(x)
            probs = F.softmax(output, dim=-1).cpu().numpy()

            for i, (lat, lon) in enumerate(batch_tiles):
                results.append({
                    "lat": lat,
                    "lon": lon,
                    "severe_prob": float(probs[i][1]),
                    "pred_class": int(np.argmax(probs[i])),
                })

            batch_tensors = []
            batch_tiles = []

    # Process remaining
    if batch_tensors:
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            output = model(x)
        probs = F.softmax(output, dim=-1).cpu().numpy()
        for i, (lat, lon) in enumerate(batch_tiles):
            results.append({
                "lat": lat,
                "lon": lon,
                "severe_prob": float(probs[i][1]),
                "pred_class": int(np.argmax(probs[i])),
            })

    return results


def results_to_geojson(results, model_type, timestamp):
    """Convert prediction results to a GeoJSON FeatureCollection."""
    features = []
    for r in results:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "source": "satellite",
                "model_type": model_type,
                "pred_class": r["pred_class"],
                "severe_prob": r["severe_prob"],
                "timestamp": timestamp,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser(description="Satellite CONUS-wide turbulence prediction")
    parser.add_argument("--model-type", choices=sorted(MODEL_FACTORIES.keys()), required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cycle-minutes", type=int, default=0,
                        help="If >0, run continuously every N minutes with rolling cache")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model_type}", flush=True)

    # Load model
    model = MODEL_FACTORIES[args.model_type]().to(device)
    raw = torch.load(str(args.weights), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)
    model.eval()

    # Generate tile grid
    tiles = generate_tiles()
    print(f"CONUS tiles: {len(tiles)} patches of {TILE_SIZE}x{TILE_SIZE} "
          f"(stride {TILE_STRIDE})", flush=True)

    cache = RollingImageCache()

    while True:
        cycle_start = time_module.time()
        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}", flush=True)
        print(f"Prediction cycle at {now.isoformat()}", flush=True)
        print(f"{'='*60}", flush=True)

        # Fetch satellite frames
        print("Fetching satellite data...", flush=True)
        t0 = time_module.time()
        conus_frames = cache.get_frames(now, NUM_FRAMES, FRAME_INTERVAL_MIN)
        print(f"Satellite data ready ({time_module.time()-t0:.1f}s). "
              f"Shape: {conus_frames.shape}", flush=True)

        # Run predictions
        print("Running inference...", flush=True)
        t0 = time_module.time()
        results = predict_tiles(model, conus_frames, tiles, device, args.batch_size)
        print(f"Inference complete ({time_module.time()-t0:.1f}s). "
              f"{len(results)} predictions", flush=True)

        # Count severe predictions
        n_severe = sum(1 for r in results if r["pred_class"] == 1)
        print(f"Severe predictions: {n_severe}/{len(results)} "
              f"({100*n_severe/len(results):.1f}%)", flush=True)

        # Write GeoJSON
        geojson = results_to_geojson(results, args.model_type, now.isoformat())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(geojson, f)
        print(f"Wrote {len(results)} features to {args.output}", flush=True)

        cycle_time = time_module.time() - cycle_start
        print(f"Cycle completed in {cycle_time:.1f}s", flush=True)

        if args.cycle_minutes <= 0:
            break

        # Wait for next cycle
        wait = max(0, args.cycle_minutes * 60 - cycle_time)
        print(f"Next cycle in {wait:.0f}s...", flush=True)
        time_module.sleep(wait)


if __name__ == "__main__":
    main()
