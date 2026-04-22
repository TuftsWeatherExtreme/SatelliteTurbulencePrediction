# predict_satellite_demo.py
# Team Celestial Blue
# Spring 2025
# Purpose: Generate 16 GeoJSON files representing 8 hours of satellite
#          turbulence predictions at 30-minute intervals, for frontend demo.
#
# Usage: python -u predict_satellite_demo.py --model-type conv3d \
#            --weights model.pth --output-dir demo_geojsons/ \
#            [--start-time "2024-03-01T12:00:00"] [--device cpu]
#
# If --start-time is not provided, uses the current time minus 8 hours
# so we're predicting over the most recent 8-hour window.

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
from train_and_test_model import ConvLSTMModel, Conv3DModel
from model_architecture import SatelliteTurbulenceModel

MODEL_FACTORIES = {
    "convlstm": ConvLSTMModel,
    "conv3d": Conv3DModel,
    # Matches checkpoints whose keys start with "cnn."
    "cnn": SatelliteTurbulenceModel,
}

TILE_SIZE = CROP_SIZE  # 128
TILE_STRIDE = 96

# Rolling cache shared across all 16 prediction windows
image_cache = {}


def round_to_interval(dt, interval_min=FRAME_INTERVAL_MIN):
    minutes = (dt.minute // interval_min) * interval_min
    return dt.replace(minute=minutes, second=0, microsecond=0)


def fetch_conus_frame(timestamp):
    """Fetch, project, and smooth a single CONUS frame. Uses cache."""
    cache_key = round_to_interval(timestamp)
    if cache_key in image_cache:
        return image_cache[cache_key]

    try:
        band_data, first_ds = sat.fetch_l1b_bands(timestamp, BANDS)
        lat, lon = sat.calculate_coordinates(first_ds)
        projected = sat.project(lat, lon, band_data)
        smoothed = sat.smooth(projected)
        image_cache[cache_key] = smoothed
        return smoothed
    except Exception as e:
        print(f"    WARNING: Failed to fetch {timestamp}: {e}", flush=True)
        fallback = np.zeros((1, GRID_RANGE["LAT"], GRID_RANGE["LON"], len(BANDS)),
                            dtype=np.float32)
        image_cache[cache_key] = fallback
        return fallback


def get_frames_for_time(prediction_time):
    """Get NUM_FRAMES satellite images ending at prediction_time."""
    frames = []
    new_fetches = 0
    cache_hits = 0

    for i in range(NUM_FRAMES):
        ts = prediction_time - timedelta(minutes=i * FRAME_INTERVAL_MIN)
        cache_key = round_to_interval(ts)
        if cache_key in image_cache:
            cache_hits += 1
        else:
            new_fetches += 1
        frames.append(fetch_conus_frame(ts))

    frames.reverse()  # oldest first
    print(f"    {cache_hits} cached, {new_fetches} new fetches", flush=True)
    return np.concatenate(frames, axis=0)


def generate_tiles():
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


def predict_tiles(model, conus_frames, tiles, device, batch_size=32, model_type="conv3d"):
    results = []
    batch_tensors = []
    batch_tiles = []
    batch_meta = []

    for row, col, clat, clon in tiles:
        tile_data = conus_frames[:, row:row+TILE_SIZE, col:col+TILE_SIZE, :]
        tile_tensor = torch.tensor(
            tile_data.transpose(0, 3, 1, 2), dtype=torch.float32
        )
        tile_tensor = torch.nan_to_num(tile_tensor, nan=0.0)
        batch_tensors.append(tile_tensor)
        batch_tiles.append((clat, clon))
        # SatelliteTurbulenceModel expects separate metadata; use [lat, lon, alt, delta_t].
        # Alt/delta_t aren't defined in this CONUS demo, so we set them to 0.
        batch_meta.append(torch.tensor([clat, clon, 0.0, 0.0], dtype=torch.float32))

        if len(batch_tensors) == batch_size:
            x = torch.stack(batch_tensors).to(device)
            meta = torch.stack(batch_meta).to(device)
            with torch.no_grad():
                output = model(x, meta) if model_type == "cnn" else model(x)
            probs = F.softmax(output, dim=-1).cpu().numpy()
            for i, (lat, lon) in enumerate(batch_tiles):
                results.append({
                    "lat": lat, "lon": lon,
                    "severe_prob": float(probs[i][1]),
                    "pred_class": int(np.argmax(probs[i])),
                })
            batch_tensors = []
            batch_tiles = []
            batch_meta = []

    if batch_tensors:
        x = torch.stack(batch_tensors).to(device)
        meta = torch.stack(batch_meta).to(device)
        with torch.no_grad():
            output = model(x, meta) if model_type == "cnn" else model(x)
        probs = F.softmax(output, dim=-1).cpu().numpy()
        for i, (lat, lon) in enumerate(batch_tiles):
            results.append({
                "lat": lat, "lon": lon,
                "severe_prob": float(probs[i][1]),
                "pred_class": int(np.argmax(probs[i])),
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate 16 GeoJSON files for 8 hours of satellite predictions (demo)"
    )
    parser.add_argument("--model-type", choices=sorted(MODEL_FACTORIES.keys()), required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-time", type=str, default=None,
                        help="Start of 8-hour window (ISO format). Default: 8 hours ago.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.start_time:
        start_time = datetime.fromisoformat(args.start_time).replace(tzinfo=timezone.utc)
    else:
        start_time = datetime.now(timezone.utc) - timedelta(hours=8)

    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model_type}", flush=True)
    print(f"8-hour window: {start_time.isoformat()} to "
          f"{(start_time + timedelta(hours=8)).isoformat()}", flush=True)

    # Load model
    model = MODEL_FACTORIES[args.model_type]().to(device)
    raw = torch.load(str(args.weights), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)
    model.eval()

    tiles = generate_tiles()
    print(f"CONUS tiles: {len(tiles)} patches", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time_module.time()

    # Generate 16 predictions at 30-minute intervals
    for step in range(16):
        prediction_time = start_time + timedelta(minutes=step * 30)
        print(f"\n{'='*60}", flush=True)
        print(f"Step {step+1}/16: {prediction_time.isoformat()}", flush=True)
        print(f"{'='*60}", flush=True)

        # Fetch satellite data
        t0 = time_module.time()
        print(f"  Fetching satellite frames...", flush=True)
        conus_frames = get_frames_for_time(prediction_time)
        print(f"  Frames ready ({time_module.time()-t0:.1f}s). "
              f"Shape: {conus_frames.shape}", flush=True)

        # Run inference
        t0 = time_module.time()
        print(f"  Running inference on {len(tiles)} tiles...", flush=True)
        results = predict_tiles(model, conus_frames, tiles, device, args.batch_size, args.model_type)
        print(f"  Inference done ({time_module.time()-t0:.1f}s)", flush=True)

        n_severe = sum(1 for r in results if r["pred_class"] == 1)
        print(f"  Severe: {n_severe}/{len(results)} ({100*n_severe/len(results):.1f}%)", flush=True)

        # Write GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                    "properties": {
                        "source": "satellite",
                        "model_type": args.model_type,
                        "pred_class": r["pred_class"],
                        "severe_prob": r["severe_prob"],
                        "timestamp": prediction_time.isoformat(),
                        "step": step,
                    },
                }
                for r in results
            ],
        }

        filename = args.output_dir / f"prediction_{step:02d}.geojson"
        with open(filename, "w") as f:
            json.dump(geojson, f)
        print(f"  Wrote {filename}", flush=True)

        print(f"  Cache size: {len(image_cache)} images", flush=True)

    total_time = time_module.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"Demo complete. 16 GeoJSONs in {total_time/60:.1f} min", flush=True)
    print(f"Total unique satellite images fetched: {len(image_cache)}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
