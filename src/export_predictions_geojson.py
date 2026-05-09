"""
Authors: Razzle Dazzle Rose
Spring 2026

Purpose
-------
Run a trained Satellite model on previously-generated Satellite model inputs
(.npz files under model_inputs/) and export predictions as a GeoJSON
FeatureCollection (points), matching the schema used by
`TurbulencePredictionFrontend/public/predictions/preds_2024_12.geojson`.

This is the satellite-side analogue of:
  NEXRADTurbulencePrediction/model_training/export_predictions_geojson.py

Expected .npz keys
------------------
Produced by `fetch_satellite_for_pireps.py`:
  - images: (15, 128, 128, 6) float32
  - lat, lon, fl
  - datetime (PIREP timestamp string)
  - turb_label (0/1), sample_weight, in_sigmet (optional)

Usage
-----
python export_predictions_geojson.py \
  --weights path/to/model.pth \
  --input-dir path/to/model_inputs \
  --output path/to/preds_2024_12_satellite.geojson \
  [--model-type cnn] \
  [--max-samples 1000] \
  [--year 2024 --month 12]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from model_architecture import SatelliteTurbulenceModel


META_MEAN = torch.tensor([40.4573, -101.0783, 21619.1992, 0.0460], dtype=torch.float32)
META_STD = torch.tensor([8.4979, 25.4030, 14091.5068, 0.2095], dtype=torch.float32)


@dataclass(frozen=True)
class Sample:
    path: Path
    lat: float
    lon: float
    fl_ft: float
    in_sigmet: float
    pirep_time: str
    true_class: float
    images: np.ndarray  # (15, 6, 128, 128)
    sample_weight: float | None


def _iter_npz_paths(input_dir: Path, year: int | None, month: int | None) -> list[Path]:
    if year is not None and month is not None:
        # Common layout: model_inputs/{year}_{month:02d}/0000001.npz
        ymdir = input_dir / f"{year}_{month:02d}"
        paths = sorted(Path(p).resolve() for p in glob.glob(str(ymdir / "**/*.npz"), recursive=True))
        return paths
    return sorted(Path(p).resolve() for p in glob.glob(str(input_dir / "**/*.npz"), recursive=True))


def _load_sample(npz_path: Path) -> Sample:
    d = np.load(npz_path)
    images = d["images"].transpose(0, 3, 1, 2)  # (15, 6, 128, 128)
    images = np.nan_to_num(images, nan=0.0)

    lat = float(d["lat"])
    lon = float(d["lon"])
    fl = float(d["fl"])
    in_sigmet = float(d["in_sigmet"]) if "in_sigmet" in d else 0.0
    pirep_time = str(d["datetime"]) if "datetime" in d else ""
    true_class = float(d["turb_label"]) if "turb_label" in d else float("nan")
    sample_weight = float(d["sample_weight"]) if "sample_weight" in d else None

    return Sample(
        path=npz_path,
        lat=lat,
        lon=lon,
        fl_ft=fl,
        in_sigmet=in_sigmet,
        pirep_time=pirep_time,
        true_class=true_class,
        images=images,
        sample_weight=sample_weight,
    )


def _normalized_meta(lat: float, lon: float, fl_ft: float, in_sigmet: float) -> torch.Tensor:
    meta = torch.tensor([lat, lon, fl_ft, in_sigmet], dtype=torch.float32)
    return (meta - META_MEAN) / META_STD


def _load_state_dict_flexible(model: torch.nn.Module, weights: Path, device: torch.device) -> None:
    raw = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Satellite inference → GeoJSON (preds_YYYY_MM style).")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--input-dir", required=True, type=Path, help="Directory containing model_inputs/**/*.npz")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model-type", default="cnn", choices=["cnn"])
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick exports.")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--month", type=int, default=None)
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {args.input_dir}")
    if not args.weights.is_file():
        raise SystemExit(f"--weights not found: {args.weights}")
    if (args.year is None) != (args.month is None):
        raise SystemExit("Provide both --year and --month, or neither.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SatelliteTurbulenceModel().to(device)
    _load_state_dict_flexible(model, args.weights, device)
    model.eval()

    npz_paths = _iter_npz_paths(args.input_dir, args.year, args.month)
    if args.max_samples is not None:
        npz_paths = npz_paths[: args.max_samples]
    if not npz_paths:
        raise SystemExit(f"No .npz files found under {args.input_dir}")

    def aircraft_class_from_sample_weight(w: float | None) -> str | None:
        if w is None:
            return None
        # From clean_pireps.py: L=0.6, M=0.8, H=1.0 (U mapped to 0.6)
        if abs(w - 0.6) < 1e-6:
            return "l"
        if abs(w - 0.8) < 1e-6:
            return "m"
        if abs(w - 1.0) < 1e-6:
            return "h"
        return None

    features: list[dict] = []
    for p in npz_paths:
        s = _load_sample(p)

        x = torch.tensor(s.images, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 15, 6, 128, 128)
        meta = _normalized_meta(s.lat, s.lon, s.fl_ft, s.in_sigmet).unsqueeze(0).to(device)  # (1,4)

        with torch.no_grad():
            logits = model(x, meta).squeeze(0)

        probs = F.softmax(logits, dim=-1).cpu().tolist()
        pred_class = int(torch.argmax(logits, dim=-1).item())
        severe_prob = float(probs[1]) if len(probs) > 1 else None
        prob_max = float(max(probs)) if probs else None
        aircraft_class = aircraft_class_from_sample_weight(s.sample_weight)

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [s.lon, s.lat]},
                "properties": {
                    "source": "satellite",
                    "aircraft_class": aircraft_class,
                    "pred_class": pred_class,
                    "probs": probs,
                    "severe_prob": severe_prob,
                    "prob_max": prob_max,
                    "true_class": s.true_class if np.isfinite(s.true_class) else None,
                    "flight_level_ft": s.fl_ft,
                    "delta_t_seconds": 0.0,
                    "pirep_time": s.pirep_time,
                    "patch_id": s.path.name,
                },
            }
        )

    collection = {"type": "FeatureCollection", "features": features}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
    print(f"Wrote {len(features)} features to {args.output}")


if __name__ == "__main__":
    main()

