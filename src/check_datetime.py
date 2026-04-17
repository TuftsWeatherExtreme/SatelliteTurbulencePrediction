#!/usr/bin/env python3
"""
Read and display the contents of a .npz file in a readable format.
Usage: python read_npz.py <path_to_file.npz>
"""

import sys
import numpy as np
from datetime import datetime, timezone


def try_parse_datetime(value):
    """Attempt to interpret a value as a datetime."""
    # numpy datetime64
    if isinstance(value, np.datetime64):
        ts = (value - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # scalar string / bytes
    if isinstance(value, (str, bytes)):
        s = value.decode() if isinstance(value, bytes) else value
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(s, fmt).isoformat()
            except ValueError:
                pass

    # numeric Unix timestamp (only if plausible range ~year 2000-2100)
    if np.issubdtype(type(value), np.floating) or np.issubdtype(type(value), np.integer):
        if 9_000_000_00 < float(value) < 40_000_000_00:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()

    return None


def summarize_array(arr):
    """Return a compact summary of a numpy array."""
    lines = []
    lines.append(f"  shape      : {arr.shape}")
    lines.append(f"  dtype      : {arr.dtype}")
    lines.append(f"  size       : {arr.size} element(s)")

    if arr.size == 0:
        lines.append("  values     : (empty)")
        return "\n".join(lines)

    # --- datetime64 arrays ---
    if np.issubdtype(arr.dtype, np.datetime64):
        flat = arr.flatten()
        lines.append(f"  *** DATETIME ARRAY DETECTED ***")
        lines.append(f"  min        : {np.datetime_as_string(flat.min())}")
        lines.append(f"  max        : {np.datetime_as_string(flat.max())}")
        preview = [str(np.datetime_as_string(v)) for v in flat[:5]]
        lines.append(f"  preview    : {preview}" + (" ..." if arr.size > 5 else ""))
        return "\n".join(lines)

    # --- string / object arrays ---
    if arr.dtype.kind in ("U", "S", "O"):
        flat = arr.flatten()
        detected = []
        for v in flat[:10]:
            dt = try_parse_datetime(v)
            if dt:
                detected.append(dt)
        if detected:
            lines.append(f"  *** POSSIBLE DATETIME VALUES DETECTED ***")
            lines.append(f"  parsed sample(s): {detected[:5]}")
        preview = [str(v) for v in flat[:5]]
        lines.append(f"  preview    : {preview}" + (" ..." if arr.size > 5 else ""))
        return "\n".join(lines)

    # --- numeric arrays ---
    flat = arr.flatten().astype(float)
    lines.append(f"  min        : {flat.min():.6g}")
    lines.append(f"  max        : {flat.max():.6g}")
    lines.append(f"  mean       : {flat.mean():.6g}")
    lines.append(f"  std        : {flat.std():.6g}")

    # check if numeric values look like Unix timestamps
    dt_hits = []
    for v in flat[:20]:
        dt = try_parse_datetime(v)
        if dt:
            dt_hits.append(dt)
    if dt_hits:
        lines.append(f"  *** VALUES MAY BE UNIX TIMESTAMPS ***")
        lines.append(f"  as UTC datetime(s): {dt_hits[:5]}")

    if arr.size <= 10:
        lines.append(f"  values     : {arr.tolist()}")
    else:
        preview = arr.flatten()[:5].tolist()
        lines.append(f"  preview    : {preview} ...")

    return "\n".join(lines)


def read_npz(path):
    print(f"\n{'='*60}")
    print(f"  NPZ FILE: {path}")
    print(f"{'='*60}\n")

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Could not load file — {e}")
        sys.exit(1)

    keys = list(data.files)
    print(f"  Arrays stored ({len(keys)} total): {keys}\n")

    for key in keys:
        arr = data[key]
        print(f"{'─'*60}")
        print(f"  KEY: '{key}'")
        print(summarize_array(arr))
        print()

    print(f"{'='*60}")
    print("  Done.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_npz.py <path_to_file.npz>")
        sys.exit(1)
    read_npz(sys.argv[1])