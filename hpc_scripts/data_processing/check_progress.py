# check_progress.py
# Compares number of .npz files generated vs total PIREPs expected
# Usage: python check_progress.py

import os
import pandas as pd

REPO = os.environ.get("SAT_REPO_PATH", "/cluster/tufts/capstone25skyblue/scleme01/SatelliteTurbulencePrediction")
PIREPS_DIR = os.path.join(REPO, "pireps")
INPUTS_DIR = os.path.join(REPO, "new_model_inputs")

# GOES-16 availability: March 2017 to April 2025
YEARS = [str(y) for y in range(2017, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

total_expected = 0
total_found = 0

print(f"{'Month':<12} {'Found':>8} {'Expected':>10} {'Progress':>10}")
print("-" * 44)

for year in YEARS:
    for month in MONTHS:
        # Skip Jan/Feb 2017
        if year == "2017" and int(month) < 3:
            continue
        # Skip May-Dec 2025 (not yet on AWS)
        if year == "2025" and int(month) > 4:
            continue

        pirep_csv = os.path.join(PIREPS_DIR, year, f"{month}_turb_pireps.csv")
        output_dir = os.path.join(INPUTS_DIR, f"{year}_{month}")

        if not os.path.exists(pirep_csv):
            continue

        # Count expected (subtract 1 for header)
        expected = sum(1 for _ in open(pirep_csv)) - 1

        # Count generated
        found = len([f for f in os.listdir(output_dir) if f.endswith(".npz")]) if os.path.exists(output_dir) else 0

        total_expected += expected
        total_found += found

        pct = (found / expected * 100) if expected > 0 else 0
        print(f"{year}_{month:<6} {found:>8} / {expected:<10} {pct:>8.1f}%")

print("-" * 44)
pct_total = (total_found / total_expected * 100) if total_expected > 0 else 0
print(f"{'TOTAL':<12} {total_found:>8} / {total_expected:<10} {pct_total:>8.1f}%")
