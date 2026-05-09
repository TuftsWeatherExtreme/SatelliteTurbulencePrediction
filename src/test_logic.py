# Authors: Razzle Dazzle Rose
# Spring 2026

import pandas as pd
import os

csv = "/cluster/tufts/capstone25skyblue/scleme01/SatelliteTurbulencePrediction/pireps/2019/05_turb_pireps.csv"

df = pd.read_csv(csv)
print("Original first 5 indices:", df.index.tolist()[:5])

df = df.sort_values('datetime').reset_index(drop=False).rename(columns={'index': 'orig_idx'})
print("After sort, orig_idx col:", df['orig_idx'].head().tolist())
print("Columns:", df.columns.tolist())

chunk = df.iloc[0:3].reset_index(drop=True)
print("Chunk rows:")
print(chunk[['orig_idx', 'datetime', 'LAT', 'LON']])

for _, row in chunk.iterrows():
    orig_idx = int(row['orig_idx'])
    path = f"/tmp/test_output/{orig_idx:07d}.npz"
    print(f"  Would save: {path}")
