# check_data_quality.py
# Authors: Razzle Dazzle Rose
# Spring 2026
# Checking if the satellite image data is pulling severe turbulence
# NOTE: took 3ish minutes to run with 7.8k files. 
# will watch to sbatch it if running with full data (about 45k)

import os, numpy as np, glob

data_dir = os.environ["SAT_REPO_PATH"] + "/model_inputs"
print(os.environ.get("SAT_REPO_PATH"))
print(os.path.join(os.environ["SAT_REPO_PATH"], "model_inputs"))

# files = glob.glob(f"{data_dir}/2017_03/*.npz")
# print("files found:", len(files))

files = sorted(glob.glob(f"{data_dir}/*/*.npz"))
print(f"files found: {len(files)}")


labels = []
issues = []
for f in files:
    d = np.load(f)
    imgs = d['images']
    label = int(d['turb_label'])
    labels.append(label)
    if imgs.shape != (15, 128, 128, 6):
        issues.append(f"{f}: bad shape {imgs.shape}")
    if np.isnan(imgs).any():
        issues.append(f"{f}: contains NaN")
    if imgs.max() == 0:
        issues.append(f"{f}: all zeros")

print(f"Checked {len(files)} files")
print(f"Label distribution: {labels.count(0)} no-turbulence, {labels.count(1)} turbulence")
print(f"Issues found: {len(issues)}")
for i in issues[:10]:
    print(" ", i)
