# dataloader_class.py
# Authors: Team Celestial Blue
# Spring 2025
# Overview: PyTorch Dataset that lazily loads satellite .npz model inputs from disk.

import torch
import glob
from torch.utils.data import Dataset
import os
import numpy as np


class SatelliteDataLoader(Dataset):

    def __init__(self, model_inputs_dir):
        """
        Builds an index of all .npz files across year_month subdirectories.
        Files are loaded lazily in __getitem__ to avoid holding everything in memory.

        Args:
            model_inputs_dir: Path to the model_inputs/ directory containing
                              {year}_{month}/ subdirectories with .npz files.
        """
        self.file_paths = sorted(glob.glob(f"{model_inputs_dir}/**/*.npz", recursive=True))
        print(f"SatelliteDataLoader: found {len(self.file_paths)} samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz = np.load(self.file_paths[idx])
        # images shape: (32, 128, 128, 6) -> reorder to (32, 6, 128, 128) for PyTorch
        images = npz['images'].transpose(0, 3, 1, 2)
        features = torch.tensor(images, dtype=torch.float32)
        features = torch.nan_to_num(features, nan=0.0)
        label = int(npz['turb_label'])
        weight = float(npz['sample_weight'])
        return features, label, weight
