# data/ae_dataset.py
# Flat image list dataset for autoencoder training.
# Returns images normalized to [-1, 1] as required by LPIPSWithDiscriminator.

import os
import random
import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset


class AEDataset(Dataset):
    """
    Reads a flat text file where each line is an absolute path to an image.
    Resizes to `image_size` (square crop), normalizes to [-1, 1].

    Expected config fields:
        config.training.dataset_path  : path to image_list.txt
        config.training.image_size    : int, e.g. 256
        config.training.random_flip   : bool (default True)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.training.image_size
        self.random_flip = config.training.get("random_flip", False)

        list_path = config.training.dataset_path
        with open(list_path, "r") as f:
            paths = [l.strip() for l in f if l.strip()]

        # Filter to only existing files (guards against stale symlinks)
        self.image_paths = [p for p in paths if os.path.isfile(p)]
        missing = len(paths) - len(self.image_paths)
        if missing > 0:
            print(f"[AEDataset] WARNING: {missing} paths missing/broken, skipping them.")

        print(f"[AEDataset] {len(self.image_paths)} images loaded from {list_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            return self._load(idx)
        except Exception as e:
            print(f"[AEDataset] Error loading {self.image_paths[idx]}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _load(self, idx):
        path = self.image_paths[idx]
        image = PIL.Image.open(path).convert("RGB")

        # Square crop from centre, then resize
        w, h = image.size
        min_side = min(w, h)
        left  = (w - min_side) // 2
        top   = (h - min_side) // 2
        image = image.crop((left, top, left + min_side, top + min_side))
        image = image.resize((self.image_size, self.image_size), PIL.Image.LANCZOS)

        if self.random_flip and random.random() < 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # [0, 255] -> [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]

        return {"image": image, "path": path}