import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TrailDataset(Dataset):
    """Trail dataset with PNG images and segmentation masks.

    Directory layout under root:
      - images/          (e.g., 000000.png)
      - segmentations/   (e.g., 000000.png, uint8 with classes 0..26 and 255 ignore)
      - train_images.txt / val_images.txt (zero-padded indices per line)
    """

    # Duplicated trail label info as a class-level constant to remove dependency on keylab.py
    CLASS_LABELS = [
        {"id": 0, "isthing": 0, "name": "Dirt"},
        {"id": 1, "isthing": 0, "name": "Grass"},
        {"id": 2, "isthing": 0, "name": "Vegetation (including trees)"},
        {"id": 3, "isthing": 0, "name": "Water"},
        {"id": 4, "isthing": 0, "name": "Sky"},
        {"id": 5, "isthing": 0, "name": "Asphalt / street"},
        {"id": 6, "isthing": 0, "name": "Gravel"},
        {"id": 7, "isthing": 0, "name": "Building"},
        {"id": 8, "isthing": 0, "name": "Rock-bed"},
        {"id": 9, "isthing": 0, "name": "Skill element"},
        {"id": 10, "isthing": 0, "name": "Fallen leaves"},
        {"id": 11, "isthing": 0, "name": "Log / root"},
        {"id": 12, "isthing": 0, "name": "Bridge"},
        {"id": 13, "isthing": 0, "name": "Asphalt"},
        {"id": 14, "isthing": 0, "name": "Path / Trail"},
        {"id": 15, "isthing": 0, "name": "Sidewalk"},
        {"id": 16, "isthing": 0, "name": "Lane lines"},
        {"id": 17, "isthing": 0, "name": "Fence"},
        {"id": 18, "isthing": 0, "name": "other"},
        {"id": 19, "isthing": 1, "name": "Pole"},
        {"id": 20, "isthing": 1, "name": "Person"},
        {"id": 21, "isthing": 1, "name": "Vehicle"},
        {"id": 22, "isthing": 1, "name": "Bicycle"},
        {"id": 23, "isthing": 1, "name": "Animal"},
        {"id": 24, "isthing": 1, "name": "Sign"},
        {"id": 25, "isthing": 1, "name": "Rock / stone"},
        {"id": 26, "isthing": 1, "name": "other"},
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        include_labels: bool = True,
        num_classes: int = 27,
        tag: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        assert split in {"train", "val"}, f"Unsupported split: {split}"

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.include_labels = include_labels
        self.num_classes = num_classes
        self.tag = tag

        images_dir = os.path.join(root, "images")
        seg_dir = os.path.join(root, "segmentations")
        split_file = os.path.join(root, f"{split}_images.txt")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            indices = [line.strip() for line in f if line.strip()]

        # Build lists of absolute file paths
        self.image_files = [os.path.join(images_dir, f"{idx}.png") for idx in indices]
        self.label_files = [os.path.join(seg_dir, f"{idx}.png") for idx in indices]

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found using split: {split_file}")

        # Basic consistency check
        for img_p, lab_p in zip(self.image_files, self.label_files):
            if not os.path.exists(img_p):
                raise FileNotFoundError(f"Image missing: {img_p}")
            if self.include_labels and not os.path.exists(lab_p):
                raise FileNotFoundError(f"Label missing: {lab_p}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        batch = {}

        img_path = self.image_files[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        batch["image"] = img
        batch["img_path"] = img_path

        if self.include_labels:
            label_path = self.label_files[index]
            label_img = Image.open(label_path)

            if self.target_transform:
                label_img = self.target_transform(label_img)

            # Ensure uint8 tensor [H, W]
            if isinstance(label_img, Image.Image):
                label_np = np.array(label_img, dtype=np.uint8)
                label = torch.from_numpy(label_np)
            else:
                # Assume transforms returned a tensor; squeeze possible channel dim
                label = label_img
                if label.ndim == 3 and label.shape[0] == 1:
                    label = label.squeeze(0)
                label = label.to(torch.uint8)

            batch["label"] = label

        return batch


