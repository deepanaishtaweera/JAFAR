import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes


class CityscapesDataset(Dataset):
    BASE_DIR = "cityscapes"
    NUM_CLASS = 19
    # Cityscapes class labels in the format {"id": idx, "isthing": X, "name": class_name}
    CLASS_LABELS = [
        {"id": 0,  "isthing": 0, "name": "road"},
        {"id": 1,  "isthing": 0, "name": "sidewalk"},
        {"id": 2,  "isthing": 0, "name": "building"},
        {"id": 3,  "isthing": 0, "name": "wall"},
        {"id": 4,  "isthing": 0, "name": "fence"},
        {"id": 5,  "isthing": 1, "name": "pole"},
        {"id": 6,  "isthing": 1, "name": "traffic light"},
        {"id": 7,  "isthing": 1, "name": "traffic sign"},
        {"id": 8,  "isthing": 0, "name": "vegetation"},
        {"id": 9,  "isthing": 0, "name": "terrain"},
        {"id": 10, "isthing": 0, "name": "sky"},
        {"id": 11, "isthing": 1, "name": "person"},
        {"id": 12, "isthing": 1, "name": "rider"},
        {"id": 13, "isthing": 1, "name": "car"},
        {"id": 14, "isthing": 1, "name": "truck"},
        {"id": 15, "isthing": 1, "name": "bus"},
        {"id": 16, "isthing": 1, "name": "train"},
        {"id": 17, "isthing": 1, "name": "motorcycle"},
        {"id": 18, "isthing": 1, "name": "bicycle"},
    ]

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        include_labels=True,
        num_classes=19,
        tag=None,
        **kwargs
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.include_labels = include_labels
        self.num_classes = num_classes
        self.tag = tag

        # fmt: off
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype("int32")
        # fmt: on

        # Get image/mask path pairs
        self.cityscapes_dataset = Cityscapes(
            root=root,
            split=split,
            mode="fine",
            target_type=["instance", "semantic"],
            transform=None,
            target_transform=None,
        )
        # fmt: on

        # if split == "train":
        #     assert self.__len__() == 2975
        # elif split == "val":
        #     assert self.__len__() == 500

    def __len__(self):
        return len(self.cityscapes_dataset)

    def __getitem__(self, index):
        batch = {}

        # Load image
        data = self.cityscapes_dataset[index]
        img = data[0]
        mask = data[1][1]

        if self.transform:
            img = self.transform(img)
        batch["image"] = img

        if self.include_labels:
            if self.target_transform:
                mask = self.target_transform(mask)
            # Convert to tensor and match COCO/ADE20K format
            mask = torch.from_numpy(self._class_to_index(np.array(mask))).to(torch.uint8)
            batch["label"] = mask.squeeze()

        return batch

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert value in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape).astype(np.int64)
