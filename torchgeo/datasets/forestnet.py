# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ForestNet dataset."""

import csv
import glob
import os
import pickle
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw
from torch import Tensor

from .geo import VisionDataset
from .utils import download_url, extract_archive, working_dir


class ForestNetDataset(VisionDataset):
    """ForestNet Dataset.

    Dataset containing images and masks for forest loss detection.

    Dataset features:

    Dataset format:


    If you use this dataset in your research, please give credit to:

    * https://arxiv.org/abs/2011.05479

    .. versionadded:: 0.3

    """

    url = "http://download.cs.stanford.edu/deep/ForestNetDataset.zip"
    md5 = "90399df485c681524190175aad3e951d"
    directory = "ForestNetDataset"
    filename = "ForestNetDataset.zip"

    label_ignore_value = 0

    class_labels = [
        "No data",
        "Timber plantation",
        "Smallholder Agriculture",
        "Grassland/shrubland",
        "Other",
    ]

    spatial_resolution = 15
    height = 332
    widht = 332

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LandCover.ai dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in ["train", "val", "test"]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.class2idx = {c: i for i, c in enumerate(self.class_labels)}

        # read datasplits
        self.indices = []
        with open(os.path.join(self.root, self.directory, split + ".csv")) as f:
            self.indices = [row for row in csv.DictReader(f, skipinitialspace=True)]

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        item = self.indices[index]
        img = self._load_image(item["example_path"])
        mask = self._load_target(item["example_path"], item["label"])
        sample = {"image": img, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @lru_cache()
    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to image

        Returns:
            the image
        """
        # Load the visible and infrared image
        visible_path = os.path.join(
            self.root, self.directory, path, "images", "visible", "composite.png"
        )
        infrared_path = os.path.join(
            self.root, self.directory, path, "images", "infrared", "composite.npy"
        )
        visible_img = np.array(Image.open(visible_path).convert("RGB"))
        infrared_img = np.load(infrared_path)
        img = np.transpose(
            np.concatenate([visible_img, infrared_img], axis=2, dtype=np.int32),
            (2, 0, 1),
        )
        tensor = torch.from_numpy(img)

        return tensor

    @lru_cache()
    def _load_target(self, path: str, label: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: filepath to mask
            label: label of mask

        Returns:
            the target mask
        """
        mask_path = os.path.join(
            self.root, self.directory, path, "forest_loss_region.pkl"
        )
        with open(mask_path, "rb") as f:
            poly_shape = pickle.load(f)

        mask = Image.new("L", (self.height, self.width), self.label_ignore_value)
        label_int = self.class2idx[label]
        shape_type = poly_shape.geom_type
        if shape_type == "Polygon":
            coords = np.array(poly_shape.exterior.coords)
            ImageDraw.Draw(mask).polygon(
                [tuple(coord) for coord in coords], outline=label_int, fill=label_int
            )
        else:
            for poly in poly_shape.geoms:
                coords = np.array(poly.exterior.coords)
                ImageDraw.Draw(mask).polygon(
                    [tuple(coord) for coord in coords],
                    outline=label_int,
                    fill=label_int,
                )

        tensor = torch.from_numpy(np.array(mask))

        return tensor

    def _load_meta_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Load bounding box and epsg.

        Args:
            item: contains meta data of indexed data point

        Returns:
            bounding box and epsg information
        """
        lat_center, lon_center = item["latitude"], item["longitude"]

        transform_center = rasterio.transform.from_origin(
            lon_center, lat_center, self.spatial_resolution, self.spatial_resolution
        )

        lon_corner, lat_corner = transform_center * [
            -self.height // 2,
            -self.width // 2,
        ]
        transform = rasterio.transform.from_origin(
            lon_corner, lat_corner, self.spatial_resolution, self.spatial_resolution
        )
        return {}

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.directory)
        if os.path.exists(pathname):
            return

        # Check if the zip file has already been downloaded
        pathname = os.path.join(self.root, self.filename)
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(self.url, self.root, md5=self.md5 if self.checksum else None)

    def _extract(self) -> None:
        """Extract the dataset.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        extract_archive(os.path.join(self.root, self.filename))
