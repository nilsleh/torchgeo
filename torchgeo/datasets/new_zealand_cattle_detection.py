# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""New Zealand Cattle Detection dataset."""

import glob
import os
from functools import lru_cache
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import VisionDataset
from .utils import check_integrity, download_url, extract_archive


class NewZealandCattleDetection(VisionDataset):
    """New Zealand Cattle Detection dataset.

    Aerial RGB dataset with point annotations of cattle as detection task.
    Target is a binary mask with point annotation of cattle location.

    Dataset features:

    * Aerial RGB images at spatial resolution of 0.1m. (500 x 500 px)
    * 655 images
    * 29803 annoted locations of cattle across all images

    Dataset format:

    * RGB images in .png format
    * point annotations in text files

    If you use this dataset in your research, please cite the following paper:

    Dr. Diab Abuaiadah, & Alexander Switzer. (2022).
    New Zealand Cattle Detection (1.0.0) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.5908869

    .. versionadded:: 0.3
    """

    filename = "images.zip"
    directory = "images"
    url = "https://zenodo.org/record/5908869/files/images.zip?download=1"
    md5 = "da6d596c3a9a0ca7a220c604b5c85580"
    height = 500
    width = 500

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new New Zealand Cattle Detection dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.ids = glob.glob(os.path.join(root, self.directory, "cow_images", "*.png"))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        img_path = self.ids[index]
        mask_path = img_path.replace(".png", ".png.mask.0.txt")
        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_target(mask_path),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    @lru_cache()
    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to image file

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    @lru_cache()
    def _load_target(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to mask file

        Returns:
            the mask
        """
        points: "np.typing.NDArray[np.int_]" = np.loadtxt(
            path, delimiter=",", dtype=np.int32
        )
        mask: "np.typing.NDArray[np.int_]" = np.zeros(
            (self.height, self.width), dtype=np.int8
        )
        mask[points[:, 1], points[:, 0]] = 1
        tensor = torch.from_numpy(mask)
        return tensor

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
            if self.checksum and not check_integrity(pathname, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
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

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.transpose(sample["image"].numpy(), (1, 2, 0))
        mask = sample["mask"].numpy()

        num_cols = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_cols += 1

        fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
