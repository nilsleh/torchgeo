# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BirdDetection dataset."""

import os
from collections.abc import Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, download_url, extract_archive


class BirdDetection(NonGeoDataset):
    """Bird Detection dataset.

    The `Bird Detection <https://zenodo.org/records/5033174>`__
    dataset contains high-resolution aerial images across the globe with bounding box annotations of birds.


    Dataset features:

    * 5128 training images with 50491 annotations
    * 197 test images with 4113 annotations
    * bounding box annotations for bird, no species labels

    Dataset format:

    * images are three-channel pngs
    * annotations are .parquet file

    If you use this dataset in your research, please cite the following source:

    * https://doi.org/10.5281/zenodo.5033174

    .. versionadded:: 0.7
    """

    dir = 'images'

    url = 'https://hf.co/datasets/torchgeo/bird_detection/resolve/c8d361c6ccf6079efa64e50fef627e345797944f/{}'

    files = [
        {'filename': 'images.tar.gzaa', 'md5': '9f231519f78279b80f5365f749392488'},
        {'filename': 'images.tar.gzab', 'md5': 'a8234b5935714a0befc5f285528f03a2'},
        {'filename': 'metadata.parquet', 'md5': '28568e4df4affb6b572c6128575a45e8'},
    ]

    valid_splits = ('train', 'test')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BirdDetection dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of {'train', 'test'} to specify the dataset split
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            AssertionError: If *split* argument is invalid.
        """
        assert split in self.valid_splits, (
            f"Split '{split}' not supported, please use one of {self.valid_splits}"
        )

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self._verify()

        self.annot_df = pd.read_parquet(
            os.path.join(self.root, 'metadata.parquet')
        )

        self.annot_df = self.annot_df[self.annot_df['split'] == self.split].reset_index(drop=True)

        # remove all entries where xmin == xmax or ymin == ymax
        self.annot_df = self.annot_df[
            (self.annot_df['xmin'] != self.annot_df['xmax'])
            & (self.annot_df['ymin'] != self.annot_df['ymax'])
        ].reset_index(drop=True)

        # group per image path to get all annotations for one sample
        self.annot_df['sample_index'] = pd.factorize(self.annot_df['image_path'])[0]
        self.annot_df = self.annot_df.set_index(['sample_index', self.annot_df.index])


    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.annot_df.index.levels[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index, labels are just ones
        """
        sample_df = self.annot_df.loc[index]

        img_path = os.path.join(self.root, self.dir, sample_df['image_path'].iloc[0])

        image = self._load_image(img_path)

        boxes, labels = self._load_target(sample_df)

        sample = {'image': image, 'bbox_xyxy': boxes, 'label': labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: np.typing.NDArray[np.uint8] = np.array(img)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, sample_df: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """Load target from a dataframe row.

        Args:
            sample_df: df subset with annotations for specific image

        Returns:
            bounding boxes and labels
        """
        boxes = torch.Tensor(
            sample_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        ).float()
        labels = torch.ones(len(boxes)).long()
        return boxes, labels

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        exists = []
        df_path = os.path.join(self.root, f'metadata.parquet')
        if os.path.exists(df_path):
            exists.append(True)
            if os.path.exists(os.path.join(self.root, self.dir)):
                exists.append(True)
        else:
            exists.append(False)

        if all(exists):
            return

        exists = []
        for file in self.files:
            path = os.path.join(self.root, file['filename'])
            if os.path.exists(path):
                if self.checksum and not check_integrity(filepath, self.md5):
                    raise RuntimeError('Dataset found, but corrupted.')
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        for file in self.files:
            download_url(
                self.url.format(file['filename']),
                self.root,
                file['filename'],
                file['md5'] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Download the dataset and extract it."""
        chunk_size = 2**15  # same as used in torchvision and ssl4eo

        image_parts = ['images.tar.gzaa', 'images.tar.gzab']

        concat_path = os.path.join(self.root, 'images.tar.gz')
        with open(concat_path, 'wb') as outfile:
            for part in image_parts:
                with open(part, 'rb') as g:
                    while chunk := g.read(chunk_size):
                        outfile.write(chunk)
        extract_archive(concat_path, self.root)
        

    def plot(
        self,
        sample: dict[str, Tensor],
        suptitle: str | None = None,
        box_alpha: float = 0.7,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            box_alpha: alpha value for boxes

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample['image'].permute((1, 2, 0)).numpy()
        boxes = sample['bbox_xyxy'].numpy()

        fig, axs = plt.subplots(ncols=1, figsize=(10, 10))

        axs.imshow(image)
        axs.axis('off')

        cm = plt.get_cmap('gist_rainbow')

        for box in boxes:
            # Horizontal box: [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=box_alpha,
                linestyle='solid',
                edgecolor='red',
                facecolor='none',
            )
            axs.add_patch(rect)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig