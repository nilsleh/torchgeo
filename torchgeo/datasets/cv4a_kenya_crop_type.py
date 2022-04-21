# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CV4A Kenya Crop Type dataset."""

import glob
import json
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from torch import Tensor

from .geo import GeoDataset
from .utils import (
    BoundingBox,
    check_integrity,
    disambiguate_timestamp,
    download_radiant_mlhub_dataset,
    extract_archive,
    percentile_normalization,
)


class CV4AKenyaCropType(GeoDataset):
    """CV4A Kenya Crop Type dataset.

    Used in a competition in the Computer Vision for Agriculture (CV4A) workshop in
    ICLR 2020. See `this website <https://registry.mlhub.earth/10.34911/rdnt.dw605x/>`__
    for dataset details.

    Consists of 4 tiles of Sentinel 2 imagery from 13 different points in time.

    Each tile has:

    * 13 multi-band observations throughout the growing season. Each observation
      includes 12 bands from Sentinel-2 L2A product, and a cloud probability layer.
      The twelve bands are [B01, B02, B03, B04, B05, B06, B07, B08, B8A,
      B09, B11, B12] (refer to Sentinel-2 documentation for more information about
      the bands). The cloud probability layer is a product of the
      Sentinel-2 atmospheric correction algorithm (Sen2Cor) and provides an estimated
      cloud probability (0-100%) per pixel. All of the bands are mapped to a common
      10 m spatial resolution grid.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.DW605X

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """

    dataset_id = "ref_african_crops_kenya_02"
    image_meta = {
        "filename": "ref_african_crops_kenya_02_source.tar.gz",
        "directory": "ref_african_crops_kenya_02_source",
        "md5": "9c2004782f6dc83abb1bf45ba4d0da46",
    }
    target_meta = {
        "filename": "ref_african_crops_kenya_02_labels.tar.gz",
        "directory": "ref_african_crops_kenya_02_labels",
        "md5": "93949abd0ae82ba564f5a933cefd8215",
    }

    all_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "CLD",
    )

    RGB_BANDS = ["B04", "B03", "B02"]

    date_format = "%Y-%m-%dT%H:%M:%SZ"

    tile_height = 3035
    tile_width = 2016

    def __init__(
        self,
        root: str = "data",
        bands: Tuple[str, ...] = all_bands,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        cache: bool = True,
    ) -> None:
        """Initialize a new CV4A Kenya Crop Type Dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: the subset of bands to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self._validate_bands(bands)

        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.checksum = checksum
        self.download = download
        self.api_key = api_key
        self.cache = cache

        self._verify()

        super().__init__(transforms)

        # fill index base on stac.json files
        i = 0
        pathname = os.path.join(root, self.image_meta["directory"], "**", "stac.json")
        self.imgdir2bbox: Dict[str, Tuple] = {}
        for filepath in glob.iglob(pathname, recursive=True):
            label_dir = os.path.basename(os.path.dirname(filepath)).rsplit("_", 1)[0]
            label_path = os.path.join(
                self.root, self.target_meta["directory"], label_dir, "labels.geojson"
            )
            label_path = label_path.replace("_01_source_", "_01_labels_")

            if i == 0:
                # neither the .tif source or label files have crs
                # stac bounding boxes are in 4326
                if crs is None:
                    crs = CRS.from_epsg(4326)
                if res is None:
                    res = 10

            # bboxes in source imagery are in epsg '4326'
            transformer = Transformer.from_crs(
                CRS.from_epsg(4326), crs.to_epsg(), always_xy=True
            )

            with open(filepath) as stac_file:
                data = json.load(stac_file)

            minx, miny, maxx, maxy = data["bbox"]

            (minx, maxx), (miny, maxy) = transformer.transform(
                [minx, maxx], [miny, maxy]
            )

            date = data["properties"]["datetime"]
            mint, maxt = disambiguate_timestamp(date, self.date_format)

            coords = (minx, maxx, miny, maxy, mint, maxt)
            # insert dict here with filepath and label
            self.index.insert(
                i,
                coords,
                {"img_dir": os.path.dirname(filepath), "label_path": label_path},
            )
            self.imgdir2bbox[os.path.dirname(filepath)] = coords
            i += 1

        if i == 0:
            raise FileNotFoundError(f"No stac.json files found in '{root}'")

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask pair and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index TxCxHxW

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # load imagery
        data = self._load_separate_images(filepaths, query)
        import pdb

        pdb.set_trace()
        # load mask
        mask = self._load_mask(filepaths, query)

        sample = {"image": data, "mask": mask, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_separate_images(
        self, filepaths: Sequence[str], query: BoundingBox
    ) -> Tensor:
        """Load seperate band images.
        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            images at this query indey
        """
        # load imagery
        img_list: List[Tensor] = []
        for filepath in filepaths:
            band_filepaths = []
            for band in getattr(self, "bands", self.all_bands):
                band_filename = os.path.join(filepath["img_dir"], band + ".tif")
                band_filepaths.append(band_filename)
            img_list.append(self._concatenate_bands(band_filepaths, query))

        # stack along time dimension
        data = torch.stack(img_list, dim=0)

        return data

    def _concatenate_bands(
        self, filepaths: Sequence[str], query: BoundingBox
    ) -> Tensor:
        """Load and merge one or more files.
        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        img_list: List[Tensor] = []
        for src in vrt_fhs:
            out_width = int(round((query.maxx - query.minx) / self.res))
            out_height = int(round((query.maxy - query.miny) / self.res))
            out_shape = (src.count, out_height, out_width)
            dest = src.read(
                out_shape=out_shape, window=from_bounds(*bounds, src.transform)
            ).astype(np.int32)

            img_list.append(torch.from_numpy(dest))

        tensor = torch.cat(img_list, dim=0)
        return tensor

    @lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            img_info: file to load and warp, as well as bbox of that image

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            img_info: filepath to load and warp, as well as bbox of that image

        Returns:
            file handle of warped VRT
        """
        # since images do not have crs or proper bounds, open them
        # by transforming them to epsg 4326 based on accompanying bbox
        # minx, maxx, miny, maxy
        coords = self.imgdir2bbox[os.path.dirname(filepath)]
        transform = rasterio.transform.from_bounds(
            west=coords[2],
            east=coords[3],
            south=coords[0],
            north=coords[1],
            width=self.tile_width,
            height=self.tile_height,
        )
        bounds = (coords[0], coords[2], coords[1], coords[3])
        # out_width = int(round((coords[2] - coords[0]) / self.res))
        # out_height = int(round((coords[3] - coords[1]) / self.res))
        out_shape = (1, self.tile_height, self.tile_width)
        import pdb

        pdb.set_trace()
        src = rasterio.open(filepath)
        k = src.read(out_shape, window=from_bounds(*bounds, transform))
        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs, transform=transform)
            src.close()
            return vrt
        else:
            return src

    def _validate_bands(self, bands: Tuple[str, ...]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided tuple of bands to load

        Raises:
            AssertionError: if ``bands`` is not a tuple
            ValueError: if an invalid band name is provided
        """
        assert isinstance(bands, tuple), "The list of bands must be a tuple"
        for band in bands:
            if band not in self.all_bands:
                raise ValueError(f"'{band}' is an invalid band name.")

    def _verify(self) -> None:
        """Verify the integrity of the dataset.
        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if extracted files already exists
        exists = []
        for directory in [self.image_meta["directory"], self.target_meta["directory"]]:
            if os.path.exists(os.path.join(self.root, directory)):
                exists.append(True)
            else:
                exists.append(False)
        if all(exists):
            return

        # check if compressed files exists
        exists = []
        for filename, md5 in zip(
            [self.image_meta["filename"], self.target_meta["filename"]],
            [self.image_meta["md5"], self.target_meta["md5"]],
        ):
            filepath = os.path.join(self.root, filename)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, md5):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath, self.root)
            else:
                exists.append(False)

        if all(exists):
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
        download_radiant_mlhub_dataset(self.dataset_id, self.root, self.api_key)

    def _extract(self) -> None:
        """Extract the dataset."""
        image_archive_path = os.path.join(self.root, self.image_meta["filename"])
        target_archive_path = os.path.join(self.root, self.target_meta["filename"])
        for filename in [image_archive_path, target_archive_path]:
            extract_archive(filename, self.root)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        time_step: int = 0,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            time_step: time step at which to access image, beginning with 0
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        if "prediction" in sample:
            prediction = sample["prediction"]
            n_cols = 3
        else:
            n_cols = 2

        image, mask = sample["image"], sample["mask"]

        assert time_step <= image.shape[0] - 1, (
            "The specified time step"
            " does not exist, image only contains {} time"
            " instances."
        ).format(image.shape[0])

        image = image[time_step, rgb_indices, :, :]

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))

        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if "prediction" in sample:
            axs[2].imshow(prediction)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
