# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo batch samplers."""

import abc
import random
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Optional, Union

import torch
from rtree.index import Index, Property
from torch.utils.data import Sampler

from torchgeo.datasets import BoundingBox, GeoDataset

from ..datasets import BoundingBox, GeoDataset
from .constants import Units
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips


class BatchGeoSampler(Sampler[list[BoundingBox]], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.BatchSampler`, :class:`BatchGeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: Optional[BoundingBox] = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        if roi is None:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)
        else:
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            hits = dataset.index.intersection(tuple(roi), objects=True)
            for hit in hits:
                bbox = BoundingBox(*hit.bounds) & roi
                self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[list[BoundingBox]]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        batch_size: int,
        length: Optional[int] = None,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            batch_size: number of samples per batch
            length: number of samples per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.batch_size = batch_size
        self.length = 0
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1
        if length is not None:
            self.length = length

    def __iter__(self) -> Iterator[list[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose random indices within that tile
            batch = []
            for _ in range(self.batch_size):
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                batch.append(bounding_box)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size


class BatchTimeWindowGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region and time of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible that vary
    both across spatial and time extent within the batch. Note that
    randomly sampled chips may overlap in both the spatial and time dimension.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        batch_size: int,
        encoder_length: int,
        prediction_length: int,
        time_unit: str,
        time_delta: int = 0,
        consecutive: bool = True,
        time_range: Optional[tuple[datetime, datetime]] = None,
        roi: Optional[BoundingBox] = None,
        size_units: Units = Units.PIXELS,
        max_samples_per_geolocation: int = None,
        length: Optional[int] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            batch_size: number of samples per batch
            length: number of samples per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units

        .. versionadded:: 0.5
        """
        super().__init__(dataset, roi)

        self.size = _to_tuple(size)

        # check time unit
        assert (
            time_unit in self.allowed_time_units
        ), f"Currently, only supporting one of {self.allowed_time_units} as time unit."
        self.time_unit = time_unit
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.consecutive = consecutive

        if size_units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0

        self.hits: list[list[float]] = []
        areas: list[float] = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

        # get all possible subsequence within the time range
        possible_input_time_ranges = self._compute_subsequences()

        # if there are a lot of possible subsequences
        # limit them
        random.shuffle(possible_input_time_ranges)
        if max_samples_per_geolocation is not None:
            possible_input_time_ranges = possible_input_time_ranges[
                :max_samples_per_geolocation
            ]

        self.possible_input_time_ranges = possible_input_time_ranges

    def __iter__(self) -> Iterator[list[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        return super().__iter__()
