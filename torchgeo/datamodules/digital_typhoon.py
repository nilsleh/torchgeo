# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Digital Typhoon Data Module."""

import copy
from typing import Any, ClassVar

from torch.utils.data import Subset

from ..datasets import DigitalTyphoonAnalysis
from ..datasets.digital_typhoon import _SampleSequenceDict
from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class DigitalTyphoonAnalysisDataModule(NonGeoDataModule):
    """Digital Typhoon Analysis Data Module."""

    valid_split_types: ClassVar[list[str]] = ['time', 'typhoon_id']

    def __init__(
        self,
        split_by: str = 'time',
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new DigitalTyphoonAnalysisDataModule instance.

        Args:
            split_by: Either 'time' or 'typhoon_id', which decides how to split
                the dataset for train, val, test
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DigitalTyphoonAnalysis`.

        """
        super().__init__(DigitalTyphoonAnalysis, batch_size, num_workers, **kwargs)

        assert (
            split_by in self.valid_split_types
        ), f'Please choose from {self.valid_split_types}'
        self.split_by = split_by

    def _split_dataset(
        self, sample_sequences: list[_SampleSequenceDict]
    ) -> tuple[list[int], list[int]]:
        """Split dataset into two parts.

        Args:
            sample_sequences: List of sample sequence dictionaries to be split

        Returns:
            a tuple of the subset datasets
        """
        if self.split_by == 'time':
            sequences = list(enumerate(sample_sequences))

            sorted_sequences = sorted(sequences, key=lambda x: x[1]['seq_id'])
            selected_indices = [x[0] for x in sorted_sequences]

            split_idx = int(len(sorted_sequences) * 0.8)
            train_indices = selected_indices[:split_idx]
            val_indices = selected_indices[split_idx:]

        else:
            sequences = list(enumerate(sample_sequences))
            train_indices, val_indices = group_shuffle_split(
                [x[1]['id'] for x in sequences], train_size=0.8, random_state=0
            )

        return train_indices, val_indices

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = DigitalTyphoonAnalysis(**self.kwargs)

        all_sample_sequences = copy.deepcopy(self.dataset.sample_sequences)

        train_indices, test_indices = self._split_dataset(self.dataset.sample_sequences)

        if stage in ['fit', 'validate']:
            # Randomly split train into train and validation sets
            index_mapping = {
                new_index: original_index
                for new_index, original_index in enumerate(train_indices)
            }
            train_sequences = [all_sample_sequences[i] for i in train_indices]
            train_indices, val_indices = self._split_dataset(train_sequences)
            train_indices = [index_mapping[i] for i in train_indices]
            val_indices = [index_mapping[i] for i in val_indices]

            # Create train val subset dataset
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)

        if stage in ['test']:
            self.test_dataset = Subset(self.dataset, test_indices)