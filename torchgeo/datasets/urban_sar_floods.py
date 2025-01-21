# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Urban SAR Floods dataset."""

import glob
import os
from collections.abc import Callable
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_and_extract_archive, extract_archive


class UrbanSARFloods(NonGeoDataset):
    """Urban SAR Floods dataset.
    
    The `Urban SAR Floods dataset <https://arxiv.org/abs/2406.04111>`_ contains SAR imagery and segmentation tasks
    for flood detection.

    Dataset features:


    Dataset format:


    Dataset classes:


    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2406.04111
    """

    valid_splits = ('train', 'val')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        fused: bool = True,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Urban SAR floods dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", or "val"
            fused: whether or not to fuse pre- and post-event images, by stacking
                them along the channel dimension, or returning them under separate image keys
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.valid_splits, f'split must be one of {self.valid_splits}, got "{split}".'

        self.root = root
        self.split = split
        self.fused = fused
        self.transforms = transforms

        self._verify()
        self.file_paths = self._get_file_paths()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        gt_path = self.file_paths[index]
        sar_path = gt_path.replace('GT', 'SAR')

        sample: dict[str, Tensor] = {}
        image = self._load_image(sar_path)
        if not self.fused:
            pre_image = image[:4]
            post_image = image[4:]
            sample["pre_image"] = image[:4]
            sample["post_image"] = image[4:]
        else:
            sample["image"] = image

        mask = self._load_mask(gt_path)
        sample["mask"] = mask

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.file_paths)

    def _load_image(self, path: str) -> Tensor:
        """Load an image from a file path.

        Args: 
            path to SAR image

        Returns: 
            image as a tensor
        """
        pass

    def _load_mask(self, path: str) -> Tensor:
        """Load a mask from a file path.

        Args: 
            path to mask

        Returns: 
            mask as a tensor
        """
        pass

    def _get_file_paths(self) -> list[str]:
        """Get file paths for the dataset.
        
        Returns:
            list of file paths for respective data split
        """
        if self.split == 'train':
            path = os.path.join(self.root, 'Train_dataset.txt')
        else:
            path = os.path.join(self.root, 'Valid_dataset.txt')
        with open(path, 'r') as f:
            file_paths: list[str] = f.readlines()
        return file_paths
        

    def _verify() -> None:
        """Verify the dataset."""
        pass

from os.path import splitext
import os
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import random
from osgeo import gdal


class BasicDataset(Dataset):
    def __init__(
        self,
        imgs_dir,
        masks_dir,
        mask_suffix_1,
        urban_mask,
        normalize_in,
        data_type,
        patch_size,
        input_dim=None,
        transform=True,
        std=False,
    ):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix_1 = mask_suffix_1  # GT
        self.urban_mask_suffix = urban_mask  # urban mask
        self.transform = transform
        self.normalize_in = normalize_in
        self.input_dim = input_dim
        self.data_type = data_type  # train / valid
        self.std = std
        self.patch_size = patch_size
        self.ids = [
            os.path.basename(file).split('_SAR')[0]
            for file in imgs_dir
            if not file.startswith('.')
        ]
        logging.info(f'Creating dataset with {len(self.imgs_dir)} examples')

    def __len__(self):
        return len(self.imgs_dir)

    @classmethod
    def standarized(cls, img, mean, std):
        mean_list = []
        std_list = []
        mean_list = list(mean)
        std_list = list(std)
        if img.shape[0] == 9:
            mean_list.append(0)
            std_list.append(1)
        return (
            (img - np.array(mean_list)[:, None, None])
            / np.array(std_list)[:, None, None]
            @ classmethod
        )

    def normalize(cls, img, max, min):
        max_list = []
        min_list = []
        max_list = list(max)
        min_list = list(min)
        if img.shape[0] == 9:
            max_list.append(1)
            min_list.append(0)
        img_out = (img - np.array(min_list)[:, None, None]) / (
            np.array(max_list)[:, None, None] - np.array(min_list)[:, None, None]
        )
        img_out[0:4, :, :] = img_out[0:4, :, :]
        img_out[img == -9999.0] = -9999.0
        img_out[img == np.nan] = -9999
        return img_out

    @classmethod
    def random_crop(cls, img, patch_size):
        row = img.shape[1]
        col = img.shape[2]
        r = random.randint(0, row - patch_size)
        c = random.randint(0, col - patch_size)
        sub_image = img[:, r : r + patch_size, c : c + patch_size]
        if sub_image.shape[1] != patch_size or sub_image.shape[2] != patch_size:
            print('image size: ', sub_image.shape)
        return sub_image

    @classmethod
    def random_horizontal_flipping(cls, img):
        rhc = random.randint(0, 1)
        img_new = np.empty(shape=img.shape)
        if rhc == 0:
            for row in range(img.shape[1]):
                img_new[:, row, :] = img[:, img.shape[0] - row - 1, :]
        else:
            img_new = img
        return img_new

    @classmethod
    def random_vertical_flipping(cls, img):
        rvc = random.randint(0, 1)
        img_new = np.empty(shape=img.shape)
        if rvc == 0:
            for col in range(img.shape[1]):
                img_new[:, :, col] = img[:, :, img.shape[1] - col - 1]
        else:
            img_new = img
        return img_new

    @classmethod
    def random_rotation(cls, img):
        rv = random.randint(0, 3)
        img_new = np.empty(shape=img.shape)
        for ii in range(img.shape[0]):
            img_new[ii, :, :] = np.rot90(img[ii, :, :], rv, axes=(0, 1))
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(img[ii, :, :])
            # axs[1].imshow(img_new[ii, :, :])
            # plt.show()
        return img_new

    def __getitem__(self, i):
        idx = self.ids[i]
        mean, std, max, min = self.normalize_in
        # mask_file = glob(self.masks_dir + idx.split('_')[0] + '_' + idx.split('_')[1] + self.mask_suffix_1 + '_'+ idx.split('_')[2]+ '.*')
        img_file = self.imgs_dir[i]
        mask_file = self.masks_dir[i]
        # mask_file = [x.replace('SAR','GT') for x in img_file]
        # mask_file = glob(self.masks_dir + idx.split('_')[0] + '_' + idx.split('_')[1] + self.mask_suffix_1 + '.*')
        # img_file = glob(self.imgs_dir + idx + '.*')
        # temp = self.masks_dir + idx.split('_')[0] + '_' + idx.split('_')[1] + self.mask_suffix_1 + '.*'
        assert len(mask_file) != 1, (
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file} --> {mask_file}'
        )
        assert len(img_file) != 1, (
            f'Either no image or multiple images found for the ID {idx}: {img_file}--> {img_file}'
        )
        mask_ds = gdal.Open(mask_file)
        mask = np.zeros((mask_ds.RasterCount, mask_ds.RasterYSize, mask_ds.RasterXSize))
        for ii in range(mask_ds.RasterCount):
            mask[ii, :, :] = mask_ds.GetRasterBand(ii + 1).ReadAsArray()
            mask[mask == -9999.0] = 0
            mask[mask == np.nan] = 0
        img_ds = gdal.Open(img_file)
        img = np.zeros((img_ds.RasterCount, img_ds.RasterYSize, img_ds.RasterXSize))
        for ii in range(8):
            # for ii in range(img_ds.RasterCount):
            temp = img_ds.GetRasterBand(ii + 1).ReadAsArray()
            temp[temp == np.nan] = 0
            # remove outlier
            temp[temp > max[ii]] = max[ii]
            temp[temp < min[ii]] = min[ii]
            img[ii, :, :] = temp
        if self.std:
            std_filename = idx + '_STD.tif'
            # file_std = os.path.join(std_folder, std_filename)
            folder_std = os.path.dirname(os.path.dirname(self.imgs_dir))
            folder_std_name = (
                os.path.basename(os.path.dirname(self.imgs_dir)).split('_')[0]
                + '_STD_'
                + os.path.basename(os.path.dirname(self.imgs_dir)).split('_')[1]
            )
            file_std = os.path.join(folder_std, folder_std_name, std_filename)
            assert os.path.exists(file_std), (
                f'Please check the path of STD file: {file_std}'
            )
            std_ds = gdal.Open(file_std)
            img_std = np.zeros(
                (std_ds.RasterCount, std_ds.RasterYSize, std_ds.RasterXSize)
            )
            for ii in range(std_ds.RasterCount):
                img_std[ii, :, :] = std_ds.GetRasterBand(ii + 1).ReadAsArray()
        img_all = np.zeros(
            (img_ds.RasterCount + 1, img_ds.RasterYSize, img_ds.RasterXSize)
        )
        img_all[0 : img_ds.RasterCount, :, :] = img
        # data SAR bands + mask bands
        if self.data_type == 'train':
            # data augmentation
            if self.std:
                img_all = np.vstack((img, img_std, mask))
            else:
                img_all = np.vstack((img, mask))
            assert (
                img_ds.RasterYSize == mask_ds.RasterYSize
                and img_ds.RasterXSize == mask_ds.RasterXSize
            ), (
                f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'
            )
            img_all = self.random_crop(img_all, self.patch_size)
            img_all = self.random_vertical_flipping(img_all)
            img_all = self.random_horizontal_flipping(img_all)
            img_all = self.random_rotation(img_all)
        elif self.data_type == 'valid':
            if self.std:
                img_all = np.vstack((img, img_std, mask))
            else:
                img_all = np.vstack((img, mask))
            img_all = self.random_crop(img_all, self.patch_size)
        img = img_all[0 : img_ds.RasterCount, :, :]
        mask = img_all[-1, :, :]
        if self.std:
            img_std = img_all[-3:-1, :, :]
        if self.input_dim:
            img_out = np.zeros((len(self.input_dim), img.shape[1], img.shape[2]))
            idx_ii = 0
            for layer in self.input_dim:
                img_out[idx_ii, :, :] = img[int(layer), :, :]
                idx_ii = idx_ii + 1
            # urban_mask[urban_mask == 0] = 0.01
            if self.transform:
                # img_out = self.standarized(img_out, mean, std)
                img_out = self.normalize(img_out, max, min)
                # img, mask = self.random_patch(img, mask)
            if self.std:
                assert (
                    img_ds.RasterYSize == std_ds.RasterYSize
                    and img_ds.RasterXSize == std_ds.RasterXSize
                ), (
                    f'Image and mask {idx} should be the same size, but are {img.shape} and {img_std.shape}'
                )
                # data normalization for std only
                std_min = 1
                std_max = 3
                img_std[img_std < std_min] = std_min
                img_std[img_std > std_max] = std_max
                img_std = (img_std - std_min) / (std_max - std_min)
                data_out_all = np.vstack((img_out[0:-1, :, :], img_std))
            else:
                data_out_all = img_out[:, :, :]
            data_out_all_nanmask = np.any(np.isnan(data_out_all), axis=0)
            data_out_all[:, data_out_all_nanmask] = 0
            mask[data_out_all_nanmask] = 0
            return {
                #'image': torch.from_numpy(img_out[0:-1,:,:]),
                'image': torch.from_numpy(data_out_all),
                self.mask_suffix_1[1:]: torch.from_numpy(mask),
                'ID': self.ids[i],
            }


class Normalization:
    def __call__(self, img, mask, mean, std):
        # amin, amax = img.min(), img.max()
        # img = (img-amin)/(amax-amin)
        img = (img - mean) / std
        return img, mask


class RandomCrop:
    def __call__(self, img, mask):
        row = img.shape[1]
        col = img.shape[2]
        patch_size = 256
        r = random.randint(0, row - patch_size)
        c = random.randint(0, col - patch_size)
        sub_image = img[:, r : r + patch_size, c : c + patch_size]
        sub_label = mask[:, r : r + patch_size, c : c + patch_size]
        return sub_image, sub_label


class ToTensor:
    def __call__(self, img, mask):
        return torch.from_numpy(img), torch.from_numpy(mask)
