# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Land Cover Net Dataset."""

from typing import List, Optional, Dict, Any, Sequence
from torchgeo.datasets import NonGeoDataset
from torch import Tensor
import matplotlib.pyplot as plt
import os
import json
import torch
import rasterio
import numpy as np


class LandCoverNet(NonGeoDataset):
    """Land Cover Net Dataset.
    
    The `LandCoverNet dataset <https://mlhub.earth/datasets?search=landcovernet>`_ consists
    of 6 dataset collections of the continents Africa, Asia, Australia, Europe, North and South America. 
    Each contains annual land cover classification labels together with three sensor sources: 
    Sentinel 1, Sentinel 2, and Landsat 8.

    Sensors:

    * Sentinel 1: Ground Range Distance (GDR) at 10m resolution, 
        includes bands [B01, B02, B03, B04, B05, B06, B07]
    * Sentinel 2: Surface Reflectance (L2A) at 10m resolution
    * Landsat 8: Surface Reflectance (Collection 2 Level 2)

    Collections:

    * Africa (af): 1980 image chips of size 256x256
    * Asia (as): 2753 image chips of size 256x256
    * Australia (au): 600 image chips of size 256x256
    * Europe (eu): 840 image chips of size 256x256
    * North America (na): 1561 image chips of size 256x256
    * South America (sa): 1200 image chips of size 256x256
    
    
    .. versionadded:: 0.5
    """

    all_sensors = ["landsat_8", "sentinel_1", "sentinel_2"]
    all_continents = ["af", "as", "au", "eu", "na", "sa"]

    collections = {
        "na": {
            "landsat_8": "ref_landcovernet_na_v1_source_landsat_8",
            "sentinel_1": "ref_landcovernet_na_v1_source_sentinel_1",
            "sentinel_2": "ref_landcovernet_na_v1_source_sentinel_2",
            "labels": "ref_landcovernet_na_v1_labels"
        }
    }

    all_band_names = {
        "landsat_8": ["B01", "B02", "B03", "B04", "B05", "B06", "B07"],
        "sentinel_1": [],
        "sentinel_2": []
    }

    def __init__(self, root: str = "data", bands: Sequence[str] = all_band_names["landsat_8"], continents: List[str] = all_continents, sensor: str = "landsat_8", api_key: Optional[str] = None, checksum: bool = False) -> None:
        """Initialize a new Land Cover Net Dataset instance.
        
        Args:
            root: root directory where dataset can be found
            bands: the subset of bands to load
            continent: specifying which collections to include in the dataset
            sensor: sensor type for which to construct dataset
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            ValueError: if *continents* or *sensor* are invalid
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        
        """
        super().__init__()
        self.root = root
        self.checksum = checksum
        
        assert set(continents).issubset(self.all_continents), f"Only a subset of these continents is valid: {self.all_continents}"
        self.continents = continents
        self.continent_ids = []
        assert sensor in self.all_sensors, f"You must choose one of these sensors {self.all_sensors}."
        self.sensor = sensor

        self._validate_bands(bands)
        self.bands = bands


        self.chip_paths = self._load_collections()

        import pdb
        pdb.set_trace()
        print(0)

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            length of dataset in integer
        """
        return len(self.chip_paths)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Returns a sample from dataset.

        Args:
            Index: index to return

        Returns:
            data and label at given index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: Dict[str, Tensor] = {"image": image, "mask": label}

        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tensor:
        """Load all source images for a chip.

        Args:
            index: position of the indexed chip

        Returns:
            a tensor of stacked source image data
        """
        source_asset_paths = self.chip_paths[index]["source"]
        images = []
        for path in source_asset_paths:
            with rasterio.open(path) as image_data:
                image_array = image_data.read(1).astype(np.int32)
                images.append(image_array)
        image_stack: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        image_tensor = torch.from_numpy(image_stack)
        return image_tensor

    def _load_target(self, index: int) -> Tensor:
        """Load label image for a chip.

        Args:
            index: position of the indexed chip

        Returns:
            a tensor of the label image data
        """
        label_asset_path = self.chip_paths[index]["target"][0]
        with rasterio.open(label_asset_path) as target_data:
            target_img = target_data.read(1).astype(np.int32)

        target_array: "np.typing.NDArray[np.int_]" = np.array(target_img)
        target_tensor = torch.from_numpy(target_array)
        return target_tensor

    def _load_items(self, item_json: str) -> Dict[str, List[str]]:
        """Loads the label item and corresponding source items.

        Args:
            item_json: a string path to the item JSON file on disk

        Returns:
            a dictionary with paths to the source and target TIF filenames
        """
        item_meta = {}

        label_data = self._read_json_data(item_json)
        label_asset_path = os.path.join(
            os.path.split(item_json)[0], label_data["assets"]["labels"]["href"]
        )
        item_meta["target"] = [label_asset_path]

        source_item_hrefs = []
        for link in label_data["links"]:
            if link["rel"] == "source" and self.sensor in link["href"]:
                source_item_hrefs.append(
                    os.path.join(self.root, link["href"].replace("../../", ""))
                )

        source_item_hrefs = sorted(source_item_hrefs)
        source_item_paths = []

        for item_href in source_item_hrefs:
            source_item_path = os.path.split(item_href)[0]
            source_data = self._read_json_data(item_href)
            source_item_assets = []
            
            for asset_key, asset_value in source_data["assets"].items():
                if asset_key in self.bands:
                    source_item_assets.append(
                        os.path.join(source_item_path, asset_value["href"])
                    )
            source_item_assets = sorted(source_item_assets)
            for source_item_asset in source_item_assets:
                source_item_paths.append(source_item_asset)

        item_meta["source"] = source_item_paths
        return item_meta

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided tuple of bands to load

        Raises:
            ValueError: if an invalid band name is provided
        """
        for band in bands:
            if band not in self.all_band_names[self.sensor]:
                raise ValueError(f"'{band}' is an invalid band name.")
            
    @staticmethod
    def _read_json_data(object_path: str) -> Any:
        """Loads a JSON file.

        Args:
            object_path: string path to the JSON file

        Returns:
            json_data: JSON object / dictionary

        """
        with open(object_path) as read_contents:
            json_data = json.load(read_contents)
        return json_data

    def _load_collections(self) -> List[Dict[str, Any]]:
        """Loads the paths to source and label assets for each collection.

        Returns:
            a dictionary with lists of filepaths to all assets for each chip/item

        Raises:
            RuntimeError if collection.json is not found in the uncompressed dataset
        """
        indexed_chips = []
        label_collection: List[str] = []
        for continent_id in self.continents:
            label_collection.append(self.collections[continent_id]["labels"])

        label_collection_path = os.path.join(self.root, label_collection[0])
        label_collection_json = os.path.join(label_collection_path, "collection.json")

        label_collection_item_hrefs = []
        for link in self._read_json_data(label_collection_json)["links"]:
            
            if link["rel"] == "item":
                label_collection_item_hrefs.append(link["href"])

        label_collection_item_hrefs = sorted(label_collection_item_hrefs)

        for label_href in label_collection_item_hrefs:
            label_json = os.path.join(label_collection_path, label_href)
            indexed_item = self._load_items(label_json)
            indexed_chips.append(indexed_item)

        return indexed_chips


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
            time_step: time step at which to access image, beginning with 0
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            ValueError: if dataset does not contain an RGB band
        """


