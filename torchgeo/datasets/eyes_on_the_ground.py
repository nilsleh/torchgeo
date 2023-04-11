"""Eyes on the Ground Dataset."""

from .geo import NonGeoDataset, List
import torch
import glob
import os
from functools import lru_cache
from torch import Tensor
from typing import Optional, Callable, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import json

from .utils import check_integrity, download_radiant_mlhub_collection, extract_archive


class EyesOnTheGround(NonGeoDataset):
    """Eyes on the Ground Dataset.

    The `Eyes on the Ground Dataset <https://mlhub.earth/data/lacuna_fund_eotg_v1>`_
    contains georeferenced and timestamped crop images from smartphone cameras along
    with several labels such as crop growth stages, crop damage, and yield estimates.
    The dataset was collected across eight Kenyan counties and led by the
    `Lacuna Fund <https://lacunafund.org/>`_. The dataset contains labels for both
    classification and regression tasks with a more elaborate explanation 
    `here <https://radiantearth.blob.core.windows.net/mlhub/koen_lacuna_fund/Documentation.pdf>`_.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.deveng.2019.100042


    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """
    label_dir = "lacuna_fund_eotg_v1_labels"
    perspective_dir = "lacuna_fund_eotg_v1_source_perspective"

    all_labels = {
        "regression": ["drought_probability", "drought_extent", "growth_sowing", "growth_vegetative", "growth_flowering", "growth_maturity", "expected_yield"],
        "classification": ["growth_stage", "damage"]
    }
    tasks = ["regression", "classification"]

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        task: str = "regression",
        labels: List[str] = all_labels["regression"],
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Eyes on the Ground Dataset.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
            RuntimeError: if ``task`` argument is invalid
            RuntimeError: if ``labels``are invalid
        """
        super().__init__()

        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.task = task
        self.labels = labels

        assert task in self.tasks, f"Invalid `task`, please choose one of {self.tasks}."
        self.task = task

        assert all(x in self.all_labels[task] for x in labels), "Invalid label."
        self.labels = labels

        if download:
            self._download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )
        
        self.collection = self.retrieve_collection()


    def retrieve_collection(self) -> List[str]:
        """Retrieve collection with sample instances and mapped paths.
        
        Returns:
            List of sample paths
        """
        return glob.glob(os.path.join(self.root, self.label_dir, "**", "vector_labels.json"))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            length of the dataset
        """
        return len(self.collection)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        label_path = self.collection[index]
        label = self._load_label(label_path)

        image = self._load_image(label_path)

        sample: Dict[str, Tensor] = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    @lru_cache
    def _load_image(self, label_path: str) -> Tensor:
        """Load a single image.

        Args:
            label_path: path of corresponding label

        Returns:
            the image
        """
        filename = label_path.replace(self.label_dir, self.perspective_dir).replace("eotg_labels", "eotg_source_perspective").replace("vector_labels.json", "imagery.jpg")
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array).permute((2, 0, 1)).float() / 255.
            return tensor
        
    def _load_label(self, label_path: str) -> Tensor:
        """Load the label for an image.
        
        Args:
            label_path: path to label
        
        Returns:
            label
        """
        with open(label_path, "r") as f:
            content = json.load(f)
            labels = torch.tensor([content["features"][0]["properties"][label] for label in self.labels])
            if self.task == "regression":
                return labels.to(dtype=torch.float32)
            else:
                return labels.to(dtype=torch.long)


    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        # for split, resources in self.md5s.items():
        #     for resource_type, md5 in resources.items():
        #         filename = "_".join([self.collection_id, split, resource_type])
        #         filename = os.path.join(self.root, filename + ".tar.gz")
        #         if not check_integrity(filename, md5 if self.checksum else None):
        #             return False
        return True

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for collection_id in self.collection_ids:
            download_radiant_mlhub_collection(collection_id, self.root, api_key)

        for split, resources in self.md5s.items():
            for resource_type in resources:
                filename = "_".join([self.collection_id, split, resource_type])
                filename = os.path.join(self.root, filename) + ".tar.gz"
                extract_archive(filename, self.root)

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        """
        image, label = sample["image"], sample["label"]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"].item()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image.permute(1, 2, 0))
        ax.axis("off")

        if show_titles:
            if self.task == "regression":
                label_str = ", ".join([label_name + f": {label[idx].item():.4f}" for idx, label_name in enumerate(self.labels)])
            title = f"Labels: {label_str}"
            if showing_predictions:
                title += f"\nPrediction: {prediction}"
            ax.set_title(title, wrap=True)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

