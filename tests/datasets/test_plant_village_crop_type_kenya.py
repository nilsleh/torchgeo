# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, PlantVillageCropTypeKenya


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_african_crops_kenya_01", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path, recursive=True):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestPlantVillageCropTypeKenya:
    @pytest.fixture(params=[True, False])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> PlantVillageCropTypeKenya:

        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.3.1")
        monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch)

        root = str(tmp_path)

        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "ref_african_crops_kenya_01",
                "ref_african_crops_kenya_01_labels.tar.gz",
            ),
            root,
        )
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "ref_african_crops_kenya_01",
                "ref_african_crops_kenya_01_source.tar.gz",
            ),
            root,
        )

        image_meta = {
            "filename": "ref_african_crops_kenya_01_source.tar.gz",
            "directory": "ref_african_crops_kenya_01_source",
            "md5": "a2d34a0d714c3e128c9e50ec8aa1decc",
        }
        target_meta = {
            "filename": "ref_african_crops_kenya_01_labels.tar.gz",
            "directory": "ref_african_crops_kenya_01_labels",
            "md5": "17a28ba9b38073ca4f1610b3dc9dbb91",
        }
        monkeypatch.setattr(PlantVillageCropTypeKenya, "image_meta", image_meta)
        monkeypatch.setattr(PlantVillageCropTypeKenya, "target_meta", target_meta)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        cache = request.param
        return PlantVillageCropTypeKenya(
            root=root, transforms=transforms, api_key="", cache=cache
        )

    def test_getitem(self, dataset: PlantVillageCropTypeKenya) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_no_stac_json_found(self, dataset: PlantVillageCropTypeKenya) -> None:
        with tarfile.open(
            os.path.join(dataset.root, "ref_african_crops_kenya_01_source.tar.gz")
        ) as f:
            f.extractall()
        stac_list = glob.glob(
            os.path.join(dataset.root, "**", "**", "stac.json"), recursive=True
        )
        for f in stac_list:
            if os.path.exists(f):
                os.remove(f)
        with pytest.raises(FileNotFoundError, match="No stac.json files found in"):
            PlantVillageCropTypeKenya(root=dataset.root, download=False, api_key="")

    # def test_download(self, tmp_path: Path) -> None:
    #     PlantVillageCropTypeKenya(root=tmp_path, download=True, api_key="")
    # def test_no_cache(self, dataset: PlantVillageCropTypeKenya) -> None:
    #     ds = PlantVillageCropTypeKenya(root=dataset.root, cache=False)
    #     x = ds[ds.bounds]
    #     assert isinstance(x, dict)

    def test_different_crs(self, dataset: PlantVillageCropTypeKenya) -> None:
        crs = CRS.from_epsg(32736)
        ds = PlantVillageCropTypeKenya(root=dataset.root, crs=crs)
        x = ds[ds.bounds]
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_corrupted(self, tmp_path: Path) -> None:
        path = os.path.join(tmp_path, "ref_african_crops_kenya_01_labels.tar.gz")
        with open(path, "w") as f:
            f.write("bad")

        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            PlantVillageCropTypeKenya(root=str(tmp_path), checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            PlantVillageCropTypeKenya(root=str(tmp_path))

    def test_already_downloaded(self, dataset: PlantVillageCropTypeKenya) -> None:
        PlantVillageCropTypeKenya(root=dataset.root, download=True, api_key="")

    def test_invalid_query(self, dataset: PlantVillageCropTypeKenya) -> None:
        query = BoundingBox(100, 100, 100, 100, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            PlantVillageCropTypeKenya(bands=["B01", "B02"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="is an invalid band name."):
            PlantVillageCropTypeKenya(bands=("foo", "bar"))

    def test_plot(self, dataset: PlantVillageCropTypeKenya) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, time_step=0, suptitle="test")
        plt.close()

    def test_plot_prediction(self, dataset: PlantVillageCropTypeKenya) -> None:
        x = dataset[dataset.bounds]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_plot_rgb(self, dataset: PlantVillageCropTypeKenya) -> None:
        dataset = PlantVillageCropTypeKenya(root=dataset.root, bands=tuple(["B01"]))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[dataset.bounds], suptitle="Single Band")
