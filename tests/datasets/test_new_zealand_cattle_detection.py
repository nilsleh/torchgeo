# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import NewZealandCattleDetection


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestNewZealandCattleDetection:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> NewZealandCattleDetection:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        data_dir = os.path.join("tests", "data", "new_zealand_cattle_detection")

        url = os.path.join(data_dir, "images.zip")

        md5 = "e3290eeb220d38591624aae44b551e65"

        monkeypatch.setattr(NewZealandCattleDetection, "url", url)
        monkeypatch.setattr(NewZealandCattleDetection, "md5", md5)
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return NewZealandCattleDetection(
            root=root, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: NewZealandCattleDetection) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["image"].ndim == 3

    def test_already_downloaded(self, dataset: NewZealandCattleDetection) -> None:
        NewZealandCattleDetection(root=dataset.root, download=True)

    def test_len(self, dataset: NewZealandCattleDetection) -> None:
        assert len(dataset) == 2

    def test_not_extracted(self, tmp_path: Path) -> None:
        url = os.path.join(
            "tests", "data", "new_zealand_cattle_detection", "images.zip"
        )
        shutil.copy(url, tmp_path)
        NewZealandCattleDetection(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "images.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            NewZealandCattleDetection(root=str(tmp_path), checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in."):
            NewZealandCattleDetection(str(tmp_path))

    def test_plot(self, dataset: NewZealandCattleDetection) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: NewZealandCattleDetection) -> None:
        x = dataset[0].copy()
        x["prediction_boxes"] = x["boxes"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()
