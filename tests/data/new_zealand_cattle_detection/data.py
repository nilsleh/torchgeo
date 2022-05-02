#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
from PIL import Image

SIZE = 32

np.random.seed(0)

PATHS = {
    "images": [
        "001_Hamilton_(2016-2017)_175.28054388784528,-37.83036803544756.png",
        "331_Kapiti_Coast_(2017)_175.14818059719508,-40.70543303376919.png",
    ],
    "annotations": [
        "001_Hamilton_(2016-2017)_175.28054388784528,-37.83036803544756.png.mask.0.txt",
        "331_Kapiti_Coast_(2017)_175.14818059719508,-40.70543303376919.png.mask.0.txt",
    ],
}


def create_annotation(path: str) -> None:
    points = np.random.randint(low=0, high=SIZE, size=(SIZE, SIZE))
    np.savetxt(path, points, delimiter=",")


def create_file(path: str) -> None:
    Z = np.random.rand(SIZE, SIZE, 3) * 255
    img = Image.fromarray(Z.astype("uint8")).convert("RGB")
    img.save(path)


if __name__ == "__main__":
    data_root = "images"
    sub_dir = "cow_images"
    # remove old data
    if os.path.isdir(data_root):
        shutil.rmtree(data_root)
    else:
        os.makedirs(data_root)

    # make subdirectory
    os.makedirs(os.path.join(data_root, sub_dir))

    for path in PATHS["images"]:
        create_file(os.path.join(data_root, sub_dir, path))

    for path in PATHS["annotations"]:
        create_annotation(os.path.join(data_root, sub_dir, path))

    # compress data
    shutil.make_archive(data_root, "zip", ".", data_root)

    # Compute checksums
    with open(data_root + ".zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{data_root}: {md5}")
