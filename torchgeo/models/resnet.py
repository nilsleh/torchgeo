# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained ResNet models."""

from typing import Any, List, Optional, Type, Union

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

MODEL_URLS = {
    "sentinel2": {
        "all": {
            "resnet50": "https://zenodo.org/record/5610000/files/resnet50-sentinel2.pt"
        }
    }
}


IN_CHANNELS = {"sentinel2": {"all": 10}}

NUM_CLASSES = {"sentinel2": 17}


def _resnet(
    sensor: str,
    bands: str,
    arch: str,
    num_outputs: Optional[int],
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    """Resnet model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385.pdf

    Args:
        sensor: imagery source which determines number of input channels
        bands: which spectral bands to consider: "all", "rgb", etc.
        arch: ResNet version specifying number of layers
        num_outputs: number of model outputs
        block: type of network block
        layers: number of layers per block
        pretrained: if True, returns a model pre-trained on ``sensor`` imagery
        progress: if True, displays a progress bar of the download to stderr

    Returns:
        A ResNet-50 model

    .. versionchanged:: 0.3 Add ``num_outputs`` argument
    """
    # Initialize a new model
    model = ResNet(block, layers, NUM_CLASSES[sensor], **kwargs)

    # Replace the first layer with the correct number of input channels
    model.conv1 = nn.Conv2d(
        IN_CHANNELS[sensor][bands],
        out_channels=64,
        kernel_size=7,
        stride=1,
        padding=2,
        bias=False,
    )

    # Load pretrained weights
    if pretrained:
        state_dict = load_state_dict_from_url(  # type: ignore[no-untyped-call]
            MODEL_URLS[sensor][bands][arch], progress=progress
        )
        model.load_state_dict(state_dict)

    # Replace last layer with the desired number of outputs
    if num_outputs is not None:
        model.fc = nn.Linear(in_features=2048, out_features=num_outputs)

    return model


def resnet50(
    sensor: str,
    bands: str,
    num_outputs: Optional[int] = None,
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-50 model.

    If you use this model in your research, please cite the following paper:

     * https://arxiv.org/pdf/1512.03385.pdf

    Args:
        sensor: imagery source which determines number of input channels
        bands: which spectral bands to consider: "all", "rgb", etc.
        num_outputs: number of model outputs
        pretrained: if True, returns a model pre-trained on ``sensor`` imagery
        progress: if True, displays a progress bar of the download to stderr

    Returns:
        A ResNet-50 model

     .. versionchanged:: 0.3 Add ``num_outputs`` argument
    """
    return _resnet(
        sensor=sensor,
        bands=bands,
        arch="resnet50",
        num_outputs=num_outputs,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )
