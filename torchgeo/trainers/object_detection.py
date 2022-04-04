# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tasks."""

from typing import Any, Dict, cast

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn.modules import Conv2d, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"


class ObjectDetectionTask(pl.LightningModule):
    """LightningModule for training models on object detection datasets."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters."""
        if self.hyperparams["detection_model"] == "faster_rcnn":
            self.model = detection.fasterrcnn_resnet50_fpn(
                pretrained=self.hyperparams["pretrained"],
                pretrained_backbone=self.hyperparams["pretrained_backbone"],
                num_classes=self.hyperparams["num_classes"],
            )

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.hyperparams["num_classes"]
            )
        elif self.hyperparams["detection_model"] == "retina net":
            self.model = detection.retinanet_resnet50_fpn(
                pretrained=self.hyperparams["pretrained"],
                pretrained_backbone=self.hyperparams["pretrained_backbone"],
                num_classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["detection_model"] == "mask r-cnn":
            self.model = detection.maskrcnn_resnet50_fpn(
                pretrained=self.hyperparams["pretrained"],
                pretrained_backbone=self.hyperparams["pretrained_backbone"],
                num_classes=self.hyperparams["num_classes"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['detection_model']}' is not valid."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

                Keyword Args:
            detection_model: Name of the segmentation model type to use
            pretrained: if True pretrained on COCO
            pretrained_backbone: if True pretrained on ImageNet
            num_classes: Number of object classes to predict (without
            including background)
            loss: Name of the loss function

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

        self.train_metrics = MetricCollection(
            [MeanAveragePrecision(box_format="xyxy")], prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model
            y: tensor of target bounding boxes
            model_mode: train or eval

        Returns:
            output from the model
        """
        self.model.train()
        loss_dict = self.model(*args, **kwargs)

        self.model.eval()
        predictions = self.model(*args, **kwargs)
        return loss_dict, predictions

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]
        y = batch["target"]
        loss_dict, y_hat = self.forward(x, y, model_mode="train")

        loss = sum(loss for loss in loss_dict.values())

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        # batch_idx = args[1]
        x = batch["image"]
        y = batch["target"]
        loss_dict, y_hat = self.forward(x, y, model_mode="val")

        loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics.update(y_hat, y)

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        loss_dict, y_hat = self.forward(x, model_mode="test")

        loss = sum(loss for loss in loss_dict.values())

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=len(x))
        self.test_metrics(y_hat, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hyperparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }
