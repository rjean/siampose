import logging
import math
import os
import pickle
import typing

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional
from pytorch_lightning.utilities import AMPType
from torch.optim.optimizer import Optimizer

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50

try:
    from pl_bolts.optimizers.lars_scheduling import LARSWrapper

    LARS_AVAILABLE = True
except ImportError:
    LARS_AVAILABLE = False
logger = logging.getLogger(__name__)


class KeypointsRegressor(pl.LightningModule):
    def __init__(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        super().__init__()
        self.save_hyperparameters(hyper_params)

        self.gpus = hyper_params.get("gpus")
        self.num_nodes = hyper_params.get("num_nodes", 1)
        self.backbone = hyper_params.get("backbone")
        self.num_samples = hyper_params.get("num_samples")
        self.batch_size = hyper_params.get("batch_size")

        self.first_conv = hyper_params.get("first_conv")
        self.maxpool1 = hyper_params.get("maxpool1")
        self.dropout = hyper_params.get("dropout")
        self.input_height = hyper_params.get("input_height")
        self.use_final_pool = hyper_params.get("use_final_pool")

        self.optim = hyper_params.get("optimizer")
        self.lars_wrapper = hyper_params.get("lars_wrapper")
        self.exclude_bn_bias = hyper_params.get("exclude_bn_bias")
        self.weight_decay = hyper_params.get("weight_decay")

        self.start_lr = hyper_params.get("start_lr")
        self.final_lr = hyper_params.get("final_lr")
        self.learning_rate = hyper_params.get("learning_rate")
        self.warmup_epochs = hyper_params.get("warmup_epochs")
        self.max_epochs = hyper_params.get("max_epoch")

        self.output_dir = hyper_params.get("output_dir")
        self.log_proj_errors = hyper_params.get("log_proj_errors")
        self.logged_proj_errors = {
            "train": {},
            "val": {},
        }  # set-to-epoch-to-uid-to-error map

        self.init_model()
        self.loss_fn = torch.nn.MSELoss()

        self.example_input_array = torch.rand(1, 3, 224, 224)

        # compute iters per epoch
        nb_gpus = len(self.gpus) if isinstance(self.gpus, (list, tuple)) else self.gpus
        assert isinstance(nb_gpus, int)
        global_batch_size = (
            self.num_nodes * nb_gpus * self.batch_size
            if nb_gpus > 0
            else self.batch_size
        )
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr,
            self.learning_rate,
            self.train_iters_per_epoch * self.warmup_epochs,
        )
        iters = np.arange(
            self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs)
        )
        cosine_lr_schedule = np.array(
            [
                self.final_lr
                + 0.5
                * (self.learning_rate - self.final_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * t
                        / (
                            self.train_iters_per_epoch
                            * (self.max_epochs - self.warmup_epochs)
                        )
                    )
                )
                for t in iters
            ]
        )

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def init_model(self):
        assert self.backbone in ["resnet18", "resnet50"]
        if self.backbone == "resnet18":
            backbone = resnet18
        else:
            backbone = resnet50

        if self.use_final_pool:
            self.encoder = backbone(
                first_conv=self.first_conv,
                maxpool1=self.maxpool1,
                return_all_feature_maps=False,
            )
            if (
                self.dropout is not None and self.dropout > 0
            ):  # @@@@ experiment with this
                self.decoder = torch.nn.Sequential(
                    torch.nn.Dropout(p=self.dropout),
                    torch.nn.Linear(2048, 18),
                )
            else:
                self.decoder = torch.nn.Linear(2048, 18)
        else:
            self.encoder = backbone(
                first_conv=self.first_conv,
                maxpool1=self.maxpool1,
                return_all_feature_maps=True,
            )
            if (
                self.dropout is not None and self.dropout > 0
            ):  # @@@@ experiment with this
                self.decoder = torch.nn.Sequential(
                    torch.nn.Dropout(p=self.dropout),
                    torch.nn.Linear(7 * 7 * 2048, 18),
                )
            else:
                self.decoder = torch.nn.Linear(7 * 7 * 2048, 18)

    def forward(self, x):
        if self.use_final_pool:
            return self.decoder(self.encoder(x)[0])
        else:
            return self.decoder(torch.flatten(self.encoder(x)[-1], 1))

    def training_step(self, batch, batch_idx):
        # @@@@@@@@@@ ADD MAX LOSS SAT, should not be > 2 per kpt?
        loss = None
        preds_stack, targets_stack = [], []
        # the loop below iterates for all frames in the frame tuple, but it should really just do one iter
        for img, pts in zip(batch["OBJ_CROPS"], batch["POINTS"]):
            preds = self(img).view(pts.shape)
            preds_stack.append(preds.detach())
            targets_stack.append(pts)
            # todo: try w/ -height/2 offset to center around 0?
            curr_loss = self.loss_fn(preds, pts / self.input_height)
            if loss is None:
                loss = curr_loss
            else:
                loss += curr_loss
        preds_stack = torch.cat(preds_stack)
        targets_stack = torch.cat(targets_stack)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        train_mae_2d = torch.nn.functional.l1_loss(
            preds_stack * self.input_height, targets_stack
        )
        self.log(
            "train_mae_2d",
            train_mae_2d,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        if self.log_proj_errors:
            if self.current_epoch not in self.logged_proj_errors["train"]:
                self.logged_proj_errors["train"][self.current_epoch] = {}
            for sample_idx, uid in enumerate(batch["UID"]):
                self.logged_proj_errors["train"][self.current_epoch][uid] = float(
                    torch.nn.functional.mse_loss(
                        preds_stack[sample_idx],
                        targets_stack[sample_idx] / self.input_height,
                    ).cpu()
                )
        self.log(
            "lr",
            self._get_latest_lr(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return {
            "loss": loss,
            "train_mae_2d": train_mae_2d,
        }

    def training_epoch_end(self, outputs: typing.List[typing.Any]) -> None:
        if self.log_proj_errors:
            log_proj_errors_dump_path = os.path.join(self.output_dir, "proj_errors.pkl")
            with open(log_proj_errors_dump_path, "wb") as fd:
                pickle.dump(self.logged_proj_errors, fd)

    def validation_step(self, batch, batch_idx):
        loss = None
        preds_stack, targets_stack = [], []
        # the loop below iterates for all frames in the frame tuple, but it should really just do one iter
        for img, pts in zip(batch["OBJ_CROPS"], batch["POINTS"]):
            preds = self(img).view(pts.shape)
            preds_stack.append(preds.detach())
            targets_stack.append(pts)
            curr_loss = self.loss_fn(preds, pts / self.input_height)
            if loss is None:
                loss = curr_loss
            else:
                loss += curr_loss
        preds_stack = torch.cat(preds_stack)
        targets_stack = torch.cat(targets_stack)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        val_mae_2d = torch.nn.functional.l1_loss(
            preds_stack * self.input_height, targets_stack
        )
        self.log(
            "val_mae_2d",
            val_mae_2d,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        if self.log_proj_errors:
            if self.current_epoch not in self.logged_proj_errors["val"]:
                self.logged_proj_errors["val"][self.current_epoch] = {}
            for sample_idx, uid in enumerate(batch["UID"]):
                self.logged_proj_errors["val"][self.current_epoch][uid] = float(
                    torch.nn.functional.mse_loss(
                        preds_stack[sample_idx],
                        targets_stack[sample_idx] / self.input_height,
                    ).cpu()
                )
        if batch_idx == 0 and hasattr(self, "_tbx_logger"):
            self._write_batch_preds_images_to_tbx(
                batch=batch, preds=preds_stack, targets=targets_stack
            )
        return {
            "val_loss": loss,
            "val_mae_2d": val_mae_2d,
        }

    def validation_epoch_end(self, outputs: typing.List[typing.Any]) -> None:
        if self.log_proj_errors:
            log_proj_errors_dump_path = os.path.join(self.output_dir, "proj_errors.pkl")
            with open(log_proj_errors_dump_path, "wb") as fd:
                pickle.dump(self.logged_proj_errors, fd)

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=["bias", "bn"]
    ):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.weight_decay
            )
        else:
            params = self.parameters()
        if self.optim == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        if self.lars_wrapper:
            assert LARS_AVAILABLE
            optimizer = LARSWrapper(
                optimizer, eta=0.001, clip=False  # trust coefficient
            )
        return optimizer

    def _get_latest_lr(self):
        capped_global_step = min(len(self.lr_schedule) - 1, self.trainer.global_step)
        return self.lr_schedule[capped_global_step]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure: typing.Optional[typing.Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        new_learning_rate = self._get_latest_lr()
        if self.lars_wrapper:
            for param_group in optimizer.optim.param_groups:
                param_group["lr"] = new_learning_rate
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_learning_rate
        if self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)

    def _write_batch_preds_images_to_tbx(self, batch, preds, targets, max_imgs=30):
        assert hasattr(self, "_tbx_logger")
        norm_std = np.asarray([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        norm_mean = np.asarray([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        for idx in range(min(len(batch["OBJ_CROPS"][0]), max_imgs, len(preds))):
            frame = batch["OBJ_CROPS"][0][idx].cpu()
            frame = (
                (frame.squeeze(0).numpy().transpose((1, 2, 0)) * norm_std) + norm_mean
            ) * 255
            frame = frame.astype(np.uint8).copy()
            for pred_pt, tgt_pt in zip(
                preds[idx], targets[idx]
            ):  # targets in green, preds in red
                # assume targets are already in abs coords, and we must scale predictions
                tgt_pt = tgt_pt[0].item(), tgt_pt[1].item()
                assert not np.isnan(tgt_pt[0]) and not np.isnan(tgt_pt[1])
                tgt_pt = int(round(tgt_pt[0])), int(round(tgt_pt[1]))
                pred_pt = pred_pt[0].item(), pred_pt[1].item()
                if np.isnan(pred_pt[0]) or np.isnan(pred_pt[0]):
                    pred_pt = (
                        -1.0,
                        -1.0,
                    )  # override nans (no clue why they happen sometimes...)
                pred_pt = int(round(pred_pt[0] * self.input_height)), int(
                    round(pred_pt[1] * self.input_height)
                )
                frame = cv.arrowedLine(
                    frame, tgt_pt, pred_pt, color=(242, 242, 34), thickness=1
                )
                frame = cv.circle(
                    frame, tgt_pt, radius=3, color=(127, 255, 112), thickness=-1
                )
                frame = cv.circle(
                    frame, pred_pt, radius=3, color=(255, 52, 52), thickness=-1
                )
            self._tbx_logger.experiment.add_image(
                batch["UID"][idx], frame, self.trainer.global_step, dataformats="HWC"
            )
