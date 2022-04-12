import logging
import math
import typing

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import AMPType
from torch.nn.modules.linear import Identity
from torch.optim.optimizer import Optimizer

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0

from siampose.models.kpts_regressor import LARS_AVAILABLE

# from pl_bolts.models.self_supervised.simsiam.models import SiameseArm
try:
    from pl_bolts.optimizers.lars_scheduling import LARSWrapper

    LARS_AVAILABLE = True
except ImportError:
    LARS_AVAILABLE = False

import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


# Credit: https://github.com/PatrickHua/SimSiam
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        """ page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


# Credit: https://github.com/PatrickHua/SimSiam
class PredictionMLP(nn.Module):
    def __init__(
        self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        """ page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing.
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


import collections


def unfreeze_batchnorm_layers(module):
    """Modifies the provided module by swapping frozen batch norm layers with unfrozen
    ones for training.
    """
    from detectron2.layers.batch_norm import FrozenBatchNorm2d

    if isinstance(module, FrozenBatchNorm2d):
        return nn.BatchNorm2d(num_features=module.num_features, eps=module.eps)

    elif isinstance(module, torch.nn.Sequential):
        return torch.nn.Sequential(*[unfreeze_batchnorm_layers(m) for m in module])
    elif isinstance(module, collections.OrderedDict):
        for mname, m in module.items():
            module[mname] = unfreeze_batchnorm_layers(m)
    elif isinstance(module, torch.nn.Module):
        for attrib, m in module.__dict__.items():
            if isinstance(m, FrozenBatchNorm2d):
                setattr(module, attrib, unfreeze_batchnorm_layers(m))
            elif isinstance(m, (torch.nn.Module, collections.OrderedDict)):
                setattr(module, attrib, unfreeze_batchnorm_layers(m))
    return module


class Resnet50Layer4(nn.Module):
    def __init__(self):
        super().__init__()

        # Evil hard-coded adaptation layer between the "detectron-2" resnet-50 backbone
        # for object detection which outputs the 14x14x1024 feature map and the resnet-50
        # we normally use, which ouputs a 2048 wide embedding. This part is copy-pasted
        # from pytorch lightning resnet-50 implementation.
        # and a feature exp

        # This part won't matter for Object Detection since it will completely removed!

        # I've removed one downsampling step at the end.
        self.bn1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(
                2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
        )
        self.bn2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.bn3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.bn3(x)
        x = self.avg_pool(x)
        return x.squeeze()


class MLP(nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_size: int = 4096, output_dim: int = 256
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        input_dim: int = 2048,
        hidden_size: int = 4096,
        output_dim: int = 256,
        predictor_input_dim: Optional[int] = None,
        hua_mlp=True,
        detectron_resnet_layer4: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.detectron_resnet_layer4 = detectron_resnet_layer4
        if (
            predictor_input_dim is None
        ):  # By default, the predictor that the projector output as a input.
            predictor_input_dim = output_dim

        if encoder is None:
            raise ValueError("Please provide an encoder.")
        # Encoder
        self.encoder = encoder
        self.detectron_encoder = False
        try:
            # Lazy loading. I don't want a hard dependency on Detectron2
            from detectron2.modeling.backbone.resnet import ResNet

            if type(self.encoder) == ResNet:
                self.detectron_encoder = True
        except:
            self.detectron_encoder = False
        # input_dim = self.encoder.fc.in_features
        # Projector
        if hua_mlp:  # Using Patrick Hua interpretation of SimSiam.
            # Will use an additional hidden layer. Linear layer will have bias.
            self.projector = ProjectionMLP(input_dim, output_dim, output_dim)
            self.predictor = PredictionMLP(predictor_input_dim, hidden_size, output_dim)
        else:  # Pytorch Lighting interepreation of SimSiam
            self.projector = MLP(input_dim, hidden_size, output_dim)
            self.predictor = MLP(output_dim, hidden_size, output_dim)
        # Predictor

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type(self.encoder) == torchvision.models.shufflenetv2.ShuffleNetV2:
            y = self.encoder(x)
        elif self.detectron_encoder:
            feature_map = self.encoder(x)["res4"]
            y = self.detectron_resnet_layer4(feature_map)
        else:
            y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


import torchvision
import cv2


def save_mosaic(filename, tensor):
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )
    # grid = torchvision.utils.make_grid(tensor)
    tensor = inv_normalize(tensor)
    torchvision.utils.save_image(tensor, filename)
    # img = img.detach().cpu().numpy().astype(np.uint8)
    # img = np.swapaxes(img,2,0)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # return cv2.imwrite(filename, img)


class SimSiam(pl.LightningModule):
    def __init__(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        super().__init__()
        self.save_hyperparameters(hyper_params)

        self.gpus = hyper_params.get("gpus")
        self.num_nodes = hyper_params.get("num_nodes", 1)
        self.backbone = hyper_params.get("backbone", "resnet50")
        self.num_samples = hyper_params.get("num_samples")
        self.num_samples_valid = hyper_params.get("num_samples_valid")
        self.batch_size = hyper_params.get("batch_size")

        self.hidden_mlp = hyper_params.get("hidden_mlp", 2048)
        self.feat_dim = hyper_params.get("feat_dim", 128)
        self.first_conv = hyper_params.get("first_conv", True)
        self.maxpool1 = hyper_params.get("maxpool1", True)

        self.optim = hyper_params.get("optimizer", "adam")
        self.lars_wrapper = hyper_params.get("lars_wrapper", False)
        self.exclude_bn_bias = hyper_params.get("exclude_bn_bias", False)
        self.weight_decay = hyper_params.get("weight_decay", 1e-6)
        self.temperature = hyper_params.get("temperature", 0.1)

        self.start_lr = hyper_params.get("start_lr", 0.0)
        self.final_lr = hyper_params.get("final_lr", 1e-6)
        self.learning_rate = hyper_params.get("learning_rate", 1e-3)
        self.warmup_epochs = hyper_params.get("warmup_epochs", 10)
        self.max_epochs = hyper_params.get("max_epochs", 100)

        self.loss_function = hyper_params.get("loss_function", "simsiam")
        self.ntxent_temp = hyper_params.get("ntxent_temp", 0.5)

        if self.loss_function == "triplet":
            # For TCN triplet, gap is 0.2. Other parameter follow baseline implementation.
            self.triplet_loss = torch.nn.TripletMarginLoss(margin=0.2)

        self.monitor_accuracy = hyper_params.get(
            "monitor_accuracy", True
        )  # Enabled by default. Can be disabled if it takes too much memory.

        self.accumulate_grad_batches_custom = hyper_params.get(
            "accumulate_grad_batches_custom", 1
        )

        self.coordconv = hyper_params.get("coordconv", None)

        self.same_crop = hyper_params.get("same_crop", False)

        # self.monitor_accuracy = False

        self.init_model()

        # compute iters per epoch
        nb_gpus = len(self.gpus) if isinstance(self.gpus, (list, tuple)) else self.gpus
        assert isinstance(nb_gpus, int)
        if self.num_samples is not None:
            # When training, compute the learning rate schedule
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

        # self.nce = torch.nn.CrossEntropyLoss()

        self.detectron_resnet_layer4 = None

        # self.log("val_loss",0)
        # self.log("train_loss", 0)

    def init_model(self):
        assert self.backbone in [
            "resnet18",
            "resnet50",
            "shufflenet_v2_x1_0",
            "resnet50_detectron",
        ]
        detectron_resnet_layer4 = None
        if self.backbone == "resnet18":
            backbone = resnet18
            backbone_network = backbone(
                first_conv=self.first_conv,
                maxpool1=self.maxpool1,
                return_all_feature_maps=False,
            )
            self.feature_dim = backbone_network.fc.in_features

        elif self.backbone == "resnet50":
            backbone = resnet50
            backbone_network = backbone(
                first_conv=self.first_conv,
                maxpool1=self.maxpool1,
                return_all_feature_maps=False,
            )
            self.feature_dim = backbone_network.fc.in_features
        elif self.backbone == "shufflenet_v2_x1_0":
            backbone = shufflenet_v2_x1_0
            backbone_network = backbone()
            self.feature_dim = backbone_network.fc.in_features
            backbone_network.fc = Identity()
        elif self.backbone == "resnet50_detectron":
            with open("examples/local/detectron_resnet50_c4_config.yaml", "r") as f:
                import yaml

                cfg = yaml.load(f, Loader=yaml.Loader)
            from detectron2.modeling.backbone.resnet import build_resnet_backbone
            from detectron2.layers import ShapeSpec

            input_shape = ShapeSpec(3)  # 3 channels RGB
            backbone_network = build_resnet_backbone(cfg, input_shape)
            backbone_network = unfreeze_batchnorm_layers(backbone_network)
            detectron_resnet_layer4 = Resnet50Layer4()
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        if self.coordconv is not None:
            from thelper.nn.coordconv import swap_coordconv_layers  # Lazy loading.
        if self.coordconv == "all":
            backbone_network = swap_coordconv_layers(backbone_network)
        if self.coordconv == "first":
            backbone_network.conv1 = swap_coordconv_layers(backbone_network.conv1)
            # backbone_network =

        self.cyclic_predictor = None
        if self.loss_function == "cyclic":
            # Use 2 stacked inputs for the predictor
            self.cyclic_predictor = PredictionMLP(
                self.feature_dim * 2, self.hidden_mlp, self.feature_dim
            )
        # else:
        # All other methods work on pairs!
        self.online_network = SiameseArm(
            backbone_network,
            input_dim=self.feature_dim,
            hidden_size=self.hidden_mlp,
            output_dim=self.feat_dim,
            detectron_resnet_layer4=detectron_resnet_layer4,
        )
        # max_batch = math.ceil(self.num_samples/self.batch_size)
        encoder, projector = self.online_network.encoder, self.online_network.projector
        if self.num_samples is not None:  # Not working on test set
            self.train_features = torch.zeros((self.num_samples, self.feature_dim))
            self.train_meta = []
            self.train_targets = -torch.ones((self.num_samples))

        self.valid_features = torch.zeros((self.num_samples_valid, self.feature_dim))
        self.valid_meta = []
        self.cuda_train_features = None

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        # https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L64-L331
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def cosine_similarity(self, a, b, version="simplified"):
        if version == "original":
            b = b.detach()  # stop gradient of backbone + projection mlp
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
            sim = -1 * (a * b).sum(-1).mean()
        elif version == "simplified":
            sim = -F.cosine_similarity(a, b.detach(), dim=-1).mean()
        else:
            raise ValueError(f"Unsupported cosine similarity version: {version}")
        return sim

    def compute_loss(self, crops):
        # Image 1 to image 2 loss

        if self.loss_function == "simsiam":
            assert len(crops) == 2, "Simsiam only works with object pairs"
            img_1, img_2 = crops
            f1, z1, h1 = self.online_network(img_1)
            f2, z2, h2 = self.online_network(img_2)
            loss = (
                self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2
            )
        elif (
            self.loss_function == "ntxent"
        ):  # Normalized temperature scaled cross entrolpy loss
            assert (
                len(crops) == 2
            ), "Negative Cross Entropy Loss only works with object pairs"
            img_1, img_2 = crops
            f1, z1, h1 = self.online_network(img_1)
            f2, z2, h2 = self.online_network(img_2)
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            loss = self.nt_xent_loss(z1, z2, self.ntxent_temp)
        elif (
            self.loss_function == "triplet"
        ):  # Normalized temperature scaled cross entrolpy loss
            assert (
                len(crops) == 3
            ), "Triplet loss requires anchors, positive and negative samples"
            anchor, positive, negative = crops
            f1, z1, h1 = self.online_network(anchor)
            f2, z2, h2 = self.online_network(positive)
            f3, z3, h3 = self.online_network(negative)
            loss = self.triplet_loss(z1, z2, z3)
        elif self.loss_function == "cyclic":
            assert len(crops) == 3, "Only triplets are supported for the Cyclic Loss"
            assert self.cyclic_predictor is not None
            img_1, img_2, img_3 = crops
            f1 = self.online_network.encoder(img_1)[0]
            z1 = self.online_network.projector(f1)
            h1 = self.online_network.predictor(z1)
            f2 = self.online_network.encoder(img_2)[0]
            z2 = self.online_network.projector(f2)
            h2 = self.online_network.predictor(z2)
            f3 = self.online_network.encoder(img_3)[0]
            z3 = self.online_network.projector(f3)
            h3 = self.online_network.predictor(z3)
            # self.cosine_similarity(predicted, real)
            z1z2 = torch.hstack([z1, z2])
            z3z2 = torch.hstack([z3, z2])
            z3_pred = self.cyclic_predictor(z1z2)
            z1_pred = self.cyclic_predictor(z3z2)
            loss = (
                self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2
            )  # 1->2, 2->1
            loss += (
                self.cosine_similarity(h3, z2) / 2 + self.cosine_similarity(h2, z3) / 2
            )  # 2->3, 3->2
            loss += (
                self.cosine_similarity(z1_pred, z1) / 2
                + self.cosine_similarity(z3_pred, z3) / 2
            )  # (1,2)->3, (3,2)->1
            loss = loss / 3  # Normalize to target loss=-1
        else:
            raise ValueError(f"The {self.loss_function} loss is not supported!")
        return loss, f1, f2

    def training_step(self, batch, batch_idx):
        crops = batch["OBJ_CROPS"]

        if batch_idx == 0:
            for i, crop in enumerate(crops):
                save_mosaic(f"img_{i}_train.jpg", crop)

        if batch_idx == 249:
            print("!")

        # assert img_1.shape==torch.Size([32, 3, 224, 224])
        uid = batch["UID"]
        y = batch["CAT_ID"]

        if self.cuda_train_features is not None:
            self.cuda_train_features = None  # Free GPU memory

        loss, f1, f2 = self.compute_loss(crops)

        # assert train_features.shape == torch.Size([32, 2048])
        self.train_meta += uid

        if self.monitor_accuracy:
            base = batch_idx * self.batch_size
            train_features = F.normalize(f1.detach(), dim=1).cpu()
            self.train_features[base : base + train_features.shape[0]] = train_features
            self.train_targets[base : base + train_features.shape[0]] = y

        # log results
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "lr",
            self.lr_schedule[self.trainer.global_step],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # assert len(batch["OBJ_CROPS"]) == 2
        crops = batch["OBJ_CROPS"]

        if batch_idx == 0:
            for i, crop in enumerate(crops):
                save_mosaic(f"img_{i}_val.jpg", crop)

        uid = batch["UID"]
        y = batch["CAT_ID"]

        # Image 1 to image 2 loss
        # f1, z1, h1 = self.online_network(img_1)
        # f2, z2, h2 = self.online_network(img_2)

        # loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2
        loss, f1, _ = self.compute_loss(crops)

        if self.monitor_accuracy:
            self.valid_meta += uid
            base = batch_idx * self.batch_size

            if self.cuda_train_features is None:  # Transfer to GPU once.
                self.cuda_train_features = self.train_features.half().cuda()
            valid_features = F.normalize(f1, dim=1).detach().half()

            similarity = torch.mm(valid_features, self.cuda_train_features.T)
            targets_idx = torch.argmax(similarity, axis=1).cpu()
            neighbor_targets = self.train_targets[targets_idx]
            match_count = (neighbor_targets == y.cpu()).sum()
            accuracy = match_count / len(neighbor_targets)

            self.valid_features[base : base + len(crops[0])] = valid_features
            self.log("val_accuracy", accuracy)

        # log results

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        return loss

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

        predictor_prefix = "encoder"
        backbone_and_encoder_parameters = [
            param for name, param in self.online_network.encoder.named_parameters()
        ]
        backbone_and_encoder_parameters += [
            param for name, param in self.online_network.projector.named_parameters()
        ]
        lr = self.learning_rate
        params = [
            {"name": "base", "params": backbone_and_encoder_parameters, "lr": lr},
            {
                "name": "predictor",
                "params": [
                    param
                    for name, param in self.online_network.predictor.named_parameters()
                ],
                "lr": lr,
            },
        ]
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
        if self.lars_wrapper:
            for param_group in optimizer.optim.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]
        else:
            for param_group in optimizer.param_groups:
                if param_group["name"] == "predictor":
                    param_group["lr"] = self.learning_rate
                else:
                    param_group["lr"] = self.lr_schedule[self.trainer.global_step]
            # param_group[0]["lr"]

        # from lightning
        # if self.trainer.amp_backend == AMPType.NATIVE:
        #    optimizer_closure()
        #    self.trainer.scaler.step(optimizer)
        if ((batch_idx + 1) % self.accumulate_grad_batches_custom) == 0:
            if self.trainer.amp_backend == AMPType.APEX:
                optimizer_closure()
                optimizer.step()
            else:
                optimizer.step(closure=optimizer_closure)
