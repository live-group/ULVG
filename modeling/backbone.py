import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import random

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone.resnet import BasicStem, BasicBlock, DeformBottleneckBlock, BottleneckBlock, ResNet

class ResNet_statistics(ResNet):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0, aug_num=1, bs=1):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__(stem, stages, num_classes=num_classes, out_features=out_features, freeze_at=freeze_at)
        # aug_num = 2
        self.feat_alpha = nn.Parameter(torch.normal(torch.ones(aug_num, bs, 256, 1, 1), 0.75 * torch.ones(aug_num, bs, 256, 1, 1)))
        self.feat_beta = nn.Parameter(torch.normal(torch.ones(aug_num, bs, 256, 1, 1), 0.75 * torch.ones(aug_num, bs, 256, 1, 1)))
        self.feat_alpha_l1 = nn.Parameter(torch.normal(torch.ones(aug_num, bs, 64, 1, 1), 0.75 * torch.ones(aug_num, bs, 64, 1, 1)))
        self.feat_beta_l1 = nn.Parameter(torch.normal(torch.ones(aug_num, bs, 64, 1, 1), 0.75 * torch.ones(aug_num, bs, 64, 1, 1)))

    def forward(self, x, aug_ids=0):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}

        x = self.stem(x)

        if self.training:
            x_aug = self.Multiply(x, self.feat_alpha_l1[aug_ids], self.feat_beta_l1[aug_ids])
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if self.training:
                if name == 'res2':
                    x_aug = stage(x_aug)
                    x_aug = self.Multiply(x_aug, self.feat_alpha[aug_ids], self.feat_beta[aug_ids])
                else:
                    x_aug = stage(x_aug)
                    # print('aug')
            if name in self._out_features:
                outputs[name] = x
                if self.training:
                    outputs[name+'_aug'] = x_aug
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def Multiply(self, x, alpha, beta):
        x_mean = x.mean((2, 3), keepdim=True)
        x_aug = alpha * x - alpha * x_mean + beta * x_mean
        return x_aug

    def Add(self, x, alpha, beta):
        x_mean = x.mean((2, 3), keepdim=True)
        x_std = x.std((2, 3), keepdim=True)
        x_aug = (x - x_mean) / (x_std + 1e-5) * (alpha + x_std) + (beta + x_mean)
        return x_aug

    def Replace(self, x, alpha, beta):
        x_mean = x.mean((2, 3), keepdim=True)
        x_std = x.std((2, 3), keepdim=True)
        x_aug = (x - x_mean) / (x_std + 1e-5) * alpha + beta
        return x_aug

@BACKBONE_REGISTRY.register()
def build_resnet_backbone_statistics(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet_statistics.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet_statistics(stem, stages, out_features=out_features, freeze_at=freeze_at, aug_num=cfg.DOMAIN_AUG_NUM, bs=cfg.SOLVER.IMS_PER_BATCH)


    
    