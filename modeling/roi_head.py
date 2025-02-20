from pydoc import classname
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import sys
import torchvision.transforms as T
import os


from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.data import MetadataCatalog

from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY,  Res5ROIHeads
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from .box_predictor import ClipFastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads_CLIP(Res5ROIHeads):
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        num_classes = self.num_classes
        self.effect2weight = nn.Sequential(
            nn.Linear( num_classes + 1, (num_classes + 1) * 10),
            nn.ReLU(inplace=True),
            nn.Linear((num_classes + 1) * 10, 1),
        )
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=512, height=1, width=1)
        )
        self.loss_lambda=cfg.SOLVER.LAMBDA

        self.aug_nums=cfg.DOMAIN_AUG_NUM
        self.feat_orig=[]
        self.feat_aug=[]


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        clip_attnpool = None,
        clip_text_diff = None,
        image_sizes = None,
        clip_cls_causal_text_bv=None,
        clip_cls_noncausal_text_bv=None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            bs = len(proposals)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )


        box_features_mean = clip_attnpool(box_features)

        cls_score_pseudo = self.box_predictor.cls_score(box_features_mean)
        clip_cls_noncausal_text_bv_up=clip_cls_noncausal_text_bv
        clip_cls_causal_text_bv_up=clip_cls_causal_text_bv
        box_features_mean_norm=box_features_mean.norm(dim=-1, keepdim=True)
        amps_interval= random.uniform(0.1, 0.5)
        clip_cls_noncausal_text_bv_up_amps = torch.stack([ amp*clip_cls_noncausal_text_bv_up for amp in torch.arange(amps_interval, 1.01, amps_interval)],dim=0)
        box_features_mean_do = box_features_mean.unsqueeze(0).unsqueeze(0) - clip_cls_noncausal_text_bv_up_amps.unsqueeze(2)*box_features_mean_norm.unsqueeze(0).unsqueeze(0)
        effect = cls_score_pseudo - torch.mean(self.box_predictor.cls_score(box_features_mean_do), dim=0)
        temper_cofe=0.4
        weight_noncausal = torch.softmax(self.effect2weight(effect)/temper_cofe,dim=0)
        clip_cls_causal_text_bv_up_amps = torch.stack([amp * clip_cls_causal_text_bv_up for amp in torch.arange(amps_interval, 1.01, amps_interval)], dim=0)
        box_features_mean_do = box_features_mean.unsqueeze(0).unsqueeze(0) + clip_cls_causal_text_bv_up_amps.unsqueeze(2)*box_features_mean_norm.unsqueeze(0).unsqueeze(0)
        effect = cls_score_pseudo - torch.mean(self.box_predictor.cls_score(box_features_mean_do), dim=0)
        weight_causal = torch.softmax(self.effect2weight(effect)/temper_cofe,dim=0)
        box_features_mean = box_features_mean - (weight_noncausal*clip_cls_noncausal_text_bv_up.unsqueeze(1)*box_features_mean_norm.unsqueeze(0)).sum(0)
        box_features_mean = box_features_mean + (weight_causal * clip_cls_causal_text_bv_up.unsqueeze(1)*box_features_mean_norm.unsqueeze(0)).sum(0)


        predictions = self.box_predictor(box_features_mean)

        if self.training:
            losses = dict()
            box_features_aug = self._shared_roi_transform([features['res4_aug']] , proposal_boxes)
            box_features_aug_mean = clip_attnpool(box_features_aug)
            cls_score_pseudo_aug = self.box_predictor.cls_score(box_features_aug_mean)
            clip_cls_noncausal_text_bv_up = clip_cls_noncausal_text_bv
            clip_cls_causal_text_bv_up = clip_cls_causal_text_bv
            box_features_aug_mean_norm = box_features_aug_mean.norm(dim=-1, keepdim=True)
            amps_interval = random.uniform(0.1, 0.5)
            clip_cls_noncausal_text_bv_up_amps_aug = torch.stack([amp * clip_cls_noncausal_text_bv_up for amp in torch.arange(amps_interval, 1.01, amps_interval)],dim=0)
            box_features_aug_mean_do = box_features_aug_mean.unsqueeze(0).unsqueeze(0) - clip_cls_noncausal_text_bv_up_amps_aug.unsqueeze(2) * box_features_aug_mean_norm.unsqueeze(0).unsqueeze(0)
            effect = cls_score_pseudo_aug - torch.mean(self.box_predictor.cls_score(box_features_aug_mean_do),dim=0)
            temper_cofe=0.4
            weight_noncausal = torch.softmax(self.effect2weight(effect)/temper_cofe, dim=0)
            clip_cls_causal_text_bv_up_amps_aug = torch.stack([amp * clip_cls_causal_text_bv_up for amp in torch.arange(amps_interval, 1.01, amps_interval)], dim=0)
            box_features_aug_mean_do = box_features_aug_mean.unsqueeze(0).unsqueeze(0) + clip_cls_causal_text_bv_up_amps_aug.unsqueeze(2) * box_features_mean_norm.unsqueeze(0).unsqueeze(0)
            effect = cls_score_pseudo_aug - torch.mean(self.box_predictor.cls_score(box_features_aug_mean_do), dim=0)
            weight_causal = torch.softmax(self.effect2weight(effect) / temper_cofe, dim=0)
            box_features_aug_mean = box_features_aug_mean - (weight_noncausal * clip_cls_noncausal_text_bv_up.unsqueeze(1) * box_features_aug_mean_norm.unsqueeze(0)).sum(0)
            box_features_aug_mean = box_features_aug_mean + (weight_causal * clip_cls_causal_text_bv_up.unsqueeze(1) * box_features_aug_mean_norm.unsqueeze(0)).sum(0)

            predictions_aug = self.box_predictor(box_features_aug_mean)
            image_Box = Boxes(torch.tensor([[0,0,image_sizes[0][0],image_sizes[0][1]]], dtype=torch.float32, device=box_features.device))
            image_features=self._shared_roi_transform([features['res4']], [image_Box for _ in range(bs)])
            image_features_aug = self._shared_roi_transform([features['res4_aug']], [image_Box for _ in range(bs)])
            image_features_atp = clip_attnpool(image_features)
            image_features_aug_atp = clip_attnpool(image_features_aug)
            image_diff = image_features_aug_atp - image_features_atp
            image_diff = image_diff / image_diff.norm(dim=-1, keepdim=True)
            losses.update( {'domain_align_loss': self.loss_lambda*(1-F.cosine_similarity(image_diff, clip_text_diff)).mean()} )


            del features
            losses.update(self.box_predictor.losses(predictions, proposals))
            losses.update({key+'_aug':value for key,value in self.box_predictor.losses(predictions_aug, proposals).items()})
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def loss_CLIP_cls(self, predictions, proposals):
        scores= predictions
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        return {"loss_cls_clip": cross_entropy(scores, gt_classes, reduction="mean"),}




