from ast import mod
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from typing import Dict,List,Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from .clip import localClip

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_CLIP(GeneralizedRCNN):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.clip_model = localClip(cfg)
        self.aug_num = cfg.DOMAIN_AUG_NUM
        self.output_dir = cfg.OUTPUT_DIR

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        aug_ids = np.random.choice(np.arange(self.aug_num), 1)[0]
        features = self.backbone(images.tensor, aug_ids)

        if self.proposal_generator is not None:
            proposals, proposal_losses= self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses \
            = self.roi_heads(images, features, proposals, gt_instances,
                            self.clip_model.enc.attnpool, self.clip_model.text_diff[aug_ids],images.image_sizes,
                             self.clip_model.class_causal_text_basisvectors, self.clip_model.class_noncausal_text_basisvectors)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None, None, None,
                                        self.clip_model.enc.attnpool, None, None, None,
                                        None, None, None,
                                        self.clip_model.class_causal_text_basisvectors, self.clip_model.class_noncausal_text_basisvectors)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results









