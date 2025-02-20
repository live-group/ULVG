# -*- coding: UTF-8 -*-
# from cgi import parse_multipart
import os
import logging
import time
from collections import OrderedDict, Counter
# import copy
import weakref

# import numpy as np

import torch
# from torch import autograd
# import torch.utils.data as torchdata

from detectron2 import model_zoo
from detectron2.config import get_cfg
# from detectron2.layers.batch_norm import FrozenBatchNorm2d
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_setup, create_ddp_model, AMPTrainer, SimpleTrainer
from detectron2.engine import default_argument_parser, hooks, HookBase
# from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, get_detection_dataset_dicts
# from detectron2.data.common import DatasetFromList, MapDataset
# from detectron2.data.samplers import InferenceSampler
# from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.evaluation import COCOEvaluator, verify_results, inference_on_dataset, print_csv_format

# from detectron2.solver import LRMultiplier
from detectron2.modeling import build_model
# from detectron2.structures import ImageList, Instances, pairwise_iou, Boxes
#
# from fvcore.common.param_scheduler import ParamScheduler
# from fvcore.common.checkpoint import Checkpointer

from data.datasets import builtin

from detectron2.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator, inference_on_dataset

from detectron2.data import build_detection_train_loader, MetadataCatalog
# import torch.utils.data as data
# from detectron2.data.dataset_mapper import DatasetMapper
# import detectron2.data.detection_utils as utils
# import detectron2.data.transforms as detT
#
# import torchvision.transforms as T
# import torchvision.transforms.functional as tF
from fvcore.nn.precise_bn import get_bn_modules

from modeling import add_stn_config
from modeling import CustomPascalVOCDetectionEvaluator
from solver.build import build_optimizer
from data.dataset_mapper import CustomDatasetMapper

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    # hack to add base yaml
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


# class CustomDatasetMapper(DatasetMapper):
#     def __init__(self, cfg, is_train) -> None:
#         super().__init__(cfg, is_train)
#         self.with_crops = cfg.INPUT.CLIP_WITH_IMG
#         self.with_random_clip_crops = cfg.INPUT.CLIP_RANDOM_CROPS
#         self.with_jitter = cfg.INPUT.IMAGE_JITTER
#         self.cropfn = T.RandomCrop  # T.RandomCrop([224,224])
#         self.aug = T.ColorJitter(brightness=.5, hue=.3)
#         self.crop_size = cfg.INPUT.RANDOM_CROP_SIZE
#
#     def __call__(self, dataset_dict):
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         # USER: Write your own image loading if it's not from a file
#         image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
#         utils.check_image_size(dataset_dict, image)
#
#         # USER: Remove if you don't do semantic/panoptic segmentation.
#         if "sem_seg_file_name" in dataset_dict:
#             sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
#         else:
#             sem_seg_gt = None
#
#         aug_input = detT.AugInput(image, sem_seg=sem_seg_gt)
#         transforms = self.augmentations(aug_input)
#         image, sem_seg_gt = aug_input.image, aug_input.sem_seg
#
#         image_shape = image.shape[:2]  # h, w
#         # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#         # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#         # Therefore it's important to use torch.Tensor.
#         dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
#
#         # USER: Remove if you don't use pre-computed proposals.
#         # Most users would not need this feature.
#         if self.proposal_topk is not None:
#             utils.transform_proposals(
#                 dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
#             )
#
#         if not self.is_train:
#             # USER: Modify this if you want to keep them for some reason.
#             dataset_dict.pop("annotations", None)
#             dataset_dict.pop("sem_seg_file_name", None)
#             return dataset_dict
#
#         if "annotations" in dataset_dict:
#             self._transform_annotations(dataset_dict, transforms, image_shape)
#
#         if self.with_jitter:
#             dataset_dict["jitter_image"] = self.aug(dataset_dict["image"])
#
#         if self.with_crops:
#             bbox = dataset_dict['instances'].gt_boxes.tensor
#             csx = (bbox[:, 0] + bbox[:, 2]) * 0.5
#             csy = (bbox[:, 1] + bbox[:, 3]) * 0.5
#             maxwh = torch.maximum(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])
#             crops = list()
#             gt_boxes = list()
#             mean = [0.48145466, 0.4578275, 0.40821073]
#             std = [0.26862954, 0.26130258, 0.27577711]
#             for cx, cy, maxdim, label, box in zip(csx, csy, maxwh, dataset_dict['instances'].gt_classes, bbox):
#
#                 if int(maxdim) < 10:
#                     continue
#                 x0 = torch.maximum(cx - maxdim * 0.5, torch.tensor(0))
#                 y0 = torch.maximum(cy - maxdim * 0.5, torch.tensor(0))
#                 try:
#                     imcrop = T.functional.resized_crop(dataset_dict['image'], top=int(y0), left=int(x0),
#                                                        height=int(maxdim), width=int(maxdim), size=224)
#                     imcrop = imcrop.flip(0) / 255  # bgr --> rgb for clip
#                     imcrop = T.functional.normalize(imcrop, mean, std)
#                     # print(x0,y0,x0+maxdim,y0+maxdim,dataset_dict['image'].shape)
#                     # print(imcrop.min(),imcrop.max() )
#                     gt_boxes.append(box.reshape(1, -1))
#                 except Exception as e:
#                     print(e)
#                     print('crops:', x0, y0, maxdim)
#                     exit()
#                 # crops.append((imcrop,label))
#                 crops.append(imcrop.unsqueeze(0))
#
#             if len(crops) == 0:
#                 dataset_dict['crops'] = []
#             else:
#                 dataset_dict['crops'] = [torch.cat(crops, 0), Boxes(torch.cat(gt_boxes, 0))]
#
#         if self.with_random_clip_crops:
#             crops = []
#             rbboxs = []
#
#             for i in range(15):
#                 p = self.cropfn.get_params(dataset_dict['image'], [self.crop_size, self.crop_size])
#                 c = tF.crop(dataset_dict['image'], *p)
#                 if self.crop_size != 224:
#                     c = tF.resize(img=c, size=224)
#                 crops.append(c)
#                 rbboxs.append(p)
#
#             crops = torch.stack(crops)
#             dataset_dict['randomcrops'] = crops
#
#             # apply same crop bbox to the jittered image
#             if self.with_jitter:
#                 jitter_crops = []
#                 for p in rbboxs:
#                     jc = tF.crop(dataset_dict['jitter_image'], *p)
#                     if self.crop_size != 224:
#                         jc = tF.resize(img=jc, size=224)
#                     jitter_crops.append(jc)
#
#                 jcrops = torch.stack(jitter_crops)
#                 dataset_dict['jitter_randomcrops'] = jcrops
#
#         return dataset_dict

class gradclipSimpleTrainer(SimpleTrainer):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, cfg, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__(model, data_loader, optimizer)
        self.domain_gradmax = cfg.SOLVER.DOMAIN_AUG_GRADMAX # 域相关参数的梯度截断阈值


    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        # clip gradient of self.model.backbone.feat_alpha and self.backbone.feat_alpha
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        # nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        param_name_list = ['backbone.feat_alpha', 'backbone.feat_beta', 'backbone.feat_alpha_l1', 'backbone.feat_beta_l1',
                           'backbone.feat_alpha_l0', 'backbone.feat_beta_l0','backbone.feat_alpha_l3', 'backbone.feat_beta_l3',]
        # param_name_list = ['backbone.feat_alpha_l1', 'backbone.feat_beta_l1',]
        torch.nn.utils.clip_grad_value_([param for name, param in self.model.named_parameters() if name in param_name_list], clip_value=self.domain_gradmax)
        # temp=torch.nn.utils.clip_grad_norm_([param for name, param in self.model.named_parameters() if name in param_name_list],max_norm=100)
        # print('total_norm', temp)
        # param_name_list = ['backbone.feat_alpha', 'backbone.feat_beta', ]
        # torch.nn.utils.clip_grad_value_([param for name, param in self.model.named_parameters() if name in param_name_list],clip_value=0.01)

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

class Trainer(DefaultTrainer):
    def __init__(self, cfg) -> None:
        # 改写DefaultTrainer的__init__函数    将trainer变成可以梯度截断的训练器SimpleTrainer --> gradclipSimpleTrainer
        super(DefaultTrainer,self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        # self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
        #     model, data_loader, optimizer
        # )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else gradclipSimpleTrainer)(
            cfg, model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))

        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)


    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if MetadataCatalog.get(dataset_name).evaluator_type == 'pascal_voc':
            return CustomPascalVOCDetectionEvaluator(dataset_name)
        else:
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=100))
        return ret



def do_test(cfg, model, model_type=''):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = CustomPascalVOCDetectionEvaluator(dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
    return results


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # for dataset_name in cfg.DATASETS.TEST:
    #     if 'daytime_clear_test' in dataset_name:
    #         trainer.register_hooks([
    #             hooks.BestCheckpointer(cfg.TEST.EVAL_SAVE_PERIOD, trainer.checkpointer, f'{dataset_name}_AP50',
    #                                    file_prefix='model_best'),
    #         ])

    trainer.train()
    # os.system("/usr/bin/shutdown")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # cfg = setup(args)
    print("Command Line Args:", args)

    main(args)
