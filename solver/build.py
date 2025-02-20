import torch
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
import copy
from detectron2.config import CfgNode
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        overrides={"feat_alpha": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_LR_TIMES},
                   "feat_beta": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_LR_TIMES},
                   "feat_alpha_l1": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_L1_LR_TIMES},
                   "feat_beta_l1": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_L1_LR_TIMES},
                   "feat_alpha_l3": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_L3_LR_TIMES},
                   "feat_beta_l3": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_L3_LR_TIMES},
                   "feat_alpha_l0": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_L0_LR_TIMES},
                   "feat_beta_l0": {"lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DOMAIN_AUG_L0_LR_TIMES},
                   },
    )
    # params=[{}]
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )


