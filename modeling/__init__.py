from .rpn import SBRPN
from .backbone import build_resnet_backbone_statistics
from .meta_arch import GeneralizedRCNN_CLIP
from .roi_head import Res5ROIHeads_CLIP
from .config import add_stn_config
from .custom_pascal_evaluation import CustomPascalVOCDetectionEvaluator