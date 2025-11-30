from yacs.config import CfgNode as CN

_CN = CN()

##############  Model    ##############
_CN.MODEL = None  # options: ['MASt3R', 'MASt3RSegFeat']
_CN.DEBUG = None
_CN.SAVE_DIR = None

_CN.RANDOM_SEED = None

_CN.FEATURE_MATCHER = CN()
_CN.FEATURE_MATCHER.TYPE = None
_CN.FEATURE_MATCHER.DUAL_SOFTMAX = CN()
_CN.FEATURE_MATCHER.DUAL_SOFTMAX.TEMPERATURE = None
_CN.FEATURE_MATCHER.DUAL_SOFTMAX.USE_DUSTBIN = None
_CN.FEATURE_MATCHER.SINKHORN = CN()
_CN.FEATURE_MATCHER.SINKHORN.NUM_IT = None
_CN.FEATURE_MATCHER.SINKHORN.DUSTBIN_SCORE_INIT = None

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
_CN.DATASET.DATA_SOURCE = None  # options: ['ScanNetPP', 'MapFree']
_CN.DATASET.SCENES = None  # scenes to use (for 7Scenes/MapFree); should be a list []; If none, use all scenes.
_CN.DATASET.SCENES_VAL = (
    None  # scenes to use for validation; should be a list []; If none, use all scenes.
)
_CN.DATASET.DATA_ROOT = None  # path to RGB dataset folder
_CN.DATASET.SEGDATA_ROOT = None  # path to segmentation data folder
_CN.DATASET.PAIRS_ROOT = None  # path to pairs data folder
_CN.DATASET.TRANSFORMS_ROOT = None  # path to transforms data folder
_CN.DATASET.M_PRIME = None  # max number of masks in segdata
_CN.DATASET.DINOV2_CACHED_ROOT = None  # path to DINOv2 cached features
_CN.DATASET.MAST3RLIKE_CACHED_ROOT = None  # path to MASt3R-like cached features
_CN.DATASET.SAM2_CACHED_ROOT = None  # path to SAM2 cached features
_CN.DATASET.SAM_AMG_ROOT = None  # path to SAM AMG cached features
_CN.DATASET.HEIGHT = None
_CN.DATASET.WIDTH = None
_CN.DATASET.RESIZE_H = None
_CN.DATASET.RESIZE_W = None

############# TRAINING #############
_CN.TRAINING = CN()
# Data Loader settings
_CN.TRAINING.BATCH_SIZE = None
_CN.TRAINING.NUM_WORKERS = None
_CN.TRAINING.PREFETCH_FACTOR = None
_CN.TRAINING.NUM_GPUS = None

# Training settings
_CN.TRAINING.LR = None
_CN.TRAINING.WEIGHT_DECAY = None
_CN.TRAINING.LR_STEP_INTERVAL = None
_CN.TRAINING.LR_STEP_GAMMA = (
    None  # multiplicative factor of LR every LR_STEP_ITERATIONS
)
_CN.TRAINING.VAL_INTERVAL = None
_CN.TRAINING.VAL_BATCHES = None
_CN.TRAINING.LOG_INTERVAL = None
_CN.TRAINING.EPOCHS = None
_CN.TRAINING.GRAD_CLIP = (
    0.0  #  Indicates the L2 norm at which to clip the gradient. Disabled if 0
)
_CN.TRAINING.ALT_LOSS_WEIGHT = None

cfg = _CN
