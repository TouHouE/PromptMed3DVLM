import logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
log_fmt = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_log = logging.FileHandler('./log/prompt_visual_encoder.log', 'w+')
file_log.setLevel(logging.DEBUG)
file_log.setFormatter(log_fmt)
logger.addHandler(file_log)

import torch.nn as nn

from .dcformer import decomp_naive, decomp_nano, decomp_small, decomp_tiny
from .vit import Vit3D
from .prompt_dcformer import MaskPromptDCFormer, PromptDCFormerConfig

def build_vision_tower(config, **kwargs):
    return VisionTower(config)


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        self.hidden_size = config.dim

        logger.debug(f'the vision_tower is {config.vision_tower}')
        logger.debug(f'the select_layer is {self.select_layer}')
        logger.debug(f'the select_feature is {self.select_feature}')

        if config.vision_tower == "vit3d":
            self.vision_tower = Vit3D(
                input_size=config.input_size,
                dim=config.dim,
                depth=config.depth,
            )
        elif config.vision_tower == "dcformer":
            self.vision_tower = decomp_small(
                input_size=config.input_size,
            )
            self.low_input_size = self.vision_tower.channels[-2]
            self.high_input_size = self.vision_tower.channels[-1]
        elif config.vision_tower == 'prompt_dcformer':
            self.vision_tower = MaskPromptDCFormer(config.vision_tower_config)
        else:
            raise ValueError(f"Unexpected vision tower: {config.vision_tower}")

    def forward(self, images, **kwargs):
        logger.debug(f"Other model inputs: {kwargs.keys()}")
        hidden_states = self.vision_tower(images, **kwargs)
        if self.select_layer == 0:
            image_features = hidden_states
        elif self.select_layer < 0:
            image_features = hidden_states[self.select_layer :]
        else:
            raise ValueError(f"Unexpected select layer: {self.select_layer}")

        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
