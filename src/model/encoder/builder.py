import json
import os
DEBUG: bool = os.environ.get('DEBUG', '0') == '1'
import logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
log_fmt = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_log = logging.FileHandler('./log/prompt_visual_encoder.log', 'a+')
file_log.setLevel(logging.DEBUG)
file_log.setFormatter(log_fmt)
logger.addHandler(file_log)
from dataclasses import fields, is_dataclass

import torch.nn as nn

from .dcformer import decomp_naive, decomp_nano, decomp_small, decomp_tiny
from .vit import Vit3D
from .m3dvit import ViT as M3DViT
from .prompt_dcformer import MaskPromptDCFormer, PromptDCFormerConfig
from .prompt_m3dvit import MaskPromptM3DViT, MaskPromptM3DViTConfig

def build_vision_tower(config, **kwargs):
    return VisionTower(config)


def wrap_matched_keys(config_obj, config_class) -> dict[str, any]:
    """Wrap all keys in config_obj to config_class if not already."""
    if not isinstance(config_obj, dict):
        try:
            config_obj: dict[str, any] = vars(config_obj)
        except Exception as e:
            raise ValueError(f"Cannot convert config_obj to dict: {e}")
    if is_dataclass(config_class):
        key_pool = set(field.name for field in fields(config_class))
    else:
        key_pool = config_class().new_keys
    filtered_config = {k: v for k, v in config_obj.items() if k in key_pool}
    return filtered_config


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        print(f'Config in VisionTower.__init__:\n{vars(config)}')
        if 'lamed' in config.model_type:
            self.hidden_size = getattr(config, 'mm_hidden_size', getattr(config, 'hidden_size', 768))
            # print(config.image_size)
            # print("WTF(getattr): "+ getattr(config, 'image_size'))
            input_size = getattr(config, 'image_size', 
                getattr(config, 'img_size', 
                    getattr(config, 'input_size', 
                        (32, 256, 256)
                    )
                )
            )
            config.depth = 12
        else:
            self.hidden_size = config.dim
            input_size = config.input_size

        logger.debug(f'the vision_tower is {config.vision_tower}')
        logger.debug(f'the select_layer is {self.select_layer}')
        logger.debug(f'the select_feature is {self.select_feature}')

        # TODO: here is debug information
        # Why using print? logger not working...
        print(f'builder.py::the vision_tower is {config.vision_tower}')
        # print(f'builder.py::the select_layer is {self.select_layer}')
        # print(f'builder.py::the select_feature is {self.select_feature}')
        if config.vision_tower == "vit3d":
            self.vision_tower = Vit3D(
                input_size=input_size,
                # dim=config.dim,
                dim=self.hidden_size,
                depth=config.depth,
            )
        elif config.vision_tower == 'm3dvit':
            self.vision_tower = M3DViT(
                in_channels=config.image_channel,
                img_size=input_size,
                patch_size=config.patch_size,
                pos_embed="perceptron",
                spatial_dims=len(config.patch_size),
                classification=True,
            )
        elif config.vision_tower == "dcformer":
            self.vision_tower = decomp_small(
                input_size=input_size,
            )
            self.low_input_size = self.vision_tower.channels[-2]
            self.high_input_size = self.vision_tower.channels[-1]
        elif config.vision_tower in ['prompt_dcformer', 'mask_prompt_dcformer']:
            self.vision_tower = MaskPromptDCFormer(PromptDCFormerConfig.small_config(config.input_size))
            self.low_input_size = self.vision_tower.channels[-2]
            self.high_input_size = self.vision_tower.channels[-1]
        elif config.vision_tower == 'mask_prompt_m3dvit':
            matched_config = wrap_matched_keys(config, MaskPromptM3DViTConfig)
            self.vision_tower = MaskPromptM3DViT(MaskPromptM3DViTConfig(**matched_config))
        else:
            raise ValueError(f"Unexpected vision tower: {config.vision_tower}")
        
    def forward(self, images, **kwargs):
        logger.debug(f"Other model inputs: {kwargs.keys()}")
        if self.config.vision_tower == 'mask_prompt_m3dvit':
            kwargs['fuse_stage'] = [4]
        
        hidden_states = self.vision_tower(images, **kwargs)
        
        # if self.config.vision_tower == 'mask_prompt_m3dvit':
        #     breakpoint()
        #     last_state, hidden_states = hidden_states

        if DEBUG:
            breakpoint()
        if self.select_layer == 0:
            image_features = hidden_states[-1]
        elif self.select_layer < 0:
            image_features = hidden_states[self.select_layer :]
            if self.config.vision_tower == 'mask_prompt_m3dvit':
                image_features = image_features[0]
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
