import json
from collections import OrderedDict
from typing import Optional, Type, Sequence, Literal
from dataclasses import dataclass, fields
import logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import torch
import transformers as HFT
import numpy as np
from torch import nn
from einops import rearrange

from .dcformer import DecompConv3D, DecompModel
from .m3dvit import ViT

POINT_PROMPT_PAD_INDEX = -1


class MaskPromptM3DViTConfig(HFT.PretrainedConfig):
    model_type: str = 'mask_prompt_vit3d'
    def __init__(
        self,
        in_channels: int = 1,

        input_size: tuple[int] = (32, 256, 256),
        axes_code: str = "SRA",
        patch_size: tuple[int] = (4, 16, 16),
        m3d_hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = 'perceptron',   # The original one is "conv"
        dropout_rate: float = .0,
        post_activation: str = "Tanh",
        classification: bool = True,     # The original one is False

        kernel_sizes: Sequence[int] = (13, 11, 9, 7),
        downsample: bool = False,
        prompt_act: str = "GELU",
        num_class: int = 512,
        prompt_hidden_size: Optional[int] = None,
        **kwargs
    ):
        _image_size = _img_size = None
        if kwargs.get('image_size') is not None:
            _image_size = kwargs.pop('image_size')
        if kwargs.get('img_size') is not None:
            _img_size = kwargs.pop('img_size')            
        image_size = None if _image_size != _img_size else _image_size
        if image_size is not None:
            print(f'INFO:|Change input_size from {input_size} to {image_size}')
            input_size = image_size


        self.in_channels = in_channels
        self.input_size = self.image_size = self.img_size = input_size
        self.axes_code = axes_code
        self.patch_size = patch_size
        self.m3d_hidden_size = self.hidden_size = m3d_hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.dropout_rate = dropout_rate
        self.post_activation = post_activation
        self.classification = classification
        
        self.kernel_sizes = kernel_sizes
        self.downsample = downsample if prompt_hidden_size is None  else False
        self.prompt_act = prompt_act
        self.num_class = num_class
        self.prompt_hidden_size = prompt_hidden_size
        super().__init__(**kwargs)
    
    

    @property
    def new_keys(self) -> list[str]:
        return [
            'input_size', 'in_channels', 'axes_code', 'patch_size', 'hidden_size', 'mlp_dim',
            'num_layers', 'num_heads', 'pos_embed', 'dropout_rate', 'post_activation', 'classification',
            'kernel_sizes', 'downsample', 'prompt_act', 'num_class', 'prompt_hidden_size'
        ]

class MaskPromptEncoder(nn.Module):
    def __init__(self, config: MaskPromptM3DViTConfig):
        super().__init__()
        # config.channels = [config.hidden_size] * 4
        if getattr(config, 'prompt_hidden_size', None) is not None:
            config.downsample = False
            logger.debug("Since `prompt_hidden_size` is provided, we set `downsample` to False in prompt encoder.")
        self.config = config
        
        self.mask_codebook = nn.Embedding(config.num_class, config.channels[0])

        if config.prompt_hidden_size is not None:            
            self.hidden_size_adapter: nn.Linear = nn.Linear(
                np.prod(config.input_size) // (4 ** 3), config.prompt_hidden_size, bias=False
            )
            self.pseudo_patch_size: Optional[tuple[int]] = config.patch_size
        else:
            self.hidden_size_adapter = nn.Identity()
        


        self.down0 = nn.Conv3d(config.channels[0], config.channels[0], 1, 4, bias=False)
        stride = 2 if config.downsample else 1
        self.down1 = nn.Sequential(
            DecompConv3D(config.channels[0], config.channels[1], stride=stride, kernel_size=config.kernel_sizes[0]),
            getattr(torch.nn, config.prompt_act, nn.GELU)(),
        )

        self.down2 = nn.Sequential(
            DecompConv3D(config.channels[1], config.channels[2], stride=stride, kernel_size=config.kernel_sizes[1]),
            getattr(torch.nn, config.prompt_act, nn.GELU)(),
        )

        self.down3 = nn.Sequential(
            DecompConv3D(config.channels[2], config.channels[3], stride=stride, kernel_size=config.kernel_sizes[2]),
            getattr(torch.nn, config.prompt_act, nn.GELU)(),
        )
        logger.debug(f'MaskPromptM3DViT.__init__|config.channels|{config.channels}')
        config.channels[3]
        config.channels[4]
        config.kernel_sizes[3]
        self.down4 = nn.Sequential(
            DecompConv3D(config.channels[3], config.channels[4], stride=stride, kernel_size=config.kernel_sizes[3]),
            getattr(torch.nn, config.prompt_act, nn.GELU)()
        )

    def forward(self, x, return_hidden_states: bool = False):
        """
            Each output tensors will following this shape: (Batch, hidden size(dim, channel, etc...), Num of embeddings(patches))            
        """
        hidden_states: list[torch.Tensor] = list()
        B, C, T, H, W = x.shape        
        
        x = self.mask_codebook(x.to(torch.int))
        x = rearrange(x, "B C H W D Dim -> B (C Dim) H W D")
        x = self.down0(x)
        logger.debug(f'MaskPromptEncoder.forward|After down0|{x.shape}')
        x = self.hidden_size_adapter(rearrange(x, "B Dim H W D -> B Dim (H W D)"))
        logger.debug(f'MaskPromptEncoder.forward|After hidden_size_adapter|{x.shape}')
        if getattr(self, "pseudo_patch_size", None) is not None:
            logger.debug(f'We contains a hidden_size_adapter, thus we will reshape the feature to a pseudo image.')
            p_t, p_h, p_w = self.pseudo_patch_size
            x = rearrange(
                x, "B Dim (pseuT pseuH pseuW)  -> B Dim pseuT pseuH pseuW", 
                pseuT=T // p_t, pseuH=H // p_h, pseuW= W // p_w
            )
        else:
            logger.debug(f'No hidden_size_adapter, keep the feature shape as is.')


        if return_hidden_states:
            hidden_states.append(rearrange(x, "B Dim H W D -> B (H W D) Dim"))

        for layer_idx, down_layer in enumerate([self.down1, self.down2, self.down3, self.down4]):
            logging.debug(f'MaskPrompt-{layer_idx + 1}|[Start]|Input Shape: {x.shape}')
            x = down_layer(x)
            logging.debug(f'MaskPrompt-{layer_idx + 1}|Feature: {x.shape}')
            if return_hidden_states:
                hidden_states.append(rearrange(x, "B Dim H W D -> B (H W D) Dim"))

        if return_hidden_states:
            return hidden_states
        return x


class MaskPromptM3DViT(HFT.PreTrainedModel):
    config_class = MaskPromptM3DViTConfig
    base_model_prefix = "mask_prompt_m3d_vit"
    
    def __init__(self, config: MaskPromptM3DViTConfig):
        if isinstance(config, dict):
            config = MaskPromptM3DViTConfig.from_dict(config)
        config.channels = [config.m3d_hidden_size] * 5
        super().__init__(config)
        self.config = config
        logger.debug(f'MaskPromptM3DViT.__init__|config|{json.dumps(vars(config), indent=2)}')

        self.prompt_encoder: MaskPromptEncoder = MaskPromptEncoder(config)
        self.backbone: ViT = ViT(
            in_channels=config.in_channels,
            img_size=config.input_size,
            patch_size=config.patch_size,
            hidden_size=config.m3d_hidden_size,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            pos_embed=config.pos_embed,
            dropout_rate=config.dropout_rate,
            post_activation=config.post_activation,            
            spatial_dims=len(config.input_size),
            classification=config.classification
        )
    
    def freeze(self, module_name: str) -> None:
        if module_name == 'vision':
            print(f'Freezing vision encoder: {self.vision_encoder.__class__.__name__}')
            self.vision_encoder.requires_grad_(False)
        elif module_name == 'prompt':
            print(f'Freezing prompt encoder: {self.prompt_encoder.__class__.__name__}')
            self.prompt_encoder.requires_grad_(False)
        else:
            print(f'Freezing all modules: {self.__class__.__name__}')
            self.requires_grad_(False)

    def forward(self, pixel_values, masks=None, no_prompt=False, return_backbone=False, fuse_stage: list[int] = None):
        """
            The `no_prompt` is meaning don't adding anything on dcformer output.
            `return_backbone` is meaning return un-fused backbone features and fused features.
        """
        if fuse_stage is None:
            fuse_stage = [0, 1, 2, 3, 4]

        logger.debug(f'Input Shape: {pixel_values.shape}')

        last_feat, feature_list = self.backbone(pixel_values)
        if no_prompt:
            logger.debug(f'argument `no_prompt` is set to True, ignore masks, return backbone results.')
            return feature_list

        if masks is None:
            logger.debug(f'No `masks` is provided, use all-zero mask instead')
            masks = torch.zeros_like(pixel_values)                    
        masks_prompt = self.prompt_encoder(masks, return_hidden_states=True)

        if return_backbone:
            fuse_list = list()
            for i, (image_embedding, mask_prompt) in enumerate(zip(feature_list, masks_prompt)):
                logger.debug(f'Layer-{i}|emb_v, emb_p|{image_embedding.shape}, {mask_prompt.shape}')
                if i in fuse_stage:
                    fuse_list.append(image_embedding + mask_prompt)
                else:
                    fuse_list.append(image_embedding)
                # fuse_list.append(image_embedding + mask_prompt)
            return feature_list, fuse_list


        for i, (image_embedding, mask_prompt) in enumerate(zip(feature_list, masks_prompt)):
            logger.debug(f'Image+Prompt|{image_embedding.shape}:image|{mask_prompt.shape}:mask prompt')
            if i in fuse_stage:
                feature_list[i] = image_embedding + mask_prompt
            else:
                feature_list[i] = image_embedding
            # feature_list[i] = image_embedding + mask_prompt
        return feature_list

    def load_backbone_state(self, state_dict: OrderedDict):
        self.backbone.load_state_dict(state_dict, strict=True)


    @property
    def channels(self) -> list[int]:
        return self.config.channels

HFT.AutoConfig.register('mask_prompt_vit3d', MaskPromptM3DViTConfig)
# HFT.AutoConfig.register('prompt_dcformer', PromptDCFormerConfig)
HFT.AutoModel.register(MaskPromptM3DViTConfig, MaskPromptM3DViT)
# print(__name__)

if __name__ == "__main__":
    # pseudo_config = {
    #     'prompt_hidden_size': 2048,
    #     'vision_pretrained_model': '/user/...'
    # }
    # # config = MaskPromptM3DViTConfig(downsample=False, prompt_hidden_size=2048)
    # key_pool = set(_field.name for _field in fields(MaskPromptM3DViTConfig))
    # print(key_pool)
    # filtered_config = {k: v for k, v in pseudo_config.items() if k in key_pool}
    config = MaskPromptM3DViTConfig(prompt_hidden_size=2048)
    
    mask_encoder = MaskPromptM3DViT(config).cuda()
    x = torch.randn((1, 1, 32, 256, 256)).cuda()

    emb_v, emb_fuse = mask_encoder(x, fuse_stage=[5], return_backbone=True)

    
    