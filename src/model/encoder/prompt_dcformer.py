import json
from collections import OrderedDict
from typing import Optional, Type, Sequence, Literal
from dataclasses import dataclass
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

import torch
import transformers as HFT
from torch import nn
from einops import rearrange

from .dcformer import DecompConv3D, DecompModel

POINT_PROMPT_PAD_INDEX = -1


@dataclass
class PromptDCFormerConfig(HFT.PretrainedConfig):
    model_type: str = "mask_prompt_dcformer"
    # Using in many Module
    input_size: Sequence[int] = (512, 512, 256)
    channels: Sequence[int] = (64, 96, 192, 384, 768)
    in_channels: int = 1
    kernel_sizes: Sequence[int] = (13, 11, 9, 7)
    # MaskPrompt Encoder usage only
    num_class: int = 512
    # PositionEncoding usage only
    scale: float = 1.
    # PromptEncoder usage only
    prompt_act: str = "GELU"
    num_point_embeddings: int = 2  # positive and negative prompt point
    num_box_embeddings: int = 2  # bounding box 2 points(top-left), (bottom-right)
    # DCFormer usage only
    num_blocks: Sequence[int] = (2, 2, 3, 5, 2)
    block_types: Sequence[Literal["C", "T"]] = ("C", "C", "C", "C")
    codebook_size: int = 8192
    model_size: Optional[Literal["tiny", "base", "small", "large"]] = None
    attn_implementation: str = 'sdpa'

    @classmethod
    def large_config(cls, input_size=(512, 512, 256)):
        return cls(
            input_size=input_size,
            num_blocks=[1, 2, 6, 12, 2],
            channels=[64, 256, 512, 1024, 2048],
        )

    @classmethod
    def base_config(cls, input_size=(512, 512, 256)):
        return cls(
            input_size=input_size,
            num_blocks=[1, 2, 6, 6, 2],
            channels=[64, 128, 256, 512, 1024],
        )

    @classmethod
    def small_config(cls, input_size=(512, 512, 256)):
        return cls(
            input_size=input_size,
            num_blocks=[1, 2, 3, 6, 2],
            channels=[64, 96, 192, 384, 768],
        )

    @classmethod
    def tiny_config(cls, input_size=(512, 512, 256)):
        return cls(
            input_size=input_size,
            num_blocks=[1, 2, 3, 3, 2],
            channels=[64, 96, 192, 384, 768],
        )
        pass

    @classmethod
    def get_default_config(cls, config_size: str):
        return getattr(cls, f'{config_size}_config')


class DCFormer(nn.Module):
    def __init__(self, config: PromptDCFormerConfig):
        super().__init__()
        self.config = config
        self.model = DecompModel(
            input_size=config.input_size,
            num_blocks=config.num_blocks,
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
            block_types=config.block_types,
            codebook_size=config.codebook_size,
        )

    def forward(self, x):
        return self.model(x)

    @property
    def channels(self):
        return self.model.channels

"""input: (1, 1, 256, 256, 128) patch_size: [4] * 3
torch.Size([1, 131072, 64])
torch.Size([1, 16384, 96])
torch.Size([1, 2048, 192])
torch.Size([1, 256, 384])
torch.Size([1, 32, 768])
"""


class MaskPromptEncoder(nn.Module):
    def __init__(self, config: PromptDCFormerConfig):
        super().__init__()
        self.config = config
        self.mask_codebook = nn.Embedding(config.num_class, config.channels[0])

        self.down0 = nn.Conv3d(config.channels[0], config.channels[0], 1, 4, bias=False)

        self.down1 = nn.Sequential(
            DecompConv3D(config.channels[0], config.channels[1], stride=2, kernel_size=config.kernel_sizes[0]),
            getattr(torch.nn, config.prompt_act, nn.GELU)(),
        )

        self.down2 = nn.Sequential(
            DecompConv3D(config.channels[1], config.channels[2], stride=2, kernel_size=config.kernel_sizes[1]),
            getattr(torch.nn, config.prompt_act, nn.GELU)(),
        )

        self.down3 = nn.Sequential(
            DecompConv3D(config.channels[2], config.channels[3], stride=2, kernel_size=config.kernel_sizes[2]),
            getattr(torch.nn, config.prompt_act, nn.GELU)(),
        )
        self.down4 = nn.Sequential(
            DecompConv3D(config.channels[3], config.channels[4], stride=2, kernel_size=config.kernel_sizes[3]),
            getattr(torch.nn, config.prompt_act, nn.GELU)()
        )

    def forward(self, x, return_hidden_states: bool = False):
        hidden_states: list[torch.Tensor] = list()
        B, C, H, W, D = x.shape
        # with torch.autocast(dtype=torch.int, device_type='cuda'):
        x = self.mask_codebook(x.to(torch.int))
        x = rearrange(x, "B C H W D Dim -> B (C Dim) H W D")
        x = self.down0(x)

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


class RandomPositionEncoding(nn.Module):
    def __init__(self, config: PromptDCFormerConfig):
        super().__init__()
        self.config = config
        scale: float = 1. if config.scale is None or config.scale <= .0 else config.scale
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, config.channels[-1]))
        )

    def encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([
            torch.sin(coords), torch.cos(coords), torch.sin(coords)
        ], dim=-1)

    def forward(self, size: tuple[int, int, int], device: Optional[str] = None,
                dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        x, y, z = size
        device = self.positional_encoding_gaussian_matrix if device is None else device
        dtype = self.positional_encoding_gaussian_matrix if dtype is None else dtype
        grid = torch.ones(size, device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - .5
        x_embed = grid.cumsum(dim=1) - .5
        z_embed = grid.cumsum(dim=2) - .5
        y_embed = y_embed / y
        x_embed = x_embed / x
        z_embed = z_embed / z
        positional_encoding = self.encoding(torch.stack([
            x_embed, y_embed, z_embed
        ], dim=-1))
        return rearrange(positional_encoding, "X Y Z C -> C X Y Z")

    def forward_coords(self, coords_input, image_size, device: Optional[str] = None,
                       dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self.encoding(coords.to(dtype))  # B x N x C


class PromptEncoder(nn.Module):

    def __init__(self, config: PromptDCFormerConfig):
        super().__init__()
        self.config = config
        # the + 1 is for none point embeddings. it should place to index-0
        self.point_codebook = nn.ModuleList(
            nn.Embedding(1, config.channels[-1]) for _ in range(config.num_point_embeddings + 1))
        self.box_codebook = nn.ModuleList(
            nn.Embedding(1, config.channels[-1]) for _ in range(config.num_box_embeddings))
        self.mask_embeds = MaskPromptEncoder(config)
        self.position_encoder = RandomPositionEncoding(config.channels[-1] // 3)

    def do_point_embeddings(self, batch_size, points, do_pad: bool = False):
        point_coords, point_labels = points
        if do_pad:
            pad_coord = torch.zeros((batch_size, 1, 3))
            pad_label = - torch.ones((batch_size, 1))
            point_coords = torch.cat([pad_coord, point_coords], dim=1)
            point_labels = torch.cat([pad_label, point_labels], dim=1)
        point_embeddings = self.position_encoder.forward_coords(
            point_coords, self.config.input_size
        )
        point_embeddings[point_labels == POINT_PROMPT_PAD_INDEX] = .0
        for cur_point_label in range(3):  # 0: -1(pad), 1: 0(neg), 2: 1(pos)
            point_embeddings[point_labels == cur_point_label - 1] += self.point_embeds[cur_point_label].weight
        return point_embeddings

    def do_box_embeddings(self, batch_size, boxes):
        boxes = boxes + .5
        coords = boxes.reshape(-1, 2, 3)  # N box, top-left, bottom-right, xyz
        boxes_embeddings = self.position_encoder.forward_coords(
            coords, self.config.input_size
        )
        for i in range(2):  # 0 -> 2, 1 -> 3
            boxes_embeddings[:, i] = boxes[:, i] + self.box_codebook[i].weight
        return boxes_embeddings

    def forward(
            self, batch_size=1,
            points: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            boxes: Optional[torch.Tensor] = None,
            masks: Optional[torch.Tensor] = None
    ):
        """
            :param: points: a tuple of :class: `torch.Tensor`, idx-0 present coordinate, idx-1 present point label.
                idx-0 should following shape: BxNx3, (3 for 3-dimension)
                idx-1 should following shape: BxNx1
            :param: boxes: a :class: `torch.Tensor`,
        """
        if masks is not None:
            masks = torch.zeros((batch_size, 1, *config.input_size))
        dense_prompt = self.mask_embeds(masks)
        coords_prompt = torch.empty((batch_size, 0, self.config.channels[-1]))
        if points is not None:
            # coords, labels = points
            point_prompt = self.do_point_embeddings(batch_size, points, do_pad=boxes is None)
            coords_prompt = torch.cat([coords_prompt, point_prompt], dim=1)
        if boxes is not None:
            box_prompt = self.do_box_embeddings(batch_size, boxes)
            coords_prompt = torch.cat([coords_prompt, box_prompt], dim=1)
        return dense_prompt, coords_prompt


class PromptDCFormer(nn.Module):
    def __init__(self, config: PromptDCFormerConfig):
        super().__init__()
        self.config = config
        self.prompt_encoder = PromptEncoder(config)
        self.dcformer = DCFormer(config)

    def forward(
            self, image,
            points=None, boxes=None, masks=None
    ):
        feature_list = self.dcformer(image)
        if all(_prompt is None for _prompt in [points, boxes, masks]):
            return feature_list
        dense_prompt, coords_prompt = self.prompt_encoder(
            image.shape[0], points, boxes, masks
        )


class MaskPromptDCFormer(nn.Module):
    def __init__(self, config: PromptDCFormerConfig):
        super().__init__()
        if isinstance(config, dict):
            config = PromptDCFormerConfig.from_dict(config)

        if config.model_size is not None:
            config = PromptDCFormerConfig.get_default_config(config.model_size)

        logger.debug(f'Final visual tower config:\n{json.dumps(config.to_dict())}')
        self.config = config
        self.dcformer = DCFormer(config)
        self.prompt_encoder = MaskPromptEncoder(config)

    def forward(self, pixel_values, masks=None, no_prompt=False):
        logger.debug(f'Input Shape: {pixel_values.shape}')

        feature_list = self.dcformer(pixel_values)
        if no_prompt:
            logger.info(f'argument `no_prompt` is set to True, ignore masks')
            return feature_list

        if masks is None:
            logger.info(f'No `masks` is provided, use all-zero mask instead')
            masks = torch.zeros_like(pixel_values)

        masks_prompt = self.prompt_encoder(masks, return_hidden_states=True)

        for i, (image_embedding, mask_prompt) in enumerate(zip(feature_list, masks_prompt)):
            logger.debug(f'Image+Prompt|{image_embedding.shape}:image|{mask_prompt.shape}:mask prompt')
            feature_list[i] = image_embedding + mask_prompt
        return feature_list

    def load_dcformer_state(self, state_dict: dict, **kwargs):
        logger.info('Loading DCFormer state dict...')
        prefix_state = OrderedDict()
        for org_key, value in state_dict.items():
            prefix_state[f'model.{org_key}'] = value
        self.dcformer.load_state_dict(prefix_state, **kwargs)

    @property
    def channels(self):
        return self.dcformer.channels

HFT.AutoConfig.register('mask_prompt_dcformer', PromptDCFormerConfig)
HFT.AutoModel.register(PromptDCFormerConfig, MaskPromptDCFormer)

if __name__ == "__main__":
    config = PromptDCFormerConfig.small_config((256, 256, 128))
    mask_encoder = MaskPromptEncoder(config)
    x = torch.randn((1, 1, 256, 256, 128))
    y = mask_encoder(x)