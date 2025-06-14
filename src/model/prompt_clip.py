from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from src.model.encoder.dcformer import (
    decomp_base,
    decomp_naive,
    decomp_nano,
    decomp_small,
    decomp_tiny,
)
from src.model.encoder.vit import Vit3D
from src.model.encoder.prompt_dcformer import PromptDCFormerConfig, MaskPromptDCFormer
from src.model.projector.mlp import MultiLayerPerceptron

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


class PromptCLIPConfig(PretrainedConfig):
    model_type = "prompt_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        input_size: tuple = (256, 256, 128),
        dim: int = 768,
        depth: int = 12,
        hidden_size: int = 512,
        mlp_depth: int = 2,
        loss_type: str = "nce",
        t_prime: float = np.log(1 / 0.07),
        bias: float = 0.0,
        efficient_loss: bool = False,
        vision_encoder: str = "dcformer",
        prompt_encoder: Literal['mask', 'point', 'full'] = 'mask',
        init_lambda: tuple[float] = (.2, 1., 1.),
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.input_size = input_size
        self.dim = dim
        self.depth = depth
        self.hidden_size = hidden_size
        self.mlp_depth = mlp_depth
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.loss_type = loss_type
        self.t_prime = t_prime
        self.bias = bias
        self.efficient_loss = efficient_loss
        self.vision_encoder = vision_encoder
        self.prompt_encoder = prompt_encoder
        self.init_lambda = init_lambda
        super().__init__(**kwargs)


class PromptCLIP(PreTrainedModel):
    config_class = PromptCLIPConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        if config.prompt_encoder == 'mask':
            self.vision_encoder = MaskPromptDCFormer(PromptDCFormerConfig.small_config(input_size=config.input_size))
        elif config.prompt_encoder in ['point', 'full']:
            raise NotImplementedError(f'Waiting for implement...')
        else:
            raise ValueError(f"Unexpected vision encoder: {config.vision_encoder}")        

        self.language_encoder = AutoModel.from_pretrained(
            config.language_model_name_or_path
        )

        self.mm_vision_proj = nn.Linear(
            self.vision_encoder.channels[-1], config.hidden_size
        )
        self.mm_language_proj = nn.Linear(
            self.language_encoder.config.dim, config.hidden_size
        )
        self.mm_fuse_proj = nn.Linear(
            self.vision_encoder.channels[-1], config.hidden_size
        )

        self.efficient_loss = config.efficient_loss
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss
        self.loss_type = config.loss_type
        self.sim_loss = nn.KLDivLoss()
        self.loss_adjuster = nn.Parameter(torch.zeros(3))
        if self.loss_type == "sigmoid":
            self.t_prime = nn.Parameter(torch.tensor(config.t_prime))
            self.bias = nn.Parameter(torch.tensor(config.bias))
            self.t_prime_fuse = nn.Parameter(torch.tensor(config.t_prime))
            self.bias_fuse = nn.Parameter(torch.tensor(config.bias))
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * config.t_prime)

    def encode_image(self, image, masks=None, image_fg=None, return_dcformer=True) -> tuple[torch.Tensor, torch.Tensor]:
        fuse_feats = self.vision_encoder(image, masks)
        image_feats = self.vision_encoder.dcformer(image_fg)

        if isinstance(image_feats, list):
            image_feats = image_feats[-1]
        image_feats = image_feats.mean(dim=1)
        image_feats = self.mm_vision_proj(image_feats)
        image_feats = F.normalize(image_feats, dim=-1)

        if isinstance(fuse_feats, list):
            fuse_feats = fuse_feats[-1]
        fuse_feats = fuse_feats.mean(dim=1)
        fuse_feats = self.mm_fuse_proj(fuse_feats)
        fuse_feats = F.normalize(fuse_feats, dim=-1)
        

        return image_feats, fuse_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)[
            "last_hidden_state"
        ]
        text_feats = text_feats[:, 0]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def get_lambda(self, t):
        """
            return: (lambda for siglip_image, lambda for siglip_fuse, lambda for alig fuse and image)
        """
        lambda1, lambda2, lambda3 = self.config.init_lambda
        return lambda1, lambda2, lambda3

    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        masks = kwargs.pop('images')
        image_fg = kwargs.pop('image_fg')
        step = kwargs.pop('step')
        image_features, fuse_features = self.encode_image(images, masks=masks, image_fg=image_fg, return_dcformer=True)

        text_features = self.encode_text(input_ids, attention_mask)

        rank = 0
        world_size = 1
        if has_distributed and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        batch_size = image_features.size(0)
        device = image_features.device
        # if self.loss_type == "sigmoid":
        if has_distributed and dist.is_initialized():
            if self.efficient_loss:
                t = torch.exp(self.t_prime)
                fuse_t = torch.exp(self.t_prime_fuse)
                loss = 0.0

                for target_rank in range(world_size):
                    if rank == target_rank:
                        target_text_features = text_features
                    else:
                        target_text_features = torch.distributed.nn.broadcast(
                            text_features.requires_grad_(), target_rank
                        )
                    # Siglip with image and text
                    local_logits_per_image = (
                        image_features @ target_text_features.T
                    ) * t + self.bias
                    local_logits_per_text = local_logits_per_image.T

                    local_logits_per_fuse2text = (
                        fuse_features @ target_text_features.T
                    ) * fuse_t + self.bias_fuse
                    local_logits_per_text2fuse = local_logits_per_fuse2text.T

                    if rank == target_rank: # The diagonal pair should be same (=1),
                        local_labels = 2 * torch.eye(
                            batch_size, device=device
                        ) - torch.ones(batch_size, batch_size, device=device)
                    else:   # all of here should be not same (= -1)
                        local_labels = -torch.ones(
                            batch_size, batch_size, device=device
                        )

                    local_logits = (
                        local_logits_per_image + local_logits_per_text
                    ) / 2.0
                    local_loss = -torch.sum(
                        F.logsigmoid(local_labels * local_logits)
                    ) / (batch_size * world_size)

                    local_fuse_logits = (
                        local_logits_per_fuse2text + local_logits_per_text2fuse
                    ) / 2.0
                    local_fuse_loss = -torch.sum(
                        F.logsigmoid(local_labels * local_fuse_logits)
                    ) / (batch_size * world_size)

                    for cur_lambda, cur_loss in zip(self.get_lambda(step), [local_loss, local_fuse_loss, self.sim_loss(image_features, fuse_features)]):
                        loss += cur_lambda * cur_loss
                    # loss += local_loss + local_fuse_loss + self.sim_loss(image_features, fuse_features)

                torch.distributed.nn.all_reduce(loss)
                torch.cuda.synchronize()

                if self.training:
                    logits = 0
            else:
                t = torch.exp(self.t_prime)
                t_fuse = torch.exp(self.t_prime_fuse)
                all_image_features, all_fuse_features, all_text_features = gather_features(
                    image_features,
                    fuse_features,
                    text_features,
                    gather_with_grad=True,
                    rank=rank,
                    world_size=world_size,
                )

                logits_per_image = (
                    all_image_features @ all_text_features.T
                ) * t + self.bias
                logits_per_text = logits_per_image.T

                logits_per_fuse2text = (
                    all_fuse_features @ all_text_features.T
                ) * t_fuse + self.bias_fuse
                logits_per_text2fuse = logits_per_fuse2text.T

                batch_size = all_image_features.size(0)

                labels = 2 * torch.eye(
                    batch_size, device=image_features.device
                ) - torch.ones(batch_size, device=image_features.device)

                logits = (logits_per_image + logits_per_text) / 2.0
                logits_fuse = (logits_per_fuse2text + logits_per_text2fuse) / 2.0
                lambda_img, lambda_fuse, lambda_img_fuse = self.get_lambda(step)
                loss = (-torch.sum(F.logsigmoid(labels * logits)) / batch_size) * lambda_img
                loss += (-torch.sum(F.logismoid(labels * logits_fuse)) / batch_size) * lambda_fuse
                loss += self.sim_loss(image_features, fuse_features) * lambda_img_fuse

        else:
            logits_per_image = (
                image_features @ text_features.T
            ) * self.t_prime + self.bias
            logits_per_text = logits_per_image.T

            logits_per_fuse = (
                fuse_features @ text_features.T
            ) * torch.exp(self.t_prime_fuse) + self.biase_fuse

            labels = 2 * torch.eye(batch_size, device=device) - torch.ones(
                batch_size, batch_size, device=device
            )

            logits = (logits_per_image + logits_per_text) / 2.0
            fuse_logits = (logits_per_fuse + logits_per_fuse.T) / 2.0
            lambda_img, lambda_fuse, lambda_img_fuse = self.get_lambda(step)
            loss = (-torch.sum(F.logsigmoid(labels * logits)) / batch_size) * lambda_img
            loss += (-torch.sum(F.logismoid(labels * fuse_logits)) / batch_size) * lambda_fuse
            loss += self.sim_loss(image_features, fuse_features) * lambda_img_fuse
       

        ret = {
            "loss": loss,
            "logits": logits,
        }

        return ret


def gather_features(
    image_features,
    text_features,
    fuse_features,
    local_loss=False,
    gather_with_grad=True,
    rank=0,
    world_size=1,
):
    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."

    if not (has_distributed and dist.is_initialized()):
        return image_features, text_features

    if gather_with_grad:
        all_image_features = torch.cat(
            torch.distributed.nn.all_gather(image_features), dim=0
        )
        all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0
        )
        all_fuse_features = torch.cat(
            torch.distributed.nn.all_gather(fuse_features), dim=0
        )
    else:
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        gathered_fuse_features = [
            torch.zeros_like(fuse_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gathered_fuse_features, fuse_features)

        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
            gathered_fuse_features[rank] = fuse_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        all_fuse_features = torch.cat(gathered_fuse_features, dim=0)

    return all_image_features, all_text_features, all_fuse_features


AutoConfig.register("prompt_clip", PromptCLIPConfig)
AutoModel.register(PromptCLIPConfig, PromptCLIP)
