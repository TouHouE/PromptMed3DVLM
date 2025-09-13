import os
import logging
from typing import Literal
DEBUG: bool = os.environ.get("DEBUG", "0") == "1"
DLOSS: bool = os.environ.get("DLOSS", "0") == "1" or DEBUG
logger = logging.getLogger(__name__)

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
from src.model.encoder.utils import wrap_matched_keys
from src.model.encoder.vit import Vit3D
from src.model.encoder.prompt_dcformer import PromptDCFormerConfig, MaskPromptDCFormer
from src.model.encoder.prompt_m3dvit import MaskPromptM3DViT, MaskPromptM3DViTConfig
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
        language_model_name_or_path: str = "medicalai/ClinicalBERT",
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
        super().__init__(**kwargs)


class PromptCLIP(PreTrainedModel):
    config_class = PromptCLIPConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if 'dcformer' in config.vision_encoder:
            self.vision_encoder = MaskPromptDCFormer(
                PromptDCFormerConfig.small_config(input_size=config.input_size)
            )
        elif 'm3dvit' in config.vision_encoder:
            self.vision_encoder = MaskPromptM3DViT(
                MaskPromptM3DViTConfig(**wrap_matched_keys(config, MaskPromptM3DViTConfig))
            )   
        else:
            raise ImplementationError(f"Unexpected vision encoder: {config.vision_encoder}")
        logger.debug(f"\n{self.vision_encoder}")

        self.language_encoder = AutoModel.from_pretrained(
            config.language_model_name_or_path
        )

        self.mm_vision_proj = nn.Linear(
            self.vision_encoder.channels[-1], config.hidden_size
        )
        if hasattr(self.language_encoder.config, 'dim'):
            lang_in_features = self.language_encoder.config.dim
        else:
            lang_in_features = self.language_encoder.config.hidden_size
        self.mm_language_proj = nn.Linear(
            lang_in_features, config.hidden_size
        )
        self.mm_fuse_proj = nn.Linear(
            self.vision_encoder.channels[-1], config.hidden_size
        )

        self.efficient_loss = config.efficient_loss
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss
        self.loss_type = config.loss_type
        self.limit_loss_type = getattr(config, "limit_loss_type", 'huber')
        logger.debug(f'limit_loss_type: {self.limit_loss_type}')

        if self.limit_loss_type == 'huber':
            self.sim_loss = nn.HuberLoss(reduction='none')
        else:
            self.sim_loss = nn.KLDivLoss(reduction='none')
        self.loss_adjuster = nn.Parameter(torch.zeros(3))   # Waiting for dynamically adjust loss factor.

        if self.loss_type == "sigmoid":
            self.t_prime = nn.Parameter(torch.tensor(config.t_prime))
            self.bias = nn.Parameter(torch.tensor(config.bias))
            self.t_prime_fuse = nn.Parameter(torch.tensor(config.t_prime))
            self.bias_fuse = nn.Parameter(torch.tensor(config.bias))
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * config.t_prime)
        self.no_roi_sig_loss = getattr(config, 'no_roi_sig_loss', False)
        self.no_fuse_sig_loss = getattr(config, 'no_fuse_sig_loss', False)

    @torch.inference_mode()
    def infer_encode_image(self, image, masks, do_mask=True) -> torch.Tensor:
        """
            This method will only usage when doing inference.
        """


        if not do_mask:
            feats = getattr(self.vision_encoder, 'dcformer', self.vision_encoder.backbone)(image)
            if isinstance(feats, tuple) and not all(torch.is_tensor(_feat) for _feat in feats):
                feats = feats[1]                
            _mm_proj = self.mm_vision_proj
        else:
            feats = self.vision_encoder(image, masks=masks)
            _mm_proj = self.mm_fuse_proj
        if isinstance(feats, list):
            feats = feats[-1]
        feats = feats.mean(dim=1)
        feats = _mm_proj(feats)
        feats = F.normalize(feats, dim=-1)
        return feats


    def encode_image(
            self, image, masks=None, image_fg=None, sim_loss=False, do_mask=True, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if masks is None:
            masks = torch.zeros_like(image)
        if image_fg is None and not self.training:    # For inference, when only given `image` and masks
            return self.infer_encode_image(image, masks, do_mask)
        fuse_feats = self.vision_encoder(image, masks)
        if hasattr(self.vision_encoder, 'dcformer'):
            image_feats = self.vision_encoder.dcformer(image_fg)
        else:
            # last_feats and all stage feats
            _, image_feats = self.vision_encoder.backbone(image_fg)

        if self.limit_loss_type == 'kld':
            limit = self.sim_loss(torch.softmax(fuse_feats[-1], dim=-1), torch.softmax(image_feats[-1], dim=-1))
        elif self.limit_loss_type == 'huber':
            limit = self.sim_loss(fuse_feats[-1], image_feats[-1])
        
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
        if sim_loss and self.limit_loss_type != 'no':
            return image_feats, fuse_feats, limit
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
        lambda1, lambda2, lambda3 = [
            getattr(self.config, f'{k}_lambda', default_lambda) for k, default_lambda in zip(
                ['fg', 'fuse', 'limit'], [.2, 1, 1]
            ) 
        ]
        if self.no_roi_sig_loss:
            lambda1 = 0
            logging.debug(f'RoI branch SigLoss factor setting to 0, because `no_roi_sig_loss` is True')
        if self.no_fuse_sig_loss:
            lambda2 = 0
            logging.debug(f'Fuse branch SigLoss factor setting to 0, because `no_roi_fuse_loss` is True')
        if self.limit_loss_type == 'no':            
            lambda3 = 0
            logging.debug(f'Embedding limit loss factor setting to 0, because `limit_loss_type` is `no`')

        return lambda1, lambda2, lambda3

    def forward(self, images, input_ids, attention_mask, labels, masks, image_fgs, **kwargs):
        step = kwargs.pop('step')
        # Image_features: the augmentation image -> E_v(FCrop(x_i, x_p))
        # Fuse_features: E_v(x_i) + E_p(x_p)
        image_features, fuse_features, limit = self.encode_image(
            images, masks=masks, image_fg=image_fgs, return_dcformer=True, sim_loss=True
        )
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
                if DLOSS:
                    print(f'Into efficient_loss')
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
                    if DLOSS:
                        print(f'text is nan? {torch.isnan(target_text_features)}')
                    # Siglip with foreground image and text
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
                    # calculate Foreground Image vs Text -> The original L_Sig
                    local_logits = (
                        local_logits_per_image + local_logits_per_text
                    ) / 2.0
                    siglip_fg = -torch.sum(
                        F.logsigmoid(local_labels * local_logits)
                    ) / (batch_size * world_size)
                    
                    # Calculate Prompt Fusion Image vs Text -> The addigional L_Sig
                    local_fuse_logits = (
                        local_logits_per_fuse2text + local_logits_per_text2fuse
                    ) / 2.0
                    siglip_fuse = -torch.sum(
                        F.logsigmoid(local_labels * local_fuse_logits)
                    ) / (batch_size * world_size)

                    # limit = self.sim_loss(image_features, fuse_features)
                    if DLOSS:
                        print(f'SigLoss[Foreground, Text]: {siglip_fg}')
                        print(f'SigLoss[PromptFuse, Text]: {siglip_fuse}')
                        print(f'KLDiv[Foreground, PromptFuse]: {limit.mean()}')
                    lambda_img, lambda_fuse, lambda_img_fuse = self.get_lambda(step)

                    loss += siglip_fg * lambda_img + siglip_fuse * lambda_fuse + limit * lambda_img_fuse
                    # for cur_lambda, cur_loss in zip(self.get_lambda(step), [siglip_fg, siglip_fuse, limit]):
                    #     loss += cur_lambda * cur_loss
                    # loss += local_loss + local_fuse_loss + self.sim_loss(image_features, fuse_features)

                torch.distributed.nn.all_reduce(loss)
                
                torch.cuda.synchronize()

                if self.training:
                    logits = 0
            else:
                if DLOSS:
                    print(f'Not into Efficient_Loss')
                t = torch.exp(self.t_prime)
                t_fuse = torch.exp(self.t_prime_fuse)
                all_image_features, all_fuse_features, all_text_features, all_limit = gather_features(
                    image_features,
                    fuse_features,
                    text_features,
                    limit,
                    gather_with_grad=True,
                    rank=rank,
                    world_size=world_size,
                )

                if DLOSS:
                    print(f'Foreground Image is Nan? {torch.isnan(all_image_features).any()}')
                    print(f'Prompt Image is Nan? {torch.isnan(all_fuse_features).any()}')
                    print(f'Text is Nan? {torch.isnan(all_text_features).any()}')
                    
                
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
                
                siglip_fg = (-torch.sum(F.logsigmoid(labels * logits)) / batch_size)
                siglip_fuse = (-torch.sum(F.logsigmoid(labels * logits_fuse)) / batch_size)
                # limit = self.sim_loss(image_features, fuse_features)
                limit = all_limit.mean()
                loss = siglip_fg * lambda_img + siglip_fuse * lambda_fuse + limit * lambda_img_fuse
                
                if DLOSS:
                    print(f'SigLoss[Foreground, Text]: {siglip_fg}')
                    print(f'SigLoss[PromptFuse, Text]: {siglip_fuse}')
                    print(f'KLDiv[Foreground, PromptFuse]: {limit}')

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
            siglip_fg = (-torch.sum(F.logsigmoid(labels * logits)) / batch_size)
            siglip_fuse = (-torch.sum(F.logsigmoid(labels * fuse_logits)) / batch_size)
            limit = limit.mean()
            loss = siglip_fg * lambda_img + siglip_fuse * lambda_fuse + limit * lambda_img_fuse            
            
            if DLOSS:
                print(f'SigLoss[Foreground, Text]: {siglip_fg}')
                print(f'SigLoss[PromptFuse, Text]: {siglip_fuse}')
                print(f'KLDiv[Foreground, PromptFuse]: {limit}')
            # loss = (-torch.sum(F.logsigmoid(labels * logits)) / batch_size) * lambda_img
            # loss += (-torch.sum(F.logismoid(labels * fuse_logits)) / batch_size) * lambda_fuse
            # loss += self.sim_loss(image_features, fuse_features) * lambda_img_fuse
       

        ret = {
            "loss": loss,
            "logits": logits,
            "siglip_fg": siglip_fg,
            "siglip_fuse": siglip_fuse,
            "sim": limit,
            'lambda_fg': lambda_img,
            'lambda_fuse': lambda_fuse,
            'lambda_sim': lambda_img_fuse
        }

        return ret

    


def gather_features(
    image_features,
    text_features,
    fuse_features,
    limit,
    local_loss=False,
    gather_with_grad=True,
    rank=0,
    world_size=1,
):
    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."

    if not (has_distributed and dist.is_initialized()):
        return image_features, text_features, fuse_features, limit

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
        all_limit = torch.cat(
            torch.distributed.nn.all_gather(limit), dim=0
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
        grathed_limit = [
            torch.zeros_like(limit) for _ in range(world_size)
        ]
        
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gathered_fuse_features, fuse_features)
        dist.all_gather(gathered_limit, limit)
        
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
            gathered_fuse_features[rank] = fuse_features
            gathered_limit[rank] = limit
            
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        all_fuse_features = torch.cat(gathered_fuse_features, dim=0)
        all_limit = torch.cat(gathered_limit, dim=0)
    
    return all_image_features, all_text_features, all_fuse_features, all_limit


AutoConfig.register("prompt_clip", PromptCLIPConfig)
AutoModel.register(PromptCLIPConfig, PromptCLIP)


if __name__ == '__main__':
    lmn = 'medicalai/ClinicalBERT'
    cfg = PromptCLIPConfig(language_model_name_or_path=lmn)
    model = PromptCLIP(cfg)
    
