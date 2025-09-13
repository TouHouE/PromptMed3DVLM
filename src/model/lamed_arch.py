from abc import ABC, abstractmethod
from functools import partial

import torch
import torch.nn as nn

from .encoder.builder import build_vision_tower
from .projector.builder import build_mm_projector
# from .segmentation_module.builder import build_segmentation_module
from .task_head.segmentation_task.builder import build_segmentation_module
# from LaMed.src.model.loss import BCELoss, BinaryDiceLoss

def hgetattr(parent, attr_name, default_value, stdout=partial(print, end=''), prefix='', suffix='\n'):
    if hasattr(parent, attr_name):
        stdout(f"{prefix}No {attr_name} fine int {parent}, return `{default_value}`{suffix}")
        return default_value
    return getattr(parent, attr_name, default_value)

class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args, change_vision_tower: bool = False):
        # if update_config_from_model_args:
        #     self.config.vision_tower = model_args.vision_tower

        self.config.image_channel = hgetattr(model_args, "image_channel", 1)
        self.config.image_size = hgetattr(model_args, 'image_size', hgetattr(model_args, 'input_size', (32, 256, 256)))
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size
        no_vision_tower = self.get_vision_tower() is None

        # vision tower
        if no_vision_tower:                    
            print(f'We build_vision_tower')
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)
        elif change_vision_tower:
            self.vision_tower = build_vision_tower(model_args)
            vit_cfg = self.vision_tower.vision_tower.config
            for attr_name in vit_cfg.new_keys:
                setattr(self.config, attr_name, getattr(vit_cfg, attr_name))
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)



                
        if model_args.pretrain_vision_model is not None:            
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            # breakpoint()
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)



        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, masks=None, enable_mask_prompt=False):
        image_features = self.get_model().get_vision_tower()(images, masks=masks, no_prompt=~enable_mask_prompt)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, masks=None, **kwargs
    ):
        enable_mask_prompt: bool = False
        if kwargs.get('enable_mask_prompt') is not None:
            enable_mask_prompt = kwargs.pop('enable_mask_prompt')

        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_images(images, masks=masks, enable_mask_prompt=enable_mask_prompt)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        target_ = target.clone().float()
        target_[target == -1] = 0
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match\n" + str(predict.shape) + '\n' + str(target.shape[0])
        predict = predict.contiguous().view(predict.shape[0], -1)
        target_ = target_.contiguous().view(target_.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target_, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        # dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]
        dice_loss_avg = dice_loss.sum() / dice_loss.shape[0]

        return dice_loss_avg

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match\n' + str(predict.shape) + '\n' + str(target.shape)
        target_ = target.clone()
        target_[target == -1] = 0

        ce_loss = self.criterion(predict, target_.float())

        return ce_loss                