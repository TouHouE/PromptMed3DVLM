import re
import os
import json
import random
import logging
import traceback as tb
from functools import partial
from typing import Type, Literal, Final
from abc import ABC, abstractmethod
from overrides import override

import overrides

# logging.basicConfig(level=logging.DEBUG)
os.makedirs("./log", exist_ok=True)
os.environ["HF_HOME"] = r"D:\huggingface"
DEBUG: bool = os.environ.get("DEBUG", "0") == "1"
logger = logging.getLogger(__name__)
log_fmt = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_log = logging.FileHandler('./log/dataset.log', 'w+')
file_log.setLevel(logging.DEBUG)
file_log.setFormatter(log_fmt)
logger.addHandler(file_log)
from os.path import join

import torch
import numpy as np
import monai.transforms as mtf
import transformers as HFT
import SimpleITK as sitk
import pandas as pd
from monai.data import set_track_meta, MetaTensor
from torch.utils.data import Dataset, ConcatDataset
from src.dataset.prompt_templates import Caption_templates, PosREC_templates, CardiacMap
from utils import myio as UIO
from utils import transforms as UT
from utils import text as UText


# Start Define Custom TypeHint
UserType = Literal['human', 'user']
BotType = Literal['gpt', 'assistant']
RoleType = Literal['system', UserType, BotType]
MessageKeyType = Literal['role', 'content']
SpeekType = dict[MessageKeyType, RoleType | str]
ConversationType = list[SpeekType]
CardiacDataKey = Literal['image', 'label', 'conversations']
CardiacData = dict[CardiacDataKey, str | ConversationType]
TorchCardiacDataKey = Literal['image', 'label', 'input_id', 'attention_mask', 'mask', 'image_file', 'mask_file']
TorchCardiacData = dict[TorchCardiacDataKey, torch.Tensor | str | MetaTensor]
# End Define Custom TypeHint


ROLE_MAP = {'gpt': 'assistant', 'human': 'user'}
PAD_EOS_SWAP_TMP_TOKEN = -100
PUBLIC_PATH: Final[str] = '/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei'

def get_prompt() -> str:
    return r"""If the input includes CT scans from at least two distinct cardiac phases, you can proceed with the requested calculation."""

class CardiacDataset(Dataset, ABC):
    def __init__(self, args, tokenizer: HFT.PreTrainedTokenizer, mode='train', ds_size=-1):
        self.args = args
        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_list = list()
        stem = UT.get_loader(args)
        self.image_loader = mtf.Compose(stem)
        self.to_tensor = mtf.ToTensord(keys=['image', 'label', 'image_fg'], allow_missing_keys=True)
        if mode == 'train':
            self.transform = mtf.Compose([
                mtf.RandRotate90d(prob=0.5, spatial_axes=(0, 1), keys=['image', 'label', 'image_fg'],
                                  allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=0, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=1, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=2, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandScaleIntensityd(factors=0.1, prob=0.5, keys=['image', 'image_fg'], allow_missing_keys=True),
                mtf.RandShiftIntensityd(offsets=0.1, prob=0.5, keys=['image', 'image_fg'], allow_missing_keys=True),
            ])
        else:
            self.transform = mtf.Identity()

        self.__make_datalist__()
        random.shuffle(self.data_list)
        self.data_list = self.data_list[:ds_size]

    @abstractmethod
    def __make_datalist__(self) -> None:
        ...

    @abstractmethod
    def load_textual_pack(self, pack: dict, visual_pack=None) -> dict[Literal['input_id', 'attention_mask', 'label'], torch.Tensor]:
        ...
    
    def load_visual_pack(self, loader_pack, do_transform=True):
        visual_pack = self.image_loader(loader_pack)
        if do_transform:
            visual_pack = self.transform(visual_pack)
        visual_pack = self.to_tensor(visual_pack)
        if visual_pack.get('label') is None and visual_pack.get('image') is not None:
            visual_pack['label'] = torch.zeros_like(visual_pack['image'])
        elif visual_pack.get('label') is None and visual_pack.get('image') is None:
            visual_pack['label'] = None
        return visual_pack

    def __getitem__(self, item):
        cur_pack = self.data_list[item]
        textual_pack = self.load_textual_pack(cur_pack)
        loader_pack = {'image': cur_pack['image']}
        if cur_pack.get('label') is not None:
            loader_pack['label'] = cur_pack['label']
        logging.debug(f'Apply to loader:\n{json.dumps(loader_pack, indent=2)}')
        visual_pack = self.load_visual_pack(loader_pack)
        return_pack = {
            'image': visual_pack['image'],
            'mask': visual_pack['label'],
            'label': textual_pack['label'],
            'attention_mask': textual_pack['attention_mask'],
            'input_id': textual_pack['input_id'],            
        }
        if 'image_fg' in visual_pack:
            return_pack['image_fg'] = visual_pack['image_fg']
        return return_pack
        
    
class VQACardiacDataset(CardiacDataset):
    image_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei'
    public_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei'

    def __init__(self, args, tokenizer, mode='train'):
        super().__init__(args, tokenizer, mode)

    @override
    def __make_datalist__(self):
        mode = self.mode
        all_pack = UIO.load_json(join(PUBLIC_PATH, f'gemini_split_{mode}.json'))
        drop_num = 0

        for _, pack in enumerate(all_pack):
            abs_pack: CardiacData | None = UIO.load_make_sure_exists(pack)

            if abs_pack is None:
                drop_num += 1
                continue
            self.data_list.append(abs_pack)
    
    @override
    def load_textual_pack(self, pack, visual_pack=None):
        convs = pack['conversations']
        history = list()
        image_is_insert = False

        for _pack in convs:
            if _pack['from'] == 'human' and not image_is_insert:
                history.append({'role': 'user', 'content': f'{self.image_tokens}\n{_pack["value"]}'})
                image_is_insert = True
                continue
            history.append({'role': ROLE_MAP[_pack['from']], 'content': _pack['value']})

        result_map: dict[Literal['input_ids', 'labels', 'attention_mask'], torch.Tensor]
        result_map = UText.preprocess(history, self.tokenizer, max_len=self.args.max_length, prompt=get_prompt())
        result_map['input_id'] = result_map.pop('input_ids')[0] # remove batch_size
        result_map['label'] = result_map.pop('labels')[0]
        result_map['attention_mask'] = result_map.pop('attention_mask')[0]

        return result_map
        
    def __len__(self):
        return len(self.data_list)


class RGCardiacDataset(CardiacDataset):
    def __init__(self, args, tokenizer, mode='train', **kwargs):
        super().__init__(args, tokenizer, mode=mode, **kwargs)
    
    @override
    def __make_datalist__(self):
        mode = self.mode
        cap_ds = UIO.load_json(join(PUBLIC_PATH, f'caption_{mode}_with_mask.json'))

        for pack in cap_ds:
            mask = random.sample(list(set(pack['mask_pool'])), 1)[0]
            pack['label'] = mask
            updated_pack = UIO.load_make_sure_exists(pack)
            if updated_pack is None:
                continue
            self.data_list.append(updated_pack)

    @override
    def load_textual_pack(self, pack, visual_pack=None):
        cur_cap = random.sample(pack['caption'], 1)[0]
        cap = cur_cap['text']
        style = cur_cap['style']
        query = random.sample(Caption_templates, 1)[0]
        query = f'{query} Generating style with {style}'
        convs = [
            {'role': 'user', 'content': f'{self.image_tokens}\n{query}'},
            {'role': 'assistant', 'content': {cap}}
        ]
        preprocessed_map: dict[Literal['input_ids', 'labels', 'attention_mask'], torch.Tensor]
        result_map: dict[Literal['input_id', 'label', 'attention_mask'], torch.Tensor] = dict()
        preprocessed_map = UText.preprocess(convs, self.tokenizer, max_len=self.args.max_length, prompt=get_prompt())
        result_map['input_id'] = preprocessed_map.pop('input_ids')[0]  # remove batch_size
        result_map['label'] = preprocessed_map.pop('labels')[0]
        result_map['attention_mask'] = preprocessed_map.pop('attention_mask')[0]
        return result_map


class TemplateCardiacDataset(CardiacDataset):
    def __init__(self, args, tokenizer, mode='train', **kwargs):
        super().__init__(args, tokenizer, mode, **kwargs)

    @override
    def __make_datalist__(self):
        mode = self.mode
        all_pack = UIO.load_json(PUBLIC_PATH, f'gemini_split_{mode}.json')
        unique_set = set()
        drop_num = 0

        for _, pack in enumerate(all_pack):
            abs_pack: CardiacData | None = UIO.load_make_sure_exists(pack)
            if abs_pack is None:
                drop_num += 1
                continue
            unique_set.add({"image": abs_pack['image'], 'label': abs_pack['label']})
        self.data_list.extend(list(unique_set))

    @override
    def load_textual_pack(self, pack, visual_pack=None):
        mask: torch.Tensor = visual_pack['label']
        unique_organ: list[int] = torch.unique(mask).tolist()
        current_organ: int = random.sample(range(1, 11), 1)[0]
        organ_name: str = CardiacMap[current_organ]
        point_coords: torch.Tensor = torch.argwhere(mask == current_organ)  # N x 4
        loc = ""
        if point_coords.shape[0] > 1:
            p0: str = ', '.join(str((point_coords[:, i].min() / mask.shape[i]).numpy()) for i in range(1, 4))
            p1: str = ', '.join(str((point_coords[:, i].max() / mask.shape[i]).numpy()) for i in range(1, 4))
            loc = f'<box_start>{p0}, {p1}<box_end>'
        case = random.sample(['cls', 'des'], 1)[0]
        query = random.sample(PosREC_templates[f'{case}_questions'], 1)[0]
        answer_yes = random.sample(PosREC_templates[f'{case}_answers'], 1)[0]
        answer_no = random.sample(PosREC_templates[f'{case}_no_questions'], 1)[0]
        if organ_not_found := current_organ not in unique_organ:
            answer_args = (organ_name,)
        else:
            answer_args = (organ_name, loc) if case == 'des' else (loc,)
        organ_not_found = current_organ not in unique_organ
        query = query.format(organ_name)
        query = f'{self.image_tokens}\n{query}'
        answer = answer_no.format(*answer_args) if organ_not_found else answer_yes.format(*answer_args)
        convs = [
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': answer},
        ]
        preprocessed_pack = UText.preprocess(convs, self.tokenizer, self.args.max_length, get_prompt())
        return_pack = {
            'input_id': preprocessed_pack.pop('input_ids')[0],
            'label': preprocessed_pack.pop('labels')[0],
            'attention_mask': preprocessed_pack.pop("attention_mask")[0],
        }
        return return_pack

    @override
    def __getitem__(self, index):
        pack = self.data_list[index]
        vloader_pack = {'image': pack['image'], 'label': pack['label']}
        vpack = self.load_visual_pack(vloader_pack)
        tpack = self.load_textual_pack(pack, vpack)
        return_pack = {
            'image': vpack['image'],
            'mask': vpack['label'],
            'input_id': tpack['input_id'],
            'label': tpack['label'],
            'attention_mask': tpack['attention_mask'],
        }

        if 'image_fg' in vpack:
            return_pack['image_fg'] = vpack['image_fg']
        return return_pack


class Stage_0_1_Dataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.dataset = ConcatDataset([
            RGCardiacDataset(args, tokenizer, mode=mode),
            TemplateCardiacDataset(args, tokenizer, mode=mode),
        ])
    def __getitem__(self, item):
        return self.dataset[item]
    def __len__(self):
        return len(self.dataset)


class Stage2Dataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        ds_list = [
            VQACardiacDataset(args, tokenizer, mode),
            RGCardiacDataset(args, tokenizer, mode),
            TemplateCardiacDataset(args, tokenizer, mode)
        ]
        self.dataset = ConcatDataset(ds_list)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    from argparse import Namespace
    import transformers as HFT
    DEBUG=True
    tokenizer_ = HFT.AutoTokenizer.from_pretrained('MagicXin/Med3DVLM-Qwen-2.5-7B')
    ds = Stage_0_1_Dataset(
        Namespace(proj_out_num=16, max_length=128, loader_type='unet-med3d-resize', input_size=(256, 256 ,128)), tokenizer=tokenizer_, mode='train'
    )
    for pack_ in iter(ds):
        for k, v in pack_.items():
            if torch.is_tensor(v):
                print(f'Key: {k} | Shape: {v.shape}')
                continue
            print(f'Key: {k} | Value: {v}')