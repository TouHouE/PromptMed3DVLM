import re
import os
import json
import random
import logging
import math
import traceback as tb
from functools import partial, cache
from typing import Type, Literal, Final
from abc import ABC, abstractmethod
from overrides import override

import overrides

# logging.basicConfig(level=logging.DEBUG)
os.makedirs("./log", exist_ok=True)
# os.environ["HF_HOME"] = r"D:\huggingface"
DEBUG: bool = os.environ.get("DEBUG", "0") == "1"
DPACK: bool = os.environ.get("DPACK", '0') == '1'
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
from src.dataset.prompt_templates import Caption_templates, PosREC_templates, CardiacMap, Caption_style, NonVisData_Intros
from src.dataset.utils import myio as UIO
from src.dataset.utils import transforms as UT
from src.dataset.utils import text as UText


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
        self.to_tensor = mtf.Compose([
            mtf.ToTensord(keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
            mtf.EnsureTyped(keys=['image', 'label', 'image_fg'], allow_missing_keys=True, device='cpu')
        ])
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
        no_mask = True
        if 'organ' in cur_pack:            
            if len(cur_pack['organ']) > 0:
                ignore_mask = torch.prod(torch.stack([visual_pack['label'] == oid for oid in cur_pack['organ']]), dim=0)
                visual_pack['label'][ignore_mask == 1] = 0
                no_mask = False
        if no_mask:
            visual_pack['label'] = torch.zeros_like(visual_pack['label'])
        

        return_pack = {
            'image': visual_pack['image'],
            'mask': visual_pack['label'],
            'label': textual_pack['label'],
            'attention_mask': textual_pack['attention_mask'],
            'input_id': textual_pack['input_id'],
            'image_file': cur_pack['image'],
            'label_file': cur_pack.get('label', "NA")
        }
        if 'image_fg' in visual_pack:
            return_pack['image_fg'] = visual_pack['image_fg']
        return return_pack
    
    def __len__(self):
        if DEBUG:
            return 10
        if isinstance(self.usage_size, float) and self.usage_size < 1:
            return int(self.usage_size * len(self.data_list))
        if self.usage_size < 0:
            return len(self.data_list[:int(self.usage_size)])
        return int(self.usage_size)
    
    def print_actual_size(self):
        print(f'size: {len(self.data_list)}')


class VQACardiacDataset(CardiacDataset):
    image_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei'
    public_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei'

    def __init__(self, args, tokenizer, mode='train', usage_size=-1):
        super().__init__(args, tokenizer, mode)
        self.usage_size = usage_size
    @override
    def __make_datalist__(self):
        mode = self.mode
        all_pack = UIO.load_json(join(PUBLIC_PATH, f'gemini_split_{mode}_v2.json'))
        drop_num = 0

        for _, pack in enumerate(all_pack):
            abs_pack: CardiacData | None = UIO.load_make_sure_exists(pack)

            if abs_pack is None:
                drop_num += 1
                continue
            self.data_list.append(abs_pack)
    
    @override
    def load_textual_pack(self, pack, visual_pack=None) -> dict[Literal['input_id', 'attention_mask', 'label'], torch.Tensor]:
        convs = pack['conversations']
        history = list()
        image_is_insert = False

        for _pack in convs:        
            history.append({'role': ROLE_MAP[_pack['from']], 'content': _pack['value']})

        result_map: dict[Literal['input_ids', 'labels', 'attention_mask'], torch.Tensor]
        result_map = UText.preprocess(history, self.tokenizer, max_len=self.args.max_length, image_tokens=self.image_tokens, args=self.args)
        result_map['input_id'] = result_map.pop('input_ids')[0] # remove batch_size
        result_map['label'] = result_map.pop('labels')[0]
        result_map['attention_mask'] = result_map.pop('attention_mask')[0]

        return result_map

    # def __len__(self):
    #     if DEBUG:
    #         return 10
    #     if isinstance(self.usage_size, float) and self.usage_size < 1:
    #         return int(self.usage_size * len(self.data_list))
    #     if self.usage_size < 0:
    #         return len(self.data_list[:int(self.usage_size)])
    #     return int(self.usage_size)

        # return len(self.data_list[:self.usage_size])

class RGCardiacDataset(CardiacDataset):
    def __init__(self, args, tokenizer, mode='train', usage_size=-1, **kwargs):
        super().__init__(args, tokenizer, mode=mode, **kwargs)
        self.usage_size = usage_size     
    
    @override
    def __make_datalist__(self):
        mode = self.mode
        cap_ds = UIO.load_json(join(PUBLIC_PATH, f'caption_{mode}_adding_mask_v2.json'))

        for pack in cap_ds:
            mask = random.sample(list(set(pack['mask_pool'])), 1)[0]
            pack['label'] = mask
            updated_pack = UIO.load_make_sure_exists(pack)
            if updated_pack is None:
                continue
            self.data_list.append(updated_pack)

    @override
    def load_textual_pack(self, pack, visual_pack=None) -> dict[Literal['input_id', 'attention_mask', 'label'], torch.Tensor]:
        cur_cap = random.sample(pack['caption'], 1)[0]
        cap = cur_cap['text']
        style = cur_cap['style']
        if style == 'Original Report':
            style = 'CCTA checklist'
        query = random.sample(Caption_templates, 1)[0]
        style = random.sample(Caption_style, 1)[0].format(style.lower())    # Already contains "."
        query = f'{query} {style}'
        sep = cur_cap['sep']
        prob = random.random()
        if DEBUG:
            print(f'The random prob for drop non_vis_data: {prob}')
        if prob < .5 or len(cur_cap['non_vis_data']) == 0:
            cap = sep.join([_ctxt for _ctxt in cur_cap['rg_template'] if _ctxt != '[rep]'])
        else:
            non_vis_data = sep.join(cur_cap['non_vis_data'])
            nvis_data_list = cur_cap['non_vis_data']
            addition_info = random.sample(NonVisData_Intros, 1)[0].format(non_vis_data)
            query = f'{addition_info}{query}'
            cap = sep.join(nvis_data_list.pop(0) if _ctxt == '[rep]' else _ctxt for _ctxt in cur_cap['rg_template'])

        convs = [
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': cap}
        ]
        preprocessed_map: dict[Literal['input_ids', 'labels', 'attention_mask'], torch.Tensor]
        result_map: dict[Literal['input_id', 'label', 'attention_mask'], torch.Tensor] = dict()
        preprocessed_map = UText.preprocess(convs, self.tokenizer, max_len=self.args.max_length, image_tokens=self.image_tokens, args=self.args)
        if DEBUG and DPACK:
            UText.show_debug_pack(preprocessed_map, self.tokenizer)
        result_map['input_id'] = preprocessed_map.pop('input_ids')[0]  # remove batch_size
        result_map['label'] = preprocessed_map.pop('labels')[0]
        result_map['attention_mask'] = preprocessed_map.pop('attention_mask')[0]
        return result_map

    @override
    def __getitem__(self, item):
        if item >= (org_len := len(self.data_list)):
            item %= org_len
        return super().__getitem__(item)


class TemplateCardiacDataset(CardiacDataset):
    def __init__(self, args, tokenizer, mode='train', usage_size=-1, **kwargs):        
        super().__init__(args, tokenizer, mode, **kwargs)
        self.usage_size = usage_size

    @override
    def __make_datalist__(self):
        mode = self.mode
        all_pack = UIO.load_json(join(PUBLIC_PATH, f'gemini_split_{mode}.json'))
        unique_set = set()
        drop_num = 0

        for _, pack in enumerate(all_pack):
            abs_pack: CardiacData | None = UIO.load_make_sure_exists(pack)
            if abs_pack is None or 'label' not in abs_pack:
                drop_num += 1
                continue            

            unique_set.add((abs_pack['image'], abs_pack['label']))  # Make sure unique
        self.data_list.extend([{'image': image_label[0], 'label': image_label[1]} for image_label in unique_set])

    @override
    def load_textual_pack(self, pack, visual_pack=None) -> dict[Literal['input_id', 'attention_mask', 'label'], torch.Tensor]:
        mask: torch.Tensor = visual_pack['label']
        unique_organ: list[int] = torch.unique(mask).tolist()
        current_organ: int = random.sample(range(1, 11), 1)[0]
        organ_name: str = CardiacMap[current_organ]
        point_coords: torch.Tensor = torch.argwhere(mask == current_organ)  # N x 4
        loc = ""
        if point_coords.shape[0] > 1:
            p0: str = ', '.join(str((point_coords[:, i].min() / mask.shape[i]).cpu().numpy()) for i in range(1, 4))
            p1: str = ', '.join(str((point_coords[:, i].max() / mask.shape[i]).cpu().numpy()) for i in range(1, 4))
            loc = f'<|box_start|>{p0}, {p1}<|box_end|>'
        case = random.sample(['cls', 'des'], 1)[0]
        query = random.sample(PosREC_templates[f'{case}_questions'], 1)[0]
        answer_yes = random.sample(PosREC_templates[f'{case}_answers'], 1)[0]
        answer_no = random.sample(PosREC_templates[f'{case}_no_answers'], 1)[0]
        if organ_not_found := current_organ not in unique_organ:
            answer_args = (organ_name,)
        else:
            answer_args = (organ_name, loc) if case == 'des' else (loc,)
        organ_not_found = current_organ not in unique_organ
        query = query.format(organ_name)
        answer = answer_no.format(*answer_args) if organ_not_found else answer_yes.format(*answer_args)
        convs = [
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': answer},
        ]
        preprocessed_pack = UText.preprocess(convs, self.tokenizer, self.args.max_length, self.image_tokens, self.args)
        if DEBUG and DPACK:
            UText.show_debug_pack(preprocessed_pack, self.tokenizer)
        return_pack = {
            'input_id': preprocessed_pack.pop('input_ids')[0],
            'label': preprocessed_pack.pop('labels')[0],
            'attention_mask': preprocessed_pack.pop("attention_mask")[0],
            "organ": current_organ
        }
        return return_pack

    @override
    def __getitem__(self, index):
        if index >= (org_max := len(self.data_list)):
            index %= org_max
        pack = self.data_list[index]
        vloader_pack = {'image': pack['image'], 'label': pack['label']}
        vpack = self.load_visual_pack(vloader_pack)
        tpack = self.load_textual_pack(pack, vpack)
        if not isinstance(tpack['organ'], list):
            tpack['organ'] = [tpack['organ']]
        ignore_mask = torch.prod(torch.stack([vpack['label'] == oid for oid in tpack['organ']]), dim=0)
        vpack['label'][ignore_mask == 1] = 0

        return_pack = {
            'image': vpack['image'],
            'mask': vpack['label'],
            'input_id': tpack['input_id'],
            'label': tpack['label'],
            'attention_mask': tpack['attention_mask'],
            'image_file': pack['image'],
            'label_file': pack.get('label', 'NA')
        }

        if 'image_fg' in vpack:
            return_pack['image_fg'] = vpack['image_fg']
        return return_pack


class Stage_0_1_Dataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):        
        rg_lambda =  getattr(args, 'rg_lambda', 5000)
        temp_lambda = getattr(args, 'temp_lambda', 5000)
            
        self.dataset = ConcatDataset([
            RGCardiacDataset(args, tokenizer, mode=mode, usage_size=rg_lambda),
            TemplateCardiacDataset(args, tokenizer, mode=mode, usage_size=temp_lambda),
        ])
    def __getitem__(self, item):
        return self.dataset[item]
    
    @cache
    def __len__(self):
        return len(self.dataset)

class Stage2Dataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        self.args = args
        vqa_size = getattr(args, 'vqa_size', .75)
        caption_size = getattr(args, 'caption_size', 2500)
        template_size = getattr(args, 'template_size', 2500)        
        ds_list = [
            VQACardiacDataset(args, tokenizer, mode, usage_size=vqa_size),
            RGCardiacDataset(args, tokenizer, mode, usage_size=caption_size),
            TemplateCardiacDataset(args, tokenizer, mode, usage_size=template_size)
        ]

        print("Each Dataset size:")
        
        for ds_name, ds in zip(['vqa', 'caption', 'template'], ds_list):
            print(f' - {ds_name:10}: {len(ds)}|{ds.print_actual_size()}')

        self.dataset = ConcatDataset(ds_list)

    def __getitem__(self, item):
        return self.dataset[item]

    @cache
    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    from argparse import Namespace
    import transformers as HFT
    DEBUG=False
    tokenizer_ = HFT.AutoTokenizer.from_pretrained('MagicXin/Med3DVLM-Qwen-2.5-7B')
    tokenizer_.add_tokens("<|nvis_data_sep|>")
    ds = Stage2Dataset(
        Namespace(proj_out_num=256, max_length=768, loader_type='unet-med3d-resize', input_size=(256, 256 ,128)), tokenizer=tokenizer_, mode='train'
    )
    exit()
    for pack_ in iter(ds):
        for k, v in pack_.items():
            if torch.is_tensor(v):
                print(f'Key: {k} | Shape: {v.shape}')
                continue
            print(f'Key: {k} | Value: {v}')
        input_ = tokenizer_.batch_decode(pack_['input_id'][pack_['input_id'] > 0][None])
        labels = tokenizer_.batch_decode(pack_['label'][pack_['label'] > 0][None])
        print("="*15 + "Text Input"+"="*15)
        print(input_[0])
        print("=" * 15 + "Text Label" + "="*15)
        print(labels[0])
        print("="*30)