import random
import os
import json
import traceback as tb
import logging
from os.path import join
from typing import Literal
from copy import deepcopy as dcp
logger = logging.getLogger(__name__)

import numpy as np
import torch
import transformers as HFT
import monai.transforms as mtf
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import Dataset
from monai.transforms import allow_missing_keys_mode
from monai.data import set_track_meta, MetaTensor
from einops import rearrange

from src.dataset.utils import transforms as UT
from src.dataset.utils import myio as UIO
from src.dataset.prompt_templates import CardiacMap


def custom_scale(pack):
    np.clip(pack['image'], -395.0, 842.0, out=pack['image'])
    pack['image'] -= 279.8117370605469
    pack['image'] /= 253.5583953857422
    return pack


class CardiacCLIPDataset(Dataset):
    # def __init__(self, args, tokenizer, mode: Literal['train', 'val', 'test']="train", test_size=1000, contains_mask=False):


    def __init__(
            self, args, tokenizer: HFT.PreTrainedTokenizer,
            mode: Literal['train', 'val', 'test']="train", test_size=1000, p_value_batch_idx=None
    ):
        """
        Initializes the CardiacCLIPDataset with the given parameters.

        Args:
            args: A namespace containing various attributes required for dataset configuration.
                - data_root: The root directory for data storage.
                - ignore_split: A boolean flag to ignore the split of training and validation data.
                - max_length: The maximum length of captions.
                - loader_type: Use to decision how preprocessor to use.
            tokenizer: A tokenizer to process textual data.
            mode (Literal['train', 'val', 'test']): Mode of operation, determines data loading and transformations. Defaults to 'train'.
            test_size (int): Number of samples to limit the dataset to in test mode. Defaults to 1000.

        Attributes:
            args: Stores the input arguments.
            data_root: The root directory for data storage, derived from args.
            tokenizer: Stores the input tokenizer.
            mode: Stores the mode of operation.
            data_list: List of data entries loaded from JSON files based on the mode.
            loader: Composed data loading pipeline using MONAI transforms.
            transform: Data transformation pipeline for training or validation based on the mode.
        """
        self.args = args        
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_list = UIO.load_json(join(args.data_root, f'caption_{mode}_adding_mask.json'))
        
        if getattr(args, 'struct_ds', "") != "":
            self.desc_pool = UIO.load_json(args.struct_ds)

        if getattr(args, 'ignore_split', False) and mode == 'test':
            self.data_list.extend(UIO.load_json(join(args.data_root, f'caption_val.json')))
            self.data_list.extend(UIO.load_json(join(args.data_root, f'caption_train.json')))
        elif getattr(args, 'ignore_split', False) and mode != 'train':
            print(f'You setting `--ignore_split` to True, but mode is not train, so `--ignore_split` will be ignored.')

        print(json.dumps(vars(args), indent=2))
        loader_comp = UT.get_loader(args)
        self.loader = mtf.Compose(loader_comp)
        if getattr(args, 'loader_type').split('-')[1] == 'm3d':
            rot_axis = (1, 2)
        else:
            rot_axis = (0, 1)


        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(prob=0.5, spatial_axes=rot_axis, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=0, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=1, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=2, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandScaleIntensityd(factors=0.1, prob=0.5, keys=['image', 'image_fg'], allow_missing_keys=True),
                mtf.RandShiftIntensityd(offsets=0.1, prob=0.5, keys=['image', 'image_fg'], allow_missing_keys=True),
                mtf.EnsureTyped(device='cpu', keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.ToTensord(dtype=torch.float, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
            ]
        )

        val_transform = mtf.Compose([mtf.ToTensord(dtype=torch.float, keys=['image', 'label', 'image_fg'], allow_missing_keys=True)])
        
        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif 'test' in mode and not getattr(args, 'calculate_p_value', False):
            self.transform = val_transform
            tmp_data_list = list()
            cnt = 0
            for pack in self.data_list:
                _pack = dcp(pack) 
                _pack['label'] = random.sample(
                    list(set(_pack['mask_pool'])), 1
                )[0]
                _pack = UIO.load_make_sure_exists(_pack)
                if _pack is None or 'label' not in _pack:
                    cnt += 1
                    continue
                tmp_data_list.append(pack)
            print(f"Drop {cnt} data")
            self.data_list = tmp_data_list
            self.data_list = self.data_list[:test_size]
        elif mode == 'test' and getattr(args, 'calculate_p_value', False):  
            self.transform = val_transform          
            self.data_list = UIO.load_json(args.p_value_json)
            self.__p_value_batch = len(self.data_list)
            self.data_list = self.data_list[p_value_batch_idx]
            tmp_data_list = list()
            cnt = 0
            for pack in self.data_list:
                _pack = dcp(pack) 
                _pack['label'] = random.sample(
                    list(set(_pack['mask_pool'])), 1
                )[0]
                _pack = UIO.load_make_sure_exists(_pack)
                if _pack is None or 'label' not in _pack:
                    cnt += 1
                    continue
                tmp_data_list.append(pack)
            print(f"Drop {cnt} data")
            self.data_list = tmp_data_list

    @property
    def p_value_batch(self) -> int:
        return self.__p_value_batch

    def __len__(self):
        # if self.mode == 'train':
        #     return len(self.data_list) * 2
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        """
        Truncate a given text to a given number of tokens.

        This function splits the input text into sentences, and then randomly selects sentences until the desired number of tokens is reached.

        Args:
            input_text (str): The text to be truncated.
            max_tokens (int): The maximum number of tokens to be kept.

        Returns:
            str: The truncated text.
        """
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def loading_visual_data(self, pack):
        vpack = {
            'image': pack['image']
        }
        if 'label' in pack:                    
            vpack['label'] = pack['label']
        if 'organ' in pack:
            vpack['organ'] = pack['organ']
        vpack: dict[str, MetaTensor] = self.loader(vpack)
        vpack = self.transform(vpack)   # It must contains `image`, and possible `label`, `image_Fg`
        return vpack

    def building_desc(self, organ: int | list[int], org_data: dict[str, str | list[int]]) -> dict[str, str | list[int]]:
        if isinstance(organ, list):
            organ = organ[0]        
        organ = int(organ)
        org_data['organ'] = [organ]
        organ_name = CardiacMap[organ]
        context = self.desc_pool[organ_name]
        start_hint = random.sample(context["ContextSettingSentences"], 1)[0]
        desc_pool = context['DescriptionPhrases']
        random_desc = random.randint(5, len(desc_pool))
        desc_list = random.sample(desc_pool, random_desc)
        start_hint = f'{start_hint} {desc_list.pop(0)}. '
        caption_list = [start_hint] + desc_list
        org_data['raw_text'] = '. '.join(caption_list)
        return org_data

    def __getitem__(self, idx):     
        if idx >= len(self.data_list):
            idx = idx % len(self.data_list)               
        data = self.data_list[idx]
        data['label'] = random.sample(
            list(set(data['mask_pool'])), 1
        )[0]
        data: dict[str, str] = UIO.load_make_sure_exists(data)        

        if data is None or 'label' not in data:
            buf_pack = self.__getitem__(random.randint(0, len(self.data_list) - 1))
            if self.mode != 'train':
                buf_pack['image'] = None                                    
            return buf_pack        
        
        try:
            vpack: dict[str, torch.Tensor | MetaTensor] = self.loading_visual_data(data)
        except Exception as e:
            os.makedirs('./visual_error', exist_ok=True)
            with open('./visual_error/content.txt', 'a+') as writer:
                writer.write(tb.format_exc() + '\n')
                writer.write("="*30 + "\n")
            print(f'Error happen: {e.args}')
            tb.print_exc()
            return self.__getitem__(idx + 1)
        ds_src = getattr(self.args, 'dataset_src', 'caption')
        
        if ds_src == 'random':
            use_desc_prob = random.random()
        elif ds_src == 'caption':
            use_desc_prob = 0
        elif ds_src == 'desc':
            use_desc_prob = 1
        else:
            use_desc_prob = 0        
        is_desc = use_desc_prob > .5 and hasattr(self, 'desc_pool')
        logger.debug(f'use_desc_prob: {use_desc_prob:.1%} [>50%?] {is_desc}')

        if self.mode != 'train':
            raw_text = data["raw_text"]                
        elif self.mode == 'train' and not is_desc:
            raw_text = random.sample(data['caption'], 1)[0]['text']
        elif self.mode == 'train' and is_desc:
            all_organ = np.unique(nib.load(data['label']).get_fdata()).tolist()  # Remove background digit `0`
            organ = random.sample(list(set(all_organ) - {0, 9}), 1)[0]
            data = self.building_desc(organ, data)
            raw_text = data['raw_text']
        

        text = self.truncate_text(raw_text, self.args.max_length)
        debug_info = f"""
        Input Text: {text}
        =========================================
        """

        logger.debug(debug_info)
        text_tensor = self.tokenizer(
            text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        ret = {
            'image': vpack['image'],
            'text': text,
            'input_id': input_id,
            'attention_mask': attention_mask,
            'question_type': "Image_text_retrieval",
            'mask': vpack.get('label', torch.zeros_like(vpack['image']))
        }

        if 'image_fg' in vpack:
            ret['image_fg'] = vpack['image_fg']
        return ret
            


class TestCardiacCLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000, contains_mask=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_list = UIO.load_json(join(args.data_root, f'caption_{mode}.json'))
        if getattr(args, 'ignore_split', False):
            self.data_list.extend(UIO.load_json(join(args.data_root, f'caption_val.json')))
            self.data_list.extend(UIO.load_json(join(args.data_root, f'caption_train.json')))

        # self.json_file = load_json_list(args.cap_data_path)
        # self.data_list = self.json_file[mode]
        self.contains_mask: bool = contains_mask
        load_kwargs = dict(allow_missing_keys=True)
        # if contains_mask:
        load_kwargs['keys'] = ['image', 'label']
        
        if args.shape_mode == 'crop':
            spacing = (.39, .39, .625)
        elif args.shape_mode == 'resize':
            spacing = (.78, .78, 1.25)
        def _slicewise(pack, scaler):
            z = pack['image'].shape[-1]
            return torch.stack([scaler(pack['image'][..., idx]) for idx in range(z)], dim=-1)
        tojpeg = mtf.ScaleIntensity(0, 255, torch.int32)
        tonorm = mtf.ScaleIntensity()
        self.loader = mtf.Compose(UT.get_normal_loader(args, keys=['image', 'label']))


        train_transform = mtf.Compose(
            [
                # mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(prob=0.10, spatial_axis=0, **load_kwargs),
                mtf.RandFlipd(prob=0.10, spatial_axis=1, **load_kwargs),
                mtf.RandFlipd(prob=0.10, spatial_axis=2, **load_kwargs),
                mtf.RandScaleIntensityd(factors=0.1, prob=0.5, **load_kwargs),
                mtf.RandShiftIntensityd(offsets=0.1, prob=0.5, **load_kwargs),

                mtf.ToTensord(dtype=torch.float, **load_kwargs),
            ]
        )

        val_transform = mtf.Compose([mtf.ToTensord(dtype=torch.float, **load_kwargs)])
        
        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif 'test' in mode:
            self.transform = val_transform
            self.data_list = self.data_list[:test_size]

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                data = UIO.load_make_sure_exists(data)

                if data is None:
                    return self.__getitem__(random.randint(0, len(self.data_list) - 1))                                
                visual_pack = {
                    'image': data['image']
                }
                if 'label' in data:                    
                    visual_pack['label'] = data['label']                
                image: dict[str, MetaTensor] = self.loader(visual_pack)
                image = self.transform(image)
                if self.mode != 'train':
                    raw_text = data["raw_text"]                
                else:
                    raw_text = random.sample(data['caption'], 1)[0]['text']
                text = self.truncate_text(raw_text, self.args.max_length)
                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image['image'],
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                if self.contains_mask and 'label' not in image:
                    ret['mask'] = torch.zeros_like(ret['image'])
                elif self.contains_mask and 'label' in image:
                    ret['mask'] = image['label']


                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)    



class CLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000):
        if getattr(args, 'cap_data_path', None) is None:
            print("Setting M3D-caption needed ")
            args.cap_data_path = '/home/jovyan/shared/uc207pr4f57t9/cardiac/M3D/M3D-Cap/M3D_Cap/M3D_Cap_nii.json'
            args.data_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/M3D/M3D-Cap'
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        # set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif 'test' in mode:
            self.transform = val_transform
            self.data_list = self.data_list[:test_size]

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        """
            Make sure the number token of input_text is < @param max_tokens
        """
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')   # Cut down all of sentence

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_path = image_path.replace(".npy", '.nii.gz')
                image_abs_path = os.path.join(self.data_root, image_path)

                # image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                text = self.truncate_text(raw_text, self.args.max_length)

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                # I change model axis defination
                image = rearrange(image, 'C D H W -> C H W D')

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

