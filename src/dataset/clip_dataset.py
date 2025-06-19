import random
import os
import json
import traceback as tb
from os.path import join

import numpy as np
import torch
import monai.transforms as mtf
import SimpleITK as sitk
from torch.utils.data import Dataset
from monai.transforms import allow_missing_keys_mode
from monai.data import set_track_meta, MetaTensor

from src.dataset.utils import transforms as UT
from src.dataset.utils import io as UIO


def custom_scale(pack):
    np.clip(pack['image'], -395.0, 842.0, out=pack['image'])
    pack['image'] -= 279.8117370605469
    pack['image'] /= 253.5583953857422
    return pack


class CardiacCLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000, contains_mask=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_list = UIO.load_json(join(args.data_root, f'caption_{mode}_adding_mask.json'))
        
        if getattr(args, 'ignore_split', False) and mode == 'test':
            self.data_list.extend(UIO.load_json(join(args.data_root, f'caption_val.json')))
            self.data_list.extend(UIO.load_json(join(args.data_root, f'caption_train.json')))
        elif getattr(args, 'ignore_split', False) and mode != 'train':
            print(f'You setting `--ignore_split` to True, but mode is not train, so `--ignore_split` will be ignored.')

        self.contains_mask: bool = contains_mask
        load_kwargs = dict(allow_missing_keys=True)
        load_kwargs['keys'] = ['image', 'label']
        print(json.dumps(vars(args), indent=2))
        loader_comp = UT.get_loader(args)
        self.image_checker = mtf.Compose([
            mtf.LoadImaged(keys=['image', 'label'], allow_missing_keys=True, image_only=True),
            mtf.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
            mtf.Orientationd(axcodes="RAS", keys=['image', 'label'], allow_missing_keys=True),
            mtf.Spacingd(keys=['image', 'label'], pixdim=(.39, .39, .625), mode=('trilinear', 'nearest'), allow_missing_keys=True)
        ])
        self.loader = mtf.Compose(loader_comp)
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(prob=0.5, spatial_axes=(0, 1), keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=0, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=1, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandFlipd(prob=0.10, spatial_axis=2, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.RandScaleIntensityd(factors=0.1, prob=0.5, keys=['image', 'image_fg'], allow_missing_keys=True),
                mtf.RandShiftIntensityd(offsets=0.1, prob=0.5, keys=['image', 'image_fg'], allow_missing_keys=True),
                mtf.EnsureTyped(device='cpu', keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
                mtf.ToTensord(dtype=torch.float, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
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

    def __getitem__(self, idx):                    
        data = self.data_list[idx]
        data['label'] = random.sample(
            list(set(data['mask_pool'])), 1
        )[0]
        data: dict[str, str] = UIO.load_make_sure_exists(data)

        if data is None:
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
            return self.__getitem__(idx + 1)

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
            'image': vpack['image'].cpu(),
            'text': text,
            'input_id': input_id,
            'attention_mask': attention_mask,
            'question_type': "Image_text_retrieval",
            'mask': vpack.get('label', torch.zeros_like(vpack['image'])).cpu()
        }

        if 'image_fg' in vpack:
            ret['image_fg'] = vpack['image_fg'].cpu()
        return ret
            


class EXPCardiacCLIPDataset(Dataset):
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

