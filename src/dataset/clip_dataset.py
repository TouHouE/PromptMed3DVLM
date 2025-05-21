import random
import os
import json
import numpy as np
import torch
import monai.transforms as mtf

from torch.utils.data import Dataset
from monai.transforms import allow_missing_keys_mode
from monai.data import set_track_meta, MetaTensor
import SimpleITK as sitk



def load_json_list(path: str) -> list[dict]:
    with open(path, 'r') as jin:
        if path.endswith('.json'):
            return json.load(jin)
        elif path.endswith('.jsonl'):
            return [json.loads(line.strip('\n')) for line in jin.readlines()]


class CardiacCLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000, contains_mask=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.json_file = load_json_list(args.cap_data_path)
        self.data_list = self.json_file[mode]
        self.contains_mask: bool = contains_mask
        load_kwargs = dict(allow_missing_keys=True)
        if contains_mask:
            load_kwargs['keys'] = ['image', 'label']
        else:
            load_kwargs['keys'] = ['image']
        self.loader = mtf.Compose([
            mtf.LoadImaged(**load_kwargs),
            mtf.EnsureChannelFirstd(**load_kwargs),
            mtf.Orientationd(axcodes='RAS', **load_kwargs),
            mtf.Spacingd(**load_kwargs, pixdim=(.39, .39, -1), mode=('trilinear', 'nearest')),
            mtf.ScaleIntensityd(keys=['image']),
            mtf.ResizeWithPadOrCropd(**load_kwargs)
        ])

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

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float, **load_kwargs),
                ]
            )
        set_track_meta(False)

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
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)
                visual_pack = {
                    'image': image_abs_path
                }
                if 'label' in data:
                    label_abs_path = os.path.join(self.data_root, data['label'])
                    visual_pack['label'] = label_abs_path

                # image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                # image = sitk.ReadImage(image_abs_path)
                # image = sitk.GetArrayFromImage(image)
                # image = np.expand_dims(image, axis=0)
                image: dict[str, MetaTensor] = self.loader(visual_pack)
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
        set_track_meta(False)

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

