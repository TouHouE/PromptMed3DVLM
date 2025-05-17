import re
import os
import json
import random
import logging
from functools import partial
# logging.basicConfig(level=logging.DEBUG)
os.makedirs("./log", exist_ok=True)
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
import SimpleITK as sitk
import pandas as pd
from monai.data import set_track_meta
from torch.utils.data import Dataset, ConcatDataset
from src.dataset.prompt_templates import Caption_templates


PAD_EOS_SWAP_TMP_TOKEN = -100

# A debugging usage method
def return_print(data, stage=None):
    if stage is not None:
        print(f'\nStart {stage}')
    if isinstance(data, dict):
        for key, value in data.items():
            if torch.is_tensor(value):
                print(f'{key}:: {value.shape}')
            else:
                print(f'{key}:: {value}')
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            if torch.is_tensor(value):
                print(f'{idx}-th:: {value.shape}')
            else:
                print(f'{idx}-th:: {value}')
    elif torch.is_tensor(data):
        print(f'{data.shape}')
    else:
        print(f'{data}')
    if stage is not None:
        print(f'End of {stage}')

    return data


def load_make_sure_exists(pack):
    public_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei'
    possible_mid_path = [
        'to_saturn',    # for Taipei_502,
        'to_saturn_yeh',
        'to_saturn_beato'
    ]
    for mid_path in possible_mid_path:
        if os.path.exists(join(public_root, mid_path, pack['image'])):
            pack['image'] = join(public_root, mid_path, pack['image'])

            if pack.get('label', None) is not None:
                pack['label'] = join(public_root, mid_path, pack['label'])
            return pack
    return None


class CardiacDataset(Dataset):
    image_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei'

    def __init__(self, args, tokenizer, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = '<im_patch>' * args.proj_out_num
        self.data_list = list()
        with open(f'/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei/gemini_split_{mode}.json', 'r', encoding='utf-8') as reader:
            all_pack = json.load(reader)
        for pack in all_pack:
            abs_pack = load_make_sure_exists(pack)
            query, answer = abs_pack['conversations']
            if query['value'] is None or answer['value'] is None:
                continue
            if len(query['value'].lower().replace('none', '').replace('<image>', '').strip()) == 0:
                continue
            if len(answer['value'].lower().replace('none', '').replace('<image>', '').strip()) == 0:
                continue

            self.data_list.append(abs_pack)

        # with open('/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei/taipei_502_vqa.jsonl', 'r') as reader:
        #     for pack in reader.readlines():
        #         pack = json.loads(pack)
        #         pack['image'] = join(self.image_root, 'to_saturn', pack['image'])
        #         if 'label' in pack:
        #             pack['label'] = join(self.image_root, 'to_saturn', pack['label'])
        #
        #         self.data_list.append(pack)
        # with open('/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei/taipei_2897_yeh_conv.jsonl', 'r') as reader:

        self.image_loader = mtf.Compose([
            mtf.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
            mtf.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
            mtf.Orientationd(keys=['image', 'label'], axcodes="RAS", allow_missing_keys=True),
            # mtf.Lambda(lambda pack: return_print(pack, 'After Orientation')),
            mtf.Spacingd(keys=['image', 'label'], pixdim=(.4, .4, -1), mode=('trilinear', 'nearest'),
                         allow_missing_keys=True),            
            mtf.ScaleIntensityd(keys=['image', 'label'], allow_missing_keys=True),
            mtf.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=(256, 256, 128), allow_missing_keys=True),
            # mtf.Lambda(lambda pack: return_print(pack, 'After Resize')),
            # mtf.Lambda(lambda pack: {key: value if value.shape[2] == 256 else value.permute(0, 1, 3, 2) for key, value in pack.items()}),
            # mtf.Lambda(lambda pack: return_print(pack, 'After Custom permute')),
            # mtf.Orientationd("SRA"),
            # Random Shit
            # mtf.RandRotate90d(prob=0.5, spatial_axes=(1, 2), keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandFlipd(prob=0.10, spatial_axis=0, keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandFlipd(prob=0.10, spatial_axis=1, keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandFlipd(prob=0.10, spatial_axis=2, keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandScaleIntensityd(factors=0.1, prob=0.5, keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandShiftIntensityd(offsets=0.1, prob=0.5, keys=['image', 'label'], allow_missing_keys=True),
            mtf.ToTensord(dtype=torch.float, keys=['image', 'label'], allow_missing_keys=True),
            # mtf.Lambda(lambda pack: return_print(pack, 'After ToTensor'))
        ])

    def __getitem__(self, idx):
        # print(f'Start Loading {idx}')
        cur_pack = self.data_list[idx]
        # cur_pack = check_image_and_download(cur_pack)
        if cur_pack is None:
            return self.__getitem__(idx + 1)
        conv = cur_pack['conversations']
        query = list(filter(lambda conv_case: conv_case['from'] == 'human', conv))[0]['value']
        if re.fullmatch(r'<image>\n.*', query) is not None:
            replace_key = '<image>\n'
        else:
            replace_key = '\n<image>'
        query = query.replace(replace_key, '')
        answer = list(filter(lambda conv_case: conv_case['from'] == 'gpt', conv))[0]['value']
        if query is None or answer is None:
            return self.__getitem__(idx + 1)

        question = self.image_tokens + query

        logger.info(f'question: {query}, answer: {answer}')
        text_tensor = self.tokenizer(
            question + " " + answer,
            max_length=self.args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]
        valid_len = torch.sum(attention_mask)
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        question_tensor = self.tokenizer(
            question,
            max_length=self.args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        question_len = torch.sum(question_tensor["attention_mask"][0])

        label = input_id.clone()
        label[:question_len] = PAD_EOS_SWAP_TMP_TOKEN
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            label[label == self.tokenizer.pad_token_id] = PAD_EOS_SWAP_TMP_TOKEN
            if valid_len < len(label):
                label[valid_len] = self.tokenizer.eos_token_id
        else:
            label[label == self.tokenizer.pad_token_id] = PAD_EOS_SWAP_TMP_TOKEN

        loader_pack = {
            'image': cur_pack['image']
        }
        if cur_pack.get('label') is not None:
            loader_pack['label'] = cur_pack['label']
        logging.debug(f'Apply to loader:\n{json.dumps(loader_pack, indent=2)}')
        try:
            visual_pack = self.image_loader(loader_pack)
        except Exception as e:
            return self.__getitem__(idx + 1)
        # print(f'image.shape:{visual["image"].shape}||{cur_pack["image"]}')

        if visual_pack.get('label') is None:            
            visual_pack['label'] = torch.zeros_like(visual_pack['image'])
        # print(f'{idx} is Done')
        # image = self.image_loader(join(self.image_root, cur_pack['image']))
        
        

        return {
            "image": visual_pack['image'],
            'mask': visual_pack['label'],
            "input_id": input_id,
            "label": label,
            "attention_mask": attention_mask,
            "question": question,
            "answer": answer,
            'image_file': cur_pack['image'],
            'label_file': cur_pack.get('label', 'None')
        }

    def __len__(self):
        return len(self.data_list)



class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, "r") as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates

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

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform
            self.data_list = self.data_list[:test_size]

    def __len__(self):
        return len(self.data_list)

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
                with open(text_abs_path, "r") as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }
                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

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

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                # image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image)

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(
                        data["Choice A"],
                        data["Choice B"],
                        data["Choice C"],
                        data["Choice D"],
                    )
                    question = question + " " + choices
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }

                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

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

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                # image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image)

                question = data["Question"]
                answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }
                if self.args.seg_enable:
                    ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextYNDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextYNDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            VQAYNDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == '__main__':
    import transformers as HFT
    from argparse import Namespace
    _tokenizer = HFT.AutoTokenizer.from_pretrained('/home/jovyan/workspace/Med3DVLM/models/VLM')
    _args = Namespace(proj_out_num=256, max_length=2048)
    ds = CardiacDataset(_args, _tokenizer)
    for pack in ds:
        pass