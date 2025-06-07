import re
import os
import json
import random
import logging
import traceback as tb
from functools import partial
from typing import Type, Literal
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
from monai.data import set_track_meta, MetaTensor
from torch.utils.data import Dataset, ConcatDataset
from src.dataset.prompt_templates import Caption_templates


# Start Define Custom TypeHint
UserType = Literal['human', 'user']
BotType = Literal['gpt', 'assistant']
RoleType = Literal['system', UserType, BotType]
MessageKeyType = Literal['role', 'value']
SpeekType = dict[MessageKeyType, RoleType | str]
ConversationType = list[SpeekType]
CardiacDataKey = Literal['image', 'label', 'conversations']
CardiacData = dict[CardiacDataKey, str | ConversationType]
TorchCardiacDataKey = Literal['image', 'label', 'input_id', 'attention_mask', 'mask', 'image_file', 'mask_file']
TorchCardiacData = dict[TorchCardiacDataKey, torch.Tensor | str | MetaTensor]
# End Define Custom TypeHint



PAD_EOS_SWAP_TMP_TOKEN = -100


def load_jfile(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as loader:
        if path.endswith('.jsonl'):
            return [json.loads(line.strip('\n')) for line in loader.readlines()]    
        return json.load(loader)


def get_prompt() -> str:
    return r"""If the input includes CT scans from at least two distinct cardiac phases, you can proceed with the requested calculation."""



# A debugging usage method
def return_print(data, stage=None) -> any:
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
    possible_root = ['/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei', '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei']
    # public_root = ''/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei''
    possible_mid_path = [
        'to_saturn',    # for Taipei_502,
        'to_saturn_yeh',
        'to_saturn_beato'
    ]
    for public_root in possible_root:
        for mid_path in possible_mid_path:
            cur_abs_path = join(public_root, mid_path, pack['image'])
            # print(f'Cur_abs_path: {cur_abs_path}')
            if os.path.exists(cur_abs_path):
                pack['image'] = cur_abs_path

                if pack.get('label', None) is not None:
                    pack['label'] = join(public_root, mid_path, pack['label'])
                    if not os.path.exists(pack['label']):
                        pack.pop('label')
                return pack
    return None

def nnunet_scale(pack):
    np.clip(pack['image'], -395.0, 842.0, out=pack['image'])
    pack['image'] -= 279.8117370605469
    pack['image'] /= 253.5583953857422
    return pack


def get_image_loader(args, mode='train'):
    axes_code: str = getattr(args, 'axes_code', 'RAS')
    final_shape: tuple[int] = getattr(args, 'input_size', (256, 256, 128))
    
    if getattr(args, 'shape_mode', 'crop') == 'resize':
        spacing = (.78, .78, 1.25)
    else:
        spacing = (.39, .39, .625)
    print(f'Preprocessor Info: \n - Axes Code: {axes_code}\n - Shape: {final_shape}\n - Assumption Resolution: {spacing}')
    stem = [
        mtf.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        mtf.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),        
        mtf.Orientationd(keys=['image', 'label'], axcodes="RAS", allow_missing_keys=True)
    ]
    
    
    stem.extend([
        mtf.Spacingd(
            keys=['image', 'label'], allow_missing_keys=True, 
            pixdim=spacing, mode=('trilinear', 'nearest')
        ),
        mtf.Orientationd(keys=['image', 'label'], axcodes=axes_code, allow_missing_keys=True),
        mtf.Lambda(lambda pack: nnunet_scale(pack)),        
        mtf.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=final_shape, allow_missing_keys=True),
    ])
    
    
    
    if mode == 'train':
        stem.extend([                                             
            # Random Shit
            mtf.RandRotate90d(prob=0.5, spatial_axes=(0, 1), keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandFlipd(prob=0.10, spatial_axis=0, keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandFlipd(prob=0.10, spatial_axis=1, keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandFlipd(prob=0.10, spatial_axis=2, keys=['image', 'label'], allow_missing_keys=True),
            mtf.RandScaleIntensityd(factors=0.1, prob=0.5, keys=['image'], allow_missing_keys=True),
            mtf.RandShiftIntensityd(offsets=0.1, prob=0.5, keys=['image'], allow_missing_keys=True),
        ])
    
    if sum(isinstance(_proc, mtf.EnsureTyped) for _proc in stem) % 2 == 1:
        stem.append(mtf.EnsureTyped(device='cpu', keys=['image', 'label'], allow_missing_keys=True))
        print(f'Make sure all of tensor will back to CPU')
    stem.append(mtf.ToTensord(dtype=torch.float, keys=['image', 'label'], allow_missing_keys=True))
    
    if getattr(args, "do_jpeg", False):
        def _slicewise_range(_pack: dict[str, torch.Tensor | np.ndarray], method: callable) -> dict[str, torch.Tensor | np.ndarray]:
            z = _pack['image'].shape[-1]
            cand_image = [method(_pack['image'][..., idx]) for idx in range(z)]               
            stacker = partial(np.stack, axis=-1)
                              
            if torch.is_tensor(_pack['image']):
                stacker = partial(torch.stack, dim=-1)                                
            _pack['image'] = stacker(cand_image)
            return _pack
        return {
            'my': mtf.Compose(stem),
            'jpeg': mtf.Compose([
            mtf.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
            # mtf.EnsureTyped(keys=['image', 'label'], allow_missing_keys=True, device='cuda', data_type='tensor'),
            mtf.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
            mtf.Orientationd(keys=['image', 'label'], axcodes="RAS", allow_missing_keys=True),
            mtf.Lambda(lambda _pack: _slicewise_range(_pack, mtf.ScaleIntensity(0, 255, np.int32))),
            mtf.Spacingd(keys=['image', 'label'], pixdim=spacing, mode=('trilinear', 'nearest'),
                         allow_missing_keys=True),            
            mtf.Orientationd(keys=['image', 'label'], axcodes=axes_code, allow_missing_keys=True),
            mtf.ScaleIntensityd(keys=['image'], allow_missing_keys=True),                
            mtf.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=final_shape, allow_missing_keys=True),
            # mtf.EnsureTyped(keys=['image', 'label'], allow_missing_keys=True, device='cpu', data_type='tensor'),
            mtf.ToTensord(dtype=torch.float, keys=['image', 'label'], allow_missing_keys=True)
        ])
        }
    
    return mtf.Compose(stem)

class CardiacDataset(Dataset):
    image_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei'
    public_root = '/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei'
    def __init__(self, args, tokenizer, mode='train'):
        """
        The args must contains:
            1. proj_out_num
            2. max_length
        @param args
        """
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = '<im_patch>' * args.proj_out_num
        self.data_list = list()
        all_pack = load_jfile(join(self.public_root, f'gemini_split_{mode}.json'))
        all_pack.extend(load_jfile(join(self.public_root, f'gemini_split_{mode}_add_phase.json')))
        
        if getattr(args, 'is_promptsubset', False):
            print(f'`--is_promptsubset` is set')
            print(f'That meaning we trying to evaluate the model training on `PromptSubset`')
            new_path = 'prompt_subset_testset.json'
            all_pack = load_jfile(join(self.public_root, new_path))
        if getattr(args, 'dataset_scale', 'full') == 'd10':
            d10_name = 'gemini_split_train_1e-1.json'
            print(f'All training data will decrease 10 scale for `dataset_scale` was set to "d10".')
            all_pack = load_jfile(join(self.public_root, d10_name))
        no_content_regex = r'(none\n){0,1}<image>(\nnone){0,1}'
        drop_num = 0
        for _, pack in enumerate(all_pack):
            abs_pack: CardiacData | None = load_make_sure_exists(pack)
            if abs_pack is None:
                with open('./missing_file.txt', 'a+') as ostream:
                    ostream.write(f"File: {pack['image']} False\n")
                # print(f'Pass {idx}')
                drop_num += 1
                continue

            query, answer = abs_pack['conversations']
            q = query['value']
            a = answer['value']

            if any(value is None for value in [q, a]):
                drop_num += 1
                continue
            q = re.sub(no_content_regex, "", q.lower())
            if len(q.strip()) == 0:
                drop_num += 1
                continue
            a = re.sub(no_content_regex, "", a.lower())
            if len(a.strip()) == 0:
                drop_num += 1
                continue
            
            self.data_list.append(abs_pack)
        print(f'Size of data list: {len(self.data_list)}, Dropped: {drop_num}, Original: {len(all_pack)}')

        # with open('/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei/taipei_502_vqa.jsonl', 'r') as reader:
        #     for pack in reader.readlines():
        #         pack = json.loads(pack)
        #         pack['image'] = join(self.image_root, 'to_saturn', pack['image'])
        #         if 'label' in pack:
        #             pack['label'] = join(self.image_root, 'to_saturn', pack['label'])
        #
        #         self.data_list.append(pack)
        # with open('/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei/taipei_2897_yeh_conv.jsonl', 'r') as reader:
        # if args.do_jpeg:
        loader_pack = get_image_loader(args, mode)
        
        # self.image_loader = get_image_loader(args, mode)
        if getattr(args, "do_jpeg", False):
            self.jpeg_loader = loader_pack['jpeg']
            self.image_loader = loader_pack['my']
            uni_pid = list(set(_pack['pid'] for _pack in self.data_list))[:args.test_size]
            self.data_list = list(filter(lambda _pack: _pack['pid'] in uni_pid, self.data_list))                        
        else:
            self.image_loader = loader_pack

    def load_visual_pack(self, loader_pack, loader):        
        visual_pack = loader(loader_pack)        
        if visual_pack.get('label') is None and visual_pack.get('image') is not None:
            visual_pack['label'] = torch.zeros_like(visual_pack['image'])
        elif visual_pack.get('label') is None and visual_pack.get('image') is None:
            visual_pack['label'] = None
        return visual_pack

    def __getitem__(self, idx):
        if getattr(self, "args.do_jpeg", False):
            print(f'Load at-{idx}')
        # print(f'Start Loading {idx}')
        cur_pack = self.data_list[idx]
        # cur_pack = check_image_and_download(cur_pack)
        if cur_pack is None:
            return self.__getitem__(idx + 1)
        conv = cur_pack['conversations']
        query = list(filter(lambda conv_case: conv_case['from'] == 'human', conv))[0]['value']
        query = re.sub('\n{0,1}<image>\n{0,1}', '', query)

        answer = list(filter(lambda conv_case: conv_case['from'] == 'gpt', conv))[0]['value']
        if query is None or answer is None:
            bypass_pack: TorchCardiacData = self.__getitem__(idx + 1)
            if self.mode == 'train':
                return bypass_pack
            # In test or val mode, we should drop current pack.
            # but we also cannot return None, because it will cause error.
            # So we return empty pack. Let `data_collator` handle it.
            for key in bypass_pack.keys():
                bypass_pack[key] = None
            return bypass_pack

        question = self.image_tokens + query

        logger.info(f'question: {query}, answer: {answer}')
        if getattr(self.args, 'apply_prompt', False):   # The text format should look like chat mode.
            convs: ConversationType = [
                {'role': 'system', 'content': get_prompt()},
                {'role': 'user', 'content': question}
            ]
            if self.mode == 'train':
                convs.append({'role': 'assistant', 'content': answer})
            formatted_text = self.tokenizer.apply_chat_template(
                convs, add_generation_prompt=True,
                tokenize=False
            )
        else:   # Following Original Med3DVLM method or maybe M3D?.
            formatted_text = f"{question} {answer}"
        
        text_tensor = self.tokenizer(
            formatted_text,
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
        visual_pack = self.load_visual_pack(loader_pack, self.image_loader)
        output_pack = {
            "image": visual_pack['image'],
            'mask': visual_pack['label'],
            "input_id": input_id,
            "label": label,
            "attention_mask": attention_mask,
            "question": question,
            "answer": answer,
            'image_file': cur_pack.get('image', 'None'),
            'label_file': cur_pack.get('label', 'None')
        }

        
        if hasattr(self, 'jpeg_loader'):
            jpeg_pack = self.load_visual_pack(loader_pack, self.jpeg_loader)
            output_pack['jpeg_image'] = jpeg_pack['image']
            
        # try:
        #     visual_pack = self.image_loader(loader_pack)
        # except Exception as e:
        #     print(f'Image Loader raise error')
        #     tb.print_exc()
            
        #     if self.mode == 'train':
        #         return self.__getitem__(idx + 1)
        #     visual_pack = {'image': None, 'label': None}            
        

        # if visual_pack.get('label') is None and visual_pack.get('image') is not None:
        #     visual_pack['label'] = torch.zeros_like(visual_pack['image'])
        # elif visual_pack.get('label') is None and visual_pack.get('image') is None:
        #     visual_pack['label'] = None
        return output_pack

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
