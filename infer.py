import argparse
import json
import os
from functools import partial
from itertools import product
from os.path import join, exists
from typing import Optional

import evaluate
import pandas as pd
import jsonlines as jsonl
import numpy as np
import torch
import transformers as HFT
from monai import transforms as MT
from tqdm.auto import tqdm

from src.model.llm import VLMQwenForCausalLM

DEBUG: bool = os.environ.get("DEBUG", "0") == "1"

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

model: HFT.PreTsc = None
tokenizer: HFT.PreTrainedTokenizer = None
image_loader: MT.Transform = None
convs_hist = list()


class DummyDebuggingModel:
    def generate(self, inputs, images=None, **kwargs):
        b = inputs.shape[0]

        return torch.randint(0, 1143, (b, 10))


def load_make_sure_exists(pack):
    possible_root = ['/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei',
                     '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei']
    possible_mid_path = [
        'to_saturn',  # for Taipei_502,
        'to_saturn_yeh',
        'to_saturn_beato'
    ]
    for (public_root, mid_path) in product(possible_root, possible_mid_path):
        cur_abs_path = join(public_root, mid_path, pack['image'])
        if exists(cur_abs_path):
            pack['image'] = cur_abs_path

            if 'label' not in pack:  # Key `label` not exists. Just return pack.
                return pack
            cur_abs_label = join(public_root, mid_path, pack['label'])

            if not exists(cur_abs_label):
                pack.pop('label')
                return pack
            pack['label'] = cur_abs_label
            return pack
    return None


def slice_scaler(pack, scaler):
    is_dict = isinstance(pack, dict)
    if is_dict:
        if 'image' not in pack:
            return pack
        image = pack['image']
    else:
        image = pack
    mz = image.shape[-1]
    stacker = partial(torch.stack, dim=-1) if torch.is_tensor(image) else partial(np.stack, axis=-1)
    image = stacker([scaler(image[..., z]) for z in range(mz)])
    if is_dict:
        pack['image'] = image
    else:
        pack = image
    return pack


def nnunet_scaler(pack):
    low, high = -395., 842.
    avg, std = 279.8117370605469, 253.5583953857422

    if isinstance(pack, dict):
        if 'image' not in pack:
            return pack
        np.clip(pack['image'], low, high, out=pack['image'])
        pack['image'] -= avg
        pack['image'] /= std
    else:
        torch.clip(pack, low, high, out=pack)
        pack -= avg
        pack /= std
    return pack


def load_model(dst_model_name):
    try:
        __model = HFT.AutoModelForCausalLM.from_pretrained(dst_model_name, torch_dtype=torch.bfloat16,
                                                           device_map='auto', trust_remote_code=True)
        print(f'Loading from AutoModel')
    except Exception as e:
        __model = VLMQwenForCausalLM.from_pretrained(dst_model_name, torch_dtype=torch.bfloat16, device_map='auto')
        print(f'Loading from VLMQwenForCausalLM')
    return __model


def postprocess_text(preds, labels):
    # 確保 preds 和 labels 都是單層的字串列表，並清理空白
    processed_preds = [pred.strip() for pred in preds]
    processed_labels = [label.strip() for label in labels]  # 不再嵌套列表
    return processed_preds, processed_labels


@torch.inference_mode()
def get_score(pack):
    score_pack = pack
    # postprocess_text 現在返回的是 ['Assistant_str'], ['Answer_str']
    decoded_preds, decoded_labels_for_bertscore = postprocess_text([pack['Assistant']], [pack['Answer']])

    # 為 BLEU, ROUGE, METEOR 準備 references 格式：list[list[str]]
    # 因為每個預測只有一個參考答案，所以需要將每個參考答案再包裝一層列表
    decoded_labels_for_bleu_rouge_meteor = [[label] for label in decoded_labels_for_bertscore]

    try:
        score_pack['BLEU'] = bleu.compute(
            predictions=decoded_preds,
            references=decoded_labels_for_bleu_rouge_meteor,  # 使用嵌套後的格式
            max_order=1
        )['bleu']
    except ZeroDivisionError:
        score_pack['BLEU'] = 0

    rouge_score = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels_for_bleu_rouge_meteor,  # 使用嵌套後的格式
        rouge_types=["rouge1"],
    )
    score_pack["ROUGE-1"] = rouge_score["rouge1"]

    meteor_score = meteor.compute(
        predictions=decoded_preds,
        references=decoded_labels_for_bleu_rouge_meteor  # 使用嵌套後的格式
    )
    score_pack["METEOR"] = meteor_score["meteor"]

    bert_score = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels_for_bertscore,  # 直接使用單層列表的格式
        lang="en",  # 注意：如果不是英文，這裡要修改
        model_type='bert-large-uncased'
    )
    score_pack["BERT-F1"] = bert_score["f1"][0]
    score_pack['BERT-PR'] = bert_score['precision'][0]
    score_pack['BERT-REC'] = bert_score['recall'][0]
    return score_pack


@torch.inference_mode()
def asking(
        text: str,
        __image: Optional[torch.Tensor | str] = None,
        __mask: Optional[torch.Tensor | str] = None,
        temp: float = 0, top_p: float = .9, max_length: int = 512
):
    global image_loader, model
    text = "<im_patch>" * 256 + text  # Following training chat template.
    pack = tokenizer(text, return_tensors="pt")

    if DEBUG:
        print(f'{type(__image)}')
    vpack = dict()

    if isinstance(__image, str):
        vpack['image'] = __image

    if isinstance(__mask, str):
        vpack['label'] = __mask
    vpack = image_loader(vpack)

    if 'image' in vpack:
        __image = vpack['image']

    if 'label' in vpack:
        __mask = vpack['label']
    del vpack

    if __image is not None:
        if __image.ndim == 4:  # Adding batch_size
            __image = __image[None]
        __image = __image.to('cuda', torch.bfloat16)

    if __mask is not None:
        if __mask.ndim == 4:
            __mask = __mask[None]
        __mask = __mask.to('cuda', torch.bfloat16)

    if DEBUG:
        print("image.shape: ", __image.shape)
        if __image.shape[-1] != 128:
            breakpoint()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            inputs=pack['input_ids'].to('cuda'),
            images=__image,
            masks=__mask,
            max_new_tokens=max_length,
            do_sample=temp > 0,
            top_p=top_p,
            temperature=temp,
        )
    output_text = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )
    return {
        "AI": output_text,
        "temp": temp,
        "top_p": top_p,
        "max_length": max_length
    }


def load_data_json():
    with open('/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei/gemini_split_test.json', 'r') as loader:
        return json.load(loader)


def load_exists_eval(args):
    tmp_eval = join(args.output_dir, args.output_name.replace(".json", '.jsonl'))
    if not exists(tmp_eval):
        return list()

    with open(tmp_eval, 'r') as loader:
        return [json.loads(line) for line in loader.readlines()]


def get_image_loader_from_args(args):
    print(args.loader_type)
    basic = [
        MT.LoadImaged(keys=['image', 'label'], allow_missing_keys=True, image_only=True),
        MT.EnsureTyped(keys=['image', 'label'], allow_missing_keys=True, device='cuda'),
        MT.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        MT.Orientationd(axcodes="RAS", keys=['image', 'label'], allow_missing_keys=True)
    ]
    if args.loader_type == 'unet-med3d':
        basic.extend([
            MT.Lambda(nnunet_scaler),
            MT.Zoomd(zoom=0.5,
                     mode=('trilinear', 'nearest'), keys=['image', 'label'],
                     allow_missing_keys=True
                     ),
            MT.ResizeWithPadOrCropd(spatial_size=(256, 256, 128), keys=['image', 'label'], allow_missing_keys=True)
        ])
    elif args.loader_type == 'minmax-med3d':
        basic.extend([
            MT.ScaleIntensityd(keys=['image'], allow_missing_keys=True),
            MT.Zoomd(zoom=0.5,
                     mode=('trilinear', 'nearest'), keys=['image', 'label'],
                     allow_missing_keys=True
                     ),
            MT.ResizeWithPadOrCropd(spatial_size=(256, 256, 128), keys=['image', 'label'], allow_missing_keys=True),
        ])
    elif args.loader_type == 'jpeg-med3d':
        basic.extend([
            MT.Lambda(partial(slice_scaler, scaler=MT.ScaleIntensity(0, 255, dtype=torch.int8))),
            MT.Lambda(partial(slice_scaler, scaler=MT.ScaleIntensity(dtype=torch.float))),
            MT.Zoomd(zoom=0.5,
                     mode=('bilinear', 'nearest'), keys=['image', 'label'],
                     allow_missing_keys=True
                     ),
            MT.ResizeWithPadOrCropd(spatial_size=(256, 256, 128), keys=['image', 'label'], allow_missing_keys=True)
        ])
    elif args.loader_type == 'm3d':
        basic.extend([
            MT.Lambda(partial(slice_scaler, scaler=MT.Intensity(0, 255, dtype=torch.int8))),
            MT.Lambda(partial(slice_scaler, scaler=MT.Intensity())),
            MT.Orientationd("SRA", keys=['image', 'label'], allow_missing_keys=True),
            MT.Zoomd(zoom=0.5,
                     mode=('bilinear', 'nearest'), keys=['image', 'label'],
                     allow_missing_keys=True
                     ),
            MT.ResizeWithPadOrCropd(spatial_size=(32, 256, 256), keys=['image', 'label'], allow_missing_keys=True)
        ])
    else:
        raise NotImplementedError(f"Loader-Type: {args.loader_type} not exists.")
    return MT.Compose(basic + [MT.ToTensord(keys=['image', 'label'], allow_missing_keys=True)])


def main(args):
    global model, tokenizer, image_loader
    os.makedirs(args.output_dir, exist_ok=True)
    with open(join(args.output_dir, 'config.json'), 'w+') as dumper:
        json.dump(vars(args), dumper, indent=2)
    if DEBUG:
        print(f'Using dummy model to avoid loading time')
        model = DummyDebuggingModel()
    else:
        model = load_model(args.model_name)
    tokenizer = HFT.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # image_loader = MT.Compose([
    #     MT.LoadImage(image_only=True),
    #     MT.EnsureType(device='cuda'),
    #     MT.EnsureChannelFirst(),
    #     MT.Orientation("RAS"),
    #     MT.Lambda(nnunet_scaler),
    #     MT.Zoom(0.5, mode='trilinear'),
    #     MT.ResizeWithPadOrCrop((256, 256, 128)),
    #     MT.ToTensor(),
    # ])
    image_loader = get_image_loader_from_args(args)
    data_list = load_data_json()
    result_list = load_exists_eval(args)
    exists_pool = [(rpack['Question'], rpack['Answer']) for rpack in result_list]
    # result_pack = list()
    pbar = tqdm(enumerate(data_list), total=len(data_list))
    JsonlWriter = jsonl.open(join(args.output_dir, args.output_name.replace(".json", ".jsonl")), 'w')
    for i, pack in pbar:
        pack = load_make_sure_exists(pack)
        if pack is None:
            continue
        query = pack['conversations'][0]['value'].replace("<image>", "").strip()
        answer = pack['conversations'][1]['value']
        if len(query) < 10 or len(answer) < 10:
            continue
        if any((query, answer) == _epack for _epack in exists_pool):
            continue
        try:
            output = asking(
                query, pack['image'], pack.get('label', None),
                temp=args.temp, top_p=args.top_p, max_length=args.max_length
            )
        except Exception as e:
            import traceback as tb
            tb.print_exc()
            print(f'Question: {query}')
        output_pack = {
            'Question': query,
            "Answer": answer,
            "Assistant": output["AI"][0].strip(),
            "pid": pack['pid'],
            'temp': args.temp,
            'top_p': args.top_p,
            'image_file': pack['image'],
            'mask_file': pack.get('label')
        }
        result_list.append(output_pack)
        JsonlWriter.write(output_pack)
    # Save pure Text
    with open(join(args.output_dir, args.output_name), 'w+') as saver:
        json.dump(result_list, saver, indent=2)

    # chunk_list = np.array_split(result_list, len(result_list) // 64)
    collector = list()

    for chunk in tqdm(result_list, total=len(result_list), desc='Taking score...'):
        rep = get_score(chunk)
        collector.append(rep)
    with open(join(args.output_dir, args.output_name.replace('.json', '_cases_score.json')), 'w+') as saver:
        json.dump(collector, saver, indent=2)

    df = pd.DataFrame(collector)
    summary = dict()
    for key in ['BLEU', 'ROUGE-1', 'METEOR', 'BERT-F1', "BERT-PR", "BERT-REC"]:
        summary[key] = df[key].describe().to_dict()

    collector2save = {
        'summary': summary,
        'cases': collector
    }
    with open(args.pred_json.replace(".json", "_summary.json"), 'w+') as saver:
        json.dump(collector2save, saver, indent=2)

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--loader_type', type=str, default='unet-med3d')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_name', type=str, default='eval.json')
    parser.add_argument('--temp', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=.9)
    parser.add_argument('--max_length', type=int, default=512)

    args = parser.parse_args()
    main(args)
