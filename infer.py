"""
For inferencing the medical VLM. 
Because of I hard coded the data path, so you need to modify the data path in method `load_make_sure_exists`.
The data json format should be like:
[
    {
        "pid": "patient_id",
        "image": "image_name.nii.gz",
        "conversations": [
            {"from": "human", "value": "The query"},
            {"from": "gpt", "value": "The standard answer"}
        ],
        "label": "mask_name.nii.gz", # If you dont have mask, please use some segmentation model to generate it. 
            But this one could be optional.
    },
    ...
]
During method `load_make_sure_exists`, it will do 2 functions:
1. Make sure the image and mask file exists. If not, it will return a None.
2. Convert the relative path to absolute path.
The absolute path is searched in several possible root path and mid path. 
eg: <possible_root>/<mid_path>/<the target file name>
"""

import argparse
import json
import os
import random
import shutil
from copy import deepcopy
from functools import partial
from itertools import product
from os.path import join, exists
from typing import Optional, Literal

import evaluate
import pandas as pd
import jsonlines as jsonl
import numpy as np
import nibabel as nib
import torch
import transformers as HFT
from monai import transforms as MT
from monai.config.type_definitions import NdarrayOrTensor
from tqdm.auto import tqdm
from torch import nn

from src.model.llm import VLMQwenForCausalLM, LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from src.dataset import prompt_templates as PT

DEBUG: bool = os.environ.get("DEBUG", "0") == "1"

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

model: HFT.AutoModelForCausalLM | VLMQwenForCausalLM = None
tokenizer: HFT.PreTrainedTokenizer = None
image_loader: MT.Transform = None
convs_hist = list()


class DummyDebuggingModel:
    def generate(self, inputs, images=None, **kwargs):
        b = inputs.shape[0]
        return torch.randint(0, 1143, (b, 10))


def making_query(query: str, args: argparse.Namespace) -> str:
    do_system_prompt = True
    if args.system_prompt.lower() in ['true', 'false']:
        do_system_prompt = args.system_prompt.lower() == 'true'

    if not args.chat_mode and not do_system_prompt:
        return "<im_patch>" * 256 + query
    convs = list()

    if args.system_prompt.lower() == 'true':
        convs.append({"role": 'system', 'content': "You are a helpful medical assistant."})
    elif exists(args.system_prompt):    # For if user save the system message in a file.
        with open(args.system_prompt, 'r', encoding='utf-8') as reader:
            convs.append({"role": 'system', 'content': reader.read()})
    else:   # For if user directly apply the system message into `--system_prompt``
        convs.append({'role': "system", 'content': args.system_prompt})

    convs.append({'role': 'user', 'content': "<im_patch>" * 256 + query})
    
    chat: str = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    return chat


def regular_all_type_ds_into_vqa(pack_list: list[dict[str, any]], task_name=None) -> list[dict[str, any]]:
    new_pack_list = list()
    pbar = tqdm(pack_list, total=len(pack_list), desc='Reformat to vqa storage type...')
    for pack in pbar:
        if 'conversations' in pack: # VQA Task
            new_pack_list.append(pack)
            continue        
        pack['label'] = random.sample(list(set(pack['mask_pool'])), 1)[0]

        # For Position detection Task                
        # for a in 
        # query = random.sample(PT.PosREC_templates['cls_questions'], 1)[0]



        # The task name are Report Generation
        ccta = list(filter(lambda x: x['style'] == 'Original Report', pack['caption']))[0]
        query = random.sample(PT.Caption_templates, 1)[0]
        style = random.sample(PT.Caption_style, 1)[0].format("CCTA checklist")
        drop_rep = ccta['sep'].join(list(filter(lambda x: x != '[rep]', ccta['rg_template'])))        

        convs0 = [
            {'from': 'human', 'value': f'{query} {style}'},
            {'from': 'gpt', 'value': drop_rep}
        ]
        dpack = deepcopy(pack)
        dpack['conversations'] = convs0
        dpack['Question Topic'] = "W/O unvisual data"
        dpack['Answer Type'] = 'CCTA checklist'
        new_pack_list.append(dpack)

        if len(ccta['non_vis_data']) < 1:
            continue
        fpack = deepcopy(pack)
        additional_info = ccta['sep'].join(ccta['non_vis_data'])
        additional_info = random.sample(PT.NonVisData_Intros, 1)[0].format(additional_info)
        convs1 = [
            {'from': 'human', 'value': f'{additional_info}{query} {style}'},
            {'from': 'gpt', 'value': ccta['text']}
        ]
        fpack['conversations'] = convs1
        fpack['Question Topic'] = "W/ unvisual data"
        fpack['Answer Type'] = "CCTA checklist"
        new_pack_list.append(fpack)
    return new_pack_list


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


def slice_scaler(pack: dict | NdarrayOrTensor, scaler: callable):
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
    """
    Following nnUNetv2's CTNormalization.
    """
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


def load_model(dst_model_name: str, auto_loading=True):
    """
        Don't apply `auto_loading`
    """
    if auto_loading:
        try:
            __model = HFT.AutoModelForCausalLM.from_pretrained(dst_model_name, torch_dtype=torch.bfloat16,
                                                            device_map='auto', trust_remote_code=True)
            print(f'Loading from AutoModel')
        except Exception as _:
            print(f"Loading Failed, Try to use actually model class to load.")
            return load_model(dst_model_name, auto_loading=False)
    vlm_model_type = load_json(join(dst_model_name, 'config.json'))['model_type']
    if 'lamed' in vlm_model_type and 'llama' in vlm_model_type:
        llm_class = LamedLlamaForCausalLM
    elif 'lamed' in vlm_model_type and 'phi' in vlm_model_type:
        llm_class = LamedPhi3ForCausalLM
    else:
        llm_class = VLMQwenForCausalLM

    print(f'Current LLM class: {llm_class.__name__}')
    # __model = VLMQwenForCausalLM.from_pretrained(dst_model_name, torch_dtype=torch.bfloat16, device_map='auto')
    __model = llm_class.from_pretrained(dst_model_name, torch_dtype=torch.bfloat16, device_map='auto')
    
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
        text: str,  # query
        __image: Optional[torch.Tensor | str] = None,
        __mask: Optional[torch.Tensor | str] = None,
        temp: float = 0, top_p: float = .9, max_length: int = 512, do_mask: bool = True
):
    global image_loader, model    
    # text = "<im_patch>" * 256 + text  # Following training chat template.
    pack = tokenizer(text, return_tensors="pt")

    if DEBUG:
        print(f'{type(__image)}')
    vpack = dict()

    if isinstance(__image, str):
        vpack['image'] = __image

    if isinstance(__mask, str):
        vpack['label'] = __mask
    vpack = image_loader(vpack)
    __image = vpack.get('image', __image)
    __mask = vpack.get('label', __mask)    
    del vpack

    if __image is not None:
        if __image.ndim == 4:  # Adding batch_size
            __image = __image[None]
        __image = __image.to('cuda', torch.bfloat16)

    if __mask is not None:
        if __mask.ndim == 4:
            __mask = __mask[None]
        __mask = __mask.to('cuda', torch.bfloat16)
    if not do_mask:
        __mask = None

    if DEBUG:
        print("image.shape: ", __image.shape)
        if __image.shape[-1] != 128:
            breakpoint()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            inputs=pack['input_ids'].to('cuda'),
            attention_mask=pack['attention_mask'].to('cuda'),
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
        "AI": [ot if ot.endswith(".") else f'{ot}.' for ot in output_text],
        'mask_prompt': __mask is not None,  # True -> apply mask, False -> no mask
        "temp": temp,
        "top_p": top_p,
        "max_length": max_length
    }


def load_json(path) -> list[dict] | dict:
    with open(path, 'r', encoding='utf-8') as loader:
        return json.load(loader)


def load_data_json(args) -> list[dict]:
    return load_json(args.data_json_path)    


def load_exists_eval(args) -> list[dict[str, any]]:
    tmp_eval = join(args.output_dir, args.output_name.replace(".json", '.jsonl'))
    if not exists(tmp_eval):
        return list()

    with open(tmp_eval, 'r', encoding='utf-8') as loader:
        return [json.loads(line) for line in loader.readlines()]


def get_image_loader_from_args(args) -> MT.Compose:
    print(args.loader_type)
    scaler_type, arch_type, shaper_type = args.loader_type.split('-')
    basic = [
        MT.LoadImaged(keys=['image', 'label'], allow_missing_keys=True, image_only=True),
        MT.EnsureTyped(keys=['image', 'label'], allow_missing_keys=True, device='cuda'),
        MT.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        MT.Orientationd(axcodes="RAS", keys=['image', 'label'], allow_missing_keys=True)
    ]
    image_zoom_mode = 'trilinear'    
    # Normalization Preprocessing Strategy
    if scaler_type == 'unet':   # Following nnUNetv2's CTNormalization.
        basic.append(MT.Lambda(nnunet_scaler))
    elif scaler_type == 'minmax':   # Directly normalize to [0, 1]
        basic.append(MT.ScaleIntensityd(key=['image'], allow_missing_keys=True))
    elif scaler_type == 'jpeg': # For simulate convert dicom slice to jpeg and normalize to [0, 1]
        image_zoom_mode = 'bilinear'
        basic.extend([
            MT.Lambda(partial(slice_scaler, scaler=ScaleIntensity(0, 255, dtype=torch.uint8))),
            MT.Lambda(partial(slice_scaler, scaler=ScaleIntensity(dtype=torch.float)))
        ])        
    else:
        raise NotImplementedError()
    
    # Shaping Preprocessing Strategy
    if shaper_type in ['zoom', 'resize']:        
        basic.append(MT.Zoomd(zoom=.5, mode=(image_zoom_mode, 'nearest'), keys=['image', 'label'], allow_missing_keys=True))        
    if arch_type == 'med3d':
        basic.append(MT.ResizeWithPadOrCropd(keys=['image', 'label'], allow_missing_keys=True, spatial_size=(256, 256, 128)))
    elif arch_type == 'm3d':    
        basic.append(MT.ResizeWithPadOrCropd(keys=['image', 'label'], allow_missing_keys=True, spatial_size=(256, 256, 32)))
        basic.append(MT.Orientationd(keys=['image', 'label'], axcodes='SRA'))
    
    return MT.Compose(basic + [MT.ToTensord(keys=['image', 'label'], allow_missing_keys=True)])


def main(args):
    global model, tokenizer, image_loader
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(join(args.output_dir, 'config.json'), 'w+', encoding='utf-8') as dumper:
        json.dump(vars(args), dumper, indent=2)
    shutil.copyfile(__file__, join(args.output_dir, 'infer_code.py'))
    
    if DEBUG:
        print('Using dummy model to avoid loading time')
        model = DummyDebuggingModel()
    else:
        model = load_model(args.model_name)
    tokenizer = HFT.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    image_loader = get_image_loader_from_args(args)
    data_list: list[dict[str, str]] = load_data_json(args)
    data_list: list[dict[str, str]] = regular_all_type_ds_into_vqa(data_list)    
    result_list: list[dict[str, str | int | float]] = load_exists_eval(args)
    exists_pool: list[tuple[str, str, int]] = [(rpack['Question'], rpack['Answer'], rpack.get("ID", -1)) for rpack in result_list]    
    last_name = args.model_name.split('/')[-1].replace("_merged", "")
    
    task: Literal["RG", "VQA"] = 'RG' if 'cap' in args.data_json_path else 'VQA'        
    
    pbar = tqdm(
        enumerate(data_list), total=len(data_list), 
        desc=f"Task: {task}| Mask: {args.mask_prompt}|Model: {last_name}"
    )
    JsonlWriter = jsonl.open(join(args.output_dir, args.output_name.replace(".json", ".jsonl")), 'w')
    cur_i = 0
    for i, pack in pbar:

        pack = load_make_sure_exists(pack)
        if pack is None:
            continue
        query = pack['conversations'][0]['value'].replace("<image>", "").strip()
        answer = pack['conversations'][1]['value']
        if len(query) < 1 or len(answer) < 1:
            continue
        if not answer.endswith("."):
            answer = f'{answer}.'
        if any((query, answer, pack.get("ID", -1)) == _epack for _epack in exists_pool):
            continue        
        try:            
            output = asking(
                making_query(query, args), pack['image'], pack.get('label', None),
                temp=args.temp, top_p=args.top_p, max_length=args.max_length, do_mask=args.mask_prompt
            )
        except Exception:
            import traceback as tb
            tb.print_exc()
            print(f'Question: {query}')

        output_pack = {
            'Question': query,
            "Answer": answer,
            "Assistant": output.pop("AI")[0].strip(),            
            "pid": pack['pid'],
            'temp': args.temp,
            'top_p': args.top_p,
            'chat_mode': args.chat_mode,
            'system_prompt': args.system_prompt,
            'image_file': pack['image'],
            'mask_file': pack.get('label')
        }
        output_pack.update(output)
        
        if 'Answer Type' in pack:
            output_pack['Answer Type'] = pack['Answer Type']
            output_pack['Question Topic'] = pack['Question Topic']
        
        if 'ID' in pack:
            output_pack['ID'] = pack['ID']
        result_list.append(output_pack)
        JsonlWriter.write(output_pack)
    # Save pure Text
    JsonlWriter.close()
    
    with open(join(args.output_dir, args.output_name), 'w+') as saver:
        json.dump(result_list, saver, indent=2)
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
    final_path = join(args.output_dir, args.output_name.replace(".json", "_summary.json"))

    with open(final_path, 'w+', encoding='utf-8') as saver:
        json.dump(collector2save, saver, indent=2)
    df = pd.DataFrame(collector)
    df.to_csv(final_path.replace('.json', '.csv'), index=False, index_label=False)
    print(f"Done, Save at : {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--loader_type', type=str, default='unet-med3d')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_name', type=str, default='eval.json')
    parser.add_argument('--temp', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=.9)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--data_json_path', type=str, default='/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei/gemini_split_test_v2.json')
    parser.add_argument('--task', type=str, choices=['caption', 'vqa', 'bbox'])
    parser.add_argument('--mask_prompt', action='store_true', default=False)
    parser.add_argument('--chat_mode', action='store_true', default=False)
    parser.add_argument('--system_prompt', type=str, default="False", help="For using default system prompt, set it to `True`. If you don't want to use any system prompt, set it to `True`. Or you can provide a path to load your custom system prompt.")
    parser.add_argument('--taks_name', type=str, default=None, choices=['pos'])
    parser.add_argument('--local_rank', help='Ignore this')
    args = parser.parse_args()
    print(f'Your config: \n{json.dumps(vars(args), indent=2)}')
    main(args)
