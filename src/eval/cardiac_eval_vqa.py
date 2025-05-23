import argparse
import csv
import os
from os.path import join
import random
import json
import re

# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from monai import transforms as mtf
import accelerate as HFA

from src.dataset.mllm_dataset import VQADataset, load_make_sure_exists, CardiacDataset
from src.model.llm import VLMQwenForCausalLM
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
IMG_TOKEN = "<im_patch>"

Accelerator = HFA.Accelerator()


def get_prompt():
    return r"""If the input includes CT scans from at least two distinct cardiac phases, you can proceed with the requested calculation."""


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="./models/Med3DVLM-Qwen-2.5-7B"
    )
    parser.add_argument(
        '--apply_mask_prompt', action='store_true', default=False, help="Only enable when PromptEncoder in VLM."
    )
    parser.add_argument(
        '--apply_system_prompt', action='store_true', default=False, help="This one is enable chat mode, meaning adding <|im_start|>user...<|im_end|>... something like that."
    )
    parser.add_argument('--vision_tower', type=str, default='dcformer')
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--test_data_path", type=str, default="/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei/gemini_split_test.json"
    )
    parser.add_argument("--close_ended", action="store_true", default=False)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/eval_vqa/",
    )

    parser.add_argument("--proj_out_num", type=int, default=256)

    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, device_map="auto", trust_remote_code=True
        )
    except Exception as e:
        model = VLMQwenForCausalLM.from_pretrained(
            args.model_name_or_path, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    import accelerate as HFA
    model = Accelerator.prepare(model)
    # model = model.to('cuda')
    return model, tokenizer


def data_collator(batch):
    # images, masks, input_ids, answers, question, image_files, mask_files = [] * 7

    # for pack in batch:
    #     if pack['image'] is None:
    #         continue


    images = torch.stack([pack['image'] for pack in batch if pack['image'] is not None])
    masks = torch.stack([pack['mask'] for pack in batch if pack['image'] is not None])
    input_ids = torch.stack([pack['input_id'] for pack in batch if pack['image'] is not None])
    answers = [pack['answer'] for pack in batch if pack['image'] is not None]
    question = [pack['question'] for pack in batch if pack['image'] is not None]
    image_files = [pack['image_file'] for pack in batch if pack['image'] is not None]
    mask_files = [pack['label_file'] for pack in batch if pack['image'] is not None]

    return {
        'images': images,
        'masks': masks,
        'input_ids': input_ids,
        'answers': answers,
        'questions': question,
        'image_files': image_files,
        'mask_files': mask_files
    }


def load_test_dataset(args, tokenizer):
    return CardiacDataset(args=args, tokenizer=tokenizer, mode='test')
    with open(args.test_data_path, 'r', encoding='utf-8') as loader:
        if args.test_data_path.endswith('.jsonl'):
            pack_list = [json.loads(line) for line in loader.readlines()]
        else:
            pack_list = json.load(loader)
    with open(args.test_data_path.replace('.json', '_add_phase.json'), 'r', encoding='utf-8') as loader:
        pack_list.extend(json.load(loader))

    filted_pack = list()
    for pack in pack_list:
        pack = load_make_sure_exists(pack)
        
        if pack is None:
            continue        
        conv = pack['conversations']
        query = list(filter(lambda conv_case: conv_case['from'] == 'human', conv))[0]['value']
        if str(query).lower() == 'none' or len(str(query)) == 0:
            continue        
        if re.fullmatch(r'<image>\n.*', query) is not None:
            replace_key = '<image>\n'
        else:
            replace_key = '\n<image>'
        query = query.replace(replace_key, '')
        answer = list(filter(lambda conv_case: conv_case['from'] == 'gpt', conv))[0]['value']                
        if str(answer).lower() == 'none' or len(str(answer))  == 0:
            continue
        pack['conversations'][0]['value'] = query
        pack['conversations'][1]['value'] = answer

        filted_pack.append(pack)
    
    return filted_pack

def get_image_loader(args):
    return mtf.Compose([
            mtf.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
            mtf.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
            mtf.Orientationd(keys=['image', 'label'], axcodes="RAS", allow_missing_keys=True),
            # mtf.Lambda(lambda pack: return_print(pack, 'After Orientation')),
            mtf.Spacingd(keys=['image', 'label'], pixdim=(.4, .4, -1), mode=('trilinear', 'nearest'),
                         allow_missing_keys=True),            
            mtf.ScaleIntensityd(keys=['image', 'label'], allow_missing_keys=True),
            mtf.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=(256, 256, 128), allow_missing_keys=True),
        ])


@torch.inference_mode()
def generation(model, tokenizer, args, pack, dtype=torch.bfloat16):    
    """
    pack: {
        'images': torch.Tensor, 'masks': torch.Tensor, 'input_ids': torch.Tensor,
        'answers': list[str], 'questions': list[str], 
        'image_files': list[str], 'mask_files': list[str]
    }
    """    
    media_pack = {'images': pack.pop('images').to('cuda', dtype), 'masks': pack.pop('masks').to('cuda', dtype)}
    vision_encoder_is_promptable = any(keyname in args.vision_tower for keyname in ['mask', 'prompt'])

    if not vision_encoder_is_promptable:    # No PromptEncoder module in Vision Encoder, thus this argument should not pass
        media_pack.pop('masks')
    
    # Start process Text data    
    chat_mode: bool = False
    sys_prompt: str = ""
    raw_question = pack['questions']    

    output_ids = model.generate(                                
        inputs=pack['input_ids'].to('cuda'),
        # images=pack['images'].to('cuda', dtype),
        # masks=pack['masks'].to('cuda', dtype),
        max_new_tokens=args.max_length,
        do_sample=args.temperature > 0,
        top_p=args.top_p,
        temperature=args.temperature,
        **media_pack
    )

    output_text = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )
    output_pack = list()
    for idx, pred in enumerate(output_text):
        output_pack.append({
            'Question': pack['questions'][idx],
            'Answer': pack['answers'][idx],
            'Assistant': pred.strip(),
            'system_prompt': sys_prompt,
            'chat mode': chat_mode,
            'image_file': pack['image_files'][idx],
            'mask_file': pack['mask_files'][idx]
        })
    return output_pack

    return {
        'Question': raw_question,
        'Answer': pack['conversations'][1]['value'],
        'Assistant': output_text.strip(),
        'system_prompt': sys_prompt,
        "chat mode": chat_mode,        
    }


def evaluate_pred(pack):
    score_pack = dict()
    decoded_preds, decoded_labels = postprocess_text([pack['Assistant']], [pack['Answer']])    
    try:
        score_pack['bleu'] = bleu.compute(
            predictions=decoded_preds, references=decoded_labels, max_order=1
        )['bleu']
    except ZeroDivisionError:
        score_pack['blue'] = 0
    rouge_score = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1"],
    )
    score_pack["rouge1"] = rouge_score["rouge1"]
    meteor_score = meteor.compute(
        predictions=decoded_preds, references=decoded_labels
    )
    score_pack["meteor"] = meteor_score["meteor"]
    bert_score = bertscore.compute(
        predictions=decoded_preds, references=decoded_labels, lang="en", model_type='bert-large-uncased'
    )
    score_pack["bert_f1"] = bert_score["f1"][0]
    score_pack['bert_pr'] = bert_score['precision'][0]
    score_pack['bert_rec'] = bert_score['recall'][0]
    return score_pack

def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)
    model, tokenizer = load_model_tokenizer(args)
    image_loader = get_image_loader(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_name = args.model_name_or_path.split("/")[-1]
    
    all_vqa_pair = load_test_dataset(args, tokenizer)
    if not isinstance(all_vqa_pair, list):
        all_vqa_pair = DataLoader(all_vqa_pair, 16, collate_fn=data_collator, num_workers=16, pin_memory=True)
    final_group = list()
    for idx, vqa_pack in tqdm(enumerate(all_vqa_pair), total=len(all_vqa_pair)):
        """
            vqa_pack: {
                'images': torch.Tensor, 'masks': torch.Tensor, 'input_ids': torch.Tensor,
                'answers': list[str], 'questions': list[str], 
                'image_files': list[str], 'mask_files': list[str]
            }
        """

        output_pack = generation(
            model, tokenizer, args, vqa_pack
        )
        if isinstance(output_pack, list):
            final_group.extend(output_pack)
        else:
            final_group.append(output_pack)
    with open(join(args.output_dir, 'result.json'), 'w+', encoding='utf-8') as writer:
        json.dump(final_group, writer)
    stats = {
        key: 0 for key in ['bleu', 'rouge1', 'meteor', 'bert_f1', 'bert_pr', 'bert_rec']
    }
    n_samp = len(final_group)
    for pack in final_group:
        score_pack = evaluate_pred(pack)
        pack.update(score_pack)
        for key, value in score_pack.items():
            stats[key] += value / n_samp

    with open(join(args.output_dir, 'result_with_scores.json'), 'w+', encoding='utf-8') as writer:
        json.dump(final_group, writer)    


if __name__ == "__main__":
    main()
