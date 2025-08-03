import argparse
import csv
import os
import random
import itertools
import functools
from os.path import join, exists
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.dataset.clip_dataset import CLIPDataset, CardiacCLIPDataset, TestCardiacCLIPDataset
from src.model.CLIP import *
from src.model.prompt_clip import PromptCLIP
from src.dataset.utils import myio as UIO

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
        '--loader_type', type=str, default='unet-med3d-zoom'
    )
    parser.add_argument(
        '--ds_type', type=str, default='cardiac'
    )
    parser.add_argument(
        "--input_size", type=int, nargs='+', default=(256, 256, 128), help="Input size for the model."
    )
    parser.add_argument(
        '--ignore_split', action='store_true', default=False, help='Ignore the train-val-test split, merge all of those together.'
    )
    parser.add_argument(
        '--is_exp', action='store_true', default=False
    )
    parser.add_argument(
        '--do_mask_prompt', action='store_true', default=False
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="./models/Med3DVLM-DCFormer-SigLIP"
    )
    parser.add_argument("--desc", type=str, default="")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument("--data_root", type=str, default="./data/")
    parser.add_argument(
        "--cap_data_path", type=str, default=None
    )
    parser.add_argument("--output_dir", type=str, default="./output/eval/")
    parser.add_argument("--save_output", type=bool, default=False)

    parser.add_argument(
        "--test_method",
        type=str, nargs="+",
        default=(
            "recall",
            "accuracy",
        ),  # ("recall", "precision", "f1_score", "accuracy")
    )
    parser.add_argument("--test_topk", type=int, default=(1, 5, 10), nargs='+')
    parser.add_argument("--test_size", type=int, default=(100, 500, 1000, 2000), nargs='+')
    parser.add_argument('--move_to_cuda', type=bool, default=False)
    parser.add_argument('--p_value_json', type=str, help="Must indicate to a .json file the json file is a list.")
    parser.add_argument('--calculate_p_value', action='store_true', default=False, help='Enable calculate p-value required or not.')
    parser.add_argument('--p_value_size', type=int, default=-1, help="how many sampling want to use, set all(-1) to default")
    return parser.parse_args(args)


def calculate_recall(similarity_matrix, k, data_pool=None):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(
        similarity_matrix.device
    )
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1))
    recall_at_k = correct_matches.float().sum(dim=1).mean()
    return recall_at_k



def calculate_precision(similarity_matrix, k, data_pool=None):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(
        similarity_matrix.device
    )
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1))
    precision_at_k = correct_matches.float().sum() / (similarity_matrix.size(0) * k)
    return precision_at_k


def calculate_f1_score(similarity_matrix, k):
    precision = calculate_precision(similarity_matrix, k)
    recall = calculate_recall(similarity_matrix, k)

    if precision + recall == 0:
        return torch.tensor(0.0).to(similarity_matrix.device)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_accuracy(similarity_matrix, k, data_pool=None):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(
        similarity_matrix.device
    )
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1)).any(dim=1)
    accuracy = correct_matches.float().mean()    
    
    return accuracy


def data_collactor_drop_na(batches):
    """ret = {
            'image': vpack['image'],
            'text': text,
            'input_id': input_id,
            'attention_mask': attention_mask,
            'question_type': "Image_text_retrieval",
            'mask': vpack.get('label', torch.zeros_like(vpack['image']))
        }"""
    return_dict = defaultdict(list)
    for pack in batches:
        if pack['image'] is None:
            continue
        for key, value in pack.items():
            return_dict[key].append(value)
    for k, v in return_dict.items():
        # print(f'Key:{k}| ', end='')
        if torch.is_tensor(v[0]):
            return_dict[k] = torch.stack(v, 0)
            # print(f'{return_dict[k].shape}')
            continue
        # print(v)
    return return_dict


def main():
    seed_everything(42)
    args = parse_args()
    print(args)    
    if args.calculate_p_value and exists(args.p_value_json):
        _full_p_value_size = len(UIO.load_json(args.p_value_json))
        print(f'Original Full size: {_full_p_value_size}')
        if args.p_value_size < 0:
            _full_p_value_size += 1 + args.p_value_size
        else:
            _full_p_value_size = min(_full_p_value_size, args.p_value_size)
        args.p_value_size = _full_p_value_size
        print(f'Final p_value_size become: {args.p_value_size}')
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        # args.model_name_or_path,
        "medicalai/ClinicalBERT",
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    # try:
    model: DEC_CLIP | PromptCLIP = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # except Exception as e:
    #     model = DEC_CLIP(DEC_CLIPConfig.from     
    model = model.to(device=device).eval()


    # results = {}
    if args.calculate_p_value:
        iterator = itertools.product(args.test_size, range(args.p_value_size))
        results = defaultdict(list)
    else:
        iterator = args.test_size
        results = dict()

    for pack in iterator:
        print(f'Get from iterator: {type(pack)}')
        if isinstance(pack, tuple):
            test_size = pack[0]
            p_value_batch_idx = pack[1]
        else:
            test_size = pack
            p_value_batch_idx = None
        
        if args.is_exp and args.ds_type == 'cardiac':
            test_dataset = TestCardiacCLIPDataset(
                args, tokenizer=tokenizer, mode='test', test_size=test_size, p_value_batch_idx=p_value_batch_idx
            )
        elif not args.is_exp and args.ds_type == 'cardiac':
            print(f'Declare a `CardiacCLIPDataset for `--ds_type` setting as `cardiac`')
            test_dataset = CardiacCLIPDataset(
                args, tokenizer=tokenizer, mode="test", test_size=test_size, p_value_batch_idx=p_value_batch_idx
            )
        elif args.ds_type == 'm3d':
            print(f'Declare a `CLIPDataset` for `--ds_type` == "m3d"')
            test_dataset = CLIPDataset(
                args, tokenizer=tokenizer, mode='test', test_size=test_size
            )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=data_collactor_drop_na
        )

        txt_feats_all = []
        img_feats_all = []
        for sample in tqdm(test_dataloader):
            input_id = sample["input_id"].to(device=device)
            attention_mask = sample["attention_mask"].to(device=device)
            image = sample["image"].to(device=device)
            masks = sample.get('masks', torch.zeros_like(image).to(device=device))

            with torch.inference_mode():
                if isinstance(model, DEC_CLIP):
                    image_features = model.encode_image(image)
                else:
                    image_features = model.encode_image(image, masks=masks, do_mask=args.do_mask_prompt)
                text_features = model.encode_text(input_id, attention_mask)
            txt_feats_all.append(text_features.detach().cpu())
            img_feats_all.append(image_features.detach().cpu())

        txt_feats_all = torch.cat(txt_feats_all, dim=0)
        img_feats_all = torch.cat(img_feats_all, dim=0)

        scores_mat = torch.matmul(img_feats_all, txt_feats_all.transpose(0, 1))

        for test_method in args.test_method:
            for test_topk in args.test_topk:
                if test_method == "recall":
                    i_to_t = calculate_recall(scores_mat, test_topk)
                    t_to_i = calculate_recall(scores_mat.transpose(0, 1), test_topk)
                    print(f"IR_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TR_{test_topk}@{test_size}: ", t_to_i)
                    results[f'IR@{test_topk}'].append(i_to_t)
                    results[f'TR@{test_topk}'].append(t_to_i)
                    # results[f"IR_{test_topk}@{test_size}"] = i_to_t
                    # results[f"TR_{test_topk}@{test_size}"] = t_to_i
                elif test_method == "precision":
                    i_to_t = calculate_precision(scores_mat, test_topk)
                    t_to_i = calculate_precision(scores_mat.transpose(0, 1), test_topk)
                    print(f"IP_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TP_{test_topk}@{test_size}: ", t_to_i)
                    results[f"IP_{test_topk}@{test_size}"] = i_to_t
                    results[f"TP_{test_topk}@{test_size}"] = t_to_i
                elif test_method == "f1_score":
                    i_to_t = calculate_f1_score(scores_mat, test_topk)
                    t_to_i = calculate_f1_score(scores_mat.transpose(0, 1), test_topk)
                    print(f"IF1_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TF1_{test_topk}@{test_size}: ", t_to_i)
                    results[f"IF1_{test_topk}@{test_size}"] = i_to_t
                    results[f"TF1_{test_topk}@{test_size}"] = t_to_i
                elif test_method == "accuracy":
                    i_to_t = calculate_accuracy(scores_mat, test_topk)
                    t_to_i = calculate_accuracy(scores_mat.transpose(0, 1), test_topk)
                    print(f"IAcc_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TAcc_{test_topk}@{test_size}: ", t_to_i)
                    results[f"IAcc_{test_topk}@{test_size}"] = i_to_t
                    results[f"TAcc_{test_topk}@{test_size}"] = t_to_i
                else:
                    raise ValueError(f"Invalid test method: {test_method}")

    if args.save_output:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        model_name = args.model_name_or_path.split("/")[-1]
        subset_size = 'full' if args.ignore_split else 'test'
        shape_mode = args.loader_type
        model_name = f'{subset_size}_{shape_mode}_{model_name}'
        desc = f"_{args.desc}" if getattr(args, "desc", None) else ""        
        output_path = join(args.output_dir, f"p_value_{model_name}{desc}_eval_retrieval.csv")
        
        if exists(output_path):
            print(f"Eval results file already exists: {output_path}")
            do_rename = input(f'Do want to rename?[Y/n]').lower()
            if do_rename == 'y':
                output_path = input("New path >>> ")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, index_label=False)
        # with open(output_path, mode="w") as outfile:
        #     writer = csv.writer(outfile)
        #     for key, value in results.items():
        #         writer.writerow([key, f"{value.item():.4f}"])
        print(f"Save eval results successfully! at: {output_path}")


if __name__ == "__main__":
    main()
