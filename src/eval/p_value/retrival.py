import argparse
import os
import random
import itertools
import functools
from collections import defaultdict
from os.path import join, exists

import numpy as np
import torch
import transformers as HFT
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.clip_dataset import CLIPDataset, CardiacCLIPDataset, TestCardiacCLIPDataset
from src.model.CLIP import *
from src.model.prompt_clip import PromptCLIP
from src.dataset.utils import myio as UIO


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


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


def calculate_recall(similarity_matrix, k, data_pool=None):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(
        similarity_matrix.device
    )
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1))
    recall_at_k = correct_matches.float().sum(dim=1).mean()
    return recall_at_k


@torch.no_grad()
@torch.inference_mode()
def extract_all_embeddings(args) -> list[dict[str, torch.Tensor]]:
    os.makedirs(args.embedding_cache_dir, exist_ok=True)
    embedding_cache_path = join(args.embedding_cache_dir, 'embedding_cache.pt')
    if exists(embedding_cache_path):
        print(f'Loading from cache_dir...')
        return torch.load(embedding_cache_path)

    tokenizer = HFT.AutoTokenizer.from_pretrained(     
        "medicalai/ClinicalBERT",
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    model: DEC_CLIP | PromptCLIP = HFT.AutoModel.from_pretrained(args.model_name, trust_remote_code=True)

    test_dataset = CardiacCLIPDataset(
        args, tokenizer, mode='test', test_size=-1, 
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collactor_drop_na
    )
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Extract Embeddings...')
    pair_pool = list()
    model.bfloat16().cuda()

    for idx, sample in pbar:
        input_id = sample["input_id"].cuda()
        attention_mask = sample["attention_mask"].cuda()
        image = sample["image"].cuda().bfloat16()
        masks = sample.get('masks', torch.zeros_like(image).cuda())

        with torch.inference_mode():
            if isinstance(model, DEC_CLIP):
                image_features = model.encode_image(image)
            else:
                image_features = model.encode_image(image, masks=masks, do_mask=args.do_mask_prompt)
            text_features = model.encode_text(input_id, attention_mask)
        pair_pool.append({
            'image': image_features.cpu(),
            'text': text_features.cpu()
        })
    
    torch.save(pair_pool, embedding_cache_path)
    return pair_pool

def main(args):
    print(f'Start to extracting embeddings...')
    embedding_pool = extract_all_embeddings(args)

    combinator = itertools.combinations(embedding_pool, args.subset_size)    
    p_value_store = defaultdict(list)

    for emb_idx, emb_subset in tqdm(enumerate(combinator), total=args.num_subset, desc="Try to calculate sampling "):
        if emb_idx >= args.num_subset:
            break
        text_pool = torch.cat([_pack['text'] for _pack in emb_subset])
        image_pool = torch.cat([_pack['image'] for _pack in emb_subset])
        scores_mat = torch.matmul(image_pool, text_pool.transpose(0, 1))
        i_to_t = calculate_recall(scores_mat, args.topk)
        t_to_i = calculate_recall(scores_mat.transpose(0, 1), args.topk)
        p_value_store[f'IR@{args.topk}'].append(i_to_t.tolist())
        p_value_store[f'TR@{args.topk}'].append(t_to_i.tolist())
    df = pd.DataFrame(p_value_store)
    value_cache_path = join(args.embedding_cache_dir, 'value_cahce.csv')
    df.to_csv(value_cache_path, index=False, index_label=False)
    print(f'Done')


    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--input_size', type=int, default=(256, 256, 128))
    # data settings
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--loader_type', type=str, default='unet-med3d-resize')
    parser.add_argument('--num_subset', type=int, default=2000)
    parser.add_argument('--subset_size', type=int, default=100)
    parser.add_argument('--output_dir', type=str)
    # evaluation settings
    parser.add_argument('--do_mask_prompt', action='store_true', default=False)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--embedding_cache_dir', type=str, default='./output/cache_dir')
    parser.add_argument('--seed', type=int, default=114514)
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)

