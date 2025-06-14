import argparse
import json
import os

import torch
import pandas as pd
import numpy as np
from evaluate import load
from tqdm.auto import tqdm

DEBUG: bool = os.environ.get("DEBUG", "0") == "0"
bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")
bertscore = load("bertscore")


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


def main(args):
    with open(args.pred_json, 'r') as loader:
        result_list = json.load(loader)

    chunk_list = np.array_split(result_list, len(result_list) // 64)
    collector = list()
    if not args.pass_score:
        for chunk in tqdm(result_list, total=len(result_list), desc='Taking score...'):
            rep = get_score(chunk)
            collector.append(rep)
        with open(args.pred_json.replace(".json", "_cases_score.json"), 'w+') as saver:
            json.dump(collector, saver, indent=2)
    else:
        with open(args.pred_json, 'r') as loader:
            collector = json.load(loader)

    df = pd.DataFrame(collector)
    summary = dict()
    for key in ['BLEU', 'ROUGE-1', 'METEOR', 'BERT-F1', "BERT-PR", "BERT-REC"]:
        summary[key] = df[key].describe().to_dict()

    collector2save = {
        'summary': summary,
        'cases': collector
    }
    with open(args.pred_json.replace(".json", args.postfix), 'w+') as saver:
        json.dump(collector2save, saver, indent=2)

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_json', type=str)
    parser.add_argument('--pass_score', action='store_true', default=False)
    parser.add_argument('--postfix', type=str, default='_Scorer.json')
    main(parser.parse_args())