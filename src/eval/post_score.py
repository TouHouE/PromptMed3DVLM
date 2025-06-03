import argparse
import json
import os
from os.path import join

import evaluate
import pandas as pd
from tqdm.auto import tqdm

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str)
    parser.add_argument('--more', action='store_true', default=False)
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as loader:
        return json.load(loader)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def get_score(decoded_preds, decoded_labels, all_key):
    pack = {key: 0 for key in all_key}
    try:
        pack['BLEU'] = bleu.compute(
            predictions=decoded_preds, references=decoded_labels, max_order=1
        )['bleu']
    except ZeroDivisionError:
        pack['BLEU'] = 0
    rouge_score = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1"],
    )
    pack["ROUGE-1"] = rouge_score["rouge1"]
    meteor_score = meteor.compute(
        predictions=decoded_preds, references=decoded_labels
    )
    pack["METEOR"] = meteor_score["meteor"]
    bert_score = bertscore.compute(
        predictions=decoded_preds, references=decoded_labels, lang="en", model_type='bert-large-uncased'
    )
    pack["BERT-F1"] = bert_score["f1"][0]
    pack['BERT-Precision'] = bert_score['precision'][0]
    pack['BERT-Recall'] = bert_score['recall'][0]
    return pack


def main(args):
    pred_pack = load_json(args.results)
    raw_key = ['BLEU', 'ROUGE-1', 'METEOR', 'BERT-F1', 'BERT-Recall', 'BERT-Precision']
    if args.more:
        actual_key = raw_key + [f'JPEG_{key}' for key in raw_key]
    else:
        actual_key = raw_key
    record_eval = {key: list() for key in actual_key}

    for pack in tqdm(pred_pack, total=len(pred_pack), desc='Calculating Score...'):
        pack['Question'] = pack['Question'].replace("<im_patch>", "")
        decoded_preds, decoded_labels = postprocess_text([pack['Assistant']], [pack['Answer']])            
        pack.update(get_score(decoded_preds, decoded_labels, raw_key))
        if args.more:
            jpeg_p, jpeg_g = postprocess_text([pack['JPEG_Assistant']], [pack['Answer']])
            jpeg_pack = {f'JPEG_{key}': v for key, v in get_score(jpeg_p, jpeg_g, raw_key).items()}
            pack.update(jpeg_pack)

    for pack in tqdm(pred_pack, total=len(pred_pack), desc='Making Summary...'):
        for key in record_eval.keys():
            record_eval[key].append(pack[key])
    pdf = pd.DataFrame(record_eval, columns=list(record_eval.keys()))    
    summary_map = dict()

    for key in pdf.columns:
        df_single = pdf[key]
        summary_map[key] = df_single.describe().to_dict()


    final_pack = {
        'Summary': summary_map,
        'Cases': pred_pack
    }
    dirname = os.path.dirname(args.results)
    bname = os.path.basename(args.results)
    with open(join(dirname, bname.replace(".json", '_ReCount.json')), 'w+', encoding='utf-8') as writer:
        json.dump(final_pack, writer)


if __name__ == '__main__':
    print("HI")
    main(get_args())