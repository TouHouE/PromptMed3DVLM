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
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as loader:
        return json.load(loader)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main(args):
    pred_pack = load_json(args.results)
    record_eval = {key: list() for key in ['BLEU', 'ROUGE-1', 'METEOR', 'BERT-F1', 'BERT-Recall', 'BERT-Precision']}

    for pack in tqdm(pred_pack, total=len(pred_pack), desc='Calculating Score...'):
        pack['Question'] = pack['Question'].replace("<im_patch>", "")
        decoded_preds, decoded_labels = postprocess_text([pack['Assistant']], [pack['Answer']])    
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
    with open(join(os.path.dirname(args.results), 'results_with_score_.json'), 'w+', encoding='utf-8') as writer:
        json.dump(final_pack, writer)


if __name__ == '__main__':
    print("HI")
    main(get_args())