import argparse
import json
import re
import os
import pandas as pd
from os.path import join, exists
from typing import Final, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from tqdm.auto import tqdm
from green_score import GREEN
from RaTEScore import RaTEScore

def load_json(path):
    with open(path, 'r') as loader:
        return json.load(loader)

def load_data_as_df(path):
    if path.endswith('.json'):
        return pd.DataFrame(load_json(path))
    elif path.endswith('.csv'):        
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def worker(model_name: str, answer_list, assistant_list, eval_dir=None):
    if 'green' in model_name.lower():
        scorer = GREEN(model_name='StanfordAIMI/GREEN-radllama2-7b', max_length=4096)
        called_scorer = lambda x, y: scorer(x, y)
    elif 'rate' in model_name.lower():
        scorer = RaTEScore()
        called_scorer = lambda x, y: scorer.compute_score(x, y)
    
    return {'model_name': model_name, 'return': called_scorer(answer_list, assistant_list)}


# cur = load_json('./output/stage2/eval_Scorer.json')

# pred_list, gt_list = list(), list()
# for pack in cur:
#     pred_list.append(pack['Assistant'].strip().rstrip('.'))
#     gt_list.append(pack['Answer'].strip().rstrip('.'))


# green_result = green(gt_list, pred_list)
# rate_result = rate.compute_score(pred_list, gt_list)

def main(args):    
    eval_dir = os.path.dirname(args.eval_path)
    print(f'Final Settings:\n{vars(args)}')
    if args.new_name is None:        
        eval_name = os.path.basename(args.eval_path)
        eval_name = re.sub(r'\.(json|csv)$', '_more_score.csv', eval_name)
    else:
        eval_name = args.new_name

    desc_dir = join(eval_dir, 'llm_report')
    total_data = load_data_as_df(args.eval_path)
    answer_list = total_data['Answer'].tolist()
    assistant_list = total_data['Assistant'].tolist()
    print("Start RaTEScoring...")
    greener = GREEN(model_name='StanfordAIMI/GREEN-radllama2-7b', max_length=4096, verbose=False)
    rater = RaTEScore()
    print(f'# of Answer: {len(answer_list)}')
    print(f'# of Assistant: {len(assistant_list)}')
    # rate_score_list = rater.compute_score(answer_list, assistant_list)
    rate_score_list = list()
    green_result_list = list()
    green_keys = ['reference', 'predictions', 'green_analysis', 'green_score',
       '(a) False report of a finding in the candidate',
       '(b) Missing a finding present in the reference',
       "(c) Misidentification of a finding's anatomic location/position",
       '(d) Misassessment of the severity of a finding',
       "(e) Mentioning a comparison that isn't in the reference",
       '(f) Omitting a comparison detailing a change from a prior study',
       'Matched Findings']

    for i, (answer_i, assistant_i) in (pbar := tqdm(
        enumerate(zip(answer_list, assistant_list)), total=len(answer_list)
    )):
        # print(f"Content: {assistant_i}")
        try:
            rate_i = rater.compute_score([answer_i], [assistant_i])            
        except IndexError:
            rate_i = 0
        try:
            green_i, *_, report_i = greener([answer_i], [assistant_i])            
            # green_keys = report_i.keys()
        except ValueError as ve:
            report_i = pd.DataFrame([{gkey: None if gkey != 'green_score' else 0 for gkey in green_keys}])
            green_i = 0
        
        
        rate_i = rate_i[0] if isinstance(rate_i, list) else rate_i
        green_result_list.append(report_i)
        pbar.set_postfix_str(f'RaTE: {rate_i:.1%} | GREEN: {green_i:.1%}')
        rate_score_list.append(rate_i)
        


    # if len(rate_score_list) != total_data.shape[0]:
    #     breakpoint()
    total_data['RaTEScore'] = rate_score_list
    try:
        report_df = pd.concat(green_result_list, axis=0).reset_index()
        total_data = pd.concat([total_data, report_df], axis=1)
    except Exception as e:
        breakpoint()
    # print("Start GREEN Scoring...")
    
    
    # *_, report_df = greener(answer_list, assistant_list)
    # total_data = pd.concat([total_data, report_df], axis=1)
    report_df.to_csv(join(eval_dir, f'GREEN_{eval_name}'), index=False, index_label=False)

    total_data.to_csv(join(eval_dir, eval_name), index=False, index_label=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--green_model', type=str, default='StanfordAIMI/GREEN-radllama2-7b')
    parser.add_argument('--do_rate', action='store_true', default=False, help='Whether to compute RaTEScore.')    
    parser.add_argument('--eval_path', type=str, help='Path to the evaluation JSON or CSV file.')
    parser.add_argument('--new_name', type=str, default=None, help='Name of the new evaluation file to save results.')
    
    args = parser.parse_args()
    main(args)