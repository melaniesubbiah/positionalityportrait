import argparse
import os
import json
import pandas as pd
import evaluate
from tqdm import tqdm
from rouge_score import rouge_scorer

bertscore = evaluate.load("bertscore")
rougeL = evaluate.load("rouge")
rouge1 = rouge_scorer.RougeScorer(rouge_types=["rouge1"], use_stemmer=False)


def score(input_data):
    pred_answers, ref_answers = [], []
    scores = {}
    index_map= {}
    idx = 0
    for index, row in tqdm(input_data.iterrows(), total=len(input_data), ncols=70):
        if pd.isna(row['summary_main']):
            continue
        pred_answer = row['summary_main'].strip()
        if pred_answer == "":
            continue
        ref_answer = row['interviewee_responses'].strip()
        if ref_answer == "":
            continue
        pred_answers.append(pred_answer)
        ref_answers.append(ref_answer)

        scores[f"{row['interview_id']}"] = {
            'gender': row['gender'],
            'race': row['race']
        }
        index_map[f"{row['interview_id']}"] = idx
        idx += 1

    print("Compute ROUGE...")
    rougeL_scores = rougeL.compute(predictions=pred_answers, references=ref_answers, use_aggregator=False)['rougeL']
    rouge_scores = {'rouge1': [], 'rougeL': rougeL_scores}
    for pa, ra in zip(pred_answers, ref_answers):
       rouge_scores['rouge1'].append(rouge1.score(ra, pa)['rouge1'].precision)

    print("Computing bertscore...")
    bert_scores = bertscore.compute(predictions=pred_answers, references=ref_answers, lang="en")

    for idx, val in scores.items():
        idx = index_map[idx]
        for key in rouge_scores.keys():
            val[f"rouge-{key}"] = rouge_scores[key][idx]
        for key in bert_scores.keys():
            if key != 'hashcode':
                val[f"bertscore-{key}"] = bert_scores[key][idx]

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='summaries')
    parser.add_argument('--model', type=str, default='Llama-3.1-8B-Instruct')
    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for file in filenames:
            if 'with_demo' in file or args.model.split('/')[-1] not in file:
                continue
            prefix = os.path.join('summary_semantics', '/'.join(dirpath.split('/')[1:]))
            print('Running', prefix, file)
            os.makedirs(prefix, exist_ok=True)
            if os.path.exists(os.path.join(prefix, f"{file[:-4]}.json")):
                print("File already exists")
                continue
            input_data = pd.read_csv(os.path.join(dirpath, file))
            scores = score(input_data)
            with open(os.path.join(prefix, f"{file[:-4]}.json"), 'w') as f:
                f.write(json.dumps(scores))
