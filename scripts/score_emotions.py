import argparse
import os
import json
import pandas as pd
from collections import defaultdict
import re
from tqdm import tqdm
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import numpy as np


STOPWORDS = set(stopwords.words("english"))

SCM_MODEL = 'data/SCM_embedding_final.bin'
SCM_DF = pd.read_csv(SCM_MODEL, sep='\t')
SCM_DICT = SCM_DF.set_index('word')[["warmth", "competence"]].apply(tuple, axis=1).to_dict()

with open('data/liwc.json', 'r') as f:
    LIWC = json.load(f)
for cat, term_list in LIWC.items():
    for i, term in enumerate(term_list):
        t = term.replace('*', r"[\w\']*")
        t = t.replace('(', r'\(').replace(')', r'\)')
        t = r'\b' + t + r'\b'
        LIWC[cat][i] = re.compile(t)

LIWC_CATEGORIES  = {
    'affect': ['Affect', 'Posemo', 'Negemo', 'Anx', 'Anger', 'Sad'],
    'social': ['Social', 'Family', 'Friend', 'Female', 'Male'],
    'cogproc': ['CogProc', 'Insight', 'Cause', 'Discrep', 'Tentat', 'Certain', 'Differ'],
    'percept': ['Percept', 'See', 'Hear', 'Feel'],
    'bio': ['Health', 'Bio', 'Body', 'Health', 'Sexual', 'Ingest'],
    'drives': ['Drives', 'Affiliation', 'Achieve', 'Power'],
    'motives': ['Reward', 'Risk'],
    'lifestyle': ['Work', 'Leisure', 'Home', 'Money', 'Relig', 'Death'],
}

VAD = pd.read_csv("data/NRC-VAD-Lexicon-v2.1.txt", sep="\t").set_index("term")


def get_vad_scores(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    scores = []
    for word in tokens:
        if word in VAD.index:
            scores.append(VAD.loc[word])
    vad_df = pd.DataFrame(scores)
    for val in ['valence', 'arousal', 'dominance']:
        vad_df[f"{val}_neg"] = vad_df[val].apply(lambda x: x if x < 0 else 0)
        vad_df[f"{val}_pos"] = vad_df[val].apply(lambda x: x if x > 0 else 0)
    vad_df = vad_df.mean()
    vad_scores = {}
    for val in ['valence', 'arousal', 'dominance']:
        vad_scores[val] = {
            f"{val}full": vad_df[val],
            f"{val}neg": vad_df[f"{val}_neg"],
            f"{val}pos": vad_df[f"{val}_pos"]
        }
    return vad_scores


def get_liwc_scores(text):
    text = text.lower()
    res_dict = defaultdict(int)
    wc = len(text.split())
    for CAT, keys in LIWC_CATEGORIES.items():
        res_dict[CAT] = defaultdict(int)
        for key in keys:
            for term in LIWC[key]:
                res_dict[CAT][key.lower()] += len(term.findall(text))

    final_dict = {}
    for key, val in res_dict.items():
        final_dict[key] = {}
        for k, v in val.items():
            final_dict[key][k] = v/wc
    return final_dict


def get_scm_scores(text):
    tokens = simple_preprocess(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    warmth, competence = [], []
    for word in tokens:
        word = word.replace('\n', '')
        if word in SCM_DICT:
            scm_val = SCM_DICT[word]
            warmth.append(scm_val[0])
            competence.append(scm_val[1])

    return {
        "SCM": {"warmth": np.mean(warmth), "competence": np.mean(competence)}
    }


def compute_metrics(summary):
    return {
        "liwc": get_liwc_scores(summary),
        "vad": get_vad_scores(summary),
        "scm": get_scm_scores(summary),
    }


def score(input_data):
    all_metrics = {}
    for index, row in tqdm(input_data.iterrows(), total=len(input_data), ncols=70):
        base_scores= {}
        if not pd.isna(row['summary_main']):
            summary_demo = row['summary_main'].strip()
            if summary_demo != "":
                base_scores = compute_metrics(summary_demo)
        interview = row['interviewee_responses'].strip()
        if interview == "":
            continue
        interview_scores = compute_metrics(interview)

        all_metrics[row['interview_id']] = {
            "base_scores": base_scores,
            "interview_scores": interview_scores,
            "gender": row['gender'],
            "race": row['race'],
        }

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='summaries')
    parser.add_argument('--model', type=str, default='Llama-3.1-8B-Instruct')
    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for file in filenames:
            if 'with_demo' in file or args.model.split('/')[-1] not in file:
                continue
            prefix = os.path.join('summary_emotions', '/'.join(dirpath.split('/')[1:]))
            print('Running', prefix, file)
            os.makedirs(prefix, exist_ok=True)
            if os.path.exists(os.path.join(prefix, f"{file[:-4]}.json")):
                print("File already exists")
                continue
            input_data = pd.read_csv(os.path.join(dirpath, file))
            all_metrics = score(input_data)
            with open(os.path.join(prefix, f"{file[:-4]}.json"), 'w') as f:
                f.write(json.dumps(all_metrics, indent=4))
