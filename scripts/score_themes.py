import argparse
import os
import json
import pandas as pd
from tqdm import tqdm


def score(input_data):
    scores = {}
    input_data = input_data[~input_data['summary_themes'].isna()]
    input_data = input_data[~input_data['interview_id'].isin([126, 92, 111])]
    for index, row in tqdm(input_data.iterrows(), total=len(input_data), ncols=70):
        scores[f"{row['interview_id']}"] = {
            'gender': row['gender'],
            'race': row['race'],
            'themes': row['summary_themes']
        }

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='summaries')
    parser.add_argument('--model', type=str, default='Llama-3.1-8B-Instruct')
    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for file in filenames:
            if args.model.split('/')[-1] not in file:
                continue
            prefix = os.path.join('summary_themes', '/'.join(dirpath.split('/')[1:]))
            print('Running', prefix, file)
            os.makedirs(prefix, exist_ok=True)
            if os.path.exists(os.path.join(prefix, f"{file[:-4]}.json")):
                print("File already exists")
                continue
            input_data = pd.read_csv(os.path.join(dirpath, file))
            scores = score(input_data)
            with open(os.path.join(prefix, f"{file[:-4]}.json"), 'w') as f:
                f.write(json.dumps(scores))
