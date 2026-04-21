import numpy as np
import argparse
import os
import json
import pandas as pd
from itertools import combinations
from collections import Counter, defaultdict


liwc_metrics = [
 'liwc-social_social',
 'liwc-social_family',
 'liwc-bio_health',
 'liwc-drives_drives',
 'liwc-drives_achieve',
 'liwc-drives_power',
 'liwc-lifestyle_work',
 'liwc-lifestyle_leisure',
]
vad_metrics = [
    'vad-valence_valencefull',
    'vad-arousal_arousalfull',
    'vad-dominance_dominancefull',
]
scm_metrics = [
    'scm-SCM_warmth',
    'scm-SCM_competence'
]
metrics_map = {
    'rouge1': 'Vocabulary similarity',
    'rougeL': 'Wording similarity',
    'bertscore': 'Semantic similarity',
    'affect': 'affective processes',
    'social': 'social processes',
    'family': 'family',
    'tentat': 'tentative',
    'health': 'health',
    'drives': 'drives',
    'achieve': 'achievement',
    'power': 'power',
    'work': 'work',
    'leisure': 'leisure',
    'valencefull': 'valence',
    'valenceneg': 'negative valence',
    'valencepos': 'positive valence',
    'arousalfull': 'arousal',
    'arousalneg': 'negative arousal',
    'arousalpos': 'positive arousal',
    'dominancefull': 'dominance',
    'dominanceneg': 'negative dominance',
    'dominancepos': 'positive dominance',
    'warmth': 'warmth',
    'competence': 'competence',
    'posemo': 'positive emotion',
    'negemo': 'negative emotion',
    'anx': 'anxiety',
    'anger': 'anger',
    'sad': 'sadness',
    'friend': 'friendship',
    'female': 'female mentions',
    'male': 'male mentions',
    'cogproc': 'cognitive processes',
    'insight': 'insight language',
    'cause': 'causal language',
    'discrep': 'discrepancy language',
    'certain': 'certain language',
    'differ': 'differentiating language',
    'percept': 'perceptions',
    'see': 'seeing',
    'hear': 'hearing',
    'feel': 'feeling',
    'bio': 'biological processes',
    'body': 'body processes',
    'sexual': 'sexual processes',
    'ingest': 'ingestion',
    'affiliation': 'affiliation',
    'reward': 'reward',
    'risk': 'risk',
    'home': 'home',
    'money': 'money',
    'relig': 'religion',
    'death': 'death'
}

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def bootstrap_means(group_a, group_b, n_boot=5000, increase=True):
    boot_t = np.zeros(n_boot)
    for i in range(n_boot):
        sample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        sample_b = np.random.choice(group_b, size=len(group_b), replace=True)
        boot_t[i] = sample_a.mean() - sample_b.mean()

    if not increase:
        p_value = np.mean(boot_t >= 0)
    else:
        p_value = np.mean(boot_t <= 0)

    return boot_t.mean(), p_value


def bootstrap_oneway(group_a, n_boot=5000, increase = True):
    boot_t = np.zeros(n_boot)
    for i in range(n_boot):
        sample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        boot_t[i] = sample_a.mean()

    if increase:
        p_value = np.mean(boot_t <= 0)
    else:
        p_value = np.mean(boot_t >= 0)

    return boot_t.mean(), p_value

def get_top_themes(all_values):
    all_counts = []
    for key, dfs in all_values.items():
        for df in dfs:
            temp_counts = Counter([])
            total = 0
            for idx, row in df.iterrows():
                summary_vals = eval(row['themes'])
                if isinstance(summary_vals, list) and len(summary_vals) > 0:
                    temp_counts.update(summary_vals)
                    total += 1
            all_counts.append(Counter({k: v / total for k, v in temp_counts.items()}))
    all_counts = Counter({k: np.mean([all_counts[i][k] for i in range(len(all_counts))]) for k in all_counts[0].keys()})
    return all_counts

def make_portrait(semantics, emotions, themes, themenames):
    demo_map = {
        'Blackman': 'BM',
        'Blackwoman': 'BW',
        'whitewoman': 'WW',
        'whiteman': 'WM'
    }
    sections = '{label: "Wording and semantics", winsOnly: true, rows: ['
    temp = []
    for metric, vals in semantics.items():
        counts = defaultdict(int)
        for val in vals:
            if val['increase']:
                counts[demo_map[val['more']]] += 1
            else:
                counts[demo_map[val['more']]] -= 1
        temp.append('{dim: "' + metrics_map[metric].capitalize() + '", BW: ' + str(counts['BW']) + ', WW: ' + str(counts['WW']) + ', BM: ' + str(counts['BM']) + ', WM: ' + str(counts['WM']) + '}')
    sections += ', '.join(temp) + ']},'
    sections += '{label: "Psychological States", winsOnly: true, rows: ['
    temp = []
    for metric, vals in emotions.items():
        counts = defaultdict(int)
        for val in vals:
            if val['increase']:
                counts[demo_map[val['more']]] += 1
            else:
                counts[demo_map[val['more']]] -= 1
        if len(vals) > 0:
            temp.append('{dim: "' + metrics_map[metric.split('_')[-1]].capitalize() + '", BW: ' + str(counts['BW']) + ', WW: ' + str(counts['WW']) + ', BM: ' + str(counts['BM']) + ', WM: ' + str(counts['WM']) + '}')
    sections += ', '.join(temp) + ']},'
    sections += '{label: "Themes", winsOnly: false, rows: ['
    temp = []
    for theme in themenames:
        if theme in themes:
            vals = themes[theme]
            counts = {'BW': '""', 'WW': '""', 'BM': '""', 'WM': '""'}
            for val in vals:
                if val['increase']:
                    counts[demo_map[val['more']]] = str(val['p'])
                else:
                    counts[demo_map[val['more']]] = str(-val['p'])
            temp.append('{dim: "' + theme.capitalize() + '", BW: ' + counts['BW'] + ', WW: ' + counts['WW'] + ', BM: ' + counts['BM'] + ', WM: ' + counts['WM'] + '}')
    sections += ', '.join(temp) + ']}'
    return sections + '];'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--attribute_subset', default=False, action='store_true')
    parser.add_argument('--seed', default=812649326, type=int)
    args = parser.parse_args()

    seed_everything(args.seed)

    modelname = args.model_path.split('/')[-1]

    # Semantics
    metrics = ['rouge-rouge1', 'rouge-rougeL', 'bertscore-precision']
    all_values = []
    for dirpath, dirnames, filenames in os.walk('summary_semantics'):
        for file in filenames:
            if file.startswith(f'{modelname}_baseline_'):
                with open(os.path.join(dirpath, file)) as f:
                    data = json.load(f)
                df = []
                for idx, v in data.items():
                    row = {"id": idx, 'race': v['race'], 'gender': v['gender']}
                    for metric in metrics:
                        if metric in v:
                            row[metric] = v[metric]
                    df.append(row)
                df = pd.DataFrame(df)
                all_values.append(df)

    significant_semantic_increases = defaultdict(list)
    result = (
        pd.concat(all_values)
        .groupby(["id", 'race', 'gender'], as_index=False)
        .mean(numeric_only=True)
    ).rename(columns={'rouge-rouge1': "rouge1", "rouge-rougeL": "rougeL", "bertscore-precision": 'bertscore'})
    result['demo'] = result.apply(lambda x: x['race'] + x['gender'], axis=1)
    for metric in ['rouge1', 'rougeL', 'bertscore']:
        for aa, bb in list(combinations(['Blackwoman', 'Blackman', 'whitewoman', 'whiteman'], 2)):
            for a, b in [(aa, bb), (bb, aa)]:
                increase = (result[result['demo'] == a][metric].mean() > 0)
                stat, p = bootstrap_means(result[result['demo'] == a][metric], result[result['demo'] == b][metric], increase=increase)
                if p < .05:
                    significant_semantic_increases[metric].append({
                        'more': a,
                        'less': b,
                        'p': p,
                        'increase': increase,
                    })

    # Emotions
    all_base_values = []
    all_interview_values = []
    for dirpath, dirnames, filenames in os.walk('summary_emotions'):
        for file in filenames:
            if file.startswith(f'{modelname}_baseline_'):
                with open(os.path.join(dirpath, file)) as f:
                    data = json.load(f)
                new_df_base, new_df_interview = [], []
                for idx, row in data.items():
                    if 'liwc' in row['base_scores']:
                        temp_interview = {"id": idx, 'race': row['race'], 'gender': row['gender']}
                        temp_base = {"id": idx, 'race': row['race'], 'gender': row['gender']}
                        interview_vals = row['interview_scores']
                        base_vals = row['base_scores']
                        for key, val in base_vals.items():
                            for k, v in val.items():
                                for kk, vv in v.items():
                                    if args.attribute_subset:
                                        if key == 'liwc' and f"{key}-{k}_{kk}" not in liwc_metrics:
                                            continue
                                        elif key == 'vad' and f"{key}-{k}_{kk}" not in vad_metrics:
                                            continue
                                    temp_interview[f"{key}-{k}_{kk}"] = interview_vals[key][k][kk]
                                    temp_base[f"{key}-{k}_{kk}"] = vv
                        new_df_interview.append(temp_interview)
                        new_df_base.append(temp_base)
                new_df_interview = pd.DataFrame(new_df_interview)
                new_df_base = pd.DataFrame(new_df_base)
                all_base_values.append(new_df_base)
                all_interview_values.append(new_df_interview)

    curr_metrics = [x for x in list(all_base_values[0].keys()) if x not in ['id', 'race', 'gender']]
    significant_emotion_increases = defaultdict(list)
    base_df = (
        pd.concat(all_base_values)
          .groupby(["id", 'race', 'gender'], as_index=False)
          .mean(numeric_only=True)
    ).rename(columns={col: col.replace('-', '_') for col in curr_metrics})
    interview_df = (
        pd.concat(all_interview_values)
          .groupby(["id", 'race', 'gender'], as_index=False)
          .mean(numeric_only=True)
    ).rename(columns={col: col.replace('-', '_') for col in curr_metrics})
    keys = ['id', 'race', 'gender']
    interview_df = interview_df.rename(columns={col: f"interview_{col}" for col in interview_df.columns if col not in keys})
    df = base_df.merge(interview_df, on=keys, how="left")
    df['demo'] = df.apply(lambda x: x['race'] + x['gender'], axis=1)
    for metric in curr_metrics:
        if 'VAD' in metric:
            metric = metric.lower()
        else:
            metric = metric.replace('-', '_')
        df[f"{metric}_combo"] = df.apply(lambda row: (row[metric]-row[f"interview_{metric}"])*2/(abs(row[f"interview_{metric}"]) + abs(row[metric])) if (abs(row[f"interview_{metric}"]) + abs(row[metric])) > 0 else 0., axis=1)
        for aa, bb in list(combinations(['Blackwoman', 'Blackman', 'whitewoman', 'whiteman'], 2)):
            increase = (df[f"{metric}_combo"].mean() > 0)
            for a, b in [(aa, bb), (bb, aa)]:
                stat, p = bootstrap_means(df[df['demo']==a][f"{metric}_combo"], df[df['demo']==b][f"{metric}_combo"], increase=increase)
                if p < .05:
                    significant_emotion_increases[metric].append({
                        'more': a,
                        'less': b,
                        'p': p,
                        'increase': increase
                    })

    # Themes
    all_values = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk('summary_themes'):
        for file in filenames:
            if file.startswith(f'{modelname}_'):
                method = 'baseline' if 'baseline' in file else 'with_demo'
                with open(os.path.join(dirpath, file), 'r') as f:
                    data = json.load(f)
                df = []
                for idx, v in data.items():
                    df.append({
                        "id": idx,
                        'themes': v['themes'],
                        'race': v['race'],
                        'gender': v['gender']
                    })
                df = pd.DataFrame(df)
                all_values[method].append(df)

    significant_value_increases = defaultdict(list)
    if args.attribute_subset:
        themenames = [x[0] for x in get_top_themes(all_values).most_common()[:20]]
    else:
        themenames = [x[0] for x in get_top_themes(all_values).most_common() if x[1] >= .05]

    base_df = (
        pd.concat(all_values['baseline'])
        .groupby(["id", 'race', 'gender'], as_index=False)['themes']
        .apply(lambda x: [eval(y) for y in x if not pd.isna(y)])
    )
    demo_df = (
        pd.concat(all_values['with_demo'])
        .groupby(["id", 'race', 'gender'], as_index=False)['themes']
        .apply(lambda x: [eval(y) for y in x if not pd.isna(y)])
    )
    keys = ['id', 'race', 'gender']
    demo_df = demo_df.rename(columns={col: f"demo_{col}" for col in demo_df.columns if col not in keys})
    df = base_df.merge(demo_df, on=keys, how="left")
    df['demo'] = df.apply(lambda x: x['race'] + x['gender'], axis=1)
    df = df[df['themes'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    df = df[df['demo_themes'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    for theme in themenames:
        df[f"{theme}_combo"] = df.apply(
            lambda row: np.mean([theme in x for x in row['demo_themes']]) - np.mean(
                [theme in x for x in row['themes']]),
            axis=1
        )

        for aa in ['Blackwoman', 'Blackman', 'whitewoman', 'whiteman']:
            increase = (df[df['demo'] == aa][f"{theme}_combo"].mean() > 0)
            stat, p = bootstrap_oneway(df[df['demo'] == aa][f"{theme}_combo"], increase = increase)
            if p < .05:
                significant_value_increases[theme].append({
                    'more': aa,
                    'p': p,
                    'increase': increase
                })

    # Portrait
    plots = make_portrait(significant_semantic_increases, significant_emotion_increases, significant_value_increases, themenames)
    with open('data/portrait_template1.txt', 'r') as f:
        template1 = f.read()
    with open('data/portrait_template2.txt', 'r') as f:
        template2 = f.read()
    os.makedirs('portraits', exist_ok=True)
    outfilename = modelname
    if args.attribute_subset:
        outfilename += "_subset"
    with open(f'portraits/{outfilename}.html', 'w') as f:
        f.write(template1 + plots + "\n" + template2)